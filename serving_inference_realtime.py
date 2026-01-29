from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import torch
import torch.cuda.amp as amp
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import cv2
import numpy as np
import queue
import threading
import time
import os
import shutil
import uuid
from pathlib import Path
import subprocess
from typing import Optional, List
import pickle
import glob
import copy
import json
from tqdm import tqdm
from omegaconf import OmegaConf
from transformers import WhisperModel, VitsModel, AutoTokenizer
import tempfile
import soundfile as sf

# Import from your original modules
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor

app = FastAPI()

# ==================== Global Variables ====================
device = None
vae = None
unet = None
pe = None
whisper = None
audio_processor = None
weight_dtype = None
timesteps = None
fp = None
args = None

# TTS models
tts_model = None
tts_tokenizer = None

# Upload directory
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

TTS_OUTPUT_DIR = Path("./tts_outputs")
TTS_OUTPUT_DIR.mkdir(exist_ok=True)

# ==================== 1. CUDA Stream for Async Processing ====================
class CUDAStreamManager:
    def __init__(self, device):
        self.device = device
        self.streams = []
        
    def get_stream(self):
        if not self.streams:
            return torch.cuda.Stream(device=self.device)
        return self.streams.pop()
    
    def return_stream(self, stream):
        self.streams.append(stream)

stream_manager = None

@torch.no_grad()
def inference_optimized_v2(audio_path: str, out_vid_name: str, fps: int, 
                          cached_data: dict, batch_size: int = 20,
                          use_amp: bool = True,
                          use_async_encoding: bool = True,
                          use_gpu_encoding: bool = True):  # NEW
    """
    Ultra-optimized inference
    """
    
    avatar_path = cached_data['avatar_path']
    temp_dir = Path(avatar_path) / 'tmp' / out_vid_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting ULTRA-optimized inference")
    total_start = time.time()
    
    # Audio Processing
    start_time = time.time()
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(
        audio_path, weight_dtype=weight_dtype
    )
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, device, weight_dtype, whisper,
        librosa_length, fps=fps,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
    )
    audio_time = time.time() - start_time
    print(f"⏱️  Audio: {audio_time:.2f}s")
    
    # Inference with Mixed Precision
    video_num = len(whisper_chunks)
    frame_buffer = [None] * video_num
    
    start_time = time.time()
    gen = datagen(whisper_chunks, cached_data['input_latent_list_cycle'], batch_size)
    
    frame_idx = 0
    with torch.cuda.amp.autocast(enabled=use_amp):
        for whisper_batch, latent_batch in tqdm(gen, total=int(np.ceil(video_num / batch_size)), desc="Inference"):
            audio_feature_batch = pe(whisper_batch.to(device))
            latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)
            
            pred_latents = unet.model(
                latent_batch, timesteps, 
                encoder_hidden_states=audio_feature_batch
            ).sample
            
            pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
            recon = vae.decode_latents(pred_latents)
            
            for res_frame in recon:
                frame_buffer[frame_idx] = res_frame
                frame_idx += 1
    
    inference_time = time.time() - start_time
    print(f"⚡ Inference: {inference_time:.2f}s ({video_num} frames, {video_num/inference_time:.1f} FPS)")
    
    # Encoding
    if use_async_encoding:
        encode_time = parallel_encode_video_gpu(
            frame_buffer, cached_data, temp_dir, 
            audio_path, out_vid_name, fps
        )
    else:
        encode_time = standard_encode_video(
            frame_buffer, cached_data, temp_dir,
            audio_path, out_vid_name, fps
        )
    
    total_time = time.time() - total_start
    
    print(f"💾 Encoding: {encode_time:.2f}s")
    print(f"📊 Total: {total_time:.2f}s")
    print(f"🎯 Breakdown: Audio={audio_time:.1f}s | Inference={inference_time:.1f}s | Encode={encode_time:.1f}s")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    output_vid = Path(cached_data['video_out_path']) / f"{out_vid_name}.mp4"
    
    # Verify file exists
    if not output_vid.exists():
        raise Exception(f"Video file not created: {output_vid}")
    
    return str(output_vid)

# ==================== 3. Parallel Frame Processing ====================
def process_single_frame(args_tuple):
    """Process a single frame (for parallel processing)"""
    idx, res_frame, bbox, ori_frame, mask, mask_crop_box = args_tuple
    
    x1, y1, x2, y2 = bbox
    res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
    combine_frame = get_image_blending(ori_frame, res_frame_resized, bbox, mask, mask_crop_box)
    
    return idx, combine_frame

def parallel_encode_video_optimized_final(frame_buffer, cached_data, temp_dir, 
                                         audio_path, out_vid_name, fps):
    """
    FINAL optimized version - fastest possible
    """
    start_time = time.time()
    output_vid = Path(cached_data['video_out_path']) / f"{out_vid_name}.mp4"
    output_vid.parent.mkdir(parents=True, exist_ok=True)
    
    print("🎬 Processing & encoding...")
    
    # Step 1: Parallel frame processing
    processing_args = []
    for idx, res_frame in enumerate(frame_buffer):
        bbox = cached_data['coord_list_cycle'][idx % len(cached_data['coord_list_cycle'])]
        ori_frame = cached_data['frame_list_cycle'][idx % len(cached_data['frame_list_cycle'])].copy()
        mask = cached_data['mask_list_cycle'][idx % len(cached_data['mask_list_cycle'])]
        mask_crop_box = cached_data['mask_coords_list_cycle'][idx % len(cached_data['mask_coords_list_cycle'])]
        processing_args.append((idx, res_frame, bbox, ori_frame, mask, mask_crop_box))
    
    num_workers = min(16, mp.cpu_count())
    processed_frames = [None] * len(frame_buffer)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_single_frame, processing_args))
    
    for idx, frame in results:
        processed_frames[idx] = frame
    
    height, width = processed_frames[0].shape[:2]
    
    # Step 2: Direct video write with OpenCV (fastest for small videos)
    temp_video_raw = temp_dir / "video_raw.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    writer = cv2.VideoWriter(
        str(temp_video_raw),
        fourcc,
        fps,
        (width, height),
        True
    )
    
    for frame in processed_frames:
        writer.write(frame)
    writer.release()
    
    # Step 3: Add audio (fast copy)
    cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-i', str(temp_video_raw),
        '-i', str(audio_path),
        '-c:v', 'copy',  # No re-encode!
        '-c:a', 'aac',
        '-b:a', '128k',
        '-shortest',
        '-movflags', '+faststart',
        str(output_vid)
    ]
    
    subprocess.run(cmd, check=True, capture_output=True)
    temp_video_raw.unlink()
    
    print(f"✅ Done: {output_vid}")
    return time.time() - start_time

def parallel_encode_video_gpu(frame_buffer, cached_data, temp_dir, 
                              audio_path, out_vid_name, fps):
    """
    Optimized encoding with GPU fallback
    """
    start_time = time.time()
    output_vid = Path(cached_data['video_out_path']) / f"{out_vid_name}.mp4"
    output_vid.parent.mkdir(parents=True, exist_ok=True)
    
    print("🎬 Processing frames in parallel...")
    
    # Parallel frame processing
    processing_args = []
    for idx, res_frame in enumerate(frame_buffer):
        bbox = cached_data['coord_list_cycle'][idx % len(cached_data['coord_list_cycle'])]
        ori_frame = cached_data['frame_list_cycle'][idx % len(cached_data['frame_list_cycle'])].copy()
        mask = cached_data['mask_list_cycle'][idx % len(cached_data['mask_list_cycle'])]
        mask_crop_box = cached_data['mask_coords_list_cycle'][idx % len(cached_data['mask_coords_list_cycle'])]
        processing_args.append((idx, res_frame, bbox, ori_frame, mask, mask_crop_box))
    
    num_workers = min(12, mp.cpu_count())
    processed_frames = [None] * len(frame_buffer)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(process_single_frame, processing_args),
            total=len(processing_args),
            desc="Processing frames"
        ))
    
    for idx, frame in results:
        processed_frames[idx] = frame
    
    height, width = processed_frames[0].shape[:2]
    
    print("🎬 Encoding video...")
    
    # Direct one-step encoding (fastest)
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', 'pipe:0',  # Read from stdin
        '-i', str(audio_path),
        '-c:v', 'libx264',
        '-preset', 'faster',  # balanced speed/quality
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        '-shortest',
        '-movflags', '+faststart',
        '-y',
        str(output_vid)
    ]
    
    try:
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        
        # Write all frames
        for frame in processed_frames:
            try:
                process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                break
        
        # Close stdin and wait
        try:
            process.stdin.close()
        except:
            pass
        
        process.wait(timeout=30)
        
        if process.returncode != 0:
            stderr = process.stderr.read().decode()
            raise Exception(f"FFmpeg failed: {stderr[:200]}")
            
        print(f"✅ Video encoded: {output_vid}")
        
    except Exception as e:
        print(f"❌ Encoding error: {e}")
        raise
    
    return time.time() - start_time

def standard_encode_video(frame_buffer, cached_data, temp_dir,
                         audio_path, out_vid_name, fps):
    """Standard encoding (original method)"""
    start_time = time.time()
    output_vid = Path(cached_data['video_out_path']) / f"{out_vid_name}.mp4"
    
    # Process frames
    processed_frames = []
    for idx, res_frame in enumerate(tqdm(frame_buffer, desc="Processing frames")):
        bbox = cached_data['coord_list_cycle'][idx % len(cached_data['coord_list_cycle'])]
        ori_frame = cached_data['frame_list_cycle'][idx % len(cached_data['frame_list_cycle'])].copy()
        x1, y1, x2, y2 = bbox
        
        res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        mask = cached_data['mask_list_cycle'][idx % len(cached_data['mask_list_cycle'])]
        mask_crop_box = cached_data['mask_coords_list_cycle'][idx % len(cached_data['mask_coords_list_cycle'])]
        
        combine_frame = get_image_blending(ori_frame, res_frame_resized, bbox, mask, mask_crop_box)
        processed_frames.append(combine_frame)
    
    # Write video
    temp_video = temp_dir / "temp_video.mp4"
    height, width = processed_frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
    
    for frame in processed_frames:
        out_writer.write(frame)
    out_writer.release()
    
    # Add audio
    cmd = f'ffmpeg -y -v warning -i "{temp_video}" -i "{audio_path}" ' \
          f'-c:v libx264 -preset ultrafast -crf 23 -c:a aac "{output_vid}"'
    subprocess.run(cmd, shell=True, check=True)
    
    return time.time() - start_time

# ==================== 4. Batch Size Auto-tuning ====================
def get_optimal_batch_size(gpu_memory_gb: float) -> int:
    """
    Auto-tune batch size based on available GPU memory
    """
    if gpu_memory_gb >= 24:  # RTX 3090, A5000
        return 32
    elif gpu_memory_gb >= 16:  # RTX 4080, A4000
        return 24
    elif gpu_memory_gb >= 12:  # RTX 3060, 4060Ti
        return 20
    elif gpu_memory_gb >= 8:   # RTX 3050, 4060
        return 16
    else:
        return 12

# ==================== 5. Model Compilation (PyTorch 2.0+) ====================
def compile_models_if_available():
    """
    Use torch.compile for 20-30% speedup (PyTorch 2.0+)
    Only compile models that are compatible
    """
    global unet, vae, pe
    
    if not hasattr(torch, 'compile'):
        print("⚠️  torch.compile not available (PyTorch < 2.0)")
        return
    
    print("🔥 Compiling compatible models with torch.compile...")
    
    try:
        # Try compiling VAE (usually safe)
        vae.vae = torch.compile(vae.vae, mode='reduce-overhead')
        print("   ✅ VAE compiled")
    except Exception as e:
        print(f"   ⚠️  VAE compilation failed: {e}")
    
    try:
        # Try compiling PE (usually safe)
        pe = torch.compile(pe, mode='reduce-overhead')
        print("   ✅ PE compiled")
    except Exception as e:
        print(f"   ⚠️  PE compilation failed: {e}")
    
    # Skip UNet compilation - it has dynamic shapes
    print("   ⏭️  Skipping UNet (dynamic shapes)")
    
    print("✅ Model compilation completed")

# ==================== GPU Cache Manager ====================
class AvatarCache:
    def __init__(self):
        self.cached_avatars = {}
        self.device = None
        
    def cache_avatar(self, avatar_id: str, avatar_data: dict):
        """Cache avatar data in GPU memory"""
        if avatar_id in self.cached_avatars:
            print(f"⚠️ Avatar {avatar_id} already cached, updating...")
            
        # Move latents to GPU and keep them there
        cached_data = {
            'input_latent_list_cycle': [
                lat.to(self.device) if not lat.is_cuda else lat 
                for lat in avatar_data['input_latent_list_cycle']
            ],
            'coord_list_cycle': avatar_data['coord_list_cycle'],
            'frame_list_cycle': avatar_data['frame_list_cycle'],
            'mask_list_cycle': avatar_data['mask_list_cycle'],
            'mask_coords_list_cycle': avatar_data['mask_coords_list_cycle'],
            'avatar_path': avatar_data['avatar_path'],
            'video_out_path': avatar_data['video_out_path']
        }
        
        self.cached_avatars[avatar_id] = cached_data
        
        # Calculate GPU memory usage
        gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"✓ Avatar {avatar_id} cached in GPU (Total GPU: {gpu_mem:.2f}GB)")
        
    def get_avatar(self, avatar_id: str):
        return self.cached_avatars.get(avatar_id)
    
    def list_avatars(self):
        return list(self.cached_avatars.keys())
    
    def clear_cache(self, avatar_id: Optional[str] = None):
        if avatar_id:
            if avatar_id in self.cached_avatars:
                del self.cached_avatars[avatar_id]
                torch.cuda.empty_cache()
                print(f"✓ Cache cleared for {avatar_id}")
        else:
            self.cached_avatars.clear()
            torch.cuda.empty_cache()
            print("✓ All cache cleared")

avatar_cache = AvatarCache()

# ==================== TTS Functions ====================
@torch.no_grad()
def synthesize_speech(text: str, output_path: str) -> dict:
    """
    Synthesize speech from text using Facebook MMS-TTS
    
    Returns:
        dict with 'audio_path', 'duration', 'latency'
    """
    print(f"🎤 Synthesizing: '{text[:50]}...'")
    
    start_time = time.time()
    
    # Tokenize
    inputs = tts_tokenizer(text, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        output = tts_model(**inputs)
        waveform = output.waveform
    
    latency = time.time() - start_time
    
    # Convert to numpy
    audio = waveform.squeeze().cpu().numpy()
    
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.95
    
    # Save
    sf.write(output_path, audio, tts_model.config.sampling_rate)
    
    duration = len(audio) / tts_model.config.sampling_rate
    
    print(f"✅ TTS completed: {latency:.2f}s latency, {duration:.2f}s audio")
    
    return {
        'audio_path': output_path,
        'duration': duration,
        'latency': latency,
        'sample_rate': tts_model.config.sampling_rate
    }

# ==================== Helper Functions ====================
def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    """Save uploaded file to destination"""
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return destination
    finally:
        upload_file.file.close()

def video2imgs(vid_path, save_path, ext='.png', cut_frame=10000000):
    """Extract frames from video"""
    cap = cv2.VideoCapture(str(vid_path))
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break
    cap.release()

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path, exist_ok=True)

# ==================== Avatar Preparation Class ====================
class AvatarPreparation:
    """Handles avatar preparation and data loading"""
    
    def __init__(self, avatar_id, video_path, bbox_shift, version="v15"):
        self.avatar_id = avatar_id
        self.video_path = video_path
        self.bbox_shift = bbox_shift
        self.version = version
        
        if version == "v15":
            self.base_path = f"./results/{version}/avatars/{avatar_id}"
        else:
            self.base_path = f"./results/avatars/{avatar_id}"
            
        self.avatar_path = self.base_path
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        
    def prepare(self, force_recreate=False):
        """Prepare avatar data"""
        
        if os.path.exists(self.avatar_path) and not force_recreate:
            print(f"✓ Avatar {self.avatar_id} already exists, loading...")
            return self.load_existing()
        
        print(f"🔨 Creating avatar: {self.avatar_id}")
        
        if os.path.exists(self.avatar_path):
            shutil.rmtree(self.avatar_path)
            
        osmakedirs([
            self.avatar_path, 
            self.full_imgs_path, 
            self.video_out_path, 
            self.mask_out_path
        ])
        
        # Save avatar info
        avatar_info = {
            "avatar_id": self.avatar_id,
            "video_path": str(self.video_path),
            "bbox_shift": self.bbox_shift,
            "version": self.version
        }
        with open(self.avatar_info_path, "w") as f:
            json.dump(avatar_info, f)
        
        # Extract frames
        print("📹 Extracting frames...")
        if os.path.isfile(self.video_path):
            video2imgs(self.video_path, self.full_imgs_path, ext='png')
        else:
            files = [f for f in sorted(os.listdir(self.video_path)) if f.endswith('.png')]
            for filename in files:
                shutil.copyfile(
                    f"{self.video_path}/{filename}", 
                    f"{self.full_imgs_path}/{filename}"
                )
        
        # Get landmarks and bbox
        print("🎯 Extracting landmarks...")
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        
        # Generate latents
        print("🧮 Generating latents...")
        input_latent_list = []
        coord_placeholder = (0.0, 0.0, 0.0, 0.0)
        
        for idx, (bbox, frame) in enumerate(zip(coord_list, frame_list)):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            
            if self.version == "v15":
                y2 = y2 + args.extra_margin
                y2 = min(y2, frame.shape[0])
                coord_list[idx] = [x1, y1, x2, y2]
                
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)
        
        # Create cycles
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        
        # Generate masks
        print("🎭 Generating masks...")
        mask_coords_list_cycle = []
        mask_list_cycle = []
        
        for i, frame in enumerate(tqdm(frame_list_cycle)):
            cv2.imwrite(f"{self.full_imgs_path}/{str(i).zfill(8)}.png", frame)
            
            x1, y1, x2, y2 = coord_list_cycle[i]
            mode = args.parsing_mode if self.version == "v15" else "raw"
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=mode)
            
            cv2.imwrite(f"{self.mask_out_path}/{str(i).zfill(8)}.png", mask)
            mask_coords_list_cycle.append(crop_box)
            mask_list_cycle.append(mask)
        
        # Save data
        print("💾 Saving data...")
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(mask_coords_list_cycle, f)
        with open(self.coords_path, 'wb') as f:
            pickle.dump(coord_list_cycle, f)
        torch.save(input_latent_list_cycle, self.latents_out_path)
        
        print(f"✅ Avatar {self.avatar_id} prepared successfully")
        
        return {
            'input_latent_list_cycle': input_latent_list_cycle,
            'coord_list_cycle': coord_list_cycle,
            'frame_list_cycle': frame_list_cycle,
            'mask_list_cycle': mask_list_cycle,
            'mask_coords_list_cycle': mask_coords_list_cycle,
            'avatar_path': self.avatar_path,
            'video_out_path': self.video_out_path
        }
    
    def load_existing(self):
        """Load existing avatar data"""
        
        input_latent_list_cycle = torch.load(self.latents_out_path)
        
        with open(self.coords_path, 'rb') as f:
            coord_list_cycle = pickle.load(f)
        
        input_img_list = sorted(
            glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        frame_list_cycle = read_imgs(input_img_list)
        
        with open(self.mask_coords_path, 'rb') as f:
            mask_coords_list_cycle = pickle.load(f)
        
        input_mask_list = sorted(
            glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]')),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )
        mask_list_cycle = read_imgs(input_mask_list)
        
        return {
            'input_latent_list_cycle': input_latent_list_cycle,
            'coord_list_cycle': coord_list_cycle,
            'frame_list_cycle': frame_list_cycle,
            'mask_list_cycle': mask_list_cycle,
            'mask_coords_list_cycle': mask_coords_list_cycle,
            'avatar_path': self.avatar_path,
            'video_out_path': self.video_out_path
        }

# ==================== Optimized Inference ====================
@torch.no_grad()
def inference_optimized(audio_path: str, out_vid_name: str, fps: int, 
                       cached_data: dict, batch_size: int = 20):
    """Optimized inference using cached GPU data"""
    
    avatar_path = cached_data['avatar_path']
    temp_dir = Path(avatar_path) / 'tmp' / out_vid_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting optimized inference")
    total_start = time.time()
    
    # ============ Audio Processing ============
    start_time = time.time()
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(
        audio_path, weight_dtype=weight_dtype
    )
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features, device, weight_dtype, whisper,
        librosa_length, fps=fps,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
    )
    audio_time = time.time() - start_time
    print(f"⏱️  Audio: {audio_time:.2f}s")
    
    # ============ Inference ============
    video_num = len(whisper_chunks)
    frame_buffer = [None] * video_num
    
    start_time = time.time()
    gen = datagen(whisper_chunks, cached_data['input_latent_list_cycle'], batch_size)
    
    frame_idx = 0
    for whisper_batch, latent_batch in tqdm(gen, total=int(np.ceil(video_num / batch_size)), desc="Inference"):
        audio_feature_batch = pe(whisper_batch.to(device))
        latent_batch = latent_batch.to(device=device, dtype=unet.model.dtype)
        
        pred_latents = unet.model(
            latent_batch, timesteps, 
            encoder_hidden_states=audio_feature_batch
        ).sample
        
        pred_latents = pred_latents.to(device=device, dtype=vae.vae.dtype)
        recon = vae.decode_latents(pred_latents)
        
        for res_frame in recon:
            frame_buffer[frame_idx] = res_frame
            frame_idx += 1
    
    inference_time = time.time() - start_time
    print(f"⚡ Inference: {inference_time:.2f}s ({video_num} frames, {video_num/inference_time:.1f} FPS)")
    
    # ============ Video Encoding ============
    start_time = time.time()
    output_vid = Path(cached_data['video_out_path']) / f"{out_vid_name}.mp4"
    
    # Process frames
    processed_frames = []
    for idx, res_frame in enumerate(tqdm(frame_buffer, desc="Processing frames")):
        bbox = cached_data['coord_list_cycle'][idx % len(cached_data['coord_list_cycle'])]
        ori_frame = cached_data['frame_list_cycle'][idx % len(cached_data['frame_list_cycle'])].copy()
        x1, y1, x2, y2 = bbox
        
        res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
        mask = cached_data['mask_list_cycle'][idx % len(cached_data['mask_list_cycle'])]
        mask_crop_box = cached_data['mask_coords_list_cycle'][idx % len(cached_data['mask_coords_list_cycle'])]
        
        combine_frame = get_image_blending(ori_frame, res_frame_resized, bbox, mask, mask_crop_box)
        processed_frames.append(combine_frame)
    
    # Write video
    temp_video = temp_dir / "temp_video.mp4"
    height, width = processed_frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
    
    for frame in processed_frames:
        out_writer.write(frame)
    out_writer.release()
    
    # Add audio
    cmd = f'ffmpeg -y -v warning -i "{temp_video}" -i "{audio_path}" ' \
          f'-c:v libx264 -preset ultrafast -crf 23 -c:a aac "{output_vid}"'
    subprocess.run(cmd, shell=True, check=True)
    
    encode_time = time.time() - start_time
    total_time = time.time() - total_start
    
    print(f"💾 Encoding: {encode_time:.2f}s")
    print(f"📊 Total: {total_time:.2f}s")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    return str(output_vid)

# ==================== FastAPI Startup ====================
@app.on_event("startup")
async def startup_event():
    """Initialize models with optimizations"""
    global device, vae, unet, pe, whisper, audio_processor, weight_dtype, timesteps, fp, args
    global tts_model, tts_tokenizer, stream_manager
    
    print("🚀 Initializing models with optimizations...")
    
    # Setup args
    args = type('Args', (), {
        'version': 'v15',
        'gpu_id': 0,
        'batch_size': 20,
        'fps': 25,
        'audio_padding_length_left': 2,
        'audio_padding_length_right': 2,
        'extra_margin': 10,
        'parsing_mode': 'jaw',
        'left_cheek_width': 90,
        'right_cheek_width': 90,
        'vae_type': 'sd-vae',
        'unet_config': './models/musetalk/musetalk.json',
        'unet_model_path': './models/musetalk/pytorch_model.bin',
        'whisper_dir': './models/whisper',
    })()
    
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    avatar_cache.device = device
    
    print(f"📱 Using device: {device}")
    
    # Initialize CUDA stream manager
    if torch.cuda.is_available():
        stream_manager = CUDAStreamManager(device)
    
    # Load MuseTalk models
    print("Loading MuseTalk models...")
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    
    timesteps = torch.tensor([0], device=device)
    pe = pe.half().to(device)
    vae.vae = vae.vae.half().to(device)
    unet.model = unet.model.half().to(device)
    
    # Enable cuDNN optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Compile models
    compile_models_if_available()
    
    # Audio processor and Whisper
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    # Face parsing
    if args.version == "v15":
        fp = FaceParsing(
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width
        )
    else:
        fp = FaceParsing()
    
    # Load TTS model
    print("Loading TTS model...")
    tts_start = time.time()
    
    model_name = "facebook/mms-tts-vie"
    tts_tokenizer = AutoTokenizer.from_pretrained(model_name)
    tts_model = VitsModel.from_pretrained(model_name).to(device)
    tts_model.eval()
    
    tts_load_time = time.time() - tts_start
    
    # Auto-tune batch size
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
        optimal_batch = get_optimal_batch_size(gpu_mem)
        args.batch_size = optimal_batch
        print(f"🎯 Auto-tuned batch size: {optimal_batch} (GPU: {gpu_mem:.1f}GB)")
    
    print(f"✅ All models loaded")
    print(f"   TTS load time: {tts_load_time:.2f}s")
    print(f"   TTS sample rate: {tts_model.config.sampling_rate}Hz")
    
    # Warmup
    print("🔥 Warming up models...")
    warmup_inference()

def warmup_inference():
    """Warmup inference"""
    try:
        # Use realistic shapes for warmup
        dummy_audio = torch.randn(1, 1, 80, 100).to(device=device, dtype=weight_dtype)
        
        # For UNet, use a realistic batch from datagen
        with torch.no_grad():
            # Warmup PE
            audio_features = pe(dummy_audio)
            
            # Warmup UNet with realistic latent size
            dummy_latents = torch.randn(1, 4, 32, 32).to(device=device, dtype=unet.model.dtype)
            
            # Don't use torch.compile for warmup if it causes issues
            _ = unet.model(
                dummy_latents, 
                timesteps, 
                encoder_hidden_states=audio_features
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print("✅ Warmup completed")
    except Exception as e:
        print(f"⚠️  Warmup failed (non-critical): {str(e)[:100]}")

@app.post("/prepare_avatar")
async def prepare_avatar(
    avatar_id: str = Form(...),
    video_file: UploadFile = File(...),
    bbox_shift: int = Form(0),
    force_recreate: bool = Form(False)
):
    """
    Prepare and cache avatar from uploaded video
    
    - **avatar_id**: Unique identifier for the avatar
    - **video_file**: Video file (mp4, avi, etc.)
    - **bbox_shift**: Bounding box shift value (default: 0)
    - **force_recreate**: Force recreate if avatar exists (default: False)
    """
    try:
        # Save uploaded video
        video_filename = f"{avatar_id}_{video_file.filename}"
        video_path = UPLOAD_DIR / video_filename
        save_upload_file(video_file, video_path)
        
        print(f"📥 Video uploaded: {video_path}")
        
        # Prepare avatar
        prep = AvatarPreparation(
            avatar_id=avatar_id,
            video_path=str(video_path),
            bbox_shift=bbox_shift,
            version=args.version
        )
        
        avatar_data = prep.prepare(force_recreate=force_recreate)
        avatar_cache.cache_avatar(avatar_id, avatar_data)
        
        return {
            "status": "success",
            "message": f"Avatar {avatar_id} prepared and cached",
            "avatar_id": avatar_id,
            "video_file": video_filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    output_name: Optional[str] = Form(None)
):
    """
    Convert text to speech using Facebook MMS-TTS Vietnamese
    
    - **text**: Vietnamese text to synthesize
    - **output_name**: Optional output filename (without extension)
    
    Returns: Audio file download
    """
    try:
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Generate output filename
        output_name = output_name or str(uuid.uuid4())
        output_path = TTS_OUTPUT_DIR / f"{output_name}.wav"
        
        # Synthesize speech
        result = synthesize_speech(text, str(output_path))
        
        return {
            "status": "success",
            "audio_path": str(output_path),
            "duration": result['duration'],
            "latency": result['latency'],
            "sample_rate": result['sample_rate'],
            "text": text,
            "text_length": len(text),
            "download_url": f"/download_audio/{output_name}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")

@app.post("/inference")
async def run_inference(
    avatar_id: str = Form(...),
    audio_file: UploadFile = File(...),
    output_name: Optional[str] = Form(None),
    fps: int = Form(25),
    batch_size: int = Form(20)
):
    """
    Run inference with uploaded audio file
    
    - **avatar_id**: ID of the cached avatar
    - **audio_file**: Audio file (wav, mp3, etc.)
    - **output_name**: Name for output video (optional)
    - **fps**: Frames per second (default: 25)
    - **batch_size**: Batch size for inference (default: 20)
    """
    try:
        # Check if avatar is cached
        cached_data = avatar_cache.get_avatar(avatar_id)
        if cached_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Avatar {avatar_id} not cached. Please prepare it first."
            )
        
        # Save uploaded audio
        audio_filename = f"{uuid.uuid4()}_{audio_file.filename}"
        audio_path = UPLOAD_DIR / audio_filename
        save_upload_file(audio_file, audio_path)
        
        print(f"📥 Audio uploaded: {audio_path}")
        
        # Generate output name
        output_name = output_name or str(uuid.uuid4())
        
        # Run inference
        output_path = inference_optimized(
            audio_path=str(audio_path),
            out_vid_name=output_name,
            fps=fps,
            cached_data=cached_data,
            batch_size=batch_size
        )
        
        # Clean up uploaded audio
        audio_path.unlink()
        
        return {
            "status": "success",
            "output_path": output_path,
            "output_name": output_name,
            "avatar_id": avatar_id,
            "download_url": f"/download/{avatar_id}/{output_name}"
        }
    except Exception as e:
        # Clean up on error
        if audio_path and audio_path.exists():
            audio_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/download/{avatar_id}/{video_name}")
async def download_video(avatar_id: str, video_name: str):
    """Download generated video"""
    video_path = Path(f"./results/{args.version}/avatars/{avatar_id}/vid_output/{video_name}.mp4")
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        video_path, 
        media_type="video/mp4", 
        filename=f"{video_name}.mp4",
        headers={"Content-Disposition": f"attachment; filename={video_name}.mp4"}
    )

@app.get("/download_audio/{audio_name}")
async def download_audio(audio_name: str):
    """Download generated audio file"""
    audio_path = TTS_OUTPUT_DIR / f"{audio_name}.wav"
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"{audio_name}.wav",
        headers={"Content-Disposition": f"attachment; filename={audio_name}.wav"}
    )

@app.post("/clear_cache")
async def clear_cache(avatar_id: Optional[str] = Form(None)):
    """Clear avatar cache to free GPU memory"""
    avatar_cache.clear_cache(avatar_id)
    return {
        "status": "success",
        "message": f"Cache cleared" + (f" for {avatar_id}" if avatar_id else " for all avatars")
    }

@app.get("/avatars")
async def list_avatars():
    """List all cached avatars"""
    return {
        "cached_avatars": avatar_cache.list_avatars(),
        "count": len(avatar_cache.list_avatars())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with system info"""
    gpu_mem = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    gpu_mem_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    
    return {
        "status": "healthy",
        "cached_avatars": avatar_cache.list_avatars(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_memory_allocated_gb": f"{gpu_mem:.2f}",
        "gpu_memory_reserved_gb": f"{gpu_mem_reserved:.2f}",
        "device": str(device),
        "tts_loaded": tts_model is not None,
        "tts_sample_rate": tts_model.config.sampling_rate if tts_model else None
    }


# ==================== 7. Update API Endpoints ====================
@app.post("/tts_and_inference")
async def tts_and_inference(
    avatar_id: str = Form(...),
    text: str = Form(...),
    output_name: Optional[str] = Form(None),
    fps: int = Form(25),
    batch_size: Optional[int] = Form(None),
    use_amp: bool = Form(True),
    use_async_encoding: bool = Form(True),
    use_gpu_encoding: bool = Form(True)  # NEW
):
    """Complete pipeline with GPU encoding"""
    try:
        cached_data = avatar_cache.get_avatar(avatar_id)
        if cached_data is None:
            raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not cached.")
        
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if batch_size is None:
            batch_size = args.batch_size
        
        print(f"🎬 Starting OPTIMIZED TTS + Inference pipeline")
        print(f"   Avatar: {avatar_id}")
        print(f"   Text: '{text[:50]}...'")
        print(f"   GPU encoding: {use_gpu_encoding}")
        
        pipeline_start = time.time()
        
        # TTS
        audio_filename = f"tts_{uuid.uuid4()}.wav"
        audio_path = TTS_OUTPUT_DIR / audio_filename
        tts_result = synthesize_speech(text, str(audio_path))
        
        # Inference
        output_name = output_name or str(uuid.uuid4())
        video_path = inference_optimized_v2(
            audio_path=str(audio_path),
            out_vid_name=output_name,
            fps=fps,
            cached_data=cached_data,
            batch_size=batch_size,
            use_amp=use_amp,
            use_async_encoding=use_async_encoding,
            use_gpu_encoding=use_gpu_encoding
        )
        
        pipeline_time = time.time() - pipeline_start
        
        print(f"\n✅ Pipeline completed in {pipeline_time:.2f}s")
        
        return {
            "status": "success",
            "video_path": video_path,
            "audio_path": str(audio_path),
            "output_name": output_name,
            "avatar_id": avatar_id,
            "text": text,
            "tts_latency": tts_result['latency'],
            "audio_duration": tts_result['duration'],
            "total_time": pipeline_time,
            "download_video_url": f"/download/{avatar_id}/{output_name}",
            "download_audio_url": f"/download_audio/{audio_filename.replace('.wav', '')}"
        }
    except Exception as e:
        if 'audio_path' in locals() and audio_path.exists():
            audio_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ==================== 8. Benchmark Endpoint ====================
@app.post("/benchmark")
async def benchmark_pipeline(
    avatar_id: str = Form(...),
    num_runs: int = Form(5)
):
    """
    Benchmark the inference pipeline
    """
    cached_data = avatar_cache.get_avatar(avatar_id)
    if cached_data is None:
        raise HTTPException(status_code=404, detail=f"Avatar {avatar_id} not found")
    
    test_text = "Xin chào, tôi là trợ lý ảo. Hôm nay bạn có khỏe không?"
    
    results = []
    for i in range(num_runs):
        print(f"\n🔄 Run {i+1}/{num_runs}")
        
        # Generate audio
        audio_filename = f"bench_{uuid.uuid4()}.wav"
        audio_path = TTS_OUTPUT_DIR / audio_filename
        tts_result = synthesize_speech(test_text, str(audio_path))
        
        # Inference
        start = time.time()
        video_path = inference_optimized_v2(
            audio_path=str(audio_path),
            out_vid_name=f"bench_{i}",
            fps=25,
            cached_data=cached_data,
            batch_size=args.batch_size,
            use_amp=True,
            use_async_encoding=True
        )
        total_time = time.time() - start
        
        results.append({
            'run': i+1,
            'tts_time': tts_result['latency'],
            'inference_time': total_time,
            'total_time': tts_result['latency'] + total_time
        })
        
        # Cleanup
        audio_path.unlink()
        Path(video_path).unlink()
    
    avg_total = np.mean([r['total_time'] for r in results])
    std_total = np.std([r['total_time'] for r in results])
    
    return {
        "status": "success",
        "avatar_id": avatar_id,
        "num_runs": num_runs,
        "results": results,
        "statistics": {
            "avg_total_time": f"{avg_total:.2f}s",
            "std_total_time": f"{std_total:.2f}s",
            "min_total_time": f"{min(r['total_time'] for r in results):.2f}s",
            "max_total_time": f"{max(r['total_time'] for r in results):.2f}s"
        }
    }