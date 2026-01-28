# musetalk inference
import os
import cv2
import copy
import torch
import pickle
import glob
import shutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from tqdm import tqdm
from transformers import WhisperModel
import uvicorn

from musetalk.utils.blending import get_image, get_image_prepare_material, get_image_blending
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import datagen, load_all_model, get_file_type, get_video_fps
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs

app = FastAPI(title="Musetalk API")

# Global models
device = None
vae = None
unet = None
pe = None
whisper = None
audio_processor = None
fp = None

UPLOAD_DIR = "./uploads"
CACHE_DIR = "./cache"
RESULT_DIR = "./results"

for d in [UPLOAD_DIR, CACHE_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

def initialize_models(
    gpu_id=0,
    unet_model_path="./models/musetalkV15/unet.pth",
    vae_type="sd-vae",
    unet_config="./models/musetalk/config.json",
    whisper_dir="./models/whisper",
    use_float16=True,
    version="v15",
    left_cheek_width=90,
    right_cheek_width=90
):
    global device, vae, unet, pe, whisper, audio_processor, fp

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load models
    vae, unet, pe = load_all_model(unet_model_path, vae_type, unet_config, device)

    if use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()

    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    # Initialize audio processor
    audio_processor = AudioProcessor(feature_extractor_path=whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    # Initialize face parser
    if version == "v15":
        fp = FaceParsing(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)
    else:
        fp = FaceParsing()
    
    print("Models initialized successfully")

def blend_frame_fast(args):
    """Fast blending using pre-cached masks"""
    i, res_frame, ori_frame, bbox, mask_array, crop_box, extra_margin, version = args

    x1, y1, x2, y2 = bbox
    if version == "v15":
        y2 = min(y2 + extra_margin, ori_frame.shape[0])

    # Resize result frame
    res_frame_resized = cv2.resize(
        res_frame.astype(np.uint8), 
        (x2-x1, y2-y1),
        interpolation=cv2.INTER_LINEAR  # Faster than LANCZOS4
    )
    
    # Use fast blending with pre-computed mask
    combine_frame = get_image_blending(
        ori_frame, 
        res_frame_resized, 
        [x1, y1, x2, y2],
        mask_array,
        crop_box
    )

    return i, combine_frame
    
@app.on_event("startup")
async def startup_event():
    initialize_models()

@app.post("/preprocess_video")
async def preprocess_video(
    video: UploadFile = File(...),
    bbox_shift: int = Form(0),
    version: str = Form("v15"),
    precache_masks: bool = Form(True)
):
    """
    Step 1: Process video, extract frames, and detect face-bounding boxes
    Run only once for each video
    """

    try:
        # Save uploaded video
        video_id = os.path.splitext(video.filename)[0]
        video_path = os.path.join(UPLOAD_DIR, video.filename)

        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        # Check if already processed
        metadata_path = os.path.join(CACHE_DIR, f"{video_id}_metadata.pkl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            return JSONResponse({
                "status": "already_cached",
                "video_id": video_id,
                "fps": metadata['fps'],
                "num_frames": metadata['num_frames'],
                "message": "Video already preprocessed"
            })
        
        # Extract frames
        save_dir_full = os.path.join(CACHE_DIR, f"{video_id}_frames")
        os.makedirs(save_dir_full, exist_ok=True)

        if get_file_type(video_path) == "video":
            cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
            os.system(cmd)
            input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
            fps = get_video_fps(video_path)
        else:
            raise ValueError("Only video files are supported")
        
        # Detect faces and get bounding boxes
        coord_save_path = os.path.join(CACHE_DIR, f"{video_id}_coords.pkl")
        
        print(f"Detecting faces for {len(input_img_list)} frames...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        
        # Save coordinates
        with open(coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)

        if precache_masks:
            print("Pre-computing face parsing masks (one-time)...")

            if version == 'v15':
                fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
            else:
                fp = FaceParsing()

            # Compute masks forr all frames
            masks_list = []
            crop_boxes_list = []

            for bbox, frame in tqdm(zip(coord_list, frame_list),
                                    total=len(frame_list),
                                    desc="Caching masks"):
                x1, y1, x2, y2 = bbox
                if version == "v15":
                    y2_adjusted = min(y2 + 10, frame.shape[0])  # extra_margin
                    mask_array, crop_box = get_image_prepare_material(
                        frame, [x1, y1, x2, y2_adjusted], 
                        upper_boundary_ratio=0.5, 
                        expand=1.5, 
                        fp=fp, 
                        mode="jaw"
                    )
                else:
                    mask_array, crop_box = get_image_prepare_material(
                        frame, bbox, 
                        upper_boundary_ratio=0.5, 
                        expand=1.5, 
                        fp=fp
                    )
                masks_list.append(mask_array)
                crop_boxes_list.append(crop_box)

            # Save masks
            masks_path = os.path.join(CACHE_DIR, f"{video_id}_masks.pkl")
            with open(masks_path, 'wb') as f:
                pickle.dump({
                    'masks': masks_list,
                    'crop_boxes': crop_boxes_list
                }, f)
            
            metadata['masks_path'] = masks_path
        
        # Cache metadata
        metadata = {
            "video_id": video_id,
            "fps": fps,
            "num_frames": len(frame_list),
            "coord_path": coord_save_path,
            "frames_dir": save_dir_full
        }
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        return JSONResponse({
            "status": "success",
            "video_id": video_id,
            "fps": fps,
            "num_frames": len(frame_list),
            "masks_cached": precache_masks,
            "message": "Video preprocessed successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/check_video/{video_id}")
async def check_video(video_id: str):
    """Check if the video has been preprocessed."""
    metadata_path = os.path.join(CACHE_DIR, f"{video_id}_metadata.pkl")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        return JSONResponse({
            "status": "cached",
            "video_id": metadata['video_id'],
            "fps": metadata['fps'],
            "num_frames": metadata['num_frames']
        })
    return JSONResponse({"status": "not_found", "message": "Video not preprocessed yet"})

@app.post("/generate_lipsync")
async def generate_lipsync(
    video_id: str = Form(...),
    audio: UploadFile = (...),
    batch_size: int = Form(8),
    extra_margin: int = Form(10),
    parsing_mode: str = Form("jaw"),
    audio_padding_left: int = Form(2),
    audio_padding_right: int = Form(2),
    version: str = Form("v15"),
    num_workers: int = Form(8)
):
    """
    Step 2: Generate lip sync video from cached video data and new audio
    This can be run multiple times with different audio
    """
    try:
        # Load cached metadata
        metadata_path = os.path.join(CACHE_DIR, f"{video_id}_metadata.pkl")
        if not os.path.exists(metadata_path):
            raise HTTPException(
                status_code=404,
                detail=f"Video '{video_id}' not preprocessed. Please call /preprocess_video first"
            )
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        # Use cached masks if available
        use_cached_masks = 'masks_path' in metadata and os.path.exists(metadata['masks_path'])

        if use_cached_masks:
            print("Loading pre-cached face parsing masks...")
            with open(metadata['masks_path'], 'rb') as f:
                mask_data = pickle.load(f)
                masks_list = mask_data['masks']
                crop_boxes_list = mask_data['crop_boxes']

        # Save audio 
        audio_filename = os.path.splitext(audio.filename)[0]
        audio_path = os.path.join(CACHE_DIR, f"temp_audio_{video_id}_{audio_filename}.wav")
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        # Load cached coordinates and frames
        with open(metadata['coord_path'], 'rb') as f:
            coord_list = pickle.load(f)

        input_img_list = sorted(glob.glob(os.path.join(metadata['frames_dir'], '*.[jpJP][pnPN]*[gG]')))
        frame_list = read_imgs(input_img_list)
        fps = metadata['fps']
        
        print(f"Processing audio with {len(frame_list)} frames at {fps} fps")

        # Process audio
        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features, device, unet.model.dtype, whisper, 
            librosa_length, fps=fps,
            audio_padding_length_left=audio_padding_left,
            audio_padding_length_right=audio_padding_right
        )

        # Prepare latents (cache này có thể optimize thêm nếu cùng video)
        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            x1, y1, x2, y2 = bbox
            if version == "v15":
                y2 = min(y2 + extra_margin, frame.shape[0])
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
        
        # Cycle for smoothing
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        
        if use_cached_masks:
            masks_list_cycle = masks_list + masks_list[::-1]
            crop_boxes_list_cycle = crop_boxes_list + crop_boxes_list[::-1]

        # Inference
        print("Starting inference...")
        video_num = len(whisper_chunks)
        timesteps = torch.tensor([0], device=device)
        gen = datagen(whisper_chunks, input_latent_list_cycle, batch_size, 0, device)

        res_frame_list = []
        total = int(np.ceil(float(video_num) / batch_size))

        for whisper_batch, latent_batch in tqdm(gen, total=total, desc="Inference"):
            audio_feature_batch = pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            res_frame_list.extend(recon)

        # Blend and save frames
        output_id = f"{video_id}_{audio_filename}"
        result_img_path = os.path.join(RESULT_DIR, output_id)
        os.makedirs(result_img_path, exist_ok=True)

        # print("Blending frames...")
        # for i, res_frame in enumerate(tqdm(res_frame_list, desc="Blending")):
        #     bbox = coord_list_cycle[i % len(coord_list_cycle)]
        #     ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
        #     x1, y1, x2, y2 = bbox
        #     if version == "v15":
        #         y2 = min(y2 + extra_margin, ori_frame.shape[0])
            
        #     res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
            
        #     if version == "v15":
        #         combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], mode=parsing_mode, fp=fp)
        #     else:
        #         combine_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], fp=fp)
            
        #     cv2.imwrite(f"{result_img_path}/{str(i).zfill(8)}.png", combine_frame)

        if use_cached_masks:
            print(f"Fast blending with {num_workers} workers...")
            
            # Prepare arguments
            blend_args = []
            for i, res_frame in enumerate(res_frame_list):
                idx = i % len(coord_list_cycle)
                blend_args.append((
                    i,
                    res_frame,
                    frame_list_cycle[idx].copy(),
                    coord_list_cycle[idx],
                    masks_list_cycle[idx],
                    crop_boxes_list_cycle[idx],
                    extra_margin,
                    version
                ))
            
            # Parallel blending
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(
                    executor.map(blend_frame_fast, blend_args),
                    total=len(blend_args),
                    desc="Blending"
                ))
            
            # Save frames
            for i, combine_frame in results:
                cv2.imwrite(f"{result_img_path}/{str(i).zfill(8)}.png", combine_frame)
        
        else:
            # Fallback to original method (slow)
            print("Warning: Using slow blending (masks not cached)")
            for i, res_frame in enumerate(tqdm(res_frame_list, desc="Blending")):
                bbox = coord_list_cycle[i % len(coord_list_cycle)]
                ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
                x1, y1, x2, y2 = bbox
                if version == "v15":
                    y2 = min(y2 + extra_margin, ori_frame.shape[0])
                
                res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                combine_frame = get_image(ori_frame, res_frame_resized, [x1, y1, x2, y2], 
                                        mode="jaw", fp=fp)
                
                cv2.imwrite(f"{result_img_path}/{str(i).zfill(8)}.png", combine_frame)

        # Generate video
        print("Generating final video...")
        output_video = os.path.join(RESULT_DIR, f"{output_id}.mp4")
        temp_vid = os.path.join(RESULT_DIR, f"temp_{output_id}.mp4")
        
        cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {result_img_path}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid}"
        os.system(cmd_img2video)
        
        cmd_combine = f"ffmpeg -y -v warning -i {audio_path} -i {temp_vid} {output_video}"
        os.system(cmd_combine)
        
        # Cleanup
        shutil.rmtree(result_img_path)
        os.remove(temp_vid)
        os.remove(audio_path)
        
        print(f"Video saved to {output_video}")
        
        return FileResponse(
            output_video, 
            media_type="video/mp4", 
            filename=f"{output_id}.mp4"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear_cache/{video_id}")
async def clear_cache(video_id: str):
    """Xóa cache của video"""
    try:
        metadata_path = os.path.join(CACHE_DIR, f"{video_id}_metadata.pkl")
        if not os.path.exists(metadata_path):
            raise HTTPException(status_code=404, detail="Video not found")
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        # Remove all cached files
        if os.path.exists(metadata['frames_dir']):
            shutil.rmtree(metadata['frames_dir'])
        if os.path.exists(metadata['coord_path']):
            os.remove(metadata['coord_path'])
        os.remove(metadata_path)
        
        return JSONResponse({"status": "success", "message": f"Cache cleared for {video_id}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "MuseTalk API",
        "endpoints": {
            "preprocess_video": "POST /preprocess_video - Upload và preprocess video (1 lần)",
            "check_video": "GET /check_video/{video_id} - Kiểm tra video đã preprocess chưa",
            "generate_lipsync": "POST /generate_lipsync - Generate lip sync với audio mới (nhiều lần)",
            "clear_cache": "DELETE /clear_cache/{video_id} - Xóa cache video"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)