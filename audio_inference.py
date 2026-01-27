# audio_inference.py - Complete fixed version

import os
import cv2
import copy
import shutil
import torch
import numpy as np
from tqdm import tqdm
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing

class AudioInference:
    def __init__(self, 
                 vae, unet, whisper_processor,
                 device,  # ✅ ADD device
                 result_dir="./results",
                 version="v15",
                 parsing_mode='jaw',
                 left_cheek_width=90,
                 right_cheek_width=90):
        
        self.vae = vae
        self.unet = unet
        self.whisper_processor = whisper_processor
        self.device = device  # ✅ Store device
        self.result_dir = result_dir
        self.version = version
        self.parsing_mode = parsing_mode
        self.timesteps = torch.tensor([0], device=device)
        
        # Initialize face parser
        if version == "v15":
            self.fp = FaceParsing(
                left_cheek_width=left_cheek_width,
                right_cheek_width=right_cheek_width
            )
        else:
            self.fp = FaceParsing()
        
        os.makedirs(result_dir, exist_ok=True)
    
    def generate(self, 
                 video_cache, 
                 audio_path, 
                 batch_size=16,
                 audio_padding_length_left=2,
                 audio_padding_length_right=2,
                 output_name=None):
        """Generate video with audio using cached video data"""
        
        print(f"\n{'='*60}")
        print(f"Generating with audio: {os.path.basename(audio_path)}")
        print(f"{'='*60}")
        
        # Extract audio features
        print("\n[1/4] Processing audio with faster-whisper...")
        whisper_chunks = self.whisper_processor.get_whisper_chunk(
            audio_path,
            fps=video_cache['fps'],
            audio_padding_length_left=audio_padding_length_left,
            audio_padding_length_right=audio_padding_length_right,
        )
        print(f"✓ Extracted {len(whisper_chunks)} audio chunks")

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB")
        
        # Run inference
        print("\n[2/4] Running ONNX inference...")
        res_frame_list = self._run_inference(
            whisper_chunks=whisper_chunks,
            latent_list_cycle=video_cache['input_latent_list_cycle'],
            batch_size=batch_size
        )
        print(f"✓ Generated {len(res_frame_list)} frames")
        
        # Composite frames
        print("\n[3/4] Compositing frames...")
        temp_dir, final_frames = self._composite_frames(
            res_frame_list=res_frame_list,
            video_cache=video_cache
        )
        print(f"✓ Composited {len(final_frames)} frames")
        
        # Create video
        print("\n[4/4] Creating final video...")
        output_path = self._create_video(
            temp_dir=temp_dir,
            audio_path=audio_path,
            video_cache=video_cache,
            output_name=output_name
        )
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        print("\n✓ Generation completed!")
        print(f"  Output: {output_path}")
        
        return output_path
    
    def _run_inference(self, whisper_chunks, latent_list_cycle, batch_size):
        """Run UNet inference with batching"""
        video_num = len(whisper_chunks)
        gen = self._datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=latent_list_cycle,
            batch_size=batch_size
        )
        
        res_frame_list = []
        total = int(np.ceil(float(video_num) / batch_size))
        
        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total, desc="Inference")):
            # UNet forward - input [B, 8, 32, 32], output [B, 4, 32, 32]
            pred_latents = self.unet.forward(latent_batch, self.timesteps, whisper_batch)
            
            # Decode latents to frames [B, H, W, 3] BGR uint8
            decoded_frames = self.vae.decode_latents(pred_latents)
            
            # Append each frame
            for frame in decoded_frames:
                res_frame_list.append(frame)
        
        return res_frame_list
    
    def _datagen(self, whisper_chunks, vae_encode_latents, batch_size):
        """Data generator for batching"""
        whisper_batch, latent_batch = [], []
        
        for i, w in enumerate(whisper_chunks):
            idx = i % len(vae_encode_latents)
            latent = vae_encode_latents[idx]
            
            # # HOTFIX: If cached latent has 16 channels, slice to 8
            # # This happens when latents were concatenated twice during caching
            # if latent.shape[1] == 16:
            #     latent = latent[:, :8, :, :]  # Take only first 8 channels
            
            whisper_batch.append(w)
            latent_batch.append(latent)

            if len(latent_batch) >= batch_size:
                whisper_batch = np.stack(whisper_batch, axis=0).astype(np.float32)
                latent_batch = np.concatenate(latent_batch, axis=0)

                yield whisper_batch, latent_batch
                whisper_batch, latent_batch = [], []

        if len(latent_batch) > 0:
            whisper_batch = np.stack(whisper_batch, axis=0).astype(np.float32)
            latent_batch = np.concatenate(latent_batch, axis=0)
            yield whisper_batch, latent_batch
    
    def _blend_face(self, background, foreground, bbox, alpha=0.95):
        """
        Blend generated face with original frame
        
        Args:
            background: Original frame
            foreground: Generated face region
            bbox: [x1, y1, x2, y2]
            alpha: Blending factor (0-1)
        """
        x1, y1, x2, y2 = bbox
        
        # Create smooth alpha mask (feathered edges)
        h, w = foreground.shape[:2]
        mask = np.ones((h, w), dtype=np.float32) * alpha
        
        # Feather the edges
        feather_size = min(20, h//10, w//10)
        if feather_size > 0:
            for i in range(feather_size):
                fade = i / feather_size
                mask[i, :] *= fade  # Top
                mask[-(i+1), :] *= fade  # Bottom
                mask[:, i] *= fade  # Left
                mask[:, -(i+1)] *= fade  # Right
        
        # Expand mask to 3 channels
        mask = np.stack([mask] * 3, axis=2)
        
        # Blend
        result = background.copy()
        result[y1:y2, x1:x2] = (
            foreground * mask + 
            result[y1:y2, x1:x2] * (1 - mask)
        ).astype(np.uint8)
        
        return result

    def _composite_frames(self, res_frame_list, video_cache):
        """Composite generated faces onto original frames"""
        temp_dir = os.path.join(self.result_dir, "temp_frames")
        os.makedirs(temp_dir, exist_ok=True)
        
        frame_list_cycle = video_cache['frame_list_cycle']
        coord_list_cycle = video_cache['coord_list_cycle']
        extra_margin = video_cache.get('extra_margin', 10)
        
        final_frames = []
        
        for i, res_frame in enumerate(tqdm(res_frame_list, desc="Compositing")):
            bbox = coord_list_cycle[i % len(coord_list_cycle)]
            ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
            x1, y1, x2, y2 = bbox
            
            if self.version == "v15":
                y2 = y2 + extra_margin
                y2 = min(y2, ori_frame.shape[0])
            
            try:
                res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
                
                if self.version == "v15":
                    combine_frame = get_image(
                        ori_frame, res_frame, [x1, y1, x2, y2], 
                        mode=self.parsing_mode, fp=self.fp
                    )
                else:
                    combine_frame = get_image(
                        ori_frame, res_frame, [x1, y1, x2, y2], 
                        fp=self.fp
                    )
                
                frame_path = f"{temp_dir}/{str(i).zfill(8)}.png"
                cv2.imwrite(frame_path, combine_frame)
                final_frames.append(frame_path)
                
            except Exception as e:
                print(f"Error frame {i}: {e}")
                continue
        
        return temp_dir, final_frames
    
    def _create_video(self, temp_dir, audio_path, video_cache, output_name):
        """Create final video with audio"""
        video_basename = video_cache['video_basename']
        audio_basename = os.path.basename(audio_path).split('.')[0]
        fps = video_cache['fps']
        
        if output_name is None:
            output_name = f"{video_basename}_{audio_basename}.mp4"
        
        output_path = os.path.join(self.result_dir, output_name)
        temp_vid_path = os.path.join(self.result_dir, f"temp_{output_name}")
        
        # Create video from frames
        cmd_img2video = (
            f"ffmpeg -y -v warning -r {fps} -f image2 "
            f"-i {temp_dir}/%08d.png -vcodec libx264 "
            f"-vf format=yuv420p -crf 18 {temp_vid_path}"
        )
        os.system(cmd_img2video)
        
        # Add audio
        cmd_combine_audio = (
            f"ffmpeg -y -v warning -i {audio_path} "
            f"-i {temp_vid_path} {output_path}"
        )
        os.system(cmd_combine_audio)
        
        # Cleanup temp video
        if os.path.exists(temp_vid_path):
            os.remove(temp_vid_path)
        
        return output_path