# audio_inference.py
import os
import cv2
import copy
import shutil
import numpy as np
from tqdm import tqdm
from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing

class AudioInference:
    """
    Run inference with different audio files using cached video data
    """
    
    def __init__(self, 
                 unet,
                 whisper_processor,
                 result_dir="./results",
                 version="v15",
                 skip_padding=False,
                 parsing_mode='jaw',
                 left_cheek_width=90,
                 right_cheek_width=90):
        
        self.unet = unet
        self.whisper_processor = whisper_processor
        self.result_dir = result_dir
        self.version = version
        self.skip_padding = skip_padding
        
        # Initialize face parser
        if version == "v15":
            self.fp = FaceParsing(
                left_cheek_width=left_cheek_width,
                right_cheek_width=right_cheek_width
            )
        else:
            self.fp = FaceParsing()
        
        self.parsing_mode = parsing_mode
        self.timesteps = np.array([0], dtype=np.int64)
        
        os.makedirs(result_dir, exist_ok=True)
    
    def generate(self, 
                 video_cache, 
                 audio_path, 
                 batch_size=16,
                 audio_padding_length_left=2,
                 audio_padding_length_right=2,
                 output_name=None):
        """
        Generate video with audio using cached video data
        
        Args:
            video_cache: Preprocessed video data from VideoPreprocessor
            audio_path: Path to audio file
            batch_size: Batch size for inference
            output_name: Custom output filename
        
        Returns:
            output_path: Path to generated video
        """
        
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
            # UNet forward
            pred_latents = self.unet.forward(latent_batch, self.timesteps, whisper_batch)
            
            # Decode
            # Note: You should pass vae instance here, simplified for demo
            # recon = self.vae.decode_latents(pred_latents)
            # For now, just append pred_latents
            for frame_latent in pred_latents:
                res_frame_list.append(frame_latent)
        
        return res_frame_list
    
    def _datagen(self, whisper_chunks, vae_encode_latents, batch_size):
        """Data generator for batching"""
        whisper_batch, latent_batch = [], []
        
        for i, w in enumerate(whisper_chunks):
            idx = i % len(vae_encode_latents)
            latent = vae_encode_latents[idx]
            whisper_batch.append(w)
            latent_batch.append(latent)

            if len(latent_batch) >= batch_size:
                whisper_batch = np.stack(whisper_batch, axis=0).astype(np.float16)
                latent_batch = np.concatenate(latent_batch, axis=0).astype(np.float16)
                yield whisper_batch, latent_batch
                whisper_batch, latent_batch = [], []

        if len(latent_batch) > 0:
            whisper_batch = np.stack(whisper_batch, axis=0).astype(np.float16)
            latent_batch = np.concatenate(latent_batch, axis=0).astype(np.float16)
            yield whisper_batch, latent_batch
    
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
                
                if self.skip_padding:
                    # Fast mode: direct paste
                    ori_frame[y1:y2, x1:x2] = res_frame
                else:
                    # Quality mode: face parsing + blending
                    if self.version == "v15":
                        ori_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], 
                                             mode=self.parsing_mode, fp=self.fp)
                    else:
                        ori_frame = get_image(ori_frame, res_frame, [x1, y1, x2, y2], 
                                             fp=self.fp)
                
                frame_path = f"{temp_dir}/{str(i).zfill(8)}.png"
                cv2.imwrite(frame_path, ori_frame)
                final_frames.append(frame_path)
                
            except Exception as e:
                print(f"Warning: Failed to composite frame {i}: {e}")
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