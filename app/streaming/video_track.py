"""Custom video track for MuseTalk WebRTC streaming"""
import torch
import cv2
import numpy as np
import queue
import threading
import fractions
from aiortc import VideoStreamTrack
from av import VideoFrame

from musetalk.utils.utils import datagen
from musetalk.utils.blending import get_image_blending
from app.core.model_loader import model_manager
from app.services.avatar_service import avatar_service


class MuseTalkVideoTrack(VideoStreamTrack):
    """
    Custom video track that streams frames from MuseTalk model inference
    """
    
    def __init__(self, avatar_id: str, audio_frames: list, fps: int = 25, 
                 batch_size: int = 4):
        """
        Initialize video track
        
        Args:
            avatar_id: ID of prepared avatar
            audio_frames: List of whisper feature chunks
            fps: Frames per second
            batch_size: Batch size for inference
        """
        super().__init__()
        
        self.avatar_id = avatar_id
        self.audio_frames = audio_frames
        self.fps = fps
        self.batch_size = batch_size
        
        # Get cached avatar data
        self.cached_data = avatar_service.get_avatar(avatar_id)
        
        # Frame management
        self.frame_queue = queue.Queue(maxsize=60)
        self.current_idx = 0
        self.total_frames = len(audio_frames)
        
        # Timing
        self.pts_increment = fractions.Fraction(1, self.fps)
        
        # Control
        self.running = True
        
        # Start background inference thread
        self.inference_thread = threading.Thread(
            target=self._inference_loop, 
            daemon=True
        )
        self.inference_thread.start()
        
        print(f"🎬 Video track initialized: {self.total_frames} frames @ {fps}fps")
    
    def _inference_loop(self):
        """Background thread that runs MuseTalk inference"""
        print("🔄 Starting inference loop...")
        
        with torch.no_grad():
            # Create data generator
            gen = datagen(
                self.audio_frames,
                self.cached_data['input_latent_list_cycle'],
                self.batch_size
            )
            
            for whisper_batch, latent_batch in gen:
                if not self.running:
                    print("⏹️ Inference loop stopped")
                    break
                
                # Run inference
                frames = self._run_inference(whisper_batch, latent_batch)
                
                # Process and queue frames
                for res_frame in frames:
                    if not self.running:
                        break
                    
                    # Blend frame with original
                    blended_frame = self._blend_frame(res_frame)
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(blended_frame, cv2.COLOR_BGR2RGB)
                    
                    # Push to queue
                    try:
                        self.frame_queue.put(rgb_frame, timeout=1.0)
                        self.current_idx += 1
                    except queue.Full:
                        print("⚠️ Frame queue full, dropping frame")
                        continue
        
        print("✅ Inference loop completed")
    
    def _run_inference(self, whisper_batch, latent_batch):
        """Run inference - already on GPU, no copy needed"""
        # whisper_batch đã ở GPU từ audio_service
        audio_feature_batch = model_manager.pe(whisper_batch)
        
        # latent_batch đã ở GPU từ cache
        # Chỉ cần convert dtype
        if latent_batch.dtype != model_manager.unet.model.dtype:
            latent_batch = latent_batch.to(dtype=model_manager.unet.model.dtype)
        
        # UNet inference
        pred_latents = model_manager.unet.model(
            latent_batch,
            model_manager.timesteps,
            encoder_hidden_states=audio_feature_batch
        ).sample
        
        # Decode
        if pred_latents.dtype != model_manager.vae.vae.dtype:
            pred_latents = pred_latents.to(dtype=model_manager.vae.vae.dtype)
        
        recon = model_manager.vae.decode_latents(pred_latents)
        return recon
    
    def _blend_frame(self, res_frame):
        """Blend generated frame with original frame"""
        idx = self.current_idx % len(self.cached_data['coord_list_cycle'])
        
        bbox = self.cached_data['coord_list_cycle'][idx]
        ori_frame = self.cached_data['frame_list_cycle'][idx].copy()
        mask = self.cached_data['mask_list_cycle'][idx]
        mask_crop_box = self.cached_data['mask_coords_list_cycle'][idx]
        
        # Resize generated frame to bbox size
        x1, y1, x2, y2 = bbox
        res_frame_resized = cv2.resize(
            res_frame.astype(np.uint8),
            (x2 - x1, y2 - y1)
        )
        
        # Blend
        combined_frame = get_image_blending(
            ori_frame, res_frame_resized, bbox, mask, mask_crop_box
        )
        
        return combined_frame
    
    async def recv(self):
        """
        Receive next video frame (called by WebRTC)
        
        Returns:
            VideoFrame for WebRTC
        """
        if self.current_idx >= self.total_frames:
            # End of stream
            self.running = False
            raise StopAsyncIteration
        
        try:
            # Get frame from queue (with timeout)
            frame = self.frame_queue.get(timeout=2.0)
            
            # Convert to VideoFrame
            video_frame = VideoFrame.from_ndarray(frame, format="rgb24")
            video_frame.pts = self.pts
            video_frame.time_base = (
                self.pts_increment.denominator / self.pts_increment.numerator
            )
            
            # Increment pts
            self.pts += int(
                self.pts_increment.denominator / self.pts_increment.numerator
            )
            
            return video_frame
            
        except queue.Empty:
            # No frame available, return black frame to maintain stream
            print("⚠️ Queue empty, returning black frame")
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            video_frame = VideoFrame.from_ndarray(black_frame, format="rgb24")
            video_frame.pts = self.pts
            video_frame.time_base = (
                self.pts_increment.denominator / self.pts_increment.numerator
            )
            self.pts += int(
                self.pts_increment.denominator / self.pts_increment.numerator
            )
            return video_frame
    
    def stop(self):
        """Stop the video track"""
        self.running = False
        if self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)