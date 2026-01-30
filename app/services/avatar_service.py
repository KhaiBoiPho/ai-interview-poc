"""Avatar preparation and caching service"""
import cv2
import glob
from pathlib import Path
from typing import Dict, Any

import torch

from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.utils.blending import get_image_prepare_material
from app.core.model_loader import model_manager
from app.config import app_config


class AvatarService:
    """Handles avatar preparation and caching"""

    def __init__(self):
        self.cache: Dict[str, Dict[str, Any]] = {}

    def prepare_avatar(self, avatar_id: str, video_path: str,
                       bbox_shift: int = 0, max_frames: int = None):
        """
                Prepare avatar from video file
        
        Args:
            avatar_id: Unique identifier for avatar
            video_path: Path to video file
            bbox_shift: Bounding box shift parameter
            max_frames: Maximum number of frames to process
            
        Returns:
            Avatar data dictionary
        """
        print(f"📹 Preparing avatar: {avatar_id}")

        # Setup directories
        avatar_path = app_config.avatar_results_dir / avatar_id
        full_imgs_path = avatar_path / "full_imgs"
        avatar_path.mkdir(parents=True, exist_ok=True)
        full_imgs_path.mkdir(parents=True, exist_ok=True)

        # Extract frames and bounding boxes
        max_str = "all" if max_frames is None else str(max_frames)
        print(f"Extract frames (max {max_str})...")
        self._extract_frames(video_path, full_imgs_path, max_frames)

        # Get landmarks and bounding boxes
        print("Detecting landmarks...")
        input_img_list = sorted(
            glob.glob(str(full_imgs_path / "*.png")) +
            glob.glob(str(full_imgs_path / "*.jpg")) +
            glob.glob(str(full_imgs_path / "*.jpeg"))
        )
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)

        # Generate latents
        print("Generating latents...")
        input_latent_list = self._generate_latents(coord_list, frame_list)

        # Create cycles
        print("Creating frame cycles...")
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

        # Generate masks
        print("Generating masks...")
        mask_list_cycle, mask_coords_list_cycle = self._generate_masks(
            frame_list_cycle, coord_list_cycle
        )

        # Create avatar data
        avatar_data = {
            'input_latent_list_cycle': input_latent_list_cycle,  # đã ở GPU
            'coord_list_cycle': coord_list_cycle,
            'frame_list_cycle': frame_list_cycle,
            'mask_list_cycle': mask_list_cycle,  # đã ở GPU
            'mask_coords_list_cycle': mask_coords_list_cycle
        }
        
        # Cache it
        self.cache[avatar_id] = avatar_data
        
        print(f"✅ Avatar '{avatar_id}' prepared and cached")
        print(f"   Total frames in cycle: {len(frame_list_cycle)}")
        
        return avatar_data
    
    def _extract_frames(self, video_path: str, output_dir: Path, max_frames: int):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        count = 0

        while True:
            if max_frames is not None and count >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(output_dir / f"{count:08d}.png"), frame)
            count += 1

        cap.release()
        print(f"Extracted {count} frames")


    def _generate_latents(self, coord_list, frame_list):
        """Generate VAE latents - MOVE TO GPU IMMEDIATELY"""
        input_latent_list = []
        device = model_manager.device
        
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == (0.0, 0.0, 0.0, 0.0):
                continue
            
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256))
            
            latents = model_manager.vae.get_latents_for_unet(resized_crop_frame)
            latents = latents.to(device)
            input_latent_list.append(latents)
        
        print(f"   Generated {len(input_latent_list)} latents (on GPU)")
        return input_latent_list
    
    def _generate_masks(self, frame_list_cycle, coord_list_cycle):
        """Generate masks và convert sang tensor ngay"""
        device = model_manager.device
        mask_list_cycle = []
        mask_coords_list_cycle = []
        
        for i, frame in enumerate(frame_list_cycle):
            x1, y1, x2, y2 = coord_list_cycle[i]
            
            mask, crop_box = get_image_prepare_material(
                frame, [x1, y1, x2, y2], 
                fp=model_manager.fp, 
                mode='raw'
            )
            
            # Convert to tensor và move to GPU ngay
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).to(device)
            else:
                mask = mask.to(device)
                
            mask_list_cycle.append(mask)
            mask_coords_list_cycle.append(crop_box)
        
        print(f"   Generated {len(mask_list_cycle)} masks (on GPU)")
        return mask_list_cycle, mask_coords_list_cycle
    
    def get_avatar(self, avatar_id: str) -> Dict[str, Any]:
        """Get cached avatar data"""
        if avatar_id not in self.cache:
            raise ValueError(f"Avatar '{avatar_id}' not found in cache")
        return self.cache[avatar_id]
    
    def has_avatar(self, avatar_id: str) -> bool:
        """Check if avatar exists in cache"""
        return avatar_id in self.cache
    
    def list_avatars(self):
        """List all cached avatars"""
        return list(self.cache.keys())
    
    def clear_cache(self):
        """Clear avatar cache"""
        self.cache.clear()


# Singleton instance
avatar_service = AvatarService()