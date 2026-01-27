# video_preprocessor.py
import os
import cv2
import glob
import pickle
from tqdm import tqdm
from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder
from musetalk.utils.utils import get_file_type, get_video_fps

class VideoPreprocessor:
    """
    Preprocess video once and cache results
    Extracts frames, face landmarks, and latents
    """
    
    def __init__(self, vae, cache_dir="./cache", version="v15", extra_margin=10):
        self.vae = vae
        self.cache_dir = cache_dir
        self.version = version
        self.extra_margin = extra_margin
        os.makedirs(cache_dir, exist_ok=True)
    
    def process_video(self, video_path, bbox_shift=0, use_saved_coord=False):
        """
        Process video and cache all necessary data
        
        Returns:
            video_cache: dict with all preprocessed data
        """
        video_basename = os.path.basename(video_path).split('.')[0]
        cache_path = os.path.join(self.cache_dir, f"{video_basename}.pkl")
        
        # Check if cache exists
        if os.path.exists(cache_path) and use_saved_coord:
            print(f"Loading cached video data from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        print(f"\n{'='*60}")
        print(f"Processing video: {video_basename}")
        print(f"{'='*60}")
        
        # Extract frames
        print("\n[1/4] Extracting frames...")
        frames_dir = os.path.join(self.cache_dir, f"{video_basename}_frames")
        input_img_list, fps = self._extract_frames(video_path, frames_dir)
        print(f"✓ Extracted {len(input_img_list)} frames at {fps} fps")
        
        # Detect faces and landmarks
        print("\n[2/4] Detecting faces and landmarks...")
        # Detect faces - check for saved coords first
        coord_pkl = os.path.join(self.cache_dir, f"{video_basename}_coords.pkl")
        if os.path.exists(coord_pkl) and use_saved_coord:
            with open(coord_pkl, 'rb') as f:
                coord_list = pickle.load(f)
            from musetalk.utils.preprocessing import read_imgs
            frame_list = read_imgs(input_img_list)
        else:
            coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
            with open(coord_pkl, 'wb') as f:
                pickle.dump(coord_list, f)
        print(f"✓ Detected faces in {len(frame_list)} frames")
        
        # Encode frames to latents
        print("\n[3/4] Encoding frames to latents...")
        input_latent_list = self._encode_frames(coord_list, frame_list)
        print(f"✓ Encoded {len(input_latent_list)} frames")
        
        # Create cycle for smooth looping
        print("\n[4/4] Creating frame cycle...")
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        
        # Prepare cache data
        video_cache = {
            'video_path': video_path,
            'video_basename': video_basename,
            'fps': fps,
            'coord_list': coord_list,
            'frame_list': frame_list,
            'input_latent_list': input_latent_list,
            'frame_list_cycle': frame_list_cycle,
            'coord_list_cycle': coord_list_cycle,
            'input_latent_list_cycle': input_latent_list_cycle,
            'num_frames': len(frame_list),
            'version': self.version,
            'extra_margin': self.extra_margin
        }
        
        # Save cache
        print(f"\n[5/5] Saving cache to {cache_path}...")
        with open(cache_path, 'wb') as f:
            pickle.dump(video_cache, f)
        
        print("\n✓ Video preprocessing completed!")
        print(f"  - Total frames: {len(frame_list)}")
        print(f"  - FPS: {fps}")
        print(f"  - Cache size: {os.path.getsize(cache_path) / (1024*1024):.2f} MB")
        
        return video_cache
    
    def _extract_frames(self, video_path, frames_dir):
        """Extract frames from video"""
        file_type = get_file_type(video_path)
        
        if file_type == "video":
            os.makedirs(frames_dir, exist_ok=True)
            cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {frames_dir}/%08d.png"
            os.system(cmd)
            input_img_list = sorted(glob.glob(os.path.join(frames_dir, '*.[jpJP][pnPN]*[gG]')))
            fps = get_video_fps(video_path)
            
        elif file_type == "image":
            input_img_list = [video_path]
            fps = 25  # Default FPS for single image
            
        elif os.path.isdir(video_path):
            input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, 
                                   key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            fps = 25  # Default FPS for image directory
            
        else:
            raise ValueError(f"{video_path} should be a video file, image, or directory")
        
        return input_img_list, fps
    
    def _encode_frames(self, coord_list, frame_list):
        """Encode frames to latents"""
        input_latent_list = []
        
        for bbox, frame in tqdm(zip(coord_list, frame_list), total=len(frame_list), desc="Encoding"):
            if bbox == coord_placeholder:
                continue
            
            x1, y1, x2, y2 = bbox
            if self.version == "v15":
                y2 = y2 + self.extra_margin
                y2 = min(y2, frame.shape[0])
            
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            
            latents = self.vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
        
        return input_latent_list