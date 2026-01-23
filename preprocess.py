# preprocess.py
import argparse
import pickle
import os
from musetalk.utils.preprocessing import get_landmark_and_bbox
from musetalk.utils.utils import get_file_type, load_all_model
import torch
import cv2
import glob

@torch.no_grad()
def preprocess_avatar(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Load VAE only (for encoding)
    vae, _, _ = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    
    if args.use_float16:
        vae.vae = vae.vae.half()
    vae.vae = vae.vae.to(device)
    
    # Extract frames from video
    video_path = args.video_path
    avatar_id = args.avatar_id
    
    if get_file_type(video_path) == "video":
        save_dir_full = f"./avatars/{avatar_id}/frames"
        os.makedirs(save_dir_full, exist_ok=True)
        cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
        os.system(cmd)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
    else:
        raise ValueError("Video path must be a video file")
    
    # Extract landmarks and coordinates
    print("Extracting landmarks and bounding boxes...")
    coord_list, frame_list = get_landmark_and_bbox(input_img_list, args.bbox_shift)
    
    # Encode frames to latents
    print("Encoding frames to latents...")
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        x1, y1, x2, y2 = bbox
        if args.version == "v15":
            y2 = y2 + args.extra_margin
            y2 = min(y2, frame.shape[0])
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)
    
    # Smooth (cycle)
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    
    # Save preprocessed data
    avatar_dir = f"./avatars/{avatar_id}"
    os.makedirs(avatar_dir, exist_ok=True)
    
    with open(f"{avatar_dir}/coords.pkl", 'wb') as f:
        pickle.dump(coord_list_cycle, f)
    
    with open(f"{avatar_dir}/frames.pkl", 'wb') as f:
        pickle.dump(frame_list_cycle, f)
    
    torch.save(input_latent_list_cycle, f"{avatar_dir}/latents.pt")
    
    # Save metadata
    metadata = {
        "avatar_id": avatar_id,
        "num_frames": len(frame_list),
        "bbox_shift": args.bbox_shift,
        "version": args.version,
        "fps": args.fps
    }
    
    with open(f"{avatar_dir}/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Preprocessing complete! Avatar saved to {avatar_dir}")
    return avatar_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--avatar_id", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--vae_type", type=str, default="sd-vae")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/config.json")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth")
    parser.add_argument("--bbox_shift", type=int, default=0)
    parser.add_argument("--extra_margin", type=int, default=10)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--use_float16", action="store_true")
    parser.add_argument("--version", type=str, default="v15", choices=["v1", "v15"])
    args = parser.parse_args()
    
    preprocess_avatar(args)