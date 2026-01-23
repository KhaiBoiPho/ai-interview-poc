# inference_fast.py
import argparse
import pickle
import os
import torch
import cv2
from tqdm import tqdm
from transformers import WhisperModel
import numpy as np

from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import load_all_model, datagen

@torch.no_grad()
def fast_inference(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Load models (UNet + PE only, VAE decoder only)
    vae, unet, pe = load_all_model(
        unet_model_path=args.unet_model_path,
        vae_type=args.vae_type,
        unet_config=args.unet_config,
        device=device
    )
    timesteps = torch.tensor([0], device=device)
    
    if args.use_float16:
        pe = pe.half()
        vae.vae = vae.vae.half()
        unet.model = unet.model.half()
    
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)
    
    # Load Whisper
    audio_processor = AudioProcessor(feature_extractor_path=args.whisper_dir)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(args.whisper_dir)
    whisper = whisper.to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)
    
    # Load preprocessed avatar data
    avatar_dir = f"./avatars/{args.avatar_id}"
    
    with open(f"{avatar_dir}/metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    with open(f"{avatar_dir}/coords.pkl", 'rb') as f:
        coord_list_cycle = pickle.load(f)
    
    with open(f"{avatar_dir}/frames.pkl", 'rb') as f:
        frame_list_cycle = pickle.load(f)
    
    input_latent_list_cycle = torch.load(f"{avatar_dir}/latents.pt")
    
    fps = metadata["fps"]
    
    # Process audio
    print("Processing audio...")
    whisper_input_features, librosa_length = audio_processor.get_audio_feature(args.audio_path)
    whisper_chunks = audio_processor.get_whisper_chunk(
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=fps,
        audio_padding_length_left=args.audio_padding_length_left,
        audio_padding_length_right=args.audio_padding_length_right,
    )
    
    # Inference
    print("Running inference...")
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    
    gen = datagen(
        whisper_chunks=whisper_chunks,
        vae_encode_latents=input_latent_list_cycle,
        batch_size=batch_size,
        delay_frame=0,
        device=device,
    )
    
    res_frame_list = []
    total = int(np.ceil(float(video_num) / batch_size))
    
    for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
        audio_feature_batch = pe(whisper_batch)
        latent_batch = latent_batch.to(dtype=unet.model.dtype)
        
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
    
    # Save frames
    output_dir = f"./outputs/{args.output_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.skip_padding:
        print("Saving frames (no padding)...")
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            cv2.imwrite(f"{output_dir}/{str(i).zfill(8)}.png", res_frame)
    else:
        print("Saving frames (with padding)...")
        for i, res_frame in enumerate(tqdm(res_frame_list)):
            bbox = coord_list_cycle[i % len(coord_list_cycle)]
            ori_frame = frame_list_cycle[i % len(frame_list_cycle)].copy()
            x1, y1, x2, y2 = bbox
            res_frame = cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))
            ori_frame[y1:y2, x1:x2] = res_frame
            cv2.imwrite(f"{output_dir}/{str(i).zfill(8)}.png", ori_frame)
    
    # Generate video
    temp_vid = f"{output_dir}/temp.mp4"
    final_vid = f"{output_dir}/output.mp4"
    
    cmd_img2video = f"ffmpeg -y -v warning -r {fps} -f image2 -i {output_dir}/%08d.png -vcodec libx264 -vf format=yuv420p -crf 18 {temp_vid}"
    os.system(cmd_img2video)
    
    cmd_combine_audio = f"ffmpeg -y -v warning -i {args.audio_path} -i {temp_vid} {final_vid}"
    os.system(cmd_combine_audio)
    
    # Cleanup
    if args.cleanup:
        for f in os.listdir(output_dir):
            if f.endswith('.png'):
                os.remove(os.path.join(output_dir, f))
        os.remove(temp_vid)
    
    print(f"✅ Output saved to {final_vid}")
    return final_vid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--avatar_id", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--output_id", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--vae_type", type=str, default="sd-vae")
    parser.add_argument("--unet_config", type=str, default="./models/musetalk/config.json")
    parser.add_argument("--unet_model_path", type=str, default="./models/musetalkV15/unet.pth")
    parser.add_argument("--whisper_dir", type=str, default="./models/whisper")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--audio_padding_length_left", type=int, default=2)
    parser.add_argument("--audio_padding_length_right", type=int, default=2)
    parser.add_argument("--use_float16", action="store_true")
    parser.add_argument("--skip_padding", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()
    
    fast_inference(args)