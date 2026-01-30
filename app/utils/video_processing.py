import cv2
import subprocess
import time
import numpy as np
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from musetalk.utils.blending import get_image_blending

def process_single_frame(args_tuple):
    """Process a single frame for parallel processing"""
    idx, res_frame, bbox, ori_frame, mask, mask_crop_box = args_tuple

    x1, y1, x2, y2 = bbox
    res_frame_resized = cv2.resize(res_frame.astype(np.unit8), (x2 - x1, y2 - y1))
    combine_frame = get_image_blending(ori_frame, res_frame_resized, bbox, mask, mask_crop_box)

    return idx, combine_frame


def parallel_encode_video_gpu(frame_buffer, cached_data, temp_dir, audio_path, out_vid_name, fps):
    """Optimized encoding with GPU acceleration"""
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

    # Direct one-step encoding
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', 'pipe:0',
        '-i', str(audio_path),
        '-c:v', 'libx264',
        '-preset', 'faster',
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
        except:  # noqa: E722
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
    """Standard encoding method (fallback)"""
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