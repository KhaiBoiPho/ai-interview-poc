# api_server.py
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import shutil
import uuid
from production_pipeline import ProductionPipeline

app = FastAPI()

# Initialize pipeline once at startup
pipeline = ProductionPipeline(
    onnx_dir="models/onnx",
    cache_dir="./cache",
    result_dir="./results",
    use_gpu=True
)

# Store video caches in memory
video_caches = {}

@app.post("/preprocess_video")
async def preprocess_video(video: UploadFile = File(...)):
    """Preprocess video and return video_id"""
    video_id = str(uuid.uuid4())
    video_path = f"./temp/{video_id}_input.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    # Save uploaded video
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # Preprocess
    video_cache = pipeline.preprocess_video(video_path, use_cache=False)
    video_caches[video_id] = video_cache
    
    return {
        "video_id": video_id,
        "num_frames": video_cache['num_frames'],
        "fps": video_cache['fps']
    }

@app.post("/generate")
async def generate(
    video_id: str = Form(...),
    audio: UploadFile = File(...)
):
    """Generate video with audio using cached video data"""
    if video_id not in video_caches:
        return {"error": "Video not found. Please preprocess first."}
    
    audio_path = f"./temp/{uuid.uuid4()}_audio.wav"
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    
    # Save uploaded audio
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    # Generate
    video_cache = video_caches[video_id]
    output_path = pipeline.generate_with_audio(
        video_cache=video_cache,
        audio_path=audio_path,
        batch_size=2
    )
    
    return FileResponse(output_path, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)