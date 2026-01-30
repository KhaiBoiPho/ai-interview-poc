"""Avatar management endpoints"""
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.avatar_service import avatar_service
from app.config import app_config


router = APIRouter(prefix="/avatar", tags=["avatar"])


@router.post("/prepare")
async def prepare_avatar(
    avatar_id: str = Form(...),
    video_file: UploadFile = File(...),
    bbox_shift: int = Form(0),
    max_frames: Optional[int] = Form(None)
):
    """
    Prepare avatar from uploaded video
    
    Args:
        avatar_id: Unique identifier for avatar
        video_file: Video file upload
        bbox_shift: Bounding box shift parameter
        max_frames: Maximum frames to process
    """
    try:
        # Save uploaded video
        video_path = app_config.upload_dir / f"{avatar_id}.mp4"

        with open(video_path, "wb") as f:
            content = await video_file.read()
            f.write(content)

        # Prepare avatar
        avatar_data = avatar_service.prepare_avatar(
            avatar_id=avatar_id,
            video_path=str(video_path),
            bbox_shift=bbox_shift,
            max_frames=max_frames
        )

        return {
            "status": "success",
            "avatar_id": avatar_id,
            "message": f"Avatar prepared with {len(avatar_data['frame_list_cycle'])} frames",
            "frames_in_cycle": len(avatar_data['frame_list_cycle'])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/list")
async def list_avatars():
    "List all cached avatars"
    avatars = avatar_service.list_avatars()

    return {
        "avatars": avatars,
        "count": len(avatars)
    }


@router.delete("/{avatar_id}")
async def delete_avatar(avatar_id: str):
    """Delete avatar from cache"""
    if not avatar_service.has_avatar(avatar_id):
        raise HTTPException(status_code=404, detail=f"Avatar '{avatar_id}' not found")
    
    # Remove from cache
    del avatar_service.cache[avatar_id]
    
    return {
        "status": "success",
        "message": f"Avatar '{avatar_id}' deleted from cache"
    }


@router.post("/clear")
async def clear_all_avatars():
    """Clear all avatars from cache"""
    avatar_service.clear_cache()
    
    return {
        "status": "success",
        "message": "All avatars cleared from cache"
    }