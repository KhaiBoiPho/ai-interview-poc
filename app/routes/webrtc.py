"""WebRTC signaling and streaming endpoints"""
import asyncio
from fastapi import APIRouter, Form, HTTPException
from aiortc import RTCPeerConnection, RTCSessionDescription

from app.services.tts_service import tts_service
from app.services.audio_service import audio_service
from app.services.avatar_service import avatar_service
from app.streaming.video_track import MuseTalkVideoTrack


router = APIRouter(prefix="/webrtc", tags=["webrtc"])

# Active peer connections
pcs = set()


@router.post("/offer")
async def webrtc_offer(
    avatar_id: str = Form(...),
    text: str = Form(...),
    sdp: str = Form(...),
    type: str = Form(...)
):
    """
    Handle WebRTC offer - TTS synthesis + video streaming
    
    Flow:
    1. Validate avatar exists
    2. Synthesize speech from text (TTS)
    3. Process audio to whisper features
    4. Create WebRTC peer connection
    5. Add video track with MuseTalk inference
    6. Return SDP answer
    
    Args:
        avatar_id: Avatar identifier
        text: Vietnamese text for TTS
        sdp: WebRTC SDP offer
        type: SDP type (should be 'offer')
    """
    try:
        # Validate avatar
        if not avatar_service.has_avatar(avatar_id):
            raise HTTPException(
                status_code=404,
                detail=f"Avatar '{avatar_id}' not found. Please prepare it first."
            )
        
        # Validate text
        if not text or len(text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        print(f"\n{'='*60}")
        print("🎯 New WebRTC request")
        print(f"   Avatar: {avatar_id}")
        print(f"   Text: '{text[:100]}...'")
        print(f"{'='*60}\n")
        
        # Step 1: TTS - synthesize speech
        print("Step 1: TTS synthesis...")
        tts_result = tts_service.synthesize(text)
        audio_path = tts_result['audio_path']
        
        # Step 2: Process audio -> whisper features
        print("Step 2: Audio processing...")
        whisper_chunks = audio_service.process_audio_for_inference(audio_path)
        
        print("\n📊 Processing complete:")
        print(f"   Audio duration: {tts_result['duration']:.2f}s")
        print(f"   TTS latency: {tts_result['latency']:.2f}s")
        print(f"   Whisper chunks: {len(whisper_chunks)}")
        print(f"   Video frames: {len(whisper_chunks)}\n")
        
        # Step 3: Create WebRTC peer connection
        print("Step 3: Setting up WebRTC...")
        pc = RTCPeerConnection()
        pcs.add(pc)
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            state = pc.connectionState
            print(f"🔗 Connection state: {state}")
            
            if state == "failed" or state == "closed":
                await pc.close()
                pcs.discard(pc)
                print("   Peer connection closed")
        
        # Step 4: Add video track with MuseTalk
        print("Step 4: Creating video track...")
        video_track = MuseTalkVideoTrack(
            avatar_id=avatar_id,
            audio_frames=whisper_chunks
        )
        pc.addTrack(video_track)
        
        # Step 5: Handle WebRTC signaling
        print("Step 5: WebRTC handshake...")
        offer = RTCSessionDescription(sdp=sdp, type=type)
        await pc.setRemoteDescription(offer)
        
        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        
        print("✅ WebRTC setup complete\n")
        
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
            "audio_duration": tts_result['duration'],
            "tts_latency": tts_result['latency'],
            "num_frames": len(whisper_chunks),
            "sample_rate": tts_result['sample_rate']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get WebRTC connection statistics"""
    return {
        "active_connections": len(pcs),
        "connections": [
            {
                "state": pc.connectionState,
                "ice_connection_state": pc.iceConnectionState,
                "ice_gathering_state": pc.iceGatheringState
            }
            for pc in pcs
        ]
    }


async def cleanup_connections():
    """Cleanup all peer connections"""
    print("🧹 Cleaning up WebRTC connections...")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros, return_exceptions=True)
    pcs.clear()
    print("   ✓ Cleanup complete")