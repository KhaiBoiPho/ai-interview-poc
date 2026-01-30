"""Main FastAPI application"""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.core.model_loader import model_manager
from app.routes import avatar, webrtc


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="MuseTalk WebRTC Streaming",
        description="Real-time AI avatar streaming with Vietnamese TTS",
        version="1.0.0"
    )
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Load all models on startup"""
        print("\n" + "="*60)
        print("🚀 Starting MuseTalk WebRTC Server")
        print("="*60 + "\n")
        
        model_manager.load_all()
        
        print("\n" + "="*60)
        print("✅ Server ready!")
        print("="*60 + "\n")
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        print("\n🛑 Shutting down...")
        await webrtc.cleanup_connections()
        print("✅ Shutdown complete\n")
    
    # Include routers
    app.include_router(avatar.router)
    app.include_router(webrtc.router)
    
    # Serve static files
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Root endpoint - serve HTML client
    @app.get("/")
    async def index():
        """Serve web client"""
        html_file = Path(__file__).parent / "static" / "index.html"
        if html_file.exists():
            return FileResponse(html_file)
        return {"message": "MuseTalk WebRTC Server", "status": "running"}
    
    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "models_loaded": model_manager.device is not None
        }
    
    return app


# Create app instance
app = create_app()