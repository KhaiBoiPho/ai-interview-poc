import torch

class CUDAStreamManager:
    """Manages CUDA stream for async processing"""

    def __init__(self, device):
        self.device = device
        self.streams = []

    def get_stream(self):
        """Get or create a CUDA stream"""
        if not self.streams:
            return torch.cuda.Stream(device=self.device)
        return self.streams.pop()
    
    def return_stream(self, stream):
        """Return stream to pool"""
        self.streams.append(stream)


stream_manager = None


def initialize_stream_manager(device):
    """Initialize global stream manager"""
    global stream_manager
    if torch.cuda.is_available():
        stream_manager = CUDAStreamManager(device=device)
    return stream_manager


def get_vram_usage(self):
    """Get current VRAM usage"""
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated(self.device) / 1024**3
    reserved = torch.cuda.memory_reserved(self.device) / 1024**3
    total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
    
    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'total_gb': total,
        'free_gb': total - reserved
    }