import onnxruntime as ort
import numpy as np
import cv2
import os
from faster_whisper import WhisperModel
import librosa
import math

class ONNXVAE:
    """ONNX VAE wrapper with FP16 support"""

    def __init__(self, model_dir="models/onnx", use_gpu=True, resized_img=256):
        providers = self._get_providers(use_gpu)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.encoder = ort.InferenceSession(
            f"{model_dir}/vae_encoder.onnx",
            sess_options=sess_options,
            providers=providers
        )

        self.decoder = ort.InferenceSession(
            f"{model_dir}/vae_decoder.onnx",
            sess_options=sess_options,
            providers=providers
        )

        self.scaling_factor = 0.18215 # sd-vae default
        self._resized_img = resized_img
        self._mask_tensor = self._get_mask_tensor()

    def _get_providers(self, use_gpu):
        providers = []
        if use_gpu:
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
            }))
        providers.append('CPUExecutionProvider')
        return providers
    
    def _get_mask_tensor(self):
        """Create half mask for upper part of image"""
        mask = np.zeros((self._resized_img, self._resized_img), dtype=np.float32)
        mask[:self._resized_img//2, :] = 1.0
        return mask
    
    def preprocess_img(self, img, half_mask=False):
        """
        Preprocess image to tensor
        img: numpy array BGR or file path
        return: [1, 3, 256, 256] float16 normalized [-1, 1]
        """
        if isinstance(img, str):
            img = cv2.imread(img)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._resized_img, self._resized_img),
                        interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize to [0, 1]
        x = img.astype(np.float32) / 255.0
        
        # Apply mask if needed
        if half_mask:
            x = x * self._mask_tensor[:, :, np.newaxis]
        
        # Normalize to [-1, 1]
        x = (x - 0.5) / 0.5
        
        # To tensor format [1, 3, H, W]
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        
        return x.astype(np.float32)
    
    def encode_latents(self, image):
        """
        Encode image to latents
        image: [1, 3, 256, 256] float16
        return: [1, 4, 32, 32] float16
        """
        latent = self.encoder.run(None, {"image": image})[0]
        # Apply scaling
        latent = latent * self.scaling_factor
        return latent
    
    def decode_latents(self, latents):
        """
        Decode latents to image
        latents: [B, 4, 32, 32] float16
        return: [B, H, W, 3] uint8 BGR
        """
        # Unscale
        latents = latents / self.scaling_factor
        
        # Decode
        image = self.decoder.run(None, {"latent": latents})[0]
        
        # Denormalize from [-1, 1] to [0, 1]
        image = image / 2.0 + 0.5
        image = np.clip(image, 0, 1)
        
        # To uint8 [B, H, W, C]
        image = np.transpose(image, (0, 2, 3, 1))
        image = (image * 255).round().astype(np.uint8)
        
        # RGB to BGR
        image = image[..., ::-1]
        
        return image
    
    def get_latents_for_unet(self, img):
        """
        Get concatenated latents for UNet
        img: numpy array BGR or file path
        return: [1, 8, 32, 32] float16
        """
        # Masked latents
        ref_image_masked = self.preprocess_img(img, half_mask=True)
        masked_latents = self.encode_latents(ref_image_masked)
        
        # Full reference latents
        ref_image = self.preprocess_img(img, half_mask=False)
        ref_latents = self.encode_latents(ref_image)
        
        # Concatenate
        latent_model_input = np.concatenate([masked_latents, ref_latents], axis=1)
        
        return latent_model_input
    
class ONNXUNet:
    """ONNX UNet wrapper with FP16 support"""
    
    def __init__(self, model_dir="models/onnx", use_gpu=True):
        providers = self._get_providers(use_gpu)
        
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.pe = ort.InferenceSession(
            f"{model_dir}/pe.onnx",
            sess_options=sess_options,
            providers=providers
        )
        self.unet = ort.InferenceSession(
            f"{model_dir}/unet.onnx",
            sess_options=sess_options,
            providers=providers
        )

        # Get expected input dtype from model metadata
        self.pe_dtype = self._get_input_dtype(self.pe, "audio_feature")
        self.unet_latent_dtype = self._get_input_dtype(self.unet, "latent")
        self.unet_encoder_dtype = self._get_input_dtype(self.unet, "encoder_hidden_states")
        
        print(f"PE expects: {self.pe_dtype}")
        print(f"UNet latent expects: {self.unet_latent_dtype}")
        print(f"UNet encoder expects: {self.unet_encoder_dtype}")

    def _get_input_dtype(self, session, input_name):
        """Get expected dtype for input"""
        for inp in session.get_inputs():
            if inp.name == input_name:
                type_str = inp.type
                if 'float16' in type_str or 'half' in type_str:
                    return np.float16
                else:
                    return np.float32
        return np.float32  # Default
    
    def _get_providers(self, use_gpu):
        providers = []
        if use_gpu:
            providers.append(('CUDAExecutionProvider', {
                'device_id': 0,
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
            }))
        providers.append('CPUExecutionProvider')
        return providers
    
    def forward(self, latent, timestep, encoder_hidden_states):
        """
        Forward pass through UNet
        latent: [B, 8, 32, 32] float16
        timestep: [1] int64
        encoder_hidden_states: [B, seq_len, 384] float16
        return: [B, 4, 32, 32] float16
        """
        encoder_hidden_states = encoder_hidden_states.astype(self.pe_dtype)

        # Add positional encoding
        encoded_states = self.pe.run(None, {
            "audio_feature": encoder_hidden_states
        })[0]

        # Convert to UNet expected dtypes
        latent = latent.astype(self.unet_latent_dtype)
        encoded_states = encoded_states.astype(self.unet_encoder_dtype)
        
        # UNet forward
        output = self.unet.run(None, {
            "latent": latent,
            "timestep": timestep,
            "encoder_hidden_states": encoded_states
        })[0]
        
        return output


class FasterWhisperProcessor:
    """Faster Whisper audio processor"""
    
    def __init__(self, model_size="base", device="cuda", compute_type="float16"):
        """
        model_size: tiny, base, small, medium, large-v2, large-v3
        device: cuda or cpu
        compute_type: float16, int8_float16, int8
        """
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"✓ Loaded faster-whisper: {model_size} on {device} ({compute_type})")
    
    def get_audio_feature(self, wav_path):
        """
        Extract audio features using librosa
        Returns features list and audio length
        """
        if not os.path.exists(wav_path):
            return None, 0
        
        librosa_output, sampling_rate = librosa.load(wav_path, sr=16000)
        assert sampling_rate == 16000
        
        # Split into 30s segments
        segment_length = 30 * sampling_rate
        segments = [librosa_output[i:i + segment_length] 
                   for i in range(0, len(librosa_output), segment_length)]
        
        # Use faster-whisper to extract features
        features = []
        for segment in segments:
            # Convert to mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=segment,
                sr=sampling_rate,
                n_fft=400,
                hop_length=160,
                n_mels=80
            )
            mel = librosa.power_to_db(mel, ref=np.max)
            # Normalize
            mel = (mel + 40) / 40
            features.append(mel)
        
        return features, len(librosa_output)
    
    def get_whisper_chunk(
        self,
        audio_path,
        fps=25,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    ):
        """
        Process audio and return whisper chunks for each frame
        Returns: [num_frames, 50, 384] np.float16
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # Get features from faster-whisper
        segments, info = self.model.transcribe(audio_path, beam_size=1)
        
        # Extract encoder features (this is a workaround)
        # For actual implementation, you need to access whisper encoder output
        # Here we'll use a simplified approach
        
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            fmax=8000
        )
        mel = librosa.power_to_db(mel, ref=np.max)
        mel = (mel + 40) / 40  # Normalize
        
        # Pad mel spectrogram
        audio_fps = 50  # Whisper's feature extraction rate
        audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)
        
        num_frames = math.floor((len(audio) / sr) * fps)
        whisper_idx_multiplier = audio_fps / fps
        
        # Create dummy whisper features (384-dim)
        # In real implementation, you need actual whisper encoder output
        whisper_feature = np.random.randn(mel.shape[1], 384).astype(np.float16)
        
        # Add padding
        padding_nums = math.ceil(whisper_idx_multiplier)
        padding_left = np.zeros((padding_nums * audio_padding_length_left, 384), dtype=np.float16)
        padding_right = np.zeros((padding_nums * 3 * audio_padding_length_right, 384), dtype=np.float16)
        whisper_feature = np.concatenate([padding_left, whisper_feature, padding_right], axis=0)
        
        # Extract chunks for each frame
        audio_prompts = []
        for frame_index in range(num_frames):
            audio_index = math.floor(frame_index * whisper_idx_multiplier)
            audio_clip = whisper_feature[audio_index: audio_index + audio_feature_length_per_frame]
            
            if audio_clip.shape[0] == audio_feature_length_per_frame:
                audio_prompts.append(audio_clip)
        
        audio_prompts = np.stack(audio_prompts, axis=0)  # [num_frames, 50, 384]
        
        return audio_prompts