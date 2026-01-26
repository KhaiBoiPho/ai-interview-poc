import os
import onnx
import json
import math
import torch
import torch.nn as nn
from onnxconverter_common import float16
from diffusers import AutoencoderKL, UNet2DConditionModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=384, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        b, seq_len, d_model = x.size()
        pe = self.pe[:, :seq_len, :]
        x = x + pe.to(x.device)
        return x

def convert_to_fp16(onnx_path):
    """Convert ONNX model to FP16"""
    print(f"Converting to FP16: {onnx_path}")
    model = onnx.load(onnx_path)
    model_fp16 = float16.convert_float_to_float16(model)
    fp16_path = onnx_path.replace('.onnx', '_fp16.onnx')
    onnx.save(model_fp16, fp16_path)
    os.remove(onnx_path)
    print(f"✓ Saved: {fp16_path}\n")
    return fp16_path

def export_vae_encoder(model_path="./models/sd-vae", output_dir="models/onnx"):
    """Export VAE Encoder to ONNX FP16"""
    print("="*60)
    print("EXPORTING VAE ENCODER")
    print("="*60)

    vae = AutoencoderKL.from_pretrained(model_path)
    vae.eval()

    # Input: [batch, 3, 256, 256] normalized image
    dummy_input = torch.randn(1, 3, 256, 256)
    output_path = os.path.join(output_dir, "vae_encoder.onnx")

    with torch.no_grad():
        torch.onnx.export(
            vae.encoder,
            dummy_input,
            output_path,
            input_names=["image"],
            output_names=["latent"],
            dynamic_axes={
                "image": {0: "batch"},
                "latent": {0: "batch"}
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    convert_to_fp16(output_path)

def export_vae_decoder(model_path="./models/sd-vae", output_dir="models/onnx"):
    """Export VAE Decoder to ONNX FP16"""
    print("="*60)
    print("EXPORTING VAE DECODER")
    print("="*60)

    vae = AutoencoderKL.from_pretrained(model_path)
    vae.eval()

    # Input: [batch, 4, 32, 32] latent (after scaling)
    dummy_latent = torch.randn(1, 4, 32, 32)
    
    output_path = os.path.join(output_dir, "vae_decoder.onnx")

    with torch.no_grad():
        torch.onnx.export(
            vae.decoder,
            dummy_latent,
            output_path,
            input_names=["latent"],
            output_names=["image"],
            dynamic_axes={
                "latent": {0: "batch"},
                "image": {0: "batch"}
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    convert_to_fp16(output_path)

def export_positional_encoding(output_dir="models/onnx"):
    """Export Positional Encoding to ONNX FP16"""
    print("="*60)
    print("EXPORTING POSITIONAL ENCODING")
    print("="*60)

    pe = PositionalEncoding(d_model=384, max_len=5000)
    pe.eval()

    # Input: [batch, seq_len, 384] from whisper chunks
    # whisper_chunks shape after rearrange: [batch, 50, 384]
    dummy_input = torch.randn(1, 50, 384)
    
    output_path = os.path.join(output_dir, "pe.onnx")

    with torch.no_grad():
        torch.onnx.export(
            pe,
            dummy_input,
            output_path,
            input_names=["audio_feature"],
            output_names=["encoded_feature"],
            dynamic_axes={
                "audio_feature": {0: "batch", 1: "seq_len"},
                "encoded_feature": {0: "batch", 1: "seq_len"}
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    
    convert_to_fp16(output_path)

def export_unet(unet_config_path="./models/musetalk/musetalk.json",
                model_path="./models/musetalkV15/unet.pth",
                output_dir="models/onnx"):
    """Export UNet to ONNX FP16"""
    print("="*60)
    print("EXPORTING UNET")
    print("="*60)
    
    # Load config
    with open(unet_config_path, 'r') as f:
        unet_config = json.load(f)
    
    # Load model
    unet = UNet2DConditionModel(**unet_config)
    weights = torch.load(model_path, map_location='cpu')
    unet.load_state_dict(weights)
    unet.eval()
    
    # Inputs:
    # - latent: [batch, 8, 32, 32] (8 = masked_latents[4] + ref_latents[4])
    # - timestep: [1] 
    # - encoder_hidden_states: [batch, 50, 384] (after PE)
    dummy_latent = torch.randn(1, 8, 32, 32)
    dummy_timestep = torch.tensor([0], dtype=torch.long)
    dummy_encoder_states = torch.randn(1, 50, 384)
    
    output_path = os.path.join(output_dir, "unet.onnx")
    
    with torch.no_grad():
        torch.onnx.export(
            unet,
            (dummy_latent, dummy_timestep, dummy_encoder_states),
            output_path,
            input_names=["latent", "timestep", "encoder_hidden_states"],
            output_names=["output_sample"],
            dynamic_axes={
                "latent": {0: "batch"},
                "encoder_hidden_states": {0: "batch", 1: "seq_len"},
                "output_sample": {0: "batch"}
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    
    convert_to_fp16(output_path)

def main():
    output_dir = "models/onnx"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("  ONNX FP16 EXPORT - MUSETALK")
    print("="*60 + "\n")
    
    # 1. VAE Encoder
    export_vae_encoder(
        model_path="./models/sd-vae",
        output_dir=output_dir
    )
    
    # 2. VAE Decoder
    export_vae_decoder(
        model_path="./models/sd-vae",
        output_dir=output_dir
    )
    
    # 3. Positional Encoding
    export_positional_encoding(output_dir=output_dir)
    
    # 4. UNet
    export_unet(
        unet_config_path="./models/musetalk/musetalk.json",
        model_path="./models/musetalkV15/unet.pth",
        output_dir=output_dir
    )

    print("\n" + "="*60)
    print("✓ EXPORT COMPLETED!")
    print("="*60)
    print(f"\nModels saved in: {output_dir}/")
    print("\nExported models:")
    print("  ✓ vae_encoder_fp16.onnx")
    print("  ✓ vae_decoder_fp16.onnx")
    print("  ✓ pe_fp16.onnx")
    print("  ✓ unet_fp16.onnx")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()