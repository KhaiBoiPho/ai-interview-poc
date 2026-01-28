# convert_to_onnx.py - Fixed version
import torch
import onnx
from onnx import numpy_helper
from onnxconverter_common import float16
import os
import json
import math
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import WhisperModel

class PositionalEncoding(torch.nn.Module):
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

def convert_to_fp16_safe(onnx_path):
    """
    Convert ONNX to FP16 with safe operations kept in FP32
    """
    print(f"Converting {onnx_path} to FP16 (safe mode)...")
    
    model = onnx.load(onnx_path)
    
    # Keep certain ops in FP32 for numerical stability
    keep_fp32_ops = {
        'Softmax', 'LayerNormalization', 'InstanceNormalization', 
        'BatchNormalization', 'ReduceMean', 'ReduceSum'
    }
    
    try:
        model_fp16 = float16.convert_float_to_float16(
            model,
            keep_io_types=True,  # Keep input/output as is
            op_block_list=keep_fp32_ops  # Keep these ops in FP32
        )
        
        fp16_path = onnx_path.replace('.onnx', '_fp16.onnx')
        onnx.save(model_fp16, fp16_path)
        
        print(f"✓ Saved: {fp16_path}\n")
        return fp16_path
    except Exception as e:
        print(f"⚠️  FP16 conversion failed: {e}")
        print(f"Keeping FP32 version: {onnx_path}\n")
        return onnx_path

def export_vae_encoder(model_path="./models/sd-vae", output_dir="models/onnx"):
    """Export VAE Encoder"""
    print("="*60)
    print("EXPORTING VAE ENCODER")
    print("="*60)
    
    vae = AutoencoderKL.from_pretrained(model_path)
    vae.eval()
    
    dummy_input = torch.randn(1, 3, 256, 256)
    output_path = os.path.join(output_dir, "vae_encoder.onnx")
    
    print(f"Exporting to {output_path}...")
    
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
            do_constant_folding=True
        )
    
    print("✓ FP32 exported")
    convert_to_fp16_safe(output_path)

def export_vae_decoder(model_path="./models/sd-vae", output_dir="models/onnx"):
    """Export VAE Decoder"""
    print("="*60)
    print("EXPORTING VAE DECODER")
    print("="*60)
    
    vae = AutoencoderKL.from_pretrained(model_path)
    vae.eval()
    
    dummy_latent = torch.randn(1, 4, 32, 32)
    output_path = os.path.join(output_dir, "vae_decoder.onnx")
    
    print(f"Exporting to {output_path}...")
    
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
            do_constant_folding=True
        )
    
    print("✓ FP32 exported")
    convert_to_fp16_safe(output_path)

def export_positional_encoding(output_dir="models/onnx"):
    """Export Positional Encoding"""
    print("="*60)
    print("EXPORTING POSITIONAL ENCODING")
    print("="*60)
    
    pe = PositionalEncoding(d_model=384, max_len=5000)
    pe.eval()
    
    dummy_input = torch.randn(1, 50, 384)
    output_path = os.path.join(output_dir, "pe.onnx")
    
    print(f"Exporting to {output_path}...")
    
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
            do_constant_folding=True
        )
    
    print("✓ FP32 exported")
    convert_to_fp16_safe(output_path)

def export_unet(unet_config_path="./models/musetalkV15/musetalk.json",
                model_path="./models/musetalkV15/unet.pth",
                output_dir="models/onnx"):
    """Export UNet"""
    print("="*60)
    print("EXPORTING UNET")
    print("="*60)
    
    with open(unet_config_path, 'r') as f:
        unet_config = json.load(f)
    
    unet = UNet2DConditionModel(**unet_config)
    weights = torch.load(model_path, map_location='cpu')
    unet.load_state_dict(weights)
    unet.eval()
    
    dummy_latent = torch.randn(1, 8, 32, 32)
    dummy_timestep = torch.tensor([0], dtype=torch.long)
    dummy_encoder_states = torch.randn(1, 50, 384)
    
    output_path = os.path.join(output_dir, "unet.onnx")
    
    print(f"Exporting to {output_path}...")
    
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
            do_constant_folding=True
        )
    
    print("✓ FP32 exported")
    convert_to_fp16_safe(output_path)

def main():
    output_dir = "models/onnx"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("  ONNX FP16 EXPORT - MUSETALK")
    print("="*60 + "\n")
    
    try:
        export_vae_encoder(
            model_path="./models/sd-vae",
            output_dir=output_dir
        )
        
        export_vae_decoder(
            model_path="./models/sd-vae",
            output_dir=output_dir
        )
        
        export_positional_encoding(output_dir=output_dir)
        
        export_unet(
            unet_config_path="./models/musetalkV15/musetalk.json",
            model_path="./models/musetalkV15/unet.pth",
            output_dir=output_dir
        )
        
        print("\n" + "="*60)
        print("✓ EXPORT COMPLETED!")
        print("="*60)
        print(f"\nModels saved in: {output_dir}/")
        print("\nExported models:")
        
        for f in os.listdir(output_dir):
            if f.endswith('.onnx'):
                size = os.path.getsize(os.path.join(output_dir, f)) / (1024*1024)
                print(f"  ✓ {f} ({size:.1f} MB)")
        
    except Exception as e:
        print(f"\n❌ Error during export: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()