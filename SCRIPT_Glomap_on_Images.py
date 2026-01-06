#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Script to run Glomap on a set of images
# --------------------------------------------------------
import os
import argparse
from pathlib import Path
import numpy as np
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.utils.image import load_images
import tempfile
import subprocess

def get_args_parser():
    parser = argparse.ArgumentParser('Glomap on images', add_help=False)
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image directory')
    parser.add_argument('--model_name', type=str, default='MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric', 
                       help='Model name')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--schedule', type=str, default='cosine', help='Learning rate schedule')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--niter', type=int, default=300, help='Number of iterations')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    return parser

def main(args):
    # Load model from local checkpoint
    model_path = Path("checkpoints/" + args.model_name + '.pth')
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {model_path}. "
            f"Please ensure the model file exists in the checkpoints folder at the root level."
        )
    
    print(f"Loading model from local checkpoint: {model_path}")
    model = AsymmetricMASt3R.from_pretrained(str(model_path)).to(args.device)
    
    # Load images
    print(f"Loading images from {args.image_dir}")
    images = load_images(args.image_dir, size=512)
    
    if len(images) == 0:
        raise ValueError(f"No images found in {args.image_dir}")
    
    print(f"Loaded {len(images)} images")
    
    # Run inference
    print("Running inference...")
    pairs = [(i, j) for i in range(len(images)) for j in range(i+1, len(images))]
    output = inference(pairs, model, args.device, batch_size=args.batch_size)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results for Glomap
    # Export camera parameters and points
    print(f"Saving results to {args.output_dir}")
    
    # Create temporary directory for Glomap input
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Export data in Glomap format
        # This is a simplified version - you may need to adjust based on actual Glomap requirements
        
        # Save images to temp directory
        import shutil
        img_dir = os.path.join(tmp_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        
        for idx, img_data in enumerate(images):
            img_path = img_data['img_path'] if 'img_path' in img_data else None
            if img_path and os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(img_dir, f"{idx:04d}.jpg"))
        
        # Run Glomap (placeholder - adjust command based on your Glomap installation)
        try:
            print("Running Glomap...")
            # Example command - adjust based on your Glomap setup
            cmd = f"glomap mapper --image_path {img_dir} --output_path {args.output_dir}"
            subprocess.run(cmd, shell=True, check=True)
            print("Glomap completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Glomap failed: {e}")
            print("Saving raw inference output instead")
            # Save raw output as fallback
            np.savez(os.path.join(args.output_dir, 'inference_output.npz'), **output)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
