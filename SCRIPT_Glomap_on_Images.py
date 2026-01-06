import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import numpy as np
import torch
import cv2


def main(args):
    # Load model
    model = AsymmetricCroCo3DStereo.from_pretrained(args.model_name).to(args.device)
    
    # Load images
    images = load_images(args.image_dir, size=512)
    
    if len(images) < 2:
        print("Error: Need at least 2 images")
        return
    
    print(f"Loaded {len(images)} images")
    
    # Fix: Generate pairs as list of tuples with image dictionaries, not indices
    pairs = [(images[i], images[j]) for i in range(len(images)) for j in range(i+1, len(images))]
    
    print(f"Generated {len(pairs)} pairs")
    
    # Run inference on pairs
    output = inference(pairs, model, args.device, batch_size=args.batch_size)
    
    # Global alignment
    scene = global_aligner(output, device=args.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=args.niter, schedule=args.schedule)
    
    print(f"Global alignment loss: {loss}")
    
    # Extract point cloud
    pts3d = scene.get_pts3d()
    confidence = scene.get_masks()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save point cloud
    pts = []
    colors = []
    for i, img in enumerate(images):
        pts_i = pts3d[i].reshape(-1, 3)
        conf_i = confidence[i].reshape(-1)
        
        # Filter by confidence
        mask = conf_i > args.min_conf_thr
        pts_i = pts_i[mask]
        
        # Get colors from image
        img_rgb = cv2.imread(img['img_path'])
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (pts3d[i].shape[1], pts3d[i].shape[0]))
        colors_i = img_rgb.reshape(-1, 3)[mask]
        
        pts.append(pts_i)
        colors.append(colors_i)
    
    pts = np.concatenate(pts, axis=0)
    colors = np.concatenate(colors, axis=0)
    
    # Save as PLY
    ply_path = output_dir / "point_cloud.ply"
    save_ply(ply_path, pts, colors)
    print(f"Saved point cloud to {ply_path}")
    
    # Save camera poses
    poses = scene.get_im_poses().detach().cpu().numpy()
    np.save(output_dir / "poses.npy", poses)
    print(f"Saved camera poses to {output_dir / 'poses.npy'}")


def save_ply(path, pts, colors):
    """Save point cloud as PLY file"""
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for pt, color in zip(pts, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Glomap on images")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--model_name", type=str, default="naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt", 
                        help="Model name")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--niter", type=int, default=300, help="Number of iterations for alignment")
    parser.add_argument("--schedule", type=str, default="cosine", help="Schedule for alignment")
    parser.add_argument("--min_conf_thr", type=float, default=3.0, help="Minimum confidence threshold")
    
    args = parser.parse_args()
    main(args)
