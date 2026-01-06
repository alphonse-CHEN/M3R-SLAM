import os
import numpy as np
from pathlib import Path
import subprocess
from kapture.converter.colmap.database import COLMAPDatabase
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

def process_images_with_glomap(image_folder, output_folder, model_path=None):
    """
    Process images using MASt3R and GLOMAP for 3D reconstruction.
    
    Args:
        image_folder: Path to folder containing input images
        output_folder: Path to folder for output files
        model_path: Path to MASt3R model weights (optional)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load MASt3R model
    if model_path is None:
        model_path = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    
    model = AsymmetricMASt3R.from_pretrained(model_path).to('cuda')
    
    # Get list of images
    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(image_paths) < 2:
        raise ValueError("Need at least 2 images for reconstruction")
    
    print(f"Found {len(image_paths)} images")
    
    # Create image pairs
    pairs = make_pairs(image_paths, scene_graph='complete', prefilter=None, symmetrize=True)
    
    # Run inference
    print("Running MASt3R inference...")
    output = inference(pairs, model, 'cuda', batch_size=1)
    
    # Global alignment
    print("Running global alignment...")
    scene = global_aligner(output, device='cuda', mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)
    
    # Extract matches for COLMAP
    print("Extracting matches...")
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()
    
    # Create COLMAP database
    db_path = os.path.join(output_folder, "database.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    
    db = COLMAPDatabase.connect(db_path)
    db.create_tables()
    
    # Add cameras and images to database
    camera_ids = {}
    image_ids = {}
    
    for idx, (img_path, focal) in enumerate(zip(image_paths, focals)):
        # Add camera
        img_name = os.path.basename(img_path)
        h, w = imgs[idx].shape[:2]
        
        camera_id = db.add_camera(
            model=1,  # SIMPLE_PINHOLE
            width=w,
            height=h,
            params=np.array([focal, w/2, h/2])
        )
        camera_ids[idx] = camera_id
        
        # Add image
        image_id = db.add_image(
            name=img_name,
            camera_id=camera_id
        )
        image_ids[idx] = image_id
    
    # Add matches
    print("Adding matches to database...")
    for i in range(len(image_paths)):
        for j in range(i+1, len(image_paths)):
            # Get matches between images i and j
            pts_i = pts3d[i].reshape(-1, 3)
            pts_j = pts3d[j].reshape(-1, 3)
            
            # Find reciprocal nearest neighbors
            matches = fast_reciprocal_NNs(
                pts_i, pts_j,
                subsample_or_initxy1=8,
                device='cuda',
                dist='dot',
                block_size=2**13
            )
            
            if len(matches) > 0:
                db.add_matches(image_ids[i], image_ids[j], matches)
    
    db.commit()
    db.close()
    
    print(f"COLMAP database created at {db_path}")
    
    # Run GLOMAP
    print("Running GLOMAP...")
    glomap_output = os.path.join(output_folder, "sparse")
    os.makedirs(glomap_output, exist_ok=True)
    
    try:
        subprocess.run([
            "glomap", "mapper",
            "--database_path", db_path,
            "--image_path", image_folder,
            "--output_path", glomap_output
        ], check=True)
        print(f"GLOMAP reconstruction completed. Output in {glomap_output}")
    except subprocess.CalledProcessError as e:
        print(f"GLOMAP failed: {e}")
        print("Make sure GLOMAP is installed and in your PATH")
    
    return scene

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process images with MASt3R and GLOMAP")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to input images")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to output folder")
    parser.add_argument("--model_path", type=str, default=None, help="Path to MASt3R model")
    
    args = parser.parse_args()
    
    scene = process_images_with_glomap(
        args.image_folder,
        args.output_folder,
        args.model_path
    )
