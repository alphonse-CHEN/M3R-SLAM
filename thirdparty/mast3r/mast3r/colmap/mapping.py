# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R colmap mapping script
# --------------------------------------------------------
import pycolmap
import os
from kapture.io.csv import kapture_from_dir
from kapture.io.csv import table_to_file
import kapture
from PIL import Image
from tqdm import tqdm
import numpy as np
from typing import Optional

from mast3r.model import AsymmetricMASt3R
from mast3r.colmap.database import COLMAPDatabase, blob_to_array
from mast3r.utils.path_to_dust3r import DUST3R_REPO_PATH

import sys
sys.path.append(str(DUST3R_REPO_PATH))
from dust3r_visloc.datasets.utils import get_resize_function, rescale_points2d

import kapture
from kapture.io.structure import delete_existing_kapture_files
from kapture.io.csv import kapture_to_dir
from kapture.io.features import get_features_fullpath, keypoints_to_file
from kapture.io.records import get_image_fullpath

from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import geotrf
from dust3r.inference import inference


def scene_prepare_images(kapture_data, topk, resize):
    """
    Prepare images for processing with MASt3R.
    
    Args:
        kapture_data: Kapture dataset object
        topk: Maximum number of images to process
        resize: Target image size for resizing
    
    Returns:
        Tuple of (records_camera, images_np, intrinsics_np)
    """
    n_imgs = len(kapture_data.records_camera)
    resize_func, _, to_orig = get_resize_function(resize, 512, "cpu")

    records_camera = []
    images_np = []
    intrinsics_np = []

    for i, (timestamp, sensor_id, image_name) in tqdm(
        enumerate(kapture.flatten(kapture_data.records_camera)), 
        total=n_imgs,
        disable=n_imgs < topk
    ):
        if i >= topk:
            break

        records_camera.append((timestamp, sensor_id, image_name))
        camera_params = kapture_data.sensors[sensor_id].camera_params

        image_path = get_image_fullpath(kapture_data.kapture_path, image_name)
        image_rgb = np.array(Image.open(image_path).convert('RGB'))
        image_rgb_torch, _, intrinsics_torch = resize_func(image_rgb, camera_params)

        images_np.append(to_numpy(image_rgb_torch).astype(np.uint8))
        intrinsics_np.append(to_numpy(intrinsics_torch))

    return records_camera, images_np, intrinsics_np


def remove_duplicates(pair_list):
    """
    Remove duplicate pairs from the list, keeping only unique unordered pairs.
    
    Args:
        pair_list: List of pairs (tuples)
    
    Returns:
        List of unique pairs
    """
    pair_set = set()
    res = []
    for i, j in pair_list:
        if (j, i) not in pair_set:
            pair_set.add((i, j))
            res.append((i, j))
    return res


def run_mast3r_matching(model, device, images_np, intrinsics_np, output_path):
    """
    Run MASt3R matching on images and save results to COLMAP database.
    
    Args:
        model: MASt3R model
        device: Device to run inference on
        images_np: List of images as numpy arrays
        intrinsics_np: List of intrinsic matrices
        output_path: Path to output COLMAP database
    
    Returns:
        Tuple of (images, keypoints, matches)
    """
    db = COLMAPDatabase.connect(output_path)
    db.create_tables()

    keypoints = {}
    descriptors = {}
    matches = {}

    images = {}
    for i, (image_np, intrinsics) in tqdm(
        enumerate(zip(images_np, intrinsics_np)), 
        total=len(images_np),
        desc="Adding images to database"
    ):
        image_id = db.add_image(f"{i:04d}.jpg", camera_id=i+1)
        images[i] = image_id
        
        width, height = image_np.shape[1], image_np.shape[0]
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        db.add_camera(
            model=1,  # PINHOLE
            width=width,
            height=height,
            params=np.array([fx, fy, cx, cy]),
            camera_id=i+1
        )

    # Generate all pairs
    pair_list = []
    n_imgs = len(images_np)
    for i in range(n_imgs):
        for j in range(i+1, n_imgs):
            pair_list.append((i, j))

    pair_list = remove_duplicates(pair_list)

    print(f"Running MASt3R on {len(pair_list)} pairs...")
    for i, j in tqdm(pair_list, desc="Processing pairs"):
        view1 = {
            'img': images_np[i],
            'idx': i,
            'instance': f'{i:04d}',
            'intrinsics': intrinsics_np[i]
        }
        view2 = {
            'img': images_np[j],
            'idx': j,
            'instance': f'{j:04d}',
            'intrinsics': intrinsics_np[j]
        }

        output = inference([tuple([view1, view2])], model, device, batch_size=1)
        pred1, pred2 = output['pred1'][0], output['pred2'][0]

        pts3d_1 = pred1['pts3d']
        pts3d_2 = pred2['pts3d_in_other_view']

        conf_1 = pred1['conf']
        conf_2 = pred2['conf']

        desc_conf_1 = pred1['desc_conf']
        desc_conf_2 = pred2['desc_conf']

        desc_1 = pred1['desc']
        desc_2 = pred2['desc']

        H, W, _ = pts3d_1.shape
        valid_1 = conf_1 > 1.001
        valid_2 = conf_2 > 1.001

        pts2d_1 = np.mgrid[:W, :H].T.astype(np.float32)
        pts2d_2 = geotrf(intrinsics_np[j], pts3d_2, norm_coords=False)
        pts2d_2 = to_numpy(pts2d_2).reshape(H, W, 2)

        valid = valid_1 & valid_2
        pts2d_1_match = pts2d_1[valid]
        pts2d_2_match = pts2d_2[valid]
        desc_conf_1_match = to_numpy(desc_conf_1.reshape(H, W)[valid])
        desc_conf_2_match = to_numpy(desc_conf_2.reshape(H, W)[valid])

        if i not in keypoints:
            keypoints[i] = pts2d_1.reshape(-1, 2)
            descriptors[i] = to_numpy(desc_1).T
            db.add_keypoints(images[i], keypoints[i])
            db.add_descriptors(images[i], descriptors[i])

        if j not in keypoints:
            keypoints[j] = pts2d_2.reshape(-1, 2)
            descriptors[j] = to_numpy(desc_2).T
            db.add_keypoints(images[j], keypoints[j])
            db.add_descriptors(images[j], descriptors[j])

        # Create matches
        if len(pts2d_1_match) > 0:
            kp1_flat = keypoints[i]
            kp2_flat = keypoints[j]

            kp1_match_idx = np.zeros(len(pts2d_1_match), dtype=np.int32)
            kp2_match_idx = np.zeros(len(pts2d_2_match), dtype=np.int32)

            for k, (p1, p2) in enumerate(zip(pts2d_1_match, pts2d_2_match)):
                idx1 = np.argmin(np.sum((kp1_flat - p1) ** 2, axis=1))
                idx2 = np.argmin(np.sum((kp2_flat - p2) ** 2, axis=1))
                kp1_match_idx[k] = idx1
                kp2_match_idx[k] = idx2

            match_matrix = np.column_stack([kp1_match_idx, kp2_match_idx])
            matches[(i, j)] = match_matrix
            db.add_matches(images[i], images[j], match_matrix)

    db.commit()
    db.close()

    return images, keypoints, matches


def pycolmap_run_triangulator(database_path, image_path, input_path, output_path):
    """
    Run COLMAP point triangulator.
    
    Args:
        database_path: Path to COLMAP database
        image_path: Path to images directory
        input_path: Path to input reconstruction
        output_path: Path to output reconstruction
    """
    pycolmap.triangulate_points(
        reconstruction=pycolmap.Reconstruction(input_path),
        database_path=database_path,
        image_path=image_path,
        output_path=output_path,
    )


def pycolmap_run_mapper(database_path, image_path, output_path, options=None):
    """
    Run COLMAP incremental mapper.
    
    Args:
        database_path: Path to COLMAP database
        image_path: Path to images directory
        output_path: Path to output reconstruction
        options: Optional mapper options
    """
    if options is None:
        options = pycolmap.IncrementalPipelineOptions()
    
    os.makedirs(output_path, exist_ok=True)
    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_path,
        output_path=output_path,
        options=options,
    )
    print(f"Reconstructed {len(maps)} models")
    return maps


def glomap_run_mapper(database_path, image_path, output_path, options=None):
    """
    Run GLOMAP mapper.
    
    Args:
        database_path: Path to COLMAP database
        image_path: Path to images directory
        output_path: Path to output reconstruction
        options: Optional mapper options
    """
    try:
        import pyglomap
    except ImportError:
        raise ImportError("GLOMAP is not installed. Please install it to use this function.")
    
    if options is None:
        options = {
            "max_num_iterations": 100,
            "ba_global_max_num_iterations": 20,
        }
    
    os.makedirs(output_path, exist_ok=True)
    reconstruction = pyglomap.glomap_mapper(
        database_path=database_path,
        image_path=image_path,
        output_path=output_path,
        **options,
    )
    
    return reconstruction


def kapture_import_image_folder_or_list(
    images_path: str,
    kapture_path: str,
    force_overwrite_existing: bool = False,
    images_list: Optional[list] = None
):
    """
    Import images from folder or list into kapture format.
    
    Args:
        images_path: Path to images directory
        kapture_path: Path to output kapture directory
        force_overwrite_existing: Whether to overwrite existing files
        images_list: Optional list of image filenames to import
    """
    os.makedirs(kapture_path, exist_ok=True)
    delete_existing_kapture_files(kapture_path, force_overwrite_existing)

    cameras = kapture.Sensors()
    images = kapture.RecordsCamera()

    if images_list is None:
        images_list = sorted([f for f in os.listdir(images_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for i, image_name in enumerate(images_list):
        image_path = os.path.join(images_path, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} does not exist, skipping...")
            continue

        img = Image.open(image_path)
        width, height = img.size

        # Create a simple pinhole camera
        camera_id = f"camera_{i}"
        focal = max(width, height)
        camera_params = [focal, focal, width/2, height/2]
        
        cameras[camera_id] = kapture.Camera(
            kapture.CameraType.PINHOLE,
            camera_params
        )

        images[i, camera_id] = image_name

    kapture_data = kapture.Kapture(
        sensors=cameras,
        records_camera=images
    )

    kapture_to_dir(kapture_path, kapture_data)
    print(f"Imported {len(images_list)} images to {kapture_path}")
