#!/usr/bin/env python3
"""
Script to join two folders of images into a new folder.
Images are renamed as: folder_stem_image_name
"""

import argparse
import pathlib
import shutil
from typing import List

# Common image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

def get_image_files(folder_path: pathlib.Path) -> List[pathlib.Path]:
    """Get all image files from a folder."""
    image_files = []
    for file_path in folder_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file_path)
    return sorted(image_files)

def join_image_folders(folder1: pathlib.Path, folder2: pathlib.Path, output_folder: pathlib.Path):
    """Join two image folders into a new folder with renamed files."""
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Get image files from both folders
    folder1_images = get_image_files(folder1)
    folder2_images = get_image_files(folder2)
    
    print(f"Found {len(folder1_images)} images in {folder1.name}")
    print(f"Found {len(folder2_images)} images in {folder2.name}")
    
    # Copy files from folder1
    for img_file in folder1_images:
        new_name = f"{folder1.stem}_{img_file.name}"
        new_path = output_folder / new_name
        shutil.copy2(img_file, new_path)
        print(f"Copied: {img_file.name} -> {new_name}")
    
    # Copy files from folder2
    for img_file in folder2_images:
        new_name = f"{folder2.stem}_{img_file.name}"
        new_path = output_folder / new_name
        shutil.copy2(img_file, new_path)
        print(f"Copied: {img_file.name} -> {new_name}")
    
    total_copied = len(folder1_images) + len(folder2_images)
    print(f"\nSuccessfully joined {total_copied} images into {output_folder}")

def main():
    parser = argparse.ArgumentParser(
        description="Join two folders of images with renamed files"
    )
    parser.add_argument("folder1", help="Path to first image folder")
    parser.add_argument("folder2", help="Path to second image folder") 
    parser.add_argument("output", help="Path to output folder")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without copying")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    folder1 = pathlib.Path(args.folder1)
    folder2 = pathlib.Path(args.folder2)
    output_folder = pathlib.Path(args.output)
    
    # Validate input folders
    if not folder1.exists() or not folder1.is_dir():
        print(f"Error: {folder1} is not a valid directory")
        return
    
    if not folder2.exists() or not folder2.is_dir():
        print(f"Error: {folder2} is not a valid directory")
        return
    
    if args.dry_run:
        print("DRY RUN - No files will be copied")
        folder1_images = get_image_files(folder1)
        folder2_images = get_image_files(folder2)
        
        print(f"\nWould copy from {folder1.name}:")
        for img_file in folder1_images:
            new_name = f"{folder1.stem}_{img_file.name}"
            print(f"  {img_file.name} -> {new_name}")
        
        print(f"\nWould copy from {folder2.name}:")
        for img_file in folder2_images:
            new_name = f"{folder2.stem}_{img_file.name}"
            print(f"  {img_file.name} -> {new_name}")
        
        print(f"\nTotal: {len(folder1_images) + len(folder2_images)} images would be copied to {output_folder}")
    else:
        join_image_folders(folder1, folder2, output_folder)

if __name__ == "__main__":
    main()