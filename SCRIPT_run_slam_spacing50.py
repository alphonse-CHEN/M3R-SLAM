#!/usr/bin/env python3
"""
Burn script to run MASt3R-SLAM with spacing=50 on cam0_389 dataset.
Runs the SLAM pipeline and visualizes results.
"""

import subprocess
import pathlib
import sys
import time

def main():
    print("=" * 80)
    print("MASt3R-SLAM Burn Script - Spacing 50")
    print("=" * 80)
    
    # Configuration
    dataset_path = "/d_disk/_DataBuffer/RopeCap/20251224_103638/parsed_data/undistort_simple/cam0_389"
    save_path = "/d_disk/_DataBuffer/RopeCap/20251224_103638/parsed_data/undistort_simple/cam0_389_SLAM"
    config_file = "config/base.yaml"
    
    # Check if dataset exists
    dataset = pathlib.Path(dataset_path)
    if not dataset.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return 1
    
    # Count images in dataset
    images = list(dataset.glob("*.png")) + list(dataset.glob("*.jpg")) + list(dataset.glob("*.jpeg"))
    print(f"\n📁 Dataset: {dataset_path}")
    print(f"   Total images: {len(images)}")
    print(f"   With spacing=50: ~{len(images)//50} images will be processed")
    
    # Prepare command
    cmd = [
        "python", "main.py",
        "--dataset", dataset_path,
        "--save-as", save_path,
        "--config", config_file,
        "--no-viz"
    ]
    
    print(f"\n🚀 Running SLAM pipeline...")
    print(f"   Command: {' '.join(cmd)}")
    print(f"   Output will be saved to: {save_path}/m3rResults/")
    print("\n" + "=" * 80)
    
    # Run the command
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print(f"✅ SLAM pipeline completed successfully!")
        print(f"⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        
        # Check output files
        output_dir = pathlib.Path(save_path) / "m3rResults"
        if output_dir.exists():
            ply_file = output_dir / "cam0_389.ply"
            txt_file = output_dir / "cam0_389.txt"
            keyframes_dir = output_dir / "keyframes" / "cam0_389"
            
            print(f"\n📊 Output files:")
            if ply_file.exists():
                size_mb = ply_file.stat().st_size / 1024 / 1024
                print(f"   ✓ Reconstruction: {ply_file.name} ({size_mb:.1f} MB)")
            else:
                print(f"   ✗ Reconstruction: {ply_file.name} (NOT FOUND)")
                
            if txt_file.exists():
                with open(txt_file) as f:
                    num_poses = len(f.readlines())
                print(f"   ✓ Trajectory: {txt_file.name} ({num_poses} poses)")
            else:
                print(f"   ✗ Trajectory: {txt_file.name} (NOT FOUND)")
                
            if keyframes_dir.exists():
                num_keyframes = len(list(keyframes_dir.glob("*.png")) + list(keyframes_dir.glob("*.jpg")))
                print(f"   ✓ Keyframes: {num_keyframes} images saved")
            else:
                print(f"   ✗ Keyframes directory not found")
        else:
            print(f"\n⚠️  Warning: Output directory not found at {output_dir}")
        
        print("\n" + "=" * 80)
        print("🎯 Next steps:")
        print(f"   • View PLY in MeshLab: {output_dir / 'cam0_389.ply'}")
        print(f"   • Check trajectory: {output_dir / 'cam0_389.txt'}")
        print(f"   • Review keyframes: {output_dir / 'keyframes' / 'cam0_389'}")
        print("=" * 80)
        
        return 0
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print("\n" + "=" * 80)
        print(f"❌ SLAM pipeline failed after {elapsed:.1f} seconds")
        print(f"   Error code: {e.returncode}")
        print("=" * 80)
        return e.returncode
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit(main())
