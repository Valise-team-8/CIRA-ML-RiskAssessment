#!/usr/bin/env python3
"""
Script to help organize large files for Google Drive upload
"""
import os
import shutil
from pathlib import Path

def create_upload_folders():
    """Create separate folders for Google Drive upload"""
    
    # Create main upload directory
    upload_dir = Path("google_drive_files")
    upload_dir.mkdir(exist_ok=True)
    
    # Define source and destination mappings
    folders_to_copy = {
        "data": "1_data_files",
        "models": "2_trained_models", 
        "results": "3_results_outputs",
        "conda_envs": "4_conda_environment"
    }
    
    print("Creating organized folders for Google Drive upload...")
    
    for source_folder, dest_name in folders_to_copy.items():
        source_path = Path(source_folder)
        dest_path = upload_dir / dest_name
        
        if source_path.exists():
            print(f"Copying {source_folder} -> {dest_path}")
            if dest_path.exists():
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path)
            
            # Calculate folder size
            total_size = sum(f.stat().st_size for f in dest_path.rglob('*') if f.is_file())
            size_mb = total_size / (1024 * 1024)
            print(f"  Size: {size_mb:.2f} MB")
        else:
            print(f"Warning: {source_folder} not found")
    
    print(f"\nAll files organized in '{upload_dir}' folder")
    print("Upload each numbered folder to Google Drive separately")
    print("Then update the links in LARGE_FILES_README.md")

if __name__ == "__main__":
    create_upload_folders()