import os
import shutil
import glob
import gc

def cleanup_previous_results():
    """Clean up all previously generated results from wrong locations"""
    
    print("🧹 CLEANING UP PREVIOUS RESULTS...")
    print("=" * 50)
    
    # Locations to clean up
    locations_to_clean = [
        "D:\\results",  # The duplicate D:\results location
        "D:\\CIRA-ML-RiskAssessment\\src\\results",  # Any src/results
        "D:\\CIRA-ML-RiskAssessment\\results",  # The correct one (we'll recreate it)
    ]
    
    # Files to clean up in root directories
    files_to_clean = [
        "D:\\metrics_report.json",
        "D:\\risk_table.csv", 
        "D:\\data_summary.json",
    ]
    
    # Pattern matches for additional cleanup
    patterns_to_clean = [
        "D:\\*.png",
        "D:\\*.json",
        "D:\\*.csv",
        "D:\\*.txt",
        "D:\\CIRA-ML-RiskAssessment\\src\\*.png",
        "D:\\CIRA-ML-RiskAssessment\\src\\*.json", 
        "D:\\CIRA-ML-RiskAssessment\\src\\*.csv",
        "D:\\CIRA-ML-RiskAssessment\\*.png",
        "D:\\CIRA-ML-RiskAssessment\\*.json",
        "D:\\CIRA-ML-RiskAssessment\\*.csv",
    ]
    
    total_cleaned = 0
    
    # Clean directories
    for location in locations_to_clean:
        if os.path.exists(location):
            try:
                file_count = len([name for name in os.listdir(location) if os.path.isfile(os.path.join(location, name))])
                print(f"🗑️  Removing directory: {location} ({file_count} files)")
                shutil.rmtree(location)
                total_cleaned += file_count
                print(f"✅ Removed: {location}")
            except Exception as e:
                print(f"⚠️  Could not remove {location}: {e}")
    
    # Clean individual files
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            try:
                size_kb = os.path.getsize(file_path) / 1024
                print(f"🗑️  Removing file: {file_path} ({size_kb:.1f} KB)")
                os.remove(file_path)
                total_cleaned += 1
                print(f"✅ Removed: {file_path}")
            except Exception as e:
                print(f"⚠️  Could not remove {file_path}: {e}")
    
    # Clean using patterns
    for pattern in patterns_to_clean:
        for file_path in glob.glob(pattern):
            if os.path.isfile(file_path):
                try:
                    size_kb = os.path.getsize(file_path) / 1024
                    print(f"🗑️  Removing pattern file: {file_path} ({size_kb:.1f} KB)")
                    os.remove(file_path)
                    total_cleaned += 1
                    print(f"✅ Removed: {file_path}")
                except Exception as e:
                    print(f"⚠️  Could not remove {file_path}: {e}")
    
    # Force garbage collection
    gc.collect()
    
    # Recreate the correct results directory
    correct_results_dir = "D:\\CIRA-ML-RiskAssessment\\results"
    os.makedirs(correct_results_dir, exist_ok=True)
    
    print("=" * 50)
    print(f"🎉 CLEANUP COMPLETED!")
    print(f"📊 Total items cleaned: {total_cleaned}")
    print(f"📁 Fresh results directory: {correct_results_dir}")
    print(f"🚀 You can now run main.py for clean results!")

if __name__ == "__main__":
    cleanup_previous_results()