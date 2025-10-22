import os
import glob

def find_results_files():
    print("🔍 Searching for results files...")
    
    # Search entire D: drive for your files
    search_patterns = [
        "*/results/risk_table.csv",
        "*/results/*.png", 
        "*/results/*.json"
    ]
    
    for pattern in search_patterns:
        print(f"\nSearching for: {pattern}")
        files = glob.glob(f"D:/{pattern}", recursive=True)
        for file in files:
            print(f"✅ FOUND: {file}")
    
    # Check common locations
    common_locations = [
        "D:\\CIRA-ML-RiskAssessment\\results",
        "D:\\results", 
        os.path.join(os.getcwd(), "results"),
        os.path.dirname(os.getcwd()) + "\\results"
    ]
    
    print(f"\n📋 Checking common locations:")
    for location in common_locations:
        if os.path.exists(location):
            print(f"✅ EXISTS: {location}")
            files = os.listdir(location)
            print(f"   Files: {files[:5]}...")  # Show first 5 files

if __name__ == "__main__":
    find_results_files()