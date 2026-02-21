"""
Fetch Benchmark Dataset
=======================
Downloads 100 clean Python functions from the sanitized MBPP
(Mostly Basic Python Problems) dataset to use for benchmarking.
This provides a credible, standardized corpus for the SSRN preprint.
"""

from datasets import load_dataset
from pathlib import Path

def fetch_mbpp_dataset():
    print("Downloading MBPP dataset...")
    # Load the sanitized version of MBPP which contains clean, standalone functions
    dataset = load_dataset("mbpp", "sanitized")
    
    clean_dir = Path(__file__).parent / "data" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing files to ensure a clean run
    for f in clean_dir.glob("*.py"):
        f.unlink()
        
    count = 0
    # Process the train/test splits to get 100 good functions
    for split in ["test", "train", "validation"]:
        for item in dataset[split]:
            code = item["code"]
            
            # Simple check to ensure it's a function and not just a script
            if "def " in code and len(code.split("\n")) > 4:
                file_path = clean_dir / f"mbpp_{item['task_id']:03d}.py"
                
                # Write the clean code
                file_path.write_text(code + "\n")
                count += 1
                
                if count >= 100:
                    break
        if count >= 100:
            break
            
    print(f"Successfully saved {count} benchmark functions to {clean_dir}")

if __name__ == "__main__":
    fetch_mbpp_dataset()
