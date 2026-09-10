#!/usr/bin/env python3
"""
Consolidate epoch_*_samples.json files into a single aggregated JSON.

Groups samples by 'original' text and collects xt_target, xt_index, recon_noisy, 
and t across all epochs.

Usage:
    python consolidate_samples.py --input-dir /path/to/eval/folder --output-file /path/to/output.json
"""

import json
import os
import argparse
import glob
from collections import defaultdict
from typing import Dict, List, Any


def consolidate_samples(input_dir: str, output_file: str) -> None:
    """
    Consolidate all epoch_*_samples.json files into a single indexed JSON.
    
    Args:
        input_dir: Directory containing epoch_*_samples.json files.
        output_file: Path to save the consolidated output JSON.
    """
    
    # Find all epoch sample files
    sample_files = sorted(glob.glob(os.path.join(input_dir, "epoch_*_samples.json")))
    
    if not sample_files:
        print(f"No epoch_*_samples.json files found in {input_dir}")
        return
    
    print(f"Found {len(sample_files)} sample files")
    
    # Group samples by original text
    consolidated: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    for sample_file in sample_files:
        # Extract epoch number from filename
        basename = os.path.basename(sample_file)
        # e.g., "epoch_10_samples.json" -> epoch = 10
        epoch_str = basename.replace("epoch_", "").replace("_samples.json", "")
        try:
            epoch = int(epoch_str)
        except ValueError:
            print(f"Warning: Could not parse epoch from {basename}, skipping")
            continue
        
        # Load the sample file
        try:
            with open(sample_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {sample_file}: {e}")
            continue
        
        # Extract samples list (handle both direct list and nested structure)
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict) and "samples" in data:
            samples = data["samples"]
        else:
            print(f"Warning: Unexpected format in {sample_file}")
            continue
        
        # Group by original
        for sample in samples:
            if "original" not in sample:
                print(f"Warning: Sample in {basename} missing 'original' key")
                continue
            
            original = sample["original"]
            
            # Extract the required fields
            entry = {
                "xt_target": sample.get("xt_target"),
                "xt_index": sample.get("xt_index"),
                "recon_noisy": sample.get("recon_noisy"),
                "t": sample.get("t"),
            }
            
            consolidated[original].append(entry)
    
    # Sort by t value in increasing order within each original group
    for original in consolidated:
        consolidated[original].sort(key=lambda x: x["t"])
    
    # Convert defaultdict to regular dict for JSON serialization
    consolidated_dict = dict(consolidated)
    
    # Save consolidated output
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    try:
        with open(output_file, 'w') as f:
            json.dump(consolidated_dict, f, indent=2)
        print(f"Consolidated {len(consolidated_dict)} unique originals")
        print(f"Saved to {output_file}")
    except IOError as e:
        print(f"Error writing output file {output_file}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate epoch_*_samples.json files into a single indexed JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python consolidate_samples.py --input-dir ./output/p2/eval --output-file ./consolidated_samples.json
    python consolidate_samples.py -i /path/to/eval -o /path/to/output.json
        """
    )
    
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Directory containing epoch_*_samples.json files",
    )
    
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        required=True,
        help="Path to save the consolidated output JSON file",
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    consolidate_samples(args.input_dir, args.output_file)


if __name__ == "__main__":
    main()
