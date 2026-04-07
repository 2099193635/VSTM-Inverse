"""
Example usage of the separated dataset generation and training modules.

This script demonstrates how to use the new modular structure where:
1. dataset_gen.py handles data loading, preprocessing, and normalization
2. train_forward_minimal.py handles model training and testing
"""

from pathlib import Path
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

from dataset_gen import prepare_dataset, find_npz_files


# Example 1: Using prepare_dataset with a single file
def example_single_file():
    """Example of preparing dataset from a single NPZ file."""
    
    npz_path = Path("results/default_project/高速客车-外部导入-vehicle-standard-20260322_065703/files/simulation_result_spatial.npz")
    
    train_loader, test_loader, component_segments, selected_indices, z_mu, z_sigma, ds, test_records = prepare_dataset(
        npz_path=npz_path,
        target_object="vehicle_full",
        components="disp,vel,acc",
        window_size=256,
        stride=64,
        pred_horizon=1,
        batch_size=8,
        test_ratio=0.2,
        dataset_dir=Path("Dataset"),
    )
    
    print("✓ Single file dataset preparation completed!")
    print(f"  Component segments: {component_segments}")
    print(f"  Z shape: {z_mu.shape}")
    print(f"  Test records: {len(test_records)}")


# Example 2: Load all files from a directory
def example_from_directory():
    """Example of preparing dataset from all NPZ files in a directory (recursive)."""
    
    npz_dir = Path("results/default_project")
    
    train_loader, test_loader, component_segments, selected_indices, z_mu, z_sigma, ds, test_records = prepare_dataset(
        npz_dir=npz_dir,  # Load all matching files recursively
        target_object="vehicle_full",
        components="disp,vel,acc",
        window_size=256,
        stride=64,
        pred_horizon=1,
        batch_size=8,
        test_ratio=0.2,
        npz_pattern="*spatial.npz",  # Pattern to match files
        dataset_dir=Path("Dataset"),
    )
    
    print("✓ Directory-based dataset preparation completed!")
    print(f"  Total training records: {len(test_records)} test records")


# Example 3: Load from a specific list of files
def example_from_file_list():
    """Example of preparing dataset from a specific list of NPZ files."""
    
    # Find all spatial NPZ files in the directory
    npz_dir = Path("results/default_project")
    npz_files = find_npz_files(npz_dir, pattern="*spatial.npz")
    
    if npz_files:
        # Use only the first few files if there are many
        file_list = npz_files[:2]  # Use up to 2 files
        
        train_loader, test_loader, component_segments, selected_indices, z_mu, z_sigma, ds, test_records = prepare_dataset(
            file_list=file_list,
            target_object="vehicle_full",
            components="disp,vel,acc",
            window_size=256,
            stride=64,
            pred_horizon=1,
            batch_size=8,
            test_ratio=0.2,
            dataset_dir=Path("Dataset"),
        )
        
        print("✓ File list-based dataset preparation completed!")
        print(f"  Files loaded: {len(file_list)}")
        print(f"  Total test records: {len(test_records)}")
    else:
        print("No NPZ files found in the directory")


# Example 4: Demo mode (synthetic data)
def example_demo_dataset():
    """Example of using demo mode for testing without real data."""
    
    train_loader, test_loader, component_segments, selected_indices, z_mu, z_sigma, ds, test_records = prepare_dataset(
        target_object="vehicle_full",
        components="disp,vel,acc",
        window_size=128,
        stride=32,
        pred_horizon=1,
        batch_size=4,
        test_ratio=0.2,
        demo=True,
        demo_records=8,
        demo_length=1000,
        dataset_dir=Path("Dataset"),
    )
    
    print("✓ Demo dataset preparation completed!")
    print(f"  Test records: {len(test_records)}")


# Example 5: Find NPZ files matching a pattern
def example_find_npz_files():
    """Example of finding NPZ files in a directory."""
    
    npz_dir = Path("results")
    
    # Find all NPZ files
    all_npz = find_npz_files(npz_dir, pattern="*.npz")
    print(f"Found {len(all_npz)} NPZ files:")
    for f in all_npz[:5]:  # Show first 5
        print(f"  - {f}")
    if len(all_npz) > 5:
        print(f"  ... and {len(all_npz) - 5} more")
    
    # Find spatial NPZ files only
    spatial_npz = find_npz_files(npz_dir, pattern="*spatial.npz")
    print(f"\nFound {len(spatial_npz)} spatial NPZ files")


if __name__ == "__main__":
    print("=" * 70)
    print("PINO Dataset Generation Examples")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("Example 1: Single File")
    print("=" * 70)
    try:
        example_single_file()
    except FileNotFoundError as e:
        print(f"⚠ Could not find data file: {e}")
    
    print("\n" + "=" * 70)
    print("Example 2: Load from Directory")
    print("=" * 70)
    try:
        example_from_directory()
    except FileNotFoundError as e:
        print(f"⚠ Could not find directory: {e}")
    
    print("\n" + "=" * 70)
    print("Example 3: Load from File List")
    print("=" * 70)
    example_from_file_list()
    
    print("\n" + "=" * 70)
    print("Example 4: Demo Mode (Synthetic Data)")
    print("=" * 70)
    example_demo_dataset()
    
    print("\n" + "=" * 70)
    print("Example 5: Find NPZ Files")
    print("=" * 70)
    try:
        example_find_npz_files()
    except FileNotFoundError as e:
        print(f"⚠ Could not find directory: {e}")
