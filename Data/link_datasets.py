"""
Interactive tool to link ant datasets to calibration and branch sets.
"""

import json
import os
from pathlib import Path
import glob

DATASET_LINKS_FILE = Path(__file__).parent / "dataset_links.json"
VIDEOS_2D_DATA_DIR = Path(__file__).parent / "Videos" / "2D_data"
CALIBRATION_SETS_FILE = Path(__file__).parent / "Calibration" / "calibration_sets.json"
BRANCH_SETS_FILE = Path(__file__).parent / "Videos" / "branch_sets.json"


def load_dataset_links():
    """Load dataset links from JSON file."""
    if DATASET_LINKS_FILE.exists():
        with open(DATASET_LINKS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_dataset_links(dataset_links):
    """Save dataset links to JSON file."""
    with open(DATASET_LINKS_FILE, 'w') as f:
        json.dump(dataset_links, f, indent=2)


def load_calibration_sets():
    """Load calibration sets."""
    if CALIBRATION_SETS_FILE.exists():
        with open(CALIBRATION_SETS_FILE, 'r') as f:
            return json.load(f)
    return {}


def load_branch_sets():
    """Load branch sets."""
    if BRANCH_SETS_FILE.exists():
        with open(BRANCH_SETS_FILE, 'r') as f:
            return json.load(f)
    return {}


def find_ant_datasets():
    """Find all available ant datasets from 2D data files."""
    if not VIDEOS_2D_DATA_DIR.exists():
        return []
    
    # Find all files with camera suffixes (L, T, R, F)
    files = glob.glob(str(VIDEOS_2D_DATA_DIR / "*.xlsx"))
    
    # Extract unique dataset names (e.g., "11U1" from "11U1L.xlsx")
    datasets = set()
    for file in files:
        name = Path(file).stem
        # Remove camera suffix if present
        if name.endswith(('L', 'T', 'R', 'F')):
            dataset_name = name[:-1]
            datasets.add(dataset_name)
    
    return sorted(list(datasets))


def link_dataset():
    """Link an ant dataset to calibration and branch sets."""
    print("\n" + "="*60)
    print("LINK ANT DATASET")
    print("="*60)
    
    # Find available datasets
    datasets = find_ant_datasets()
    if not datasets:
        print("\nNo ant datasets found in 2D_data folder.")
        return
    
    print("\nAvailable ant datasets:")
    for i, name in enumerate(datasets, 1):
        print(f"  {i}. {name}")
    
    dataset_name = input("\nEnter ant dataset name: ").strip()
    if dataset_name not in datasets:
        print(f"Error: Dataset '{dataset_name}' not found.")
        return
    
    dataset_links = load_dataset_links()
    
    # Load calibration sets
    calibration_sets = load_calibration_sets()
    if not calibration_sets:
        print("Error: No calibration sets found. Please create calibration sets first.")
        return
    
    print("\nAvailable calibration sets:")
    for i, name in enumerate(calibration_sets.keys(), 1):
        print(f"  {i}. {name}")
    
    cal_set = input("\nEnter calibration set name: ").strip()
    if cal_set not in calibration_sets:
        print(f"Error: Calibration set '{cal_set}' not found.")
        return
    
    # Load branch sets
    branch_sets = load_branch_sets()
    if not branch_sets:
        print("Error: No branch sets found. Please create branch sets first.")
        return
    
    print("\nAvailable branch sets:")
    for i, name in enumerate(branch_sets.keys(), 1):
        print(f"  {i}. {name}")
    
    branch_set = input("\nEnter branch set name: ").strip()
    if branch_set not in branch_sets:
        print(f"Error: Branch set '{branch_set}' not found.")
        return
    
    # Determine species from dataset name (11U/12U = WR, 21U/22U = NWR)
    if dataset_name.startswith(('11U', '12U')):
        species = 'WR'
    elif dataset_name.startswith(('21U', '22U')):
        species = 'NWR'
    else:
        species = input(f"Enter species (WR/NWR) for {dataset_name}: ").strip().upper()
        if species not in ['WR', 'NWR']:
            print("Error: Species must be WR or NWR")
            return
    
    # Save link
    dataset_links[dataset_name] = {
        "calibration_set": cal_set,
        "branch_set": branch_set,
        "species": species
    }
    
    save_dataset_links(dataset_links)
    print(f"\n✓ Linked '{dataset_name}' to:")
    print(f"  Calibration set: {cal_set}")
    print(f"  Branch set: {branch_set}")
    print(f"  Species: {species}")


def view_link():
    """View links for an ant dataset."""
    dataset_links = load_dataset_links()
    
    if not dataset_links:
        print("\nNo dataset links found.")
        return
    
    print("\n" + "="*60)
    print("VIEW DATASET LINK")
    print("="*60)
    print("\nLinked datasets:")
    for i, name in enumerate(dataset_links.keys(), 1):
        print(f"  {i}. {name}")
    
    dataset_name = input("\nEnter dataset name: ").strip()
    if dataset_name not in dataset_links:
        print(f"Error: Dataset '{dataset_name}' not found.")
        return
    
    link = dataset_links[dataset_name]
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Calibration Set: {link.get('calibration_set', 'N/A')}")
    print(f"Branch Set: {link.get('branch_set', 'N/A')}")
    print(f"Species: {link.get('species', 'N/A')}")


def update_link():
    """Update an existing link."""
    dataset_links = load_dataset_links()
    
    if not dataset_links:
        print("\nNo dataset links found.")
        return
    
    print("\n" + "="*60)
    print("UPDATE DATASET LINK")
    print("="*60)
    print("\nLinked datasets:")
    for i, name in enumerate(dataset_links.keys(), 1):
        print(f"  {i}. {name}")
    
    dataset_name = input("\nEnter dataset name to update: ").strip()
    if dataset_name not in dataset_links:
        print(f"Error: Dataset '{dataset_name}' not found.")
        return
    
    # Show current link
    current = dataset_links[dataset_name]
    print(f"\nCurrent link:")
    print(f"  Calibration Set: {current.get('calibration_set', 'N/A')}")
    print(f"  Branch Set: {current.get('branch_set', 'N/A')}")
    print(f"  Species: {current.get('species', 'N/A')}")
    
    # Update calibration set
    calibration_sets = load_calibration_sets()
    if calibration_sets:
        print("\nAvailable calibration sets:")
        for i, name in enumerate(calibration_sets.keys(), 1):
            print(f"  {i}. {name}")
        cal_set = input("\nEnter new calibration set name (or press Enter to keep current): ").strip()
        if cal_set:
            if cal_set in calibration_sets:
                dataset_links[dataset_name]['calibration_set'] = cal_set
            else:
                print(f"Error: Calibration set '{cal_set}' not found.")
                return
    
    # Update branch set
    branch_sets = load_branch_sets()
    if branch_sets:
        print("\nAvailable branch sets:")
        for i, name in enumerate(branch_sets.keys(), 1):
            print(f"  {i}. {name}")
        branch_set = input("\nEnter new branch set name (or press Enter to keep current): ").strip()
        if branch_set:
            if branch_set in branch_sets:
                dataset_links[dataset_name]['branch_set'] = branch_set
            else:
                print(f"Error: Branch set '{branch_set}' not found.")
                return
    
    # Update species
    species = input(f"\nEnter species (WR/NWR, or press Enter to keep current): ").strip().upper()
    if species:
        if species in ['WR', 'NWR']:
            dataset_links[dataset_name]['species'] = species
        else:
            print("Error: Species must be WR or NWR")
            return
    
    save_dataset_links(dataset_links)
    print(f"\n✓ Link updated for '{dataset_name}'")


def delete_link():
    """Delete a dataset link."""
    dataset_links = load_dataset_links()
    
    if not dataset_links:
        print("\nNo dataset links found.")
        return
    
    print("\n" + "="*60)
    print("DELETE DATASET LINK")
    print("="*60)
    print("\nLinked datasets:")
    for i, name in enumerate(dataset_links.keys(), 1):
        print(f"  {i}. {name}")
    
    dataset_name = input("\nEnter dataset name to delete: ").strip()
    if dataset_name not in dataset_links:
        print(f"Error: Dataset '{dataset_name}' not found.")
        return
    
    confirm = input(f"Are you sure you want to delete link for '{dataset_name}'? (yes/no): ").strip().lower()
    if confirm == 'yes':
        del dataset_links[dataset_name]
        save_dataset_links(dataset_links)
        print(f"✓ Link deleted for '{dataset_name}'")
    else:
        print("Cancelled.")


def list_links():
    """List all dataset links."""
    dataset_links = load_dataset_links()
    
    if not dataset_links:
        print("\nNo dataset links found.")
        return
    
    print("\n" + "="*60)
    print("DATASET LINKS")
    print("="*60)
    print(f"\nTotal: {len(dataset_links)}")
    print("\nDataset".ljust(15) + "Calibration Set".ljust(25) + "Branch Set".ljust(25) + "Species")
    print("-" * 80)
    
    for name, link in dataset_links.items():
        cal_set = link.get('calibration_set', 'N/A')
        branch_set = link.get('branch_set', 'N/A')
        species = link.get('species', 'N/A')
        print(f"{name.ljust(15)}{cal_set.ljust(25)}{branch_set.ljust(25)}{species}")


def main():
    """Main menu."""
    while True:
        print("\n" + "="*60)
        print("DATASET LINKING MANAGER")
        print("="*60)
        print("\n1. Link dataset")
        print("2. View link")
        print("3. Update link")
        print("4. Delete link")
        print("5. List all links")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            link_dataset()
        elif choice == '2':
            view_link()
        elif choice == '3':
            update_link()
        elif choice == '4':
            delete_link()
        elif choice == '5':
            list_links()
        elif choice == '6':
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please enter 1-6.")


if __name__ == "__main__":
    main()
