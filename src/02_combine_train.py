"""
02_combine_train.py
-------------------
Consolidates all 51 individual training CSV files from the CICIoMT2024 dataset
into a single unified DataFrame (D_train) with hierarchical label columns:
    - Class    : binary label (BENIGN / ATTACK)
    - Category : 6-class label (BENIGN, DDoS, DoS, MQTT, RECON, SPOOFING)
    - Attack   : 19-class label (specific attack type)
    - source_file : name of the original CSV (metadata, excluded from the features)

Usage:
    python 02_combine_train.py --input_dir /path/to/train/csvs --output_file CICIoMT2024_train.csv

Arguments:
    --input_dir   Directory containing the 51 training CSV files (default: ./train)
    --output_file Output CSV file path (default: CICIoMT2024_train.csv)
"""

import argparse
import pandas as pd
import os
import glob

# ======================================================
# Label mapping (based on Table 6 of the reference paper)
# ======================================================
LABEL_MAPPING = {
    # BENIGN
    "Benign": ("BENIGN", "Benign", "Benign"),

    # SPOOFING
    "ARP_Spoofing": ("ATTACK", "SPOOFING", "ARP spoofing"),

    # RECON
    "Recon-OS_Scan":    ("ATTACK", "RECON", "OS scan"),
    "Recon-Ping_Sweep": ("ATTACK", "RECON", "Ping sweep"),
    "Recon-Port_Scan":  ("ATTACK", "RECON", "Port scan"),
    "Recon-VulScan":    ("ATTACK", "RECON", "Recon VulScan"),

    # MQTT
    "MQTT-Malformed_Data":    ("ATTACK", "MQTT", "Malformed data"),
    "MQTT-DoS-Connect_Flood": ("ATTACK", "MQTT", "DoS connect flood"),
    "MQTT-DoS-Publish_Flood": ("ATTACK", "MQTT", "DoS publish flood"),
    "MQTT-DDoS-Connect_Flood":("ATTACK", "MQTT", "DDoS connect flood"),
    "MQTT-DDoS-Publish_Flood":("ATTACK", "MQTT", "DDoS publish flood"),

    # DoS
    **{f"TCP_IP-DoS-{proto}{num}": ("ATTACK", "DoS", f"DoS {proto}")
       for proto, nums in [("ICMP", [1,2,3,4]), ("SYN", [1,2,3,4]),
                           ("TCP",  [1,2,3,4]), ("UDP", [1,2,3,4])]
       for num in nums},

    # DDoS
    **{f"TCP_IP-DDoS-{proto}{num}": ("ATTACK", "DDoS", f"DDoS {proto}")
       for proto, nums in [("ICMP", [1,2,3,4,5,6,7,8]), ("SYN", [1,2,3,4]),
                           ("TCP",  [1,2,3,4]),           ("UDP", [1,2,3,4,5,6,7,8])]
       for num in nums},
}


def get_labels(filename: str) -> tuple:
    """Extract hierarchical labels from a training CSV filename."""
    base_name = os.path.basename(filename)
    attack_name = base_name.split("_train")[0]
    return LABEL_MAPPING.get(attack_name, ("UNKNOWN", "UNKNOWN", "UNKNOWN"))


def process_files(folder_path: str) -> pd.DataFrame:
    """Load and label all CSV files in folder_path."""
    all_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in: {folder_path}")

    print(f"Found {len(all_files)} CSV files in '{folder_path}'.")
    dfs = []

    for file in all_files:
        df = pd.read_csv(file)
        class_, category, attack = get_labels(file)
        if class_ == "UNKNOWN":
            print(f"  [WARNING] Unrecognized file pattern: {os.path.basename(file)}")
        df["Class"]    = class_
        df["Category"] = category
        df["Attack"]   = attack
        df["source_file"] = os.path.basename(file)  # used only for the per-file scaling ablation; never a feature
        dfs.append(df)
        print(f"  Loaded: {os.path.basename(file)} -> {class_} / {category} / {attack}")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined training dataset shape: {combined.shape}")
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Combine CICIoMT2024 training CSV files into D_train."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./train",
        help="Directory containing the training CSV files (default: ./train)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="CICIoMT2024_train.csv",
        help="Output CSV file path (default: CICIoMT2024_train.csv)"
    )
    args = parser.parse_args()

    final_df = process_files(args.input_dir)
    final_df.to_csv(args.output_file, index=False)
    print(f"\nD_train saved to: {args.output_file}")

    print("\nLabel distribution:")
    print(final_df.groupby(["Class", "Category", "Attack"])
                  .size()
                  .reset_index(name="Count")
                  .to_string(index=False))


if __name__ == "__main__":
    main()
