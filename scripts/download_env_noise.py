import pandas as pd
import subprocess
import os
import time

# --- CONFIGURATION ---
# Environmental noise ID
TARGET_IDS = [
    "/m/06_y0by"    # Environmental noise
]

DOWNLOAD_GOAL = 500      # How many files you want total
CSV_FILE = "unbalanced_train_segments.csv"
OUTPUT_DIR = "noise_dataset"

# Create the output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"Reading {CSV_FILE}...")

# Read the CSV
try:
    df = pd.read_csv(CSV_FILE, 
                     skiprows=3, 
                     header=None, 
                     names=['YTID', 'start', 'end', 'labels'],
                     skipinitialspace=True)
except FileNotFoundError:
    print(f"Error: Could not find {CSV_FILE}. Make sure it is in this folder.")
    exit()

# Filter for rows that contain ANY of the noise IDs
print(f"Filtering for Noise-related IDs: {', '.join(TARGET_IDS)}...")
noise_df = df[df['labels'].str.contains('|'.join(TARGET_IDS), na=False, regex=True)]

# Shuffle the dataframe to randomize which noise type we download
noise_df = noise_df.sample(frac=1).reset_index(drop=True)

total_available = len(noise_df)
print(f"Found {total_available} potential noise segments.")

if total_available == 0:
    print("Warning: No matches found. Check your CSV file format.")
    exit()

print(f"Starting download loop. Aiming for {DOWNLOAD_GOAL} files.")
print("-------------------------------------------------------")

success_count = 0
processed_count = 0

for index, row in noise_df.iterrows():
    # Stop if we reached our goal
    if success_count >= DOWNLOAD_GOAL:
        print(f"\nGoal reached! {success_count} files downloaded.")
        break

    ytid = row['YTID']
    start_time = row['start']
    end_time = row['end']
    
    # Create the filename (e.g., noise_dataset/VIDEOID.wav)
    output_filename = os.path.join(OUTPUT_DIR, f"{ytid}.wav")

    # Skip if file already exists
    if os.path.exists(output_filename):
        print(f"Skipping {ytid} (Already exists)")
        success_count += 1
        continue

    # Construct the YouTube URL
    url = f"https://www.youtube.com/watch?v={ytid}"

    # yt-dlp command
    cmd = [
        "yt-dlp",
        "-x", 
        "--audio-format", "wav",
        "--download-sections", f"*{start_time}-{end_time}",
        "--force-keyframes-at-cuts",
        "-o", output_filename,
        "--quiet",
        "--no-warnings",
        url
    ]

    try:
        # Run the download command
        subprocess.run(cmd, check=True)
        success_count += 1
        print(f"[{success_count}/{DOWNLOAD_GOAL}] Success: {ytid}")
        
        # IMPORTANT: Pause briefly to avoid YouTube IP bans
        time.sleep(1)

    except subprocess.CalledProcessError:
        # This catches errors (video deleted, private, etc.)
        print(f"Failed: {ytid} (Video unavailable)")
    
    processed_count += 1

print("-------------------------------------------------------")
print(f"Process complete.")
print(f"Total processed: {processed_count}")
print(f"Total downloaded: {success_count}")
