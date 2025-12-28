import pandas as pd
import subprocess
import os
import time

# --- CONFIGURATION ---
# Human-related sound IDs
TARGET_IDS = [
    "/m/09x0r",     # Speech
    "/m/05zppz",    # Male speech, man speaking
    "/m/01h8n0",    # Conversation
    "/m/02qldy",    # Narration, monologue
    "/m/0261r1",    # Babbling
    "/m/0brhx",     # Speech synthesizer
    "/m/07p6fty",   # Shout
    "/m/07q4ntr",   # Bellow
    "/m/07rwj3x",   # Whoop
    "/m/01w250",    # Whistling
    "/m/07q0yl5",   # Snort
    "/m/01b_21",    # Cough
    "/m/0dl9sf8",   # Throat clearing
    "/m/01hsr_",    # Sneeze
    "/m/07ppn3j",   # Sniff
    "/m/06h7j"      # Run
]

DOWNLOAD_GOAL_PER_CLASS = 100  # How many files per class
CSV_FILE = "unbalanced_train_segments.csv"
OUTPUT_DIR = "human_sounds"

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

print(f"Downloading {DOWNLOAD_GOAL_PER_CLASS} files for each of {len(TARGET_IDS)} human sound classes.")
print("="*70)

total_downloaded = 0

# Process each ID separately
for target_id in TARGET_IDS:
    print(f"\n--- Processing {target_id} ---")
    
    # Filter for this specific ID
    filtered_df = df[df['labels'].str.contains(target_id, na=False, regex=False)]
    
    total_available = len(filtered_df)
    print(f"Found {total_available} segments for {target_id}")
    
    if total_available == 0:
        print(f"Warning: No segments found for {target_id}, skipping...")
        continue
    
    # Shuffle to get random samples
    filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)
    
    success_count = 0
    processed_count = 0
    
    for index, row in filtered_df.iterrows():
        # Stop if we reached our goal for this class
        if success_count >= DOWNLOAD_GOAL_PER_CLASS:
            print(f"Goal reached for {target_id}! {success_count} files downloaded.")
            break
        
        ytid = row['YTID']
        start_time = row['start']
        end_time = row['end']
        
        # Create the filename with ID prefix for organization
        # Convert /m/xyz to m_xyz for valid filename
        id_prefix = target_id.replace('/', '_').replace('_', '', 1)
        output_filename = os.path.join(OUTPUT_DIR, f"{id_prefix}_{ytid}.wav")
        
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
            total_downloaded += 1
            print(f"[{success_count}/{DOWNLOAD_GOAL_PER_CLASS}] Success: {ytid}")
            
            # Pause briefly to avoid YouTube IP bans
            time.sleep(1)
        
        except subprocess.CalledProcessError:
            print(f"Failed: {ytid} (Video unavailable)")
        
        processed_count += 1

print("\n" + "="*70)
print(f"Process complete!")
print(f"Total files downloaded across all classes: {total_downloaded}")
