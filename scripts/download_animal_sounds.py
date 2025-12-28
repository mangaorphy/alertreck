import pandas as pd
import subprocess
import os
import time

# --- CONFIGURATION ---
# Organize IDs by category
CATEGORIES = {
    "quadrupeds": [
        "/m/01280g",    # Wild animals
        "/m/0cdnk",     # Roaring cats (lions, tigers)
        "/m/04cvmfc",   # Roar
        "/m/01z5f"      # Canidae, dogs, wolves
    ],
    "birds": [
        "/m/015p6",     # Bird
        "/m/020bb7",    # Bird vocalization, bird call, bird song
        "/m/07pggtn",   # Chirp, tweet
        "/m/07sx8x_",   # Squawk
        "/m/0h0rv",     # Pigeon, dove
        "/m/07r_25d",   # Coo
        "/m/04s8yn",    # Crow
        "/m/07r5c2p",   # Caw
        "/m/09d5_",     # Owl
        "/m/07r_80w",   # Hoot
        "/m/05_wcq"     # Bird flight, flapping wings
    ],
    "insects": [
        "/m/03vt0",     # Insect
        "/m/09xqv",     # Cricket
        "/m/09f96",     # Mosquito
        "/m/0h2mp",     # Fly, housefly
        "/m/07pjwq1",   # Buzz
        "/m/01h3n",     # Bee, wasp, etc.
        "/m/09ld4",     # Frog
        "/m/07st88b"    # Croak
    ]
}

DOWNLOAD_GOAL_PER_CLASS = 500  # How many files per class
CSV_FILE = "unbalanced_train_segments.csv"
OUTPUT_DIR = "animal_sound"

# Create the output directory and subdirectories
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for category in CATEGORIES.keys():
    category_dir = os.path.join(OUTPUT_DIR, category)
    if not os.path.exists(category_dir):
        os.makedirs(category_dir)

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

# Flatten all IDs and create a mapping to their category
id_to_category = {}
for category, ids in CATEGORIES.items():
    for id_val in ids:
        id_to_category[id_val] = category

print(f"Downloading {DOWNLOAD_GOAL_PER_CLASS} files for each ID across {len(CATEGORIES)} categories.")
print("="*70)

total_downloaded = 0

# Process each category (skip quadrupeds and birds - already have enough files)
for category, target_ids in CATEGORIES.items():
    # Skip quadrupeds and birds since they already have files
    if category in ["quadrupeds", "birds"]:
        print(f"\nSkipping {category.upper()} - already downloaded")
        continue
    
    print(f"\n{'='*70}")
    print(f"CATEGORY: {category.upper()}")
    print(f"{'='*70}")
    
    # Process each ID in this category
    for target_id in target_ids:
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
            category_dir = os.path.join(OUTPUT_DIR, category)
            output_filename = os.path.join(category_dir, f"{id_prefix}_{ytid}.wav")
            
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
