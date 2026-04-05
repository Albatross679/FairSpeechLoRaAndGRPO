#!/usr/bin/env python3
"""
Extract ALL test-split clips from the Common Voice tar.gz.
Single-pass streaming approach (iterates archive once, extracts matches).
Based on the bootcamp's extract_female_fast.py pattern.
"""

import csv
import os
import tarfile
import time

CV_DIR = "/users/PAS2030/srishti/bootcamp/data/commonvoice/cv-corpus-24.0-2025-12-05/en"
TAR_FILE = "/users/PAS2030/srishti/bootcamp/data/commonvoice/cv-corpus-24.0-2025-12-05-en.tar.gz"
CLIPS_DIR = os.path.join(CV_DIR, "clips")
EXTRACT_TO = "/users/PAS2030/srishti/bootcamp/data/commonvoice/"

# 1. Build set of needed clip paths (as they appear inside the tar)
test_tsv = os.path.join(CV_DIR, "test.tsv")
needed = set()
with open(test_tsv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        # Path inside tar: cv-corpus-24.0-2025-12-05/en/clips/<filename>
        needed.add(f"cv-corpus-24.0-2025-12-05/en/clips/{row['path']}")

# 2. Check which are already extracted
existing = set(os.listdir(CLIPS_DIR))
already_have = sum(1 for n in needed if os.path.basename(n) in existing)
to_extract = len(needed) - already_have

print(f"Total test clips needed:  {len(needed)}")
print(f"Already on disk:          {already_have}")
print(f"Need to extract:          {to_extract}")

if to_extract == 0:
    print("All test clips already extracted!")
    exit(0)

# 3. Single-pass extraction
extracted = 0
skipped = 0
start = time.time()

print(f"\nScanning {TAR_FILE} ({os.path.getsize(TAR_FILE) / 1e9:.1f} GB)...")
print("This iterates the archive once — expect ~15-20 min.\n")

with tarfile.open(TAR_FILE, "r:gz") as tar:
    for member in tar:
        if member.name in needed:
            # Skip if already on disk
            if os.path.basename(member.name) in existing:
                skipped += 1
                continue
            tar.extract(member, path=EXTRACT_TO)
            extracted += 1
            if extracted % 2000 == 0:
                elapsed = time.time() - start
                rate = extracted / elapsed if elapsed > 0 else 1
                remaining = (to_extract - extracted) / rate if rate > 0 else 0
                print(f"  Extracted {extracted:,}/{to_extract:,} "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")
        else:
            skipped += 1

elapsed = time.time() - start
final_count = len(os.listdir(CLIPS_DIR))
print(f"\nDone! Extracted {extracted:,} new clips, skipped {skipped:,}")
print(f"Total clips now on disk: {final_count:,}")
print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
