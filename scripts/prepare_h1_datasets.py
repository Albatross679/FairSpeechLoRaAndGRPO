import os
import csv
import shutil
import re
from datasets import load_dataset
import soundfile as sf

H1_DIR = "/users/PAS2030/srishti/asr_fairness/data/h1"
os.makedirs(H1_DIR, exist_ok=True)
os.makedirs(os.path.join(H1_DIR, 'audio'), exist_ok=True)

manifest = []

def clean_coraal_text(text):
    """Normalize CORAAL sociolinguistic conventions for baseline WER computation."""
    # Remove sociolinguistic angle bracket tags e.g. <ts>, <sigh>
    text = re.sub(r'<[^>]+>', '', text)
    # Remove parentheses tags e.g. (laughing)
    text = re.sub(r'\([^)]+\)', '', text)
    # Remove square brackets if any
    text = re.sub(r'\[[^\]]+\]', '', text)
    # Remove / delimiters that are used for tagging, but keep the word if it's not a redacted token
    # Redacted tokens like /RD-NAME/ will just become RD-NAME which whisper might transcribe differently,
    # but that's perfectly fine for relative fairness eval. Or we can just strip them completely.
    text = text.replace('/', '')
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==============================================================================
# 1. PROCESS CORAAL
# ==============================================================================
print("Processing CORAAL...")

coraal_csv = "/users/PAS2030/srishti/asr_fairness/data/asr-disparities/input/CORAAL_transcripts.csv"
coraal_audio_base = "/users/PAS2030/srishti/asr_fairness/data/asr-disparities/input/CORAAL_audio"

count = 0
with open(coraal_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row['segment_filename']
        src_path = None
        
        prefix = filename.split('_')[0].lower()
        possible_path = os.path.join(coraal_audio_base, prefix, filename)
        if os.path.exists(possible_path):
            src_path = possible_path
            
        if not src_path:
            continue
            
        dst_path = os.path.join(H1_DIR, 'audio', filename)
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            
        raw_text = row['content'].strip()
        cleaned_text = clean_coraal_text(raw_text)
        
        manifest.append({
            'utterance_id': filename.replace('.wav', ''),
            'audio_path': dst_path,
            'sentence': cleaned_text,          # Cleaned for standard WER computation
            'sentence_raw': raw_text,          # Preserved for sociolinguistic analysis
            'dataset': 'coraal',
            'dialect': 'african_american',
            'age': row['age'],
            'gender': row['gender']
        })
        count += 1
        
        if count >= 2000:
            break

print(f"Added {count} CORAAL utterances.")

# ==============================================================================
# 2. PROCESS LIBRISPEECH
# ==============================================================================
print("Processing LibriSpeech (test-clean)...")

try:
    import urllib.request
    import tarfile
    import glob
    
    ls_tar_url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    ls_tar_path = os.path.join(H1_DIR, "test-clean.tar.gz")
    ls_ext_dir = os.path.join(H1_DIR, "librispeech_raw")
    
    if not os.path.exists(ls_ext_dir):
        if not os.path.exists(ls_tar_path):
            print("Downloading LibriSpeech test-clean from OpenSLR...")
            urllib.request.urlretrieve(ls_tar_url, ls_tar_path)
        print("Extracting LibriSpeech...")
        with tarfile.open(ls_tar_path, "r:gz") as tar:
            tar.extractall(path=ls_ext_dir)
            
    count = 0
    trans_files = glob.glob(os.path.join(ls_ext_dir, "**", "*.txt"), recursive=True)
    for t_file in trans_files:
        if count >= 2000:
            break
        with open(t_file, "r") as f:
            for line in f:
                if count >= 2000:
                    break
                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue
                file_id = parts[0]
                txt = parts[1]
                
                flac_path = os.path.join(os.path.dirname(t_file), f"{file_id}.flac")
                if not os.path.exists(flac_path):
                    continue
                    
                dst_path = os.path.join(H1_DIR, 'audio', f"ls_{file_id}.wav")
                if not os.path.exists(dst_path):
                    # Convert FLAC to WAV to keep things standard 16kHz
                    data, sr = sf.read(flac_path)
                    sf.write(dst_path, data, sr)
                
                manifest.append({
                    'utterance_id': f"ls_{file_id}",
                    'audio_path': dst_path,
                    'sentence': txt.lower(), # Keep it somewhat standard, LS uses caps for trans
                    'sentence_raw': txt,
                    'dataset': 'librispeech',
                    'dialect': 'standard_american',
                    'age': 'unknown',
                    'gender': 'unknown'
                })
                count += 1
                
    print(f"Added {count} LibriSpeech utterances.")
except Exception as e:
    print("Warning: Failed to load LibriSpeech:", e)

# Clean up the raw tar file to save space and file quota
try:
    if os.path.exists(ls_tar_path):
        os.remove(ls_tar_path)
        print("Cleaned up LibriSpeech tar file.")
    if os.path.exists(ls_ext_dir):
        shutil.rmtree(ls_ext_dir)
        print("Cleaned up raw LibriSpeech extracted dir.")
except:
    pass

# ==============================================================================
# 3. SAVE MANIFEST
# ==============================================================================
out_csv = os.path.join(H1_DIR, 'h1_manifest.csv')
with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['utterance_id', 'audio_path', 'sentence', 'sentence_raw', 'dataset', 'dialect', 'age', 'gender'])
    writer.writeheader()
    writer.writerows(manifest)

print(f"\nWritten {len(manifest)} utterances to {out_csv}")
