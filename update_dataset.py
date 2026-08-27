import os
import zipfile
import io
import requests
import pandas as pd

DATASET_DIR = "Dataset"
OUTPUT_FILE = os.path.join(DATASET_DIR, "database.csv")

# Direct official PHMSA link for 2010-present hazardous liquids
PHMSA_URL = "https://www.phmsa.dot.gov/sites/phmsa.dot.gov/files/data_stats/pipeline/hazardous_liquid_2010_present.zip"

print("Downloading latest 2010–2026 PHMSA incident records...")
response = requests.get(PHMSA_URL, headers={"User-Agent": "Mozilla/5.0"})

if response.status_code == 200:
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Locate the CSV in the zip file
        csv_filename = [f for f in z.namelist() if f.endswith('.csv') or f.endswith('.txt')][0]
        with z.open(csv_filename) as f:
            df = pd.read_csv(f, low_memory=False)
            
    os.makedirs(DATASET_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Updated dataset successfully saved with {len(df)} records covering up to 2026!")
else:
    print("Direct download link unavailable. Please download the latest CSV from the PHMSA portal manually.")