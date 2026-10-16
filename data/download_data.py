import urllib.request
import os

DATA_URL = "https://raw.githubusercontent.com/pr1m8/Credit-Default/main/credit_risk_dataset.csv"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "credit_risk_dataset.csv")

def download_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(OUTPUT_FILE):
        print(f"Downloading dataset from {DATA_URL}...")
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response, open(OUTPUT_FILE, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Dataset successfully downloaded to: {OUTPUT_FILE} ({len(data)} bytes)")
    else:
        print(f"Dataset already exists at: {OUTPUT_FILE}")

if __name__ == "__main__":
    download_dataset()
