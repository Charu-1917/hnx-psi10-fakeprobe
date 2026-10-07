import urllib.request
import os

url = 'https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth'
dest = r'C:\Users\yaso0\.gemini\antigravity-ide\brain\fe3baf12-faf4-4c52-9aae-e0bcb76cb59c\scratch\syncnet_python\detectors\s3fd\weights\sfd_face.pth'

os.makedirs(os.path.dirname(dest), exist_ok=True)

req = urllib.request.Request(
    url, 
    data=None, 
    headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
)

print(f"Starting download from {url}...")
try:
    with urllib.request.urlopen(req, timeout=15) as response, open(dest, 'wb') as out_file:
        total_length = response.getheader('content-length')
        print(f"Total size: {total_length} bytes")
        
        if total_length is None:
            out_file.write(response.read())
        else:
            total_length = int(total_length)
            downloaded = 0
            while True:
                buffer = response.read(8192 * 4)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if downloaded % (8192 * 4 * 100) == 0:
                    print(f"Downloaded {downloaded}/{total_length} bytes ({(downloaded/total_length)*100:.2f}%)")
    print("Download complete.")
except Exception as e:
    print(f"Download failed: {e}")
