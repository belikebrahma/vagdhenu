import time
import requests
import subprocess
import os
import sys

PORT = 8011  # Using a custom port for local test
url = f"http://127.0.0.1:{PORT}/tts"

# Start the uvicorn server in a subprocess
server_process = subprocess.Popen(
    [
        sys.executable, "-m", "uvicorn", "src.api:app",
        "--host", "127.0.0.1",
        "--port", str(PORT)
    ],
    env={**os.environ, "VAGDHENU_DEVICE": "cpu"}
)

print("Waiting for Vagdhenu API server to start and load models...", flush=True)
connected = False
for i in range(30):
    if server_process.poll() is not None:
        print("Server process terminated unexpectedly.", flush=True)
        break
    try:
        # Send a quick request to see if uvicorn is accepting connections
        response = requests.post(url, json={"text": "test", "meter": "anushtubh"})
        connected = True
        print(f"Connected to server successfully after {i*2} seconds!", flush=True)
        break
    except requests.exceptions.ConnectionError:
        time.sleep(2)

if not connected:
    print("Failed to connect to server within timeout limit.", flush=True)

try:
    payload = {
        "text": "ॐ त्र्यम्बकं यजामहे सुगन्धिं पुष्टिवर्धनम्। उर्वारुकमिव बन्धनान्मृत्योर्मुक्षीय मामृतात्॥",
        "meter": "anushtubh",
        "speed": 0.90,
        "seed": 42,
        "format": "wav",
        "reverb": "temple",
        "chorus": True,
        "tanpura": "C#",
        "pause_duration": 0.85
    }

    print("Sending POST request to generate audio...", flush=True)
    start_time = time.time()
    response = requests.post(url, json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        data = response.json()
        print("\n--- Success ---")
        print(f"Response Time: {elapsed:.2f} seconds")
        print(f"Generated URL: {data['url']}")
        print(f"Duration: {data['dur']:.2f} seconds")
        print(f"From Cache: {data['cached']}")
        
        # Test cache hit
        print("\nTesting cache hit (should be near instant)...", flush=True)
        start_time = time.time()
        cached_response = requests.post(url, json=payload)
        cache_elapsed = time.time() - start_time
        
        if cached_response.status_code == 200:
            cached_data = cached_response.json()
            print(f"Cache Response Time: {cache_elapsed:.4f} seconds")
            print(f"From Cache: {cached_data['cached']}")
        else:
            print(f"Cache test failed: {cached_response.status_code}")
    else:
        print(f"Generation failed (Status {response.status_code}): {response.text}")

finally:
    print("\nStopping local API server...", flush=True)
    server_process.terminate()
    server_process.wait()
