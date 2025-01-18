import subprocess
import shutil
from pydub.utils import which

# Method 1: Using shutil
print("ffmpeg path (shutil):", shutil.which("ffmpeg"))

# Method 2: Using pydub's utility
print("ffmpeg path (pydub):", which("ffmpeg"))

# Method 3: Try to run ffmpeg
try:
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    print("ffmpeg version output:", result.stdout[:200])  # Print first 200 chars
except FileNotFoundError:
    print("ffmpeg not found in PATH")
except Exception as e:
    print(f"Error running ffmpeg: {e}")

# Print current PATH
import os
print("\nCurrent PATH:")
for path in os.environ['PATH'].split(os.pathsep):
    print(path)