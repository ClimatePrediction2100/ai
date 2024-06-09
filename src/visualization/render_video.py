import subprocess
import sys
import os
sys.path.append('../..')
import config

output_dir = os.path.join(config.ROOT_DIR, 'results', 'frames_5')
os.makedirs(output_dir, exist_ok=True)

# Creating video using ffmpeg
video_filename = os.path.join(config.ROOT_DIR, 'results', 'temperature_video_5.mp4')
subprocess.run([
    'ffmpeg', '-framerate', '12', '-i', os.path.join(output_dir, 'frame_%04d.png'),
    '-c:v', 'libx264', '-profile:v', 'high', '-crf', '20', '-pix_fmt', 'yuv420p',
    video_filename
])

print(f"Video saved as {video_filename}")