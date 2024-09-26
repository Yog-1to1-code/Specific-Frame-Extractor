import cv2
import torch
import os
from datetime import datetime, timedelta

def time_to_seconds(time_str):
    """Convert HH:MM:SS.FF to seconds."""
    h, m, s = time_str.split(':')
    s, f = s.split('.')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(f) / 100

def extract_frames(video_path, timestamps_file, output_folder):
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read timestamps from file
    with open(timestamps_file, 'r') as f:
        timestamps = [line.strip() for line in f.readlines()]

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for timestamp in timestamps:
        # Convert timestamp to frame number
        seconds = time_to_seconds(timestamp)
        frame_number = int(seconds * fps)

        if frame_number >= total_frames:
            print(f"Warning: Frame {frame_number} exceeds total frames in video. Skipping.")
            continue

        # Set the video to the correct frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame at timestamp {timestamp}")
            continue

        # If GPU is available, process the frame on GPU
        if device.type == 'cuda':
            frame_tensor = torch.from_numpy(frame).to(device)
            # Perform any GPU-based processing here if needed
            frame = frame_tensor.cpu().numpy()

        # Save the frame
        output_path = os.path.join(output_folder, f"frame_{timestamp.replace(':', '_')}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Saved frame for timestamp {timestamp}")

    cap.release()
    print("Frame extraction completed.")

# Usage
video_path = "path/to/your/video.mp4"
timestamps_file = "path/to/your/timestamps.txt"
output_folder = "path/to/output/folder"

extract_frames(video_path, timestamps_file, output_folder)
