import os
import cv2
import numpy as np
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Folder paths
video_folder_path = 'VIDEOS'
images_folder_path = 'IMAGES'
counter_file = 'frame_counter.txt'

# Frame extraction parameters
crop_margin = 5
frame_interval = 5  # Extract every 5th frame (adjustable)

def get_video_range():
    """Read the current video range from the counter file."""
    if os.path.exists(counter_file):
        with open(counter_file, 'r') as file:
            start_video = int(file.read().strip())
    else:
        start_video = 1
    return start_video

def update_counter(next_start):
    """Update the counter file with the next starting video number."""
    with open(counter_file, 'w') as file:
        file.write(str(next_start))
    logging.info(f"Counter updated to: {next_start}")

def detect_video_area(frame):
    """
    Detects the embedded video area by scanning from center outward in 4 directions
    and stopping at black regions.
    
    Returns:
        (x, y, width, height) or None if detection fails
    """
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold for "black" pixel
    black_thresh = 10
    black_ratio_thresh = 0.95  # % of black pixels in a row/col to consider it 'black'

    center_x = width // 2
    center_y = height // 2

    # --- Find top boundary (scan upward) ---
    for y in range(center_y, 0, -1):
        row = gray[y, :]
        black_ratio = np.mean(row < black_thresh)
        if black_ratio > black_ratio_thresh:
            start_y = y + 1
            break
    else:
        start_y = 0

    # --- Find bottom boundary (scan downward) ---
    for y in range(center_y, height):
        row = gray[y, :]
        black_ratio = np.mean(row < black_thresh)
        if black_ratio > black_ratio_thresh:
            end_y = y - 1
            break
    else:
        end_y = height - 1

    # --- Find left boundary (scan left) ---
    for x in range(center_x, 0, -1):
        col = gray[:, x]
        black_ratio = np.mean(col < black_thresh)
        if black_ratio > black_ratio_thresh:
            start_x = x + 1
            break
    else:
        start_x = 0

    # --- Find right boundary (scan right) ---
    for x in range(center_x, width):
        col = gray[:, x]
        black_ratio = np.mean(col < black_thresh)
        if black_ratio > black_ratio_thresh:
            end_x = x - 1
            break
    else:
        end_x = width - 1

    # Compute width and height
    box_width = end_x - start_x
    box_height = end_y - start_y

    # Final sanity check
    if box_width <= 0 or box_height <= 0:
        return None

    return (start_x + crop_margin, start_y + crop_margin, 
            box_width - 2*crop_margin, box_height - 2*crop_margin)

def extract_frames_from_video(video_path, video_number):
    """
    Extract frames from a single video and save cropped frames to IMAGES folder.
    
    Args:
        video_path: Path to the video file
        video_number: Video number for naming extracted frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Failed to open video file: {video_path}")
        return 0
    
    # Detect video area from first frame
    ret, first_frame = cap.read()
    if not ret:
        logging.error(f"Could not read first frame from {video_path}")
        cap.release()
        return 0
    
    video_area = detect_video_area(first_frame)
    if not video_area:
        logging.error(f"Could not detect video area in {video_path}")
        cap.release()
        return 0
    
    x, y, w, h = video_area
    logging.info(f"Detected video area for Video_{video_number}: {w}x{h} at ({x}, {y})")
    
    # Reset to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    extracted_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract every Nth frame
        if frame_count % frame_interval == 0:
            # Crop the frame using the detected area
            cropped_frame = frame[y:y+h, x:x+w]
            
            # Save the cropped frame
            output_filename = f"Video_{video_number}_frame_{frame_count:05d}.jpg"
            output_path = os.path.join(images_folder_path, output_filename)
            cv2.imwrite(output_path, cropped_frame)
            extracted_count += 1
        
        frame_count += 1
        
        # Progress indicator
        if frame_count % 100 == 0:
            print(f"  Processing Video_{video_number}: {frame_count}/{total_frames} frames")
    
    cap.release()
    logging.info(f"✅ Video_{video_number}: Extracted {extracted_count} frames from {total_frames} total frames")
    return extracted_count

def process_video_range(start_video, videos_per_batch=10):
    """
    Process a range of videos and extract frames.
    
    Args:
        start_video: Starting video number
        videos_per_batch: Number of videos to process in this batch
    """
    # Create IMAGES folder if it doesn't exist
    os.makedirs(images_folder_path, exist_ok=True)
    
    total_extracted = 0
    processed_videos = 0
    
    for i in range(videos_per_batch):
        video_number = start_video + i
        video_filename = f"Video_{video_number}.mp4"
        video_path = os.path.join(video_folder_path, video_filename)
        
        # Check if video exists
        if not os.path.exists(video_path):
            logging.warning(f"Video not found: {video_filename} - Skipping")
            continue
        
        logging.info(f"Processing {video_filename} ({i+1}/{videos_per_batch})")
        extracted = extract_frames_from_video(video_path, video_number)
        total_extracted += extracted
        processed_videos += 1
    
    logging.info(f"\n{'='*60}")
    logging.info(f"BATCH COMPLETE!")
    logging.info(f"Processed: {processed_videos} videos")
    logging.info(f"Total frames extracted: {total_extracted}")
    logging.info(f"Frames saved to: {images_folder_path}/")
    logging.info(f"{'='*60}\n")
    
    # Update counter for next batch
    next_start = start_video + videos_per_batch
    update_counter(next_start)

def main():
    """Main function to extract frames from a batch of videos."""
    
    # Get the starting video number
    start_video = get_video_range()
    logging.info(f"Starting frame extraction from Video_{start_video}")
    
    # Process 10 videos at a time (configurable)
    videos_per_batch = 10
    
    # Process the video range
    process_video_range(start_video, videos_per_batch)
    
    logging.info(f"Next run will start from Video_{start_video + videos_per_batch}")

if __name__ == "__main__":
    main()
