import os
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from moviepy.video.fx import all as vfx

def process_video(
    input_video_path: str, 
    output_base_path: str, 
    subfolder: str, 
    desired_duration: float = 3.0, 
    resize_dim: tuple = (224, 224), 
    fps: int = 30
):
    """
    Process a single video file by:
      1. Adjusting its speed to make it exactly `desired_duration` seconds.
      2. Resizing it to `resize_dim`.
      3. Extracting frames at `fps` for the entire `desired_duration`.
      4. Saving those frames in an output folder.

    :param input_video_path: Path to the input video file.
    :param output_base_path: Base path where processed frames will be saved.
    :param subfolder: Name of the category folder (e.g. "extreme-helpless").
    :param desired_duration: Final duration (in seconds) of the processed clip.
    :param resize_dim: (width, height) for resizing frames.
    :param fps: Frames per second to extract.
    """
    # Get video name (without extension) to create an output subfolder
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    # Create an output directory for the frames of this video
    output_dir = os.path.join(output_base_path, subfolder, video_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load the clip
        clip = VideoFileClip(input_video_path)
        original_duration = clip.duration

        if original_duration <= 0:
            print(f"[WARNING] Video {input_video_path} has invalid duration. Skipping.")
            clip.close()
            return

        # Calculate speed factor to get the final clip to desired_duration
        # speedx(factor > 1) = speed up, factor < 1 = slow down
        speed_factor = original_duration / desired_duration

        # Apply speed change and resize
        processed_clip = clip.fx(vfx.speedx, speed_factor).resize(newsize=resize_dim)

        # Generate times at which to sample frames (0 to desired_duration)
        # e.g. 30 FPS * 3 seconds = 90 frames
        total_frames = int(fps * desired_duration)
        times = np.linspace(0, desired_duration, num=total_frames, endpoint=False)

        # Extract and save frames
        for i, t in enumerate(times):
            frame = processed_clip.get_frame(t)  
            # Convert RGB (MoviePy) to BGR (OpenCV) 
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame_filename = os.path.join(output_dir, f"frame_{i:03d}.jpg")
            cv2.imwrite(frame_filename, frame_bgr)

        print(f"[INFO] Processed and saved {total_frames} frames for {input_video_path} -> {output_dir}")

        clip.close()
        processed_clip.close()

    except Exception as e:
        print(f"[ERROR] Failed to process {input_video_path}. Error: {e}")


def main():
    """
    Processes all raw videos in each class folder and stores extracted frames 
    into the `processed_frames/` directory relative to the project root.
    """
    # locate the project directory
    project_root = os.path.abspath(os.path.dirname(__file__))

    # Class folders with raw .mp4 files
    subfolders = ["extreme-helpless", "little_helplessness", "no-helpless"]

    # Output folder to store extracted frames
    output_base_path = os.path.join(project_root, "processed_frames")
    os.makedirs(output_base_path, exist_ok=True)

    # Video processing parameters
    desired_duration = 3.0  # seconds
    resize_dim = (224, 224) # final frame size
    fps = 30                # frames per second

    # Iterate through each class subfolder
    for subfolder in subfolders:
        folder_path = os.path.join(project_root, subfolder)

        if not os.path.isdir(folder_path):
            print(f"[WARNING] Folder '{folder_path}' not found. Skipping.")
            continue

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
                input_video_path = os.path.join(folder_path, file_name)

                process_video(
                    input_video_path=input_video_path,
                    output_base_path=output_base_path,
                    subfolder=subfolder,
                    desired_duration=desired_duration,
                    resize_dim=resize_dim,
                    fps=fps
                )


if __name__ == "__main__":
    main()
