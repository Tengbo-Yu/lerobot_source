#!/usr/bin/env python3
"""
Extract RGB images from npz file and save as video.
"""

import sys
import os
import argparse
import numpy as np
import cv2

def extract_video(
    npz_path: str,
    output_path: str = None,
    fps: int = 30,
    combine_views: bool = True,
):
    """
    Extract RGB images from npz and save as video.
    
    Args:
        npz_path: Path to npz file
        output_path: Output video path (auto-generated if None)
        fps: Video FPS
        combine_views: If True, combine primary and wrist images side by side
    """
    print(f"Loading data from: {npz_path}")
    
    # Load npz data
    data = np.load(npz_path, allow_pickle=True)
    rgb = data['rgb']  # (n_frames, 128, 128, 6)
    
    n_frames = len(rgb)
    print(f"Total frames: {n_frames}")
    print(f"RGB shape: {rgb.shape}")
    
    # Parse rgb: (n_frames, 128, 128, 6) -> (n_frames, 128, 128, 2, 3)
    # 6 channels = 2 images x 3 channels (RGB)
    rgb = rgb.reshape(n_frames, 128, 128, 2, 3)
    
    primary = rgb[:, :, :, 0, :]   # (n_frames, 128, 128, 3)
    wrist = rgb[:, :, :, 1, :]     # (n_frames, 128, 128, 3)
    
    print(f"Primary image shape: {primary.shape}")
    print(f"Wrist image shape: {wrist.shape}")
    
    # Generate output path
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        output_path = f"{base_name}.mp4"
    
    # Prepare video writer
    if combine_views:
        # Combine primary and wrist side by side
        # (128, 128, 3) + (128, 128, 3) -> (128, 256, 3)
        combined = np.concatenate([primary, wrist], axis=2)
        frame_size = (256, 128)  # width, height
    else:
        # Save primary view only
        combined = primary
        frame_size = (128, 128)
    
    # Convert RGB (BGR for OpenCV)
    combined = (combined * 255).astype(np.uint8)
    
    # Add step number text
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(n_frames):
        frame = combined[i].copy()
        cv2.putText(frame, f"Step {i}/{n_frames-1}", (10, 20), 
                    font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        combined[i] = frame
    
    # Write video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    print(f"Writing video to: {output_path}")
    for i in range(n_frames):
        out.write(combined[i])
        if (i + 1) % 30 == 0:
            print(f"  Frame {i + 1}/{n_frames}")
    
    out.release()
    print(f"Video saved: {output_path}")
    
    # Also save individual frames as images
    frames_dir = os.path.splitext(output_path)[0] + "_frames"
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
        print(f"Saving frames to: {frames_dir}")
        for i in range(n_frames):
            cv2.imwrite(f"{frames_dir}/frame_{i:04d}.png", combined[i])
            if (i + 1) % 30 == 0:
                print(f"  Frame {i + 1}/{n_frames}")
        print(f"Frames saved: {frames_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract video from npz RGB data")
    parser.add_argument("--data.path", type=str, 
                        default="/media/raid/workspace/tengbo/any4lerobot/data/raw_data/InterceptMedium-v0/train_data_1.npz",
                        help="Path to npz file")
    parser.add_argument("--output", type=str, default=None,
                        help="Output video path (auto-generated if None)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video FPS")
    parser.add_argument("--no-combine", action="store_true",
                        help="Don't combine primary and wrist views")
    
    args = parser.parse_args()
    
    extract_video(
        npz_path=getattr(args, "data.path"),
        output_path=getattr(args, "output"),
        fps=getattr(args, "fps"),
        combine_views=not getattr(args, "no_combine"),
    )


if __name__ == "__main__":
    main()
