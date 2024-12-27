import cv2
import os
import numpy as np
import matplotlib.pyplot as plt 
import copy
from scipy.ndimage import convolve

def gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(- ((x-(size-1)/2)**2 + (y-(size-1)/2)**2) / (2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def generate_deartifact_filter():
    # init base filter
    base_filter = np.zeros((400, 400), np.float32)
    
    kernel_size = 9  # Size of the kernel (should be an odd number)
    sigma = 5  # Standard deviation of the Gaussian

    intensity = 1
    base_filter[50:350, 100] = intensity 
    base_filter[50:350, 300] = intensity 
    
    gaussian_k = gaussian_kernel(kernel_size, sigma)
    # Convolve the image with the base fileter
    base_filter_gaussian = convolve(base_filter, gaussian_k)
    base_filter_gaussian = 1 - base_filter_gaussian / np.max(base_filter_gaussian)

    return base_filter_gaussian

def apply_frequency_domain_filter(img, frquency_domain_filter):
    img_filtered_color = np.zeros(img.shape, np.float32)
    for ch in range(3):
        gray_img = img[:, :, ch]
        f_img = np.fft.fft2(gray_img)
        f_img_shifted = np.fft.fftshift(f_img)
        f_img_shifted_ = copy.deepcopy(f_img_shifted)
        f_img_shifted_amp = np.abs(f_img_shifted_)
        f_img_shifted_phase = np.angle(f_img_shifted_)
        # Apply the Gaussian high-pass filter
        f_img_shifted_amp_filtered = f_img_shifted_amp * frquency_domain_filter
        f_img_shifted_filtered = f_img_shifted_amp_filtered * (np.cos(f_img_shifted_phase) + 1j * np.sin(f_img_shifted_phase))
        f_img_filtered = np.fft.ifftshift(f_img_shifted_filtered)
        img_filtered = np.real(np.fft.ifft2(f_img_filtered))
        img_filtered_color[:, :, ch] = img_filtered
    return img_filtered_color

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extract frames from a video and save them to a folder.

    Parameters:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where frames will be saved.
        frame_rate (int): Save one frame every `frame_rate` frames.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    filter_ = generate_deartifact_filter()

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the frame.")
            break

        frame = frame[:, :768, :]
        frame = cv2.resize(frame, [400, 400])

        frame = apply_frequency_domain_filter(frame, filter_)
        width = frame.shape[1]
        height = frame.shape[0]
        focal_length = 246.8

        # Define camera matrix (assuming no skew, simple focal length, and center of image as principal point)
        center = (width / 2, height / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], 
                                [0, focal_length, center[1]], 
                                [0, 0, 1]], dtype="double")
        # Define distortion coefficients (k1, k2, p1, p2, k3)
        # Positive values cause barrel distortion, negative values cause pincushion distortion
        # dist_coeff = np.array([-0.184048271181615, 0.0242569027226932, 0, 0, 0])
        dist_coeff = np.array([-0.184, 0.0243, 0, 0, 0])

        # Apply distortion
        distorted_imgblock = cv2.undistort(frame, camera_matrix, dist_coeff)
        distorted_imgblock = cv2.resize(distorted_imgblock, [512, 512])

        # Save every `frame_rate` frame
        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, distorted_imgblock)
            print(f"Saved: {frame_filename}")
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extraction complete. {saved_count} frames saved to {output_folder}")

# Example usage
video_path = "/data/hdd/Data/SkinSight_video/20241221/output_video_0_rgbd.mp4"  # Replace with your video file path
output_folder = "/home/xiaoyun/Code/Data/NVDS/demo_videos/SkinSight"  # Replace with your desired output folder
frame_rate = 1  # Save one frame every 10 frames

extract_frames(video_path, output_folder, frame_rate)
