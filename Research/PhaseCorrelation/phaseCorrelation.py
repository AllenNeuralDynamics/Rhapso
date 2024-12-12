"""
phaseCorrelation.py

This script performs phase correlation between two images to estimate the translation between them.
It includes functions to normalize FFT data, calculate the Phase Correlation Matrix (PCM), visualize the PCM,
describe the translation direction, and summarize the results.

How to run:
1. Ensure you have the required libraries installed:
   - OpenCV (cv2)
   - NumPy (numpy)
   - Matplotlib (matplotlib)
   
   You can install them using pip:
   pip install opencv-python-headless numpy matplotlib

2. Place the images you want to compare in the same directory as this script.
The script currently reads 'apple.jpg' as the original image.

3. Run the script using Python:
python phaseCorrelation.py

4. The script will read the image, perform phase correlation, and output the results.
It will also save a visualization of the Phase Correlation Matrix (PCM) to a file.

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize_fft(fft_data):
    """
    Normalize the FFT data by dividing by its magnitude.
    """
    magnitude = np.abs(fft_data)
    # Avoid division by zero
    normalized_fft = fft_data / (magnitude + 1e-10)
    return normalized_fft

def calculate_pcm(img1, img2):
    """
    Calculate the Phase Correlation Matrix (PCM) between two images.
    """
    fft1 = np.fft.fft2(img1)
    fft2 = np.fft.fft2(img2)
    fft1_normalized = normalize_fft(fft1)
    fft2_normalized = normalize_fft(fft2)
    fft2_conj = np.conj(fft2_normalized)
    cross_power_spectrum = fft1_normalized * fft2_conj
    pcm = np.fft.ifft2(cross_power_spectrum)
    pcm_magnitude = np.abs(pcm)
    return pcm, pcm_magnitude

def visualize_pcm(pcm_magnitude, save_path):
    plt.imshow(pcm_magnitude, cmap='gray')
    plt.colorbar()
    plt.title('Phase Correlation Matrix (PCM)')
    plt.savefig(save_path)
    plt.close()

def describe_translation(x, y):
    vertical = "bottom" if y > 0 else "top"
    horizontal = "right" if x > 0 else "left"
    return f"shifted {vertical} {horizontal}"

def summary_of_results(real_x, real_y, estimated_x, estimated_y, score):
    confidence = "high" if score > 0.8 else "moderate" if score > 0.5 else "low"
    accuracy_x = abs((real_x - estimated_x) / real_x) * 100 if real_x != 0 else 0
    accuracy_y = abs((real_y - estimated_y) / real_y) * 100 if real_y != 0 else 0
    print(f"\nSummary of Results:\nConfidence level: {confidence} ({score * 100:.2f}% confident)")

if __name__ == "__main__":
    original_image = cv2.imread('apple.jpg', cv2.IMREAD_GRAYSCALE)
    height, width = original_image.shape
    crop_height = height // 2
    crop_width = width // 2
    translate_x = np.random.randint(-width // 4, width // 4)
    translate_y = np.random.randint(-height // 4, height // 4)

    img1 = original_image[:crop_height, :crop_width]
    img2 = np.roll(original_image, shift=(translate_y, translate_x), axis=(0, 1))[:crop_height, :crop_width]

    cv2.imwrite('slice1.jpg', img1)
    cv2.imwrite('slice2.jpg', img2)

    pcm, pcm_magnitude = calculate_pcm(img1, img2)
    visualize_pcm_path = 'phase_correlation_matrix.png'
    visualize_pcm(pcm_magnitude, visualize_pcm_path)
    print(f"\nPhase Correlation Matrix visualization saved as '{visualize_pcm_path}'")

    direction_description = describe_translation(translate_x, translate_y)
    print(f"\nReal translation: X = {translate_x}, Y = {translate_y} ({direction_description})\n")

    (shift_x, shift_y), phase_corr_score = cv2.phaseCorrelate(np.float32(img1), np.float32(img2))
    print(f"Estimated translation offset: X = {shift_x:.2f}, Y = {shift_y:.2f}\nCorrelation Score: {phase_corr_score:.2f}\n")

    summary_of_results(translate_x, translate_y, shift_x, shift_y, phase_corr_score)
