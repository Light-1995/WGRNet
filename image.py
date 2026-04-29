import h5py
import numpy as np
import torch
import cv2
from model.thops import mean
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

data = h5py.File('E:\pansharpening-main\dataset\WV3\\test\\reduced\\test_wv3_multiExm1.h5', 'r')
print("---keys---")
print(data.keys())
print("---key size---")
print(data["gt"].shape)
index = 5
gt1 = data["gt"][index]

print(gt1.shape, "before downsampling")
gt1 = np.array(gt1, dtype=np.float32) / 2047.0
gt_image = torch.from_numpy(gt1)
out_mean_gt = mean(gt_image, dim=0, keepdim=True)
print("GT data size:", out_mean_gt.size())

lms1 = data["ms"][index]
print(lms1.shape, "before upsampling")
channels, original_height, original_width = lms1.shape

target_height = 4 * original_height
target_width = 4 * original_width

lms1_resized = np.zeros((channels, target_height, target_width), dtype=lms1.dtype)
for c in range(channels):
    lms1_resized[c, :, :] = cv2.resize(lms1[c, :, :], (target_width, target_height), interpolation=cv2.INTER_CUBIC)

print(lms1_resized.shape, "after upsampling")

lms1 = np.array(lms1_resized, dtype=np.float32) / 2047.0
bms_image = torch.from_numpy(lms1)
out_mean_bms = mean(bms_image, dim=0, keepdim=True)
print(out_mean_bms.size())

pan1 = data["pan"][index]
print(pan1.shape, "before downsampling")
pan1 = np.array(pan1, dtype=np.float32) / 2047.0
out_pan = torch.from_numpy(pan1)
print(out_pan.size())

average_image = (out_pan + out_mean_bms) / 2.0
gt_np = out_mean_gt.numpy()
bms_np = out_mean_bms.numpy()
pan_np = out_pan.numpy()
average_np = average_image.numpy()

print("Normalized pan range:", np.min(pan_np), np.max(pan_np))
print("Normalized gt range:", np.min(gt_np), np.max(gt_np))

save_path_gt = 'C:/Users/Administrator/Desktop/mean/mean_gt.png'
save_path_lms = 'C:/Users/Administrator/Desktop/mean/mean_lms.png'
save_path_pan = 'C:/Users/Administrator/Desktop/mean/pan.png'
save_path_average = 'C:/Users/Administrator/Desktop/mean/average.png'

gt_scaled_gray = cv2.normalize(gt_np * 2047, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)[0, :, :]
bms_scaled_gray = cv2.normalize(bms_np * 2047, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)[0, :, :]
pan_scaled_gray = ((cv2.normalize(pan_np * 2047, None, 0, 255, cv2.NORM_MINMAX)+0)*1).astype(np.uint8)[0, :, :]
average_scaled_gray = ((cv2.normalize(average_np * 2047, None, 0, 255, cv2.NORM_MINMAX)+0)*1).astype(np.uint8)[0, :, :]

print("Normalized pan range:", np.min(pan_scaled_gray), np.max(pan_scaled_gray))
print("Normalized gt range:", np.min(gt_scaled_gray), np.max(gt_scaled_gray))
print("Normalized average range:", np.min(average_scaled_gray), np.max(average_scaled_gray))

hist_gt = cv2.calcHist([gt_scaled_gray], [0], None, [256], [0, 256])
hist_bms = cv2.calcHist([bms_scaled_gray], [0], None, [256], [0, 256])
hist_pan = cv2.calcHist([pan_scaled_gray], [0], None, [256], [0, 256])
hist_average = cv2.calcHist([average_scaled_gray], [0], None, [256], [0, 256])

plt.plot(hist_gt, color='blue', label='GT')
plt.plot(hist_bms, color='green', label='BMS')
plt.plot(hist_pan, color='red', label='PAN')
plt.plot(hist_average, color='yellow', label='AVG(PAN+BMS)')

plt.title('Histogram of Grayscale Images')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

cv2.imwrite(save_path_gt, gt_scaled_gray, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite(save_path_lms, bms_scaled_gray, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite(save_path_pan, pan_scaled_gray, [cv2.IMWRITE_PNG_COMPRESSION, 0])
cv2.imwrite(save_path_average, average_scaled_gray, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def plot_fft_spectrum(image, title):
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    plt.figure(figsize=(8, 8))
    plt.imshow(magnitude_spectrum)
    plt.title(title)
    plt.colorbar()
    plt.show()

plot_fft_spectrum(gt_scaled_gray, 'GT Fourier Transform Spectrum')
plot_fft_spectrum(bms_scaled_gray, 'BMS Fourier Transform Spectrum')
plot_fft_spectrum(pan_scaled_gray, 'PAN Fourier Transform Spectrum')

def calculate_spectrum_difference(image1, image2):
    f_transform1 = fft2(image1)
    f_transform2 = fft2(image2)
    f_transform_shifted1 = fftshift(f_transform1)
    f_transform_shifted2 = fftshift(f_transform2)
    magnitude_spectrum1 = np.log(np.abs(f_transform_shifted1) + 1)
    magnitude_spectrum2 = np.log(np.abs(f_transform_shifted2) + 1)
    spectrum_difference = magnitude_spectrum1 - magnitude_spectrum2
    return spectrum_difference

spectrum_diff_gt_bms = calculate_spectrum_difference(gt_scaled_gray, bms_scaled_gray)
spectrum_diff_gt_pan = calculate_spectrum_difference(gt_scaled_gray, pan_scaled_gray)
spectrum_diff_gt_average = calculate_spectrum_difference(gt_scaled_gray, average_scaled_gray)
spectrum_diff_pan_bms = calculate_spectrum_difference(pan_scaled_gray, bms_scaled_gray)

plt.figure(figsize=(8, 8))
plt.imshow(spectrum_diff_gt_bms)
plt.title('Spectrum Difference (GT - BMS)')
plt.colorbar()
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(spectrum_diff_gt_pan)
plt.title('Spectrum Difference (GT - PAN)')
plt.colorbar()
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(spectrum_diff_gt_average)
plt.title('Spectrum Difference (GT - Average)')
plt.colorbar()
plt.show()

plt.figure(figsize=(8, 8))
plt.imshow(spectrum_diff_pan_bms)
plt.title('Spectrum Difference (PAN - Average)')
plt.colorbar()
plt.show()