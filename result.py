import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_correlation(img1, img2):
    num_bands = img1.shape[2]
    corr = []
    for b in range(num_bands):
        x = img1[:, :, b].flatten()
        y = img2[:, :, b].flatten()
        corr.append(np.corrcoef(x, y)[0, 1])
    return np.mean(corr)


def spectral_distortion(fused, ms):
    num_bands = ms.shape[2]
    diffs = []
    for i in range(num_bands):
        for j in range(i+1, num_bands):
            rho_f = np.corrcoef(fused[:, :, i].flatten(), fused[:, :, j].flatten())[0, 1]
            rho_m = np.corrcoef(ms[:, :, i].flatten(), ms[:, :, j].flatten())[0, 1]
            diffs.append(abs(rho_f - rho_m))
    return np.mean(diffs)


def spatial_distortion(fused, pan, ms):
    num_bands = ms.shape[2]
    diffs = []
    for i in range(num_bands):
        rho_f = np.corrcoef(fused[:, :, i].flatten(), pan.flatten())[0, 1]
        rho_m = np.corrcoef(ms[:, :, i].flatten(), pan.flatten())[0, 1]
        diffs.append(abs(rho_f - rho_m))
    return np.mean(diffs)


def qnr(fused, ms, pan, alpha=1, beta=1):
    d_lambda = spectral_distortion(fused, ms)
    d_s = spatial_distortion(fused, pan, ms)
    qnr_value = (1 - d_lambda) ** alpha * (1 - d_s) ** beta
    return qnr_value, d_lambda, d_s