import numpy as np
from scipy.ndimage import gaussian_filter

def MTF2(fused, sensor, ratio):
    sensor_sigma = {
        'WV3': 0.4,
        'IKONOS': 0.5,
        'pleiades': 0.35,
        'none': 0.4
    }
    sigma = sensor_sigma.get(sensor, 0.4) * ratio
    fused_lp = gaussian_filter(fused, sigma=(sigma, sigma, 0))
    return fused_lp


def q2n(ms_exp, fused_degraded, block_h, block_w):
    H, W, C = ms_exp.shape
    num_blocks_h = H // block_h
    num_blocks_w = W // block_w
    Q2n_list = []

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            h_start = i * block_h
            w_start = j * block_w
            block_ms = ms_exp[h_start:h_start+block_h, w_start:w_start+block_w, :]
            block_fused = fused_degraded[h_start:h_start+block_h, w_start:w_start+block_w, :]
            
            rho = []
            for b1 in range(C):
                for b2 in range(b1+1, C):
                    x = block_ms[:, :, b1].flatten()
                    y = block_ms[:, :, b2].flatten()
                    rho_m = np.corrcoef(x, y)
                    
                    x_f = block_fused[:, :, b1].flatten()
                    y_f = block_fused[:, :, b2].flatten()
                    rho_f = np.corrcoef(x_f, y_f)
                    
                    rho.append(rho_f / rho_m if rho_m != 0 else 1.0)
            Q2n_list.append(np.mean(rho))
    return np.mean(Q2n_list), None<websource>source_group_web_1</websource>


def D_lambda_K(fused, ms_exp, ratio, sensor, S):
    H, W, C = fused.shape
    if H % S != 0 or W % S != 0:
        raise ValueError("number of rows/columns must be multiple of the block size")
    fused_degraded = MTF2(fused, sensor, ratio)
    Q2n_index, _ = q2n(ms_exp, fused_degraded, S, S)
    Dl = 1 - Q2n_index
    return Dl


def D_s2(ps_ms, ms_exp, ms, pan, ratio, S, alpha=1):
    H, W, C = ps_ms.shape
    if H % S != 0 or W % S != 0:
        raise ValueError("number of rows/columns must be multiple of the block size")
    
    num_blocks_h = H // S
    num_blocks_w = W // S
    rho_list = []

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            h_start = i * S
            w_start = j * S
            for b in range(C):
                block_fused = ps_ms[h_start:h_start+S, w_start:w_start+S, b].flatten()
                block_ms = ms_exp[h_start:h_start+S, w_start:w_start+S, b].flatten()
                rho_f = np.corrcoef(block_fused, pan[h_start:h_start+S, w_start:w_start+S].flatten())
                rho_m = np.corrcoef(block_ms, pan[h_start:h_start+S, w_start:w_start+S].flatten())
                rho_list.append(abs(rho_f - rho_m))
    Ds = np.mean(rho_list)
    return Ds<websource>source_group_web_2</websource>


def HQNR(ps_ms, ms, ms_exp, pan, S=32, sensor='pleiades', ratio=4):
    Dl = D_lambda_K(ps_ms, ms_exp, ratio, sensor, S)
    Ds = D_s2(ps_ms, ms_exp, ms, pan, ratio, S)
    HQNR_value = (1 - Dl) * (1 - Ds)
    return HQNR_value, Dl, Ds