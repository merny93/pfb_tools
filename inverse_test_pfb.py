import ctypes
import numpy as np


def inverse_pfb_fft(data, ntap, window_func = sinc_hanning, thresh =0.0):
    ##undo the DFT as it was applied so along the horizontals
    ##thus vertical columns of this matrix represent mixed points as discribed above
    psudo_ts = np.fft.irfft(data, axis=1)
    
    ##get the window function vals and shape it 
    window_mat = get_window(ntap, psudo_ts.shape[1], window = window_func).reshape(ntap, -1)
    
    #init the SW_p matrix
    SW_P = np.zeros(psudo_ts.shape, dtype=psudo_ts.dtype)
    SW_P[:ntap,:] = window_mat

    ##switch into freq domain
    SW_P_Ft = np.fft.rfft(SW_P, axis = 0)
    psudo_ts_Ft = np.fft.rfft(psudo_ts, axis = 0)

    ##filter if we want to
    if thresh > 0:
        filt = np.abs(SW_P_Ft)**2/(thresh**2 + np.abs(SW_P_Ft)**2)*(1+ thresh**2)
        SW_P_Ft = SW_P_Ft * filt

    ##deconvolve:
    ts_Ft = SW_P_Ft / np.conj(psudo_ts_Ft)

    ##return to time domain
    reconstructed_ts = np.fft.irfft(ts_Ft, axis=0)

    return reconstructed_ts