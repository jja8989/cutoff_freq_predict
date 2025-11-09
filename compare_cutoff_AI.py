import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import platform
from scipy.fft import fft, ifft
from sklearn.metrics import mean_squared_error
import torch
from model import CutoffPredictorCNN_GRU
from NCDialog import create_dialog


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
matplotlib.use("TkAgg")


plt.rcParams['font.family'] = 'DejaVu Sans' 
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['mathtext.fontset'] = 'dejavusans' 
plt.rcParams['font.sans-serif'] = ['NanumGothic.ttf', 'sans-serif']



def compute_algorithmic_cutoff(ch1, time, DefInput):
    y_fft = fft(ch1)
    N = len(ch1)
    Fs = DefInput
    T = N / Fs
    freq = np.arange(N) / T

    FreqSearchRange = np.linspace(np.log10(freq[1]), np.log10(freq[-1] / 2), 60)


    MSE = np.zeros(len(FreqSearchRange))
    for i in range(len(FreqSearchRange) - 1):
        f_val = 10 ** FreqSearchRange[i]
        f_idx = np.argmax(freq > f_val)
        filt_fft = y_fft.copy()
        filt_fft[f_idx:N - f_idx] = 0
        ch1_return = np.real(ifft(filt_fft))
        MSE[i] = mean_squared_error(ch1, ch1_return)

    MSE2 = np.zeros(len(FreqSearchRange))
    for i in range(len(FreqSearchRange) - 1):
        f_val2 = 10 ** FreqSearchRange[i]
        f_idx2 = np.argmax(freq > f_val2)
        final_fft2 = y_fft.copy()
        final_fft2[:f_idx2] = 0
        ch1_return2 = np.real(ifft(final_fft2))
        MSE2[i] = mean_squared_error(ch1, ch1_return2)

    mse_mean = np.mean(MSE2)
    idx_closest = np.argmin(np.abs(MSE2 - mse_mean))

    log_mse = np.log10(MSE)
    grad_mse = np.gradient(log_mse)
    
    error_change_rate = np.gradient(FreqSearchRange) / grad_mse

    cutoff_idx_rel = np.argmin(error_change_rate[idx_closest:-1])
    cutoff_idx = idx_closest + cutoff_idx_rel
    cutoff_alg = 10 ** FreqSearchRange[cutoff_idx] * 1.15

    return cutoff_alg


def predict_cutoff_DL(signal, Fs_value, model_path="checkpoints3/best_model2.pth"):
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
    print(f"[INFO] Loaded model with y_mean={y_mean:.4f}, y_std={y_std:.4f}")

    model = CutoffPredictorCNN_GRU().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    model.eval()
    
    sig = np.asarray(signal, dtype=np.float32)
    mean, std = np.mean(sig), np.std(sig)
    if std > 1e-8:
        sig = (sig - mean) / std

    x = torch.tensor(sig, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    L = x.shape[-1]
    lengths = torch.tensor([L], dtype=torch.long).to(DEVICE)
    Fs = torch.tensor([np.log10(Fs_value)], dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pred_norm = model(x, Fs, lengths).item()

    pred_log = pred_norm * y_std + y_mean
    cutoff_pred = 10 ** pred_log
    print(f"[PRED] log10 cutoff = {pred_log:.4f} → cutoff = {cutoff_pred:.2f} Hz")

    return cutoff_pred

def compute_full_pipeline(ch1, time, cap_area, cutoff_freq, DefInput, para1, para3, para4, para5):
    y_fft = fft(ch1)
    N = len(ch1)
    Fs = DefInput
    T = N / Fs
    freq = np.arange(N) / T

    pos = np.argmax(freq > cutoff_freq)
    filt_fft = y_fft.copy()
    filt_fft[pos:N - pos] = 0
    ch1_return = np.real(ifft(filt_fft))

    tap = np.min(ch1[:round(0.1 * len(time))]) * 1.1
    offset_candidates = np.linspace(-tap, tap, 200)
    
    MSE = [mean_squared_error(ch1_return, o * np.ones_like(ch1_return)) for o in offset_candidates]
    
    offset = offset_candidates[np.argmin(MSE)]
    ch1_return -= offset

    current_data = ch1_return / 50
    del_time = np.diff(time)
    del_charge = del_time * current_data[:-1]
    accum_charge = np.cumsum(del_charge)
    net_accum_charge_density = accum_charge / cap_area
    
    Q_charge = np.max(np.real(net_accum_charge_density))
    Q_charge_position = np.argmax(np.real(net_accum_charge_density)) 
    
    
    end_half_index = Q_charge_position + round(0.5 * (len(time) - Q_charge_position))
    slope_range2 = end_half_index - Q_charge_position
    slope_finder2 = np.zeros(slope_range2)

    for i in range(slope_range2 - 1):
        idx = Q_charge_position + i
        slope_finder2[i] = (net_accum_charge_density[-1] - net_accum_charge_density[idx]) / (time[-1] - time[idx])

    max_slope2 = np.argmin(slope_finder2) + Q_charge_position

    delay_ns = para5 + 0.6 * para4
    delay_s = delay_ns * 1e-9       
    
    Qres_position = np.where(time > time[max_slope2] + delay_s)[0][0]

    Q_res = np.min(np.real(net_accum_charge_density[Qres_position:]))
    Qres_position = np.where(net_accum_charge_density == Q_res)[0][0]
    
    Q_discharge = Q_charge - Q_res


    slope_range = int(max_slope2 * 0.7)
    slope_finder = np.zeros(slope_range)

    for i in range(slope_range):
        slope_finder[i] = (net_accum_charge_density[max_slope2] - net_accum_charge_density[i]) / \
                        (time[max_slope2] - time[i])

    MaxSlope = np.argmax(slope_finder)
    MaxSlope = round(0.75 * MaxSlope)
    
    polyfit_x = time[:MaxSlope]
    polyfit_y = net_accum_charge_density[:MaxSlope]
    slope, intercept = np.polyfit(polyfit_x, polyfit_y, 1)

    drift = slope * time[:-1] + intercept
    Qres_adjust = net_accum_charge_density[:] - drift

    NewQ_charge = Qres_adjust[Q_charge_position]
    NewQ_res = np.min(np.real(Qres_adjust[Qres_position:]))
    NewQ_discharge = NewQ_charge - NewQ_res
    ErrorRatio = 100 * NewQ_res / NewQ_charge
    
    if para1 == 'Negative biased':
        ch1 = -ch1
        ch1_return = -ch1_return
        net_accum_charge_density = -net_accum_charge_density
        Qres_adjust = - Qres_adjust
        
        Q_charge = -Q_charge
        Q_discharge = -Q_discharge
        Q_res = -Q_res
        
        NewQ_charge = -NewQ_charge
        NewQ_discharge = -NewQ_discharge
        NewQ_res = -NewQ_res
        
        offset = -offset
        
    if ErrorRatio > para3:
        Q_charge = NewQ_charge
        Q_res = NewQ_res
        Q_discharge = NewQ_discharge
        net_accum_charge_density = Qres_adjust

    return {
        "ch1" : ch1,
        "cutoff": cutoff_freq,
        "signal": ch1_return,
        "charge": net_accum_charge_density,
        "Q_charge_pos" : Q_charge_position,
        "Q_charge": Q_charge,
        "Q_res_pos" : Qres_position,
        "Q_res": Q_res,
        "Q_discharge": Q_discharge,
        "ErrorRatio": ErrorRatio,
        "offset": offset,
    }


# ======================================================
# 🔹 비교 시각화
# ======================================================
def compare_two_cutoffs(ch1, time, cap_area, DefInput, para1, cutoff_alg, para3, para4, para5):
    cutoff_dl = predict_cutoff_DL(ch1, DefInput)
    
    print(f"\n📊 Algorithm cutoff: {cutoff_alg:.3e} Hz")
    print(f"🤖 Model cutoff:     {cutoff_dl:.3e} Hz\n")

    result_alg = compute_full_pipeline(ch1, time, cap_area, cutoff_alg, DefInput, para1, para3, para4, para5)
    result_dl = compute_full_pipeline(ch1, time, cap_area, cutoff_dl, DefInput, para1, para3, para4, para5)

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    fig.subplots_adjust(top=0.85, hspace=0.5, wspace=0.3)
    fig.suptitle(f"Algorithm vs Model Cutoff Comparison", fontsize=18, color='red')

    axes[0, 0].plot(time, result_alg["ch1"], 'gray', lw=0.8, label='Raw')
    axes[0, 0].plot(time, np.ones_like(time)*result_alg["offset"], 'b--', lw=1, label='Offset')
    axes[0, 0].plot(time, result_alg["signal"], 'r', lw=1.5,
                    label=f'Denoised ({result_alg["cutoff"]/1e6:.2f} MHz)')
    axes[0, 0].set_title("Algorithm Denoised"); axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(time, result_dl["ch1"], 'gray', lw=0.8, label='Raw')
    axes[0, 1].plot(time, np.ones_like(time)*result_dl["offset"], 'b--', lw=1, label='Offset')
    axes[0, 1].plot(time, result_dl["signal"], 'b', lw=1.5,
                    label=f'Denoised ({result_dl["cutoff"]/1e6:.2f} MHz)')
    axes[0, 1].set_title("Model Denoised"); axes[0, 1].legend(); axes[0, 1].grid(True)


    ErrorRate_indic = f"Qres size = {result_alg["ErrorRatio"]:.3f}% of Q₍charge₎"
    Q_charge_indic = f"Q₍charge₎ = {result_alg["Q_charge"]:.3e} [C/m²]"
    Q_discharge_indic = f"Q₍discharge₎ = {result_alg["Q_discharge"]:.3e} [C/m²]"
    Q_res_indic = f"Q₍res₎ = {result_alg["Q_res"]:.3e} [C/m²]"
    
    axes[1, 0].plot(time[:-1], result_alg["charge"], 'r')
    axes[1, 0].plot(time[result_alg["Q_charge_pos"]], result_alg["charge"][result_alg["Q_charge_pos"]], 'bo')
    axes[1, 0].plot(time[result_alg["Q_res_pos"]], result_alg["charge"][result_alg["Q_res_pos"]], 'bo')
    axes[1, 0].text(0.6 * time[-1], 0.6 * result_alg["Q_charge"], ErrorRate_indic, fontsize=8)
    axes[1, 0].text(0.6 * time[-1], 0.7 * result_alg["Q_charge"], Q_res_indic, fontsize=8)
    axes[1, 0].text(0.6 * time[-1], 0.8 * result_alg["Q_charge"], Q_discharge_indic, fontsize=8)
    axes[1, 0].text(0.6 * time[-1], 0.9 * result_alg["Q_charge"], Q_charge_indic, fontsize=8)

    axes[1, 0].set_title("Algorithm Charge Density"); axes[1, 0].grid(True)
    
    ErrorRate_indic = f"Qres size = {result_dl["ErrorRatio"]:.3f}% of Q₍charge₎"
    Q_charge_indic = f"Q₍charge₎ = {result_dl["Q_charge"]:.3e} [C/m²]"
    Q_discharge_indic = f"Q₍discharge₎ = {result_dl["Q_discharge"]:.3e} [C/m²]"
    Q_res_indic = f"Q₍res₎ = {result_dl["Q_res"]:.3e} [C/m²]"
    
    axes[1, 1].plot(time[:-1], result_dl["charge"], 'b')
    axes[1, 1].plot(time[result_dl["Q_charge_pos"]], result_dl["charge"][result_dl["Q_charge_pos"]], 'ro')
    axes[1, 1].plot(time[result_dl["Q_res_pos"]], result_dl["charge"][result_dl["Q_res_pos"]], 'ro')
    axes[1, 1].text(0.6 * time[-1], 0.6 * result_dl["Q_charge"], ErrorRate_indic, fontsize=8)
    axes[1, 1].text(0.6 * time[-1], 0.7 * result_dl["Q_charge"], Q_res_indic, fontsize=8)
    axes[1, 1].text(0.6 * time[-1], 0.8 * result_dl["Q_charge"], Q_discharge_indic, fontsize=8)
    axes[1, 1].text(0.6 * time[-1], 0.9 * result_dl["Q_charge"], Q_charge_indic, fontsize=8)

    axes[1, 1].set_title("Model Charge Density"); axes[1, 1].grid(True)

    plt.show()


# ======================================================
# 🔹 메인 루틴
# ======================================================
def main_load_and_config():
    file_path = filedialog.askopenfilename()
    if not file_path:
        raise FileNotFoundError("파일이 선택되지 않았습니다.")

    data = np.loadtxt(file_path, skiprows=1)
    time, ch1, ch2 = data[:, 0], data[:, 1], data[:, 2]
    dt_mean = np.mean(np.gradient(time))
    DefInput = 1 / dt_mean

    params, _ = create_dialog()
    
    try:
        para1 = params["BiasSign"]
        para2 = float(params["CapacitorArea"])
        para3 = float(params["QresTolerance"])
        para4 = float(params["PulseWidth"])
        para5 = float(params["PulseRiseFall"])
    except ValueError:
        raise ValueError("Input parameters are not valid number.")


    if para2 < 0 or para3 <= 0 or para3 >= 100 or para4 <= 0 or para5 <= 0:
        raise ValueError("Input parameter condition error : Verify whether it is valid.")

    cap_area = para2 * 1e-12
    
    cutoff_alg = compute_algorithmic_cutoff(ch1, time, DefInput)

    compare_two_cutoffs(ch1, time, cap_area, DefInput, para1, cutoff_alg, para3, para4, para5)


if __name__ == "__main__":
    main_load_and_config()
