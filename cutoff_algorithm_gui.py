import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import os
from Dialog import *
from scipy.fft import fft, ifft
from sklearn.metrics import mean_squared_error
import matplotlib.font_manager as fm
import platform


plt.rcParams['font.family'] = 'DejaVu Sans' 
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['mathtext.fontset'] = 'dejavusans' 
plt.rcParams['font.sans-serif'] = ['NanumGothic.ttf', 'sans-serif']


matplotlib.use("TkAgg")

def sci_notation_tex(x, precision=3):
    mantissa, exp = f"{x:.{precision}e}".split("e")
    exp = int(exp)
    return rf"{mantissa}\times 10^{{{exp}}}"


def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    return file_path

def main_load_and_config():

    file_path = select_file()
    if not file_path:
        raise FileNotFoundError("File was not selected")
    
    dat_file = os.path.basename(file_path)
    folder_path  = os.path.dirname(file_path)


    data = np.loadtxt(file_path, skiprows=1)
    time = data[:, 0]
    ch1 = data[:, 1]
    ch2 = data[:, 2]


    dt_mean = np.mean(np.gradient(time))
    if dt_mean <= 0:
        raise ValueError("Time step is not valid")
    DefInput = 1 / dt_mean

    params, file_path = create_dialog()
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

    prompt1 = para1
    cap_area = para2 * 1e-12  # um^2 → m^2
    
    y_fft = fft(ch1)  # FFT of ch1
    N = len(ch1)      # Number of samples
    Fs = DefInput     # Sampling frequency (Hz)
    T = N / Fs        # Total time duration

    k = np.arange(N)        # [0, 1, ..., N-1]
    freq = k / T            # frequency = k / total time
    
    FreqInit = np.log10(freq[1])        
    FreqLast = np.log10(freq[-1] / 2)   
    FreqSearchRange = np.linspace(FreqInit, FreqLast, 60)  # shape: (60,)


    MSE = np.zeros(len(FreqSearchRange))

    for i in range(len(FreqSearchRange) - 1):
        filter_freq_value = 10 ** FreqSearchRange[i]

        filter_freq_pos = np.argmax(freq > filter_freq_value)

        final_fft = y_fft.copy()
        final_fft[filter_freq_pos : N - filter_freq_pos] = 0

        ch1_return = np.real(ifft(final_fft))

        MSE[i] = mean_squared_error(ch1, ch1_return)
        
        
    MSE2 = np.zeros(len(FreqSearchRange))
    
    for i in range(len(FreqSearchRange) - 1):
        filter_freq_value2 = 10 ** FreqSearchRange[i]
        
        filter_freq_pos2 = np.argmax(freq > filter_freq_value2)
        
        final_fft2 = y_fft.copy()
        final_fft2[:filter_freq_pos2] = 0
        
        ch1_return2 = np.real(ifft(final_fft2))
        MSE2[i] = mean_squared_error(ch1, ch1_return2)
        
    mse_mean = np.mean(MSE2)
    idx_closest = np.argmin(np.abs(MSE2 - mse_mean))


    original_mse = np.gradient(MSE)
    log_mse = np.log10(MSE)
    grad_mse = np.gradient(log_mse)
    
    original_grad = original_mse / np.gradient(FreqSearchRange)
    
    log_original = grad_mse / np.gradient(FreqSearchRange) 
    
    
    error_change_rate = np.gradient(FreqSearchRange) / grad_mse
    
    cutoff_idx_rel = np.argmin(error_change_rate[idx_closest:-1])
    cutoff_idx = idx_closest + cutoff_idx_rel

    original_cutoff_freq = 10 ** FreqSearchRange[cutoff_idx]
    cutoff_freq = 10 ** FreqSearchRange[cutoff_idx] * 1.15

    filter_freq_pos = np.argmax(freq > cutoff_freq)
    
    final_fftout = np.copy(y_fft)
    final_fftout[filter_freq_pos : N - filter_freq_pos] = 0
    
    ch1_return = np.real(ifft(final_fftout))

    tap = np.min(ch1[:round(0.1 * len(time))]) * 1.1

    OSCErrorMean = np.mean(ch1_return)

    if np.abs(OSCErrorMean) > np.abs(tap):
        OSCErrorMean = np.mean(ch1[:round(0.1 * len(time))])

    ch1_return_beforeAdj = ch1_return.copy()

    ch1_return = ch1_return - OSCErrorMean

    current_data = ch1_return / 50  # V = IR → I = V / R
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

    ErrorRatio = 100 * Q_res / Q_charge

    
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
    NewQres_position = np.where(Qres_adjust == NewQ_res)[0][0]
    
    NewQ_discharge = NewQ_charge - NewQ_res
    NewErrorRatio = 100 * NewQ_res / NewQ_charge
    

    if para1 == 'Negative biased':
        ch1 = -ch1
        ch1_return = -ch1_return
        net_accum_charge_density = -net_accum_charge_density
        Q_charge = -Q_charge
        Q_discharge = -Q_discharge
        Q_res = -Q_res
        
        NewQ_charge = -NewQ_charge
        NewQ_discharge = -NewQ_discharge
        NewQ_res = -NewQ_res
        
        OSCErrorMean = - OSCErrorMean
                
    plt.figure(figsize=(8, 5))
    plt.plot(time, ch1)
    plt.title("Original Signal", pad=15)
    plt.xlabel('Time')
    plt.ylabel('Voltage')
    plt.grid(True)
    plt.tight_layout()
    
    plt.figure(figsize=(8, 5))
    plt.subplot(2,1,1)
    plt.title("FFT Magnitude (Linear Scale)")
    plt.plot(freq[1:], np.abs(y_fft[1:]))
    plt.title("FFT Magnitude Spectrum (Linear scale)", pad=15)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|FFT(ch1)|")
    plt.grid(True)
    
    plt.subplot(2,1,2)
    plt.semilogx(freq[1:], np.abs(y_fft[1:]))
    plt.title("FFT Magnitude Spectrum (Log scale)", pad=15)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|FFT(ch1)|")
    plt.grid(True, which='both')

    plt.subplots_adjust(hspace=1.5)
    
    import matplotlib.ticker as ticker

    
    plt.figure(figsize=(8, 5))
    plt.loglog(10 ** FreqSearchRange, MSE, label="Low-pass filter MSE")
    plt.loglog(10 ** FreqSearchRange[:-1], MSE2[:-1], label="High-pass filter MSE")
    plt.axhline(mse_mean, color='gray', linestyle=':', label='Mean of High-pass filter MSE')
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("MSE")
    plt.yticks(fontname="Liberation Sans")
    plt.title("MSE Curves for Cutoff Frequency Search", pad=15)
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    
    plt.figure(figsize=(8, 5))
    plt.plot(10 ** FreqSearchRange, np.sign(error_change_rate) * np.log(np.abs(error_change_rate)), label="Inverse Gradient")
    plt.plot(10 ** FreqSearchRange, log_original, label="Original Gradient")
    plt.xscale('log')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: fr'$10^{{{int(v)}}}$')
    )
    
    plt.axvline(original_cutoff_freq, color='r', linestyle='--', label=f"Cutoff ≈ {cutoff_freq/1e6:.2f} MHz")
    # plt.axvline(cutoff_freq, color='r', linestyle='--', label=f"Cutoff ≈ {cutoff_freq/1e6:.2f} MHz")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Gradient")
    plt.title("Inverse Gradient Curve of Log MSE", pad=15)
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    
    plt.figure(figsize=(8, 5))
    plt.plot(time, ch1, 'k-', label='Raw Signal')
    plt.plot(time, ch1_return, 'r-', linewidth=2, label='Denoised Signal')
    plt.plot(time, np.ones_like(time) * OSCErrorMean, 'b--', label='Offset')
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title("Time-domain Signal Before and After FFT Denoising", pad=15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    state = 1
    

    bbox_style = dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="gray",
        alpha=0.85
    )


    while state > 0:
        if state == 1:

            Q_charge_indic = (
                    rf"$\mathbf{{Q_{{charge}} = {sci_notation_tex(Q_charge)}\ [C/m^2]}}$"
                )

            Q_discharge_indic = (
                rf"$\mathbf{{Q_{{discharge}} = {sci_notation_tex(Q_discharge)}\ [C/m^2]}}$"
            )

            Q_res_indic = (
                rf"$\mathbf{{Q_{{res}} = {sci_notation_tex(Q_res)}\ [C/m^2]}}$"
            )

            ErrorRate_indic = (
                rf"$\mathbf{{Q_{{res}} / Q_{{charge}} = {ErrorRatio:.3f}\%}}$"
            )

            Area_indic = (
                rf"$\mathbf{{Area_{{cap}} = {cap_area * 1e12:.2e}}}\ [\mu m^2]$"
            )
                
            MaxFreq_indic = rf"$\mathbf{{Max\ FFT\ freq.\ = {sci_notation_tex(DefInput, 2)}}}$"
            FreqIncrement_indic = rf"$\mathbf{{FFT\ freq.\ inc.\ = {sci_notation_tex(1/T, 2)}}}$"
            CutOffFreq_indic = rf"$\mathbf{{CutOff\ freq.\ = {sci_notation_tex(cutoff_freq, 2)}}}$"
    
            fig, ax = plt.subplots(1, 2, figsize=(16, 5))  
            fig.subplots_adjust(top=0.85, wspace=0.3, bottom=0.2)

            # --- Subplot 1: Current plot ---
            ax[0].plot(time, ch1, 'k-', label='Raw')
            ax[0].plot(time, np.ones_like(time) * OSCErrorMean, 'b--', label='Offset')
            ax[0].plot(time, ch1_return, 'r-', linewidth=2, label='Denoised')

            info_text1 = (
                f"FFT Info.\n"
                f"{MaxFreq_indic}\n"
                f"{FreqIncrement_indic}\n"
                f"{CutOffFreq_indic}"
            )

            ax[0].text(
                0.5*time[-1], max(ch1),
                info_text1,
                fontsize=11,
                va="top",
                bbox=bbox_style,
                linespacing=1.6
            )


            ax[0].set_xlabel('Time [s]', fontsize=18, fontweight='bold')
            ax[0].set_ylabel('Voltage [V]', fontsize=18, fontweight='bold')
            ax[0].set_title('OSC measurment profile, noise removed', fontsize=15)
            ax[0].grid(True)
            ax[0].legend(loc='lower left', bbox_to_anchor=(0.05, 0.05))
            # ax[0].legend()

            # --- Subplot 2: Charge density plot ---
            ax[1].plot(time[:-1], net_accum_charge_density, 'k-')
            ax[1].plot(time[Q_charge_position], net_accum_charge_density[Q_charge_position], 'bo')
            ax[1].plot(time[Qres_position], net_accum_charge_density[Qres_position], 'bo')

            info_text2 = (
                f"{Area_indic}\n"
                f"{Q_charge_indic}\n"
                f"{Q_discharge_indic}\n"
                f"{Q_res_indic}\n"
                f"{ErrorRate_indic}"
            )

            ax[1].text(
                0.5*time[-1], Q_charge,
                info_text2,
                fontsize=11,
                va="top",
                bbox=bbox_style,
                linespacing=1.6
            )

            ax[1].set_xlabel('Time [s]', fontsize=18, fontweight='bold')
            ax[1].set_ylabel('ChargeDensity [C/m²]', fontsize=18, fontweight='bold')
            ax[1].set_title('Accumulation charge density', fontsize=15)
            ax[1].grid(True)

            fig.suptitle(dat_file, color='red', fontsize=20)
                    
            if abs(ErrorRatio) > para3:
                state = 10
            else:
                state = -1  
                ValueUpdate = 0
            
        if state == 2:

            Q_charge_indic = (
                    rf"$\mathbf{{Q_{{charge}} = {sci_notation_tex(NewQ_charge)}\ [C/m^2]}}$"
                )

            Q_discharge_indic = (
                rf"$\mathbf{{Q_{{discharge}} = {sci_notation_tex(NewQ_discharge)}\ [C/m^2]}}$"
            )

            Q_res_indic = (
                rf"$\mathbf{{Q_{{res}} = {sci_notation_tex(NewQ_res)}\ [C/m^2]}}$"
            )

            ErrorRate_indic = (
                rf"$\mathbf{{Q_{{res}} / Q_{{charge}} = {NewErrorRatio:.3f}\%}}$"
            )

            Area_indic = (
                rf"$\mathbf{{Area_{{cap}} = {cap_area * 1e12:.2e}}}\ [\mu m^2]$"
            )
                
            MaxFreq_indic = rf"$\mathbf{{Max\ FFT\ freq.\ = {sci_notation_tex(DefInput, 2)}}}$"
            FreqIncrement_indic = rf"$\mathbf{{FFT\ freq.\ inc.\ = {sci_notation_tex(1/T, 2)}}}$"
            CutOffFreq_indic = rf"$\mathbf{{CutOff\ freq.\ = {sci_notation_tex(cutoff_freq, 2)}}}$"


            fig, ax = plt.subplots(1, 2, figsize=(14, 6))
            fig.subplots_adjust(top=0.85)
            
            ax[0].plot(time, ch1, 'k-', label='Raw')
            ax[0].plot(time, np.ones_like(time) * OSCErrorMean, 'b--', label='Offset')
            ax[0].plot(time, ch1_return, 'r-', linewidth=2, label='Denoised')


            info_text1 = (
                f"FFT Info.\n"
                f"{MaxFreq_indic}\n"
                f"{FreqIncrement_indic}\n"
                f"{CutOffFreq_indic}"
            )


            ax[0].text(
                0.6*time[-1], max(ch1),
                info_text1,
                fontsize=11,
                va="top",
                bbox=bbox_style,
                linespacing=1.6
            )


            ax[0].set_xlabel('Time [s]', fontsize=18, fontweight='bold')
            ax[0].set_ylabel('Current × 50Ω [V]', fontsize=18, fontweight='bold')
            ax[0].set_title('OSC measurment profile, noise removed', fontsize=15)
            ax[0].grid(True)

            ax[1].plot(time[:-1], Qres_adjust, 'k-')
            ax[1].plot(time[Q_charge_position], Qres_adjust[Q_charge_position], 'bo')
            ax[1].plot(time[NewQres_position], Qres_adjust[NewQres_position], 'bo')

            info_text2 = (
                f"{Area_indic}\n"
                f"{Q_charge_indic}\n"
                f"{Q_discharge_indic}\n"
                f"{Q_res_indic}\n"
                f"{ErrorRate_indic}"
            )

            ax[1].text(
                0.5*time[-1], Q_charge,
                info_text2,
                fontsize=11,
                va="top",
                bbox=bbox_style,
                linespacing=1.6
            )

            ax[1].set_xlabel('Time [s]', fontsize=18, fontweight='bold')
            ax[1].set_ylabel('ChargeDensity [C/m²]', fontsize=18, fontweight='bold')
            ax[1].set_title('Accumulation charge density', fontsize=15)
            ax[1].grid(True)
            
            root = tk.Tk()
            root.withdraw()
            response = messagebox.askquestion("Dialog 1", "Proceed?\n(Yes, No)")
            
            if response == 'no':
                state = 1
            else:
                state = -1
                ValueUpdate = 1
                
        if state == 10:
            response = messagebox.askquestion(
                "Dialog 1",
                "A large Q_res has been detected.\nWould you like to apply Q_res correction?\n(Yes: Re-plot with correction, No: Keep original)"
            )
            if response == 'yes':
                state = 2
            else:
                state = -1
                ValueUpdate = 0

    for i in plt.get_fignums():
        fig = plt.figure(i)
        fig.savefig(f"Figure{i}.png", dpi=400, bbox_inches='tight')
    plt.show()                

    pic_name = os.path.splitext(dat_file)[0] + '.jpg'

    full_path = os.path.join(folder_path, pic_name)

    fig.savefig(full_path, format='jpg', dpi=400, bbox_inches='tight')
    
if __name__ == "__main__":
    main_load_and_config()