import tkinter as tk
from tkinter import ttk
import json
import os
import inspect

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.stack()[0].filename))
PARAM_FILE = os.path.join(CURRENT_DIR, "NCDialogPara.json")

def load_params():
    try:
        with open(PARAM_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "BiasSign": "Positive biased",
            "CapacitorArea": "10000",
            "QresTolerance": "4",
            "PulseWidth": "500",
            "PulseRiseFall": "20"
        }

def save_params(params):
    with open(PARAM_FILE, "w") as f:
        json.dump(params, f, indent=4)

def create_dialog():
    params = load_params()
    result = {}

    root = tk.Tk()
    root.title("Parameter Settings")
    root.geometry("520x450")

    tk.Label(root, text="Condition Settings", font=("Arial", 14, "bold"), fg="red").pack(pady=10)

    # Bias Sign
    tk.Label(root, text="Pulse Bias Sign", font=("Arial", 12, "bold")).pack()
    bias_var = tk.StringVar(value=params["BiasSign"])
    ttk.Combobox(root, textvariable=bias_var, values=["Positive biased", "Negative biased"]).pack()

    # Capacitor Area
    tk.Label(root, text="Capacitor Area [um²]", font=("Arial", 12, "bold")).pack()
    cap_area_var = tk.StringVar(value=params["CapacitorArea"])
    tk.Entry(root, textvariable=cap_area_var).pack()

    # Qres Tolerance
    tk.Label(root, text="Qres Tolerance [%]", font=("Arial", 12, "bold")).pack()
    qres_tol_var = tk.StringVar(value=params["QresTolerance"])
    tk.Entry(root, textvariable=qres_tol_var).pack()

    # Pulse Width
    tk.Label(root, text="Pulse Width [ns]", font=("Arial", 12, "bold")).pack()
    width_var = tk.StringVar(value=params["PulseWidth"])
    tk.Entry(root, textvariable=width_var).pack()

    # Rise/Fall
    tk.Label(root, text="Pulse Rise/Fall [ns]", font=("Arial", 12, "bold")).pack()
    risefall_var = tk.StringVar(value=params["PulseRiseFall"])
    tk.Entry(root, textvariable=risefall_var).pack()

    def on_save():
        result.update({
            "BiasSign": bias_var.get(),
            "CapacitorArea": cap_area_var.get(),
            "QresTolerance": qres_tol_var.get(),
            "PulseWidth": width_var.get(),
            "PulseRiseFall": risefall_var.get()
        })
        save_params(result)
        root.quit()
        root.destroy()

    tk.Button(root, text="Save & Continue", font=("Arial", 13), fg="blue", command=on_save).pack(pady=15)
    root.mainloop()

    return result, PARAM_FILE

if __name__ == "__main__":
    params, file_path = create_dialog()