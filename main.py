from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()

class SolarInput(BaseModel):
    G: float  # Solar Irradiance (W/m^2)
    T_a: float  # Ambient Temperature (Degrees Celsius)

def newton_raphson_solve(f, df, I0, tol):
    iter_count = 0
    I1 = I0 - (f(I0) / df(I0))
    while abs(I1 - I0) > tol:
        I0 = I1
        I1 = I0 - (f(I0) / df(I0))
        iter_count += 1
    return I1, iter_count

@app.post("/calculate_iv_pv")
def calculate_iv_pv(data: SolarInput):
    G = data.G
    T_a = data.T_a
    
    NMOT = 42
    T = T_a + (NMOT - 20) * G / 800 + 273.15
    T_stc = 25 + 273.15
    k = 1.381e-23
    q = 1.602e-19
    T_ref = 44.6 + 273.15
    VT = (k * T_ref) / q
    tol = 1e-4
    Isc = 14.01
    Ns = 72
    
    C = [10.025, 8.6836e-9, 1.1214, 0.2061, 1767.1]
    a1, a2, a3, a4 = 0.0103, 0.1514, -3.47, -22.5634
    Ipv_ref, Is_ref, n, Rs_ref, Rsh_ref = C[0], C[1], 1.07, C[3], C[4]
    G_ref = 714
    
    Ipv_stc = (1000 / G_ref) * (Ipv_ref + a1 * (T_stc - T_ref))
    Ipv_meas = (G / G_ref) * (Ipv_ref + a1 * (T - T_ref))
    
    Is_stc = Is_ref * ((T_stc / T_ref) ** (a2 / n)) * np.exp((1.12 * ((T_stc / T_ref) - 1)) / (n * VT))
    Is_meas = Is_ref * ((T / T_ref) ** (a2 / C[2])) * np.exp((1.12 * ((T / T_ref) - 1)) / (n * VT))
    
    Rs_stc = Rs_ref * ((T_stc / T_ref) ** a3)
    Rs_meas = Rs_ref * ((T / T_ref) ** a3)
    
    Rsh_stc = Rsh_ref * ((T_stc / T_ref) ** a4)
    Rsh_meas = Rsh_ref * ((T / T_ref) ** a4)
    
    voltage, current1, current2 = [], [], []
    C[0], C[1], C[3], C[4] = Ipv_meas, Is_meas, Rs_meas, Rsh_meas
    I0 = 0.5 * Isc
    
    for V in range(51):
        f = lambda I: I - C[0] + (C[1] * (np.exp((V + (I * C[3])) / (Ns * C[2] * VT)) - 1)) + ((V + (I * C[3])) / C[4])
        df = lambda I: 1 + ((C[1] * C[3]) / (C[2] * Ns * VT)) * np.exp((V + (I * C[3])) / (C[2] * Ns * VT)) + (C[3] / C[4])
        I1, _ = newton_raphson_solve(f, df, I0, tol)
        voltage.append(V)
        current1.append(I1)
    
    C[0], C[1], C[3], C[4] = Ipv_stc, Is_stc, Rs_stc, Rsh_stc
    
    for V in range(51):
        f = lambda I: I - C[0] + (C[1] * (np.exp((V + (I * C[3])) / (Ns * n * VT)) - 1)) + ((V + (I * C[3])) / C[4])
        df = lambda I: 1 + ((C[1] * C[3]) / (n * Ns * VT)) * np.exp((V + (I * C[3])) / (n * Ns * VT)) + (C[3] / C[4])
        I1, _ = newton_raphson_solve(f, df, I0, tol)
        current2.append(I1)
    
    power_meas = np.array(voltage) * np.array(current1)
    power_stc = np.array(voltage) * np.array(current2)
    
    return {
        "voltage": voltage,
        "current_measured": current1,
        "current_stc": current2,
        "power_measured": power_meas.tolist(),
        "power_stc": power_stc.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
