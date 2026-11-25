"""Euler y Runge-Kutta 4 programados a mano (sin SciPy). y puede ser escalar."""
import numpy as np

def euler(f, y0, t):
    y = np.empty(len(t)); y[0] = y0
    for n in range(len(t) - 1):
        h = t[n + 1] - t[n]
        y[n + 1] = y[n] + h * f(t[n], y[n])
    return y

def rk4(f, y0, t):
    y = np.empty(len(t)); y[0] = y0
    for n in range(len(t) - 1):
        h = t[n + 1] - t[n]
        k1 = f(t[n], y[n]); k2 = f(t[n] + h / 2, y[n] + h * k1 / 2)
        k3 = f(t[n] + h / 2, y[n] + h * k2 / 2); k4 = f(t[n] + h, y[n] + h * k3)
        y[n + 1] = y[n] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y

def orden(errs):
    """Orden observado p = log2(E(h)/E(h/2)) para errores consecutivos."""
    return [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]
