"""Física del flash de cámara: circuito RC en serie con interruptor de dos posiciones."""
from dataclasses import dataclass
import numpy as np
from metodos import rk4

EPS0 = 8.854e-12   # F/m

@dataclass
class Flash:
    V: float = 400.0      # V, fuente
    C: float = 220e-6     # F
    Rc: float = 10e3      # ohm, resistencia de carga
    Rd: float = 10.0      # ohm, lámpara ideal (resistencia constante)
    Rs: float = 0.5       # ohm, resistencia en serie del circuito de descarga (modelo real)
    K0: float = 60.0      # ohm·A^(1/2), impedancia de la lámpara real: V_lámpara = K0·sqrt(I)

    @property
    def tau_c(self): return self.Rc * self.C
    @property
    def tau_d(self): return self.Rd * self.C
    @property
    def Q(self): return self.C * self.V
    @property
    def U(self): return 0.5 * self.C * self.V ** 2

    # ---- carga (solución exacta) ----
    def vc_carga(self, t): return self.V * (1 - np.exp(-np.asarray(t) / self.tau_c))
    def i_carga(self, t): return self.V / self.Rc * np.exp(-np.asarray(t) / self.tau_c)
    def t_fraccion(self, f): return -self.tau_c * np.log(1 - f)

    # ---- descarga ideal (Rd constante) ----
    def vc_ideal(self, v0, t): return v0 * np.exp(-np.asarray(t) / self.tau_d)
    def i_ideal(self, v0, t): return v0 / self.Rd * np.exp(-np.asarray(t) / self.tau_d)

    # ---- descarga con lámpara real: Rs·I + K0·sqrt(I) = V_C ----
    def i_real(self, v):
        v = np.maximum(v, 0.0)
        x = (-self.K0 + np.sqrt(self.K0 ** 2 + 4 * self.Rs * v)) / (2 * self.Rs)
        return x ** 2
    def f_real(self, t, v): return -self.i_real(v) / self.C
    def descarga_real(self, v0, t): return rk4(self.f_real, v0, t)

def energia_lampara(P, t):
    return float(np.sum(0.5 * (P[1:] + P[:-1]) * np.diff(t)))   # regla del trapecio

def ancho_medio(P, t):
    """Duración a media altura (FWHM) de la potencia de la lámpara."""
    m = P >= 0.5 * P.max(); idx = np.where(m)[0]
    return float(t[idx[-1]] - t[idx[0]])

def exposicion_stops(f): return float(2 * np.log2(f))   # U ∝ f²
def area_placas(C, d, er=1.0): return C * d / (EPS0 * er)   # C = ε0·εr·A/d
