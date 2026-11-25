import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from PIL import Image
from modelo import *
from metodos import euler, rk4, orden

st.set_page_config(page_title="Flash RC · Física 2", layout="wide")
st.title("El flash que tarda segundos en cargarse y milisegundos en disparar")
st.caption("Circuito RC en serie con interruptor de dos posiciones. Mueve los parámetros: todo se recalcula.")
st.latex(r"R_c\frac{dq}{dt}+\frac{q}{C}=V\;\Rightarrow\;V_C=V\left(1-e^{-t/\tau_c}\right),\qquad \tau_c=R_cC,\;\;\tau_d=R_dC")

sb = st.sidebar; sb.header("Parámetros")
V = sb.slider("Voltaje V (V)", 50, 450, 400, 10)
C = sb.slider("Capacitancia C (µF)", 10, 1000, 220, 10)
Rc = sb.slider("Resistencia de carga Rc (kΩ)", 1.0, 100.0, 10.0, 1.0)
Rd = sb.slider("Lámpara ideal Rd (Ω)", 1.0, 50.0, 10.0, 1.0)
sb.subheader("Lámpara real")
Rs = sb.slider("Resistencia en serie Rs (Ω)", 0.1, 5.0, 0.5, 0.1)
K0 = sb.slider("Impedancia K0 (Ω·A^½)", 20.0, 120.0, 60.0, 5.0)
fl = sb.slider("Indicador «listo» (%)", 50, 99, 95) / 100
F = Flash(V, C * 1e-6, Rc * 1e3, Rd, Rs, K0)
v0 = fl * V

m = st.columns(6)
m[0].metric("Carga máx. CV", f"{F.Q*1e3:.1f} mC"); m[1].metric("Energía ½CV²", f"{F.U:.2f} J")
m[2].metric("τ carga", f"{F.tau_c:.3g} s"); m[3].metric("τ descarga", f"{F.tau_d*1e3:.3g} ms")
m[4].metric(f"Listo al {fl*100:.0f} %", f"{F.t_fraccion(fl):.2f} s"); m[5].metric("τc / τd", f"{F.tau_c/F.tau_d:.0f}")

T1, T2, T3, T4, T5 = st.tabs(["Carga y descarga", "Lámpara real", "Disparo y foto", "Métodos numéricos", "Física del condensador"])

with T1:
    u = st.slider("Instante en unidades de τ (enlaza las cuatro gráficas)", 0.0, 5.0, 1.0, 0.05)
    tc, td = np.linspace(0, 5 * F.tau_c, 400), np.linspace(0, 5 * F.tau_d, 400)
    fig, ax = plt.subplots(2, 2, figsize=(11, 6))
    ax[0, 0].plot(tc, F.vc_carga(tc), c="#d98a00"); ax[0, 0].axhline(v0, ls="--", c="g"); ax[0, 0].set(title="Carga: V_C (V)", xlabel="t (s)")
    ax[1, 0].plot(tc, F.i_carga(tc) * 1e3, c="#1f77b4"); ax[1, 0].set(title="Carga: corriente (mA)", xlabel="t (s)")
    ax[0, 1].plot(td * 1e3, F.vc_ideal(v0, td), c="#d98a00"); ax[0, 1].set(title="Descarga: V_C (V)", xlabel="t (ms)")
    ax[1, 1].plot(td * 1e3, F.i_ideal(v0, td), c="#d62728"); ax[1, 1].set(title="Descarga: corriente (A)", xlabel="t (ms)")
    for a, x in ((ax[0, 0], u * F.tau_c), (ax[1, 0], u * F.tau_c), (ax[0, 1], u * F.tau_d * 1e3), (ax[1, 1], u * F.tau_d * 1e3)):
        a.axvline(x, c="k", lw=1); a.grid(alpha=.3)
    plt.tight_layout(); st.pyplot(fig)
    st.progress(float(F.vc_carga(u * F.tau_c) / V), text=f"Condensador al {F.vc_carga(u*F.tau_c)/V*100:.1f} % a t = {u*F.tau_c:.3g} s")
    st.write(f"**t = {u:.2f} τ** → carga: V_C = {F.vc_carga(u*F.tau_c):.1f} V, i = {F.i_carga(u*F.tau_c)*1e3:.2f} mA · "
             f"descarga: V_C = {F.vc_ideal(v0, u*F.tau_d):.1f} V, i = {F.i_ideal(v0, u*F.tau_d):.1f} A")

with T2:
    st.latex(r"V_C = R_s I + K_0\sqrt{I}\;\Rightarrow\; C\frac{dV_C}{dt}=-I(V_C)\quad\text{(no lineal: se resuelve con RK4)}")
    t = np.linspace(0, 30e-3, 6001); vi, vr = F.vc_ideal(v0, t), F.descarga_real(v0, t)
    Ii, Ir = vi / Rd, F.i_real(vr); Pi, Pr = vi ** 2 / Rd, F.K0 * Ir ** 1.5
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for a, (yi, yr, ti) in zip(ax, ((vi, vr, "V_C (V)"), (Ii, Ir, "Corriente (A)"), (Pi / 1e3, Pr / 1e3, "Potencia en la lámpara (kW)"))):
        a.plot(t * 1e3, yi, label="Rd constante"); a.plot(t * 1e3, yr, label="Lámpara real"); a.set(title=ti, xlabel="t (ms)"); a.grid(alpha=.3)
    ax[0].legend(); plt.tight_layout(); st.pyplot(fig)
    d = pd.DataFrame({"Rd constante": [Ii.max(), Pi.max() / 1e3, energia_lampara(Pi, t), ancho_medio(Pi, t) * 1e3],
                      "Lámpara real": [Ir.max(), Pr.max() / 1e3, energia_lampara(Pr, t), ancho_medio(Pr, t) * 1e3]},
                     index=["Corriente pico (A)", "Potencia pico (kW)", "Energía en la lámpara, 30 ms (J)", "Duración a media altura (ms)"])
    st.dataframe(d.round(3)); st.caption("Con la lámpara real la corriente cae más rápido que e^(−t/τ) al inicio pero deja una cola larga: no hay τ único.")

with T3:
    fd = st.slider("Disparar cuando el condensador está al (%)", 20, 100, int(fl * 100)) / 100
    Ud = fd ** 2 * F.U; st_ = exposicion_stops(fd)
    a, b, c = st.columns(3); a.metric("Energía del destello", f"{Ud:.2f} J"); b.metric("Exposición", f"{st_:+.2f} pasos"); c.metric("Tiempo de carga", f"{F.t_fraccion(min(fd,.999)):.2f} s")
    st.latex(r"U=\tfrac12 C V_C^2=f^2\,U_{\max}\;\Rightarrow\;\text{exposición}=2\log_2 f\ \text{(pasos)}")
    amb = st.slider("Luz ambiente (%)", 3, 60, 18) / 100
    img = st.camera_input("Toma una foto con tu cámara") or st.file_uploader("…o sube una imagen", type=["png", "jpg", "jpeg"])
    if img is not None:
        im = Image.open(img).convert("RGB"); A = np.asarray(im, float) / 255
        out = np.clip(A * amb + A * min(1.0, 1.6 * Ud / 17.6), 0, 1)
        c1, c2 = st.columns(2); c1.image(np.clip(A * amb, 0, 1), caption="Sin flash"); c2.image(out, caption=f"Con flash ({Ud:.1f} J)")
    st.caption("Modelo simple: brillo añadido ∝ energía del destello (referencia 17,6 J = 1,6 de ganancia).")

with T4:
    esc = st.radio("Problema", ["Carga ideal (hay solución exacta)", "Descarga con lámpara real (referencia: RK4 muy fino)"], horizontal=True)
    if esc.startswith("Carga"):
        f, y0, tf = (lambda t, v: (V - v) / F.tau_c), 0.0, 4.0; ref = float(F.vc_carga(tf)); hs = [1, .5, .25, .125, .0625]
    else:
        f, y0, tf = F.f_real, v0, 2e-3; ref = rk4(f, y0, np.linspace(0, tf, 20001))[-1]; hs = [tf / n for n in (5, 10, 20, 40, 80)]
    rows = {}
    for nm, mt in (("Euler", euler), ("RK4", rk4)):
        rows[nm] = [abs(mt(f, y0, np.linspace(0, tf, int(round(tf / h)) + 1))[-1] - ref) for h in hs]
    p = {k: [np.nan] + orden(v) for k, v in rows.items()}
    st.dataframe(pd.DataFrame({"h": hs, "Error Euler": rows["Euler"], "p Euler": p["Euler"], "Error RK4": rows["RK4"], "p RK4": p["RK4"]}).style.format({"h": "{:.4g}", "Error Euler": "{:.2e}", "Error RK4": "{:.2e}", "p Euler": "{:.2f}", "p RK4": "{:.2f}"}))
    fig, ax = plt.subplots(figsize=(5.5, 3.4)); ax.loglog(hs, rows["Euler"], "s-", label="Euler"); ax.loglog(hs, rows["RK4"], "o-", label="RK4")
    ax.set(xlabel="paso h", ylabel="error absoluto"); ax.legend(); ax.grid(alpha=.3, which="both"); st.pyplot(fig)
    st.subheader("Estabilidad de Euler en la carga")
    h = st.slider("Paso h (s)", 0.1, 8.0, 5.0, 0.1); ts = np.arange(0, 12 * h + 1e-9, h) if h > 1 else np.arange(0, 12.01, h)
    ye = euler(lambda t, v: (V - v) / F.tau_c, 0.0, ts); tt = np.linspace(0, ts[-1], 300)
    fig, ax = plt.subplots(figsize=(7, 3)); ax.plot(tt, F.vc_carga(tt), label="exacta"); ax.plot(ts, ye, "o-", label="Euler"); ax.set_ylim(-2 * V, 2 * V); ax.legend(); ax.grid(alpha=.3); st.pyplot(fig)
    st.info(f"Criterio: estable si h < 2τc = {2*F.tau_c:.2f} s. Con h = {h:.1f} s el método {'es estable' if h < 2*F.tau_c else 'diverge'}.")

with T5:
    st.latex(r"C=\frac{\varepsilon_0\varepsilon_r A}{d},\qquad E=\frac{V}{d},\qquad u=\tfrac12\varepsilon_0\varepsilon_r E^2")
    d_um = st.slider("Separación entre placas d (µm)", 1.0, 100.0, 10.0); er = st.slider("Constante dieléctrica εr", 1.0, 100.0, 1.0)
    A = area_placas(F.C, d_um * 1e-6, er); Ef = V / (d_um * 1e-6)
    a, b, c = st.columns(3); a.metric("Área de placas necesaria", f"{A:.3g} m²"); b.metric("Campo entre placas", f"{Ef/1e6:.1f} MV/m")
    c.metric("Densidad de energía", f"{0.5*EPS0*er*Ef**2/1e6:.2f} MJ/m³")
    st.caption("Un condensador tan grande no cabe en un flash con placas planas: los electrolíticos enrollan láminas y usan un óxido muy delgado y de εr alto. El campo de ruptura del aire (≈ 3 MV/m) limita la separación mínima sin dieléctrico.")
    st.write("**Balance de energía.** La fuente entrega CV²; la mitad queda en el condensador y la otra mitad se disipa en Rc, sin importar su valor.")
    a, b, c = st.columns(3); a.metric("Entregada por la fuente", f"{F.C*V**2:.1f} J"); b.metric("Almacenada", f"{F.U:.1f} J"); c.metric("Disipada en Rc", f"{F.U:.1f} J")
