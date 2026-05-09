
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

R        = 50.0
R_root   = 0.2 * R
B        = 3
U0       = 10.0
rho      = 1.225
TSRs     = [6, 8, 10]
MAX_ITER = 30
TOL      = 1e-3

def get_twist_baseline(r): return 14.0 * (1.0 - r / R)
def get_chord_baseline(r): return 3.0 * (1.0 - r / R) + 1.0

polar_filename = 'polar DU95W180 (3).xlsx'
if not os.path.exists(polar_filename):
    polar_filename = r'C:\Users\SID-DRW\OneDrive\Escritorio\Master courses\wake\polar DU95W180 (3).xlsx'

try:
    df_polar  = pd.read_excel(polar_filename, skiprows=3).dropna()
    cl_interp = interp1d(df_polar.iloc[:, 0], df_polar.iloc[:, 1], bounds_error=False, fill_value="extrapolate")
    cd_interp = interp1d(df_polar.iloc[:, 0], df_polar.iloc[:, 2], bounds_error=False, fill_value="extrapolate")
    alphas_test  = np.linspace(-5, 15, 200)
    glide_ratio  = cl_interp(alphas_test) / (cd_interp(alphas_test) + 1e-6)
    alpha_design = alphas_test[np.argmax(glide_ratio)]
    cl_design    = float(cl_interp(alpha_design))
except Exception as e:
    print(f"Failed to load polar: {e}"); exit()

def ainduction(CT):
    CT = np.atleast_1d(CT)
    a  = np.zeros_like(CT)
    CT1, CT2 = 1.816, 2 * np.sqrt(1.816) - 1.816
    hi = CT >= CT2
    a[ hi] = 1 + (CT[hi] - CT1) / (4 * (np.sqrt(CT1) - 1))
    a[~hi] = 0.5 - 0.5 * np.sqrt(np.clip(1 - CT[~hi], 0, 1))
    return a

def CTfunction(a, glauert=True):
    a  = np.atleast_1d(a)
    CT = 4 * a * (1 - a)
    if glauert:
        a1   = 1 - np.sqrt(1.816) / 2
        mask = a > a1
        CT[mask] = 1.816 - 4 * (np.sqrt(1.816) - 1) * (1 - a[mask])
    return CT

def get_geometry(r, tsr, use_optimized):
    if use_optimized:
        phi_opt = np.arctan(0.75 / (tsr * r / R))
        twist   = np.degrees(phi_opt) - alpha_design
        chord   = (8 * np.pi * r / (B * cl_design)) * (np.sin(phi_opt)**2 / np.cos(phi_opt)) * (0.25 / 0.75)
    else:
        twist, chord = get_twist_baseline(r), get_chord_baseline(r)
    return twist, chord

def prandtl_factor(r, phi):
    def _F(exp_arg): return (2 / np.pi) * np.arccos(np.exp(-np.clip(exp_arg, 0, 50)))
    F_tip  = _F((B / 2) * (R - r)      / (r * np.sin(phi) + 1e-6))
    F_root = _F((B / 2) * (r - R_root) / (r * np.sin(phi) + 1e-6))
    return np.maximum(F_tip * F_root, 1e-4), F_tip, F_root

def solve_bem(tsr, n_elem=50, pitch_deg=-2.0, use_optimized=False,
              spacing='constant', track_conv=False):
    Omega = tsr * U0 / R
    if spacing == 'cosine':
        beta    = np.linspace(0, np.pi, n_elem + 1)
        r_edges = R_root + (R - R_root) * 0.5 * (1 - np.cos(beta))
    else:
        r_edges = np.linspace(R_root, R, n_elem + 1)
    r_mid = (r_edges[:-1] + r_edges[1:]) / 2.0
    dr    = np.diff(r_edges)

    a_vec  = np.full(n_elem, 0.3)
    ap_vec = np.full(n_elem, 0.01)

    thrust_history = []
    for _ in range(MAX_ITER):
        it_thrust = 0.0
        for i, r in enumerate(r_mid):
            twist, chord = get_geometry(r, tsr, use_optimized)
            phi      = np.arctan2((1 - a_vec[i]) * U0, (1 + ap_vec[i]) * Omega * r)
            alpha    = np.degrees(phi) - (twist + pitch_deg)
            Cl, Cd   = float(cl_interp(alpha)), float(cd_interp(alpha))
            Cn       = Cl * np.cos(phi) + Cd * np.sin(phi)
            Ct_blade = Cl * np.sin(phi) - Cd * np.cos(phi)
            F, _, _  = prandtl_factor(r, phi)
            CT_loc   = (B * chord * Cn * (1 - a_vec[i])**2) / (2 * np.pi * r * np.sin(phi)**2 + 1e-8)
            a_new    = float(ainduction(np.array([CT_loc / F]))[0])
            ap_new   = (B * chord * Ct_blade) / (
                8 * np.pi * r * F * np.sin(phi) * np.cos(phi) - B * chord * Ct_blade + 1e-8)
            a_vec[i]  = 0.1 * a_new  + 0.9 * a_vec[i]
            ap_vec[i] = 0.1 * ap_new + 0.9 * ap_vec[i]
            Vrel2     = (U0 * (1 - a_vec[i]))**2 + (Omega * r * (1 + ap_vec[i]))**2
            it_thrust += B * 0.5 * rho * Vrel2 * chord * Cn * dr[i]
        thrust_history.append(it_thrust)
        if len(thrust_history) > 1 and abs(thrust_history[-1] - thrust_history[-2]) < TOL:
            break

    if track_conv:
        return thrust_history

    keys = ['r_R', 'a', 'ap', 'alpha', 'phi', 'pN', 'pT',
            'chord', 'twist', 'Cl', 'Cd', 'Pstag_drop', 'F', 'F_tip', 'F_root']
    data = {k: [] for k in keys}
    total_thrust = total_torque = 0.0

    for i, r in enumerate(r_mid):
        twist, chord = get_geometry(r, tsr, use_optimized)
        phi      = np.arctan2((1 - a_vec[i]) * U0, (1 + ap_vec[i]) * Omega * r)
        alpha    = np.degrees(phi) - (twist + pitch_deg)
        Cl, Cd   = float(cl_interp(alpha)), float(cd_interp(alpha))
        Cn       = Cl * np.cos(phi) + Cd * np.sin(phi)
        Ct_blade = Cl * np.sin(phi) - Cd * np.cos(phi)
        F, F_tip, F_root = prandtl_factor(r, phi)
        Vrel2 = (U0 * (1 - a_vec[i]))**2 + (Omega * r * (1 + ap_vec[i]))**2
        pN    = 0.5 * rho * Vrel2 * chord * Cn
        pT    = 0.5 * rho * Vrel2 * chord * Ct_blade
        total_thrust += B * pN * dr[i]
        total_torque += B * pT * r * dr[i]
        for k, v in zip(keys, [
            r / R, a_vec[i], ap_vec[i], alpha, np.degrees(phi),
            pN, pT, chord, twist, Cl, Cd,
            0.5 * rho * U0**2 * CTfunction(a_vec[i])[0],
            F, F_tip, F_root
        ]):
            data[k].append(v)

    Area = np.pi * R**2
    Cp   = (total_torque * Omega) / (0.5 * rho * Area * U0**3)
    Ct   = total_thrust           / (0.5 * rho * Area * U0**2)
    return Cp, Ct, total_thrust, total_torque, data

def solve_bem_spanwise(tsr, twist_offsets, n_elem=50, relax=0.1):
    Omega   = tsr * U0 / R
    r_edges = np.linspace(R_root, R, n_elem + 1)
    r_mid   = (r_edges[:-1] + r_edges[1:]) / 2.0
    dr      = np.diff(r_edges)
    a_vec   = np.full(n_elem, 1/3)
    ap_vec  = np.full(n_elem, 0.01)

    for _ in range(MAX_ITER):
        a_old = a_vec.copy()
        for i, r in enumerate(r_mid):
            twist, chord = get_geometry(r, tsr, use_optimized=True)
            phi      = np.arctan2((1 - a_vec[i]) * U0, (1 + ap_vec[i]) * Omega * r)
            alpha    = np.degrees(phi) - (twist + twist_offsets[i])
            Cl, Cd   = float(cl_interp(alpha)), float(cd_interp(alpha))
            Cn       = Cl * np.cos(phi) + Cd * np.sin(phi)
            Ct_blade = Cl * np.sin(phi) - Cd * np.cos(phi)
            F, _, _  = prandtl_factor(r, phi)
            CT_loc   = (B * chord * Cn * (1 - a_vec[i])**2) / (2 * np.pi * r * np.sin(phi)**2 + 1e-8)
            a_new    = float(ainduction(np.array([CT_loc / F]))[0])
            ap_new   = (B * chord * Ct_blade) / (
                8 * np.pi * r * F * np.sin(phi) * np.cos(phi) - B * chord * Ct_blade + 1e-8)
            a_vec[i]  = relax * a_new  + (1 - relax) * a_vec[i]
            ap_vec[i] = relax * ap_new + (1 - relax) * ap_vec[i]
        if np.max(np.abs(a_vec - a_old)) < TOL:
            break

    total_thrust = total_torque = 0.0
    span_data = {'r_R': [], 'a': [], 'alpha': [], 'pN': [], 'pT': [], 'Cl': [], 'Cd': []}
    for i, r in enumerate(r_mid):
        twist, chord = get_geometry(r, tsr, use_optimized=True)
        phi      = np.arctan2((1 - a_vec[i]) * U0, (1 + ap_vec[i]) * Omega * r)
        alpha    = np.degrees(phi) - (twist + twist_offsets[i])
        Cl, Cd   = float(cl_interp(alpha)), float(cd_interp(alpha))
        Cn       = Cl * np.cos(phi) + Cd * np.sin(phi)
        Ct_blade = Cl * np.sin(phi) - Cd * np.cos(phi)
        Vrel2    = (U0 * (1 - a_vec[i]))**2 + (Omega * r * (1 + ap_vec[i]))**2
        pN       = 0.5 * rho * Vrel2 * chord * Cn
        pT       = 0.5 * rho * Vrel2 * chord * Ct_blade
        total_thrust += B * pN * dr[i]
        total_torque += B * pT * r * dr[i]
        for k, v in zip(span_data, [r/R, a_vec[i], alpha, pN, pT, Cl, Cd]):
            span_data[k].append(v)

    Area = np.pi * R**2
    Cp   = (total_torque * Omega) / (0.5 * rho * Area * U0**3)
    Ct   = total_thrust           / (0.5 * rho * Area * U0**2)
    return Cp, Ct, span_data

CT_TARGET = 0.75
N_OPT     = 50

print("Finding initial guess via collective pitch sweep...")
pitches_init = np.linspace(-5, 5, 101)
cts_init     = [solve_bem_spanwise(8, np.full(N_OPT, p))[1] for p in pitches_init]
best_p0      = pitches_init[np.argmin(np.abs(np.array(cts_init) - CT_TARGET))]
x0           = np.full(N_OPT, best_p0)
print(f"  Initial collective pitch: {best_p0:.2f} deg")

call_count = [0]

def objective(offsets):
    Cp, _, _ = solve_bem_spanwise(8, offsets, n_elem=N_OPT)
    return -Cp

def ct_constraint(offsets):
    _, Ct, _ = solve_bem_spanwise(8, offsets, n_elem=N_OPT)
    return Ct - CT_TARGET

def callback(xk):
    call_count[0] += 1
    if call_count[0] % 10 == 0:
        Cp, Ct, _ = solve_bem_spanwise(8, xk, n_elem=N_OPT)
        print(f"  Iter {call_count[0]:3d}:  Cp = {Cp:.4f},  Ct = {Ct:.4f}")

print("\nRunning spanwise pitch optimisation (SLSQP)...")
result = minimize(
    objective, x0,
    method='SLSQP',
    bounds=[(-10, 10)] * N_OPT,
    constraints={'type': 'eq', 'fun': ct_constraint},
    callback=callback,
    options={'maxiter': 500, 'ftol': 1e-7}
)
opt_offsets              = result.x
Cp_sw,  Ct_sw,  span_sw  = solve_bem_spanwise(8, opt_offsets,          n_elem=N_OPT)
Cp_ref, Ct_ref, span_ref = solve_bem_spanwise(8, np.full(N_OPT, best_p0), n_elem=N_OPT)

print(f"\nReference (collective {best_p0:.2f} deg) →  Cp = {Cp_ref:.4f},  Ct = {Ct_ref:.4f}")
print(f"Spanwise optimised               →  Cp = {Cp_sw:.4f},  Ct = {Ct_sw:.4f}")
print(f"Optimiser: {result.message}")

def twin_axis_plot(title, xlabel, ylabel_left, ylabel_right, left_series, right_series):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    for x, y, lbl, ls in left_series:
        ax1.plot(x, y, ls, label=lbl)
    ax1.set_xlabel(xlabel); ax1.set_ylabel(ylabel_left)
    ax1.set_xlim(0.2, 1.0); ax1.legend(loc='upper left'); ax1.grid(True)
    ax2 = ax1.twinx()
    for x, y, lbl, ls in right_series:
        ax2.plot(x, y, ls, label=lbl)
    ax2.set_ylabel(ylabel_right); ax2.legend(loc='upper right')
    plt.title(title); plt.tight_layout(); plt.show()

baseline_results = {}
summary = {'TSR': [], 'Thrust': [], 'Torque': [], 'Cp': [], 'Ct': []}

for tsr in TSRs:
    cp, ct, thr, torq, span = solve_bem(tsr)
    baseline_results[tsr] = span
    summary['TSR'].append(tsr);     summary['Thrust'].append(thr)
    summary['Torque'].append(torq); summary['Cp'].append(cp); summary['Ct'].append(ct)
    print(f"TSR {tsr}: Cp={cp:.4f}, Ct={ct:.4f}, Thrust={thr/1000:.2f} kN")

pitches = np.linspace(-2, 2, 41)
opt_cts = [solve_bem(8, pitch_deg=p, use_optimized=True)[1] for p in pitches]
best_p  = pitches[np.argmin(np.abs(np.array(opt_cts) - 0.75))]
final_cp, final_ct, _, _, opt_span = solve_bem(8, pitch_deg=best_p, use_optimized=True)


rR         = {t: baseline_results[t]['r_R'] for t in TSRs}
tsr_colors = {6: 'tab:blue', 8: 'tab:orange', 10: 'tab:green'}

plt.figure(figsize=(10, 5))
for t in TSRs:
    c = tsr_colors[t]
    plt.plot(rR[t], baseline_results[t]['alpha'], color=c, linestyle='-',  label=f'α (TSR {t})')
    plt.plot(rR[t], baseline_results[t]['phi'],   color=c, linestyle='--', label=f'φ (TSR {t})')
plt.xlabel('r/R'); plt.ylabel('Angle [deg]'); plt.xlim(0.2, 1.0)
plt.title('Spanwise AoA and Inflow Angle')
plt.legend(loc='upper right', ncol=2); plt.grid(True); plt.tight_layout(); plt.show()

twin_axis_plot(
    "Spanwise Axial and Azimuthal Inductions", "r/R", "Axial Induction (a)", "Azimuthal Induction (a')",
    [(rR[t], baseline_results[t]['a'],  f'a (TSR {t})',  '-') for t in TSRs],
    [(rR[t], baseline_results[t]['ap'], f"a' (TSR {t})", ':') for t in TSRs],
)
twin_axis_plot(
    "Spanwise Thrust and Azimuthal Loading", "r/R", "Normal Load pN [N/m]", "Azimuthal Load pT [N/m]",
    [(rR[t], baseline_results[t]['pN'], f'pN (TSR {t})', '-') for t in TSRs],
    [(rR[t], baseline_results[t]['pT'], f'pT (TSR {t})', ':') for t in TSRs],
)

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(summary['TSR'], summary['Thrust'], 'ro-', label='Thrust'); ax1.set_ylabel('Thrust [N]', color='r')
ax2 = ax1.twinx()
ax2.plot(summary['TSR'], summary['Torque'], 'bo-', label='Torque'); ax2.set_ylabel('Torque [Nm]', color='b')
ax1.set_xlabel('TSR'); plt.title("Total Thrust and Torque vs TSR"); ax1.grid(True); plt.tight_layout(); plt.show()

r_idx  = int(0.7 * len(opt_span['r_R']))
P_inf  = 101325
P_tot  = P_inf + 0.5 * rho * U0**2
P_drop = opt_span['Pstag_drop'][r_idx]
plt.figure(figsize=(8, 5))
plt.plot([-2, 0, 0, 2], [P_tot, P_tot, P_tot - P_drop, P_tot - P_drop], 'r-o', linewidth=2)
plt.axvline(0, color='k', linestyle='--', alpha=0.5)
plt.title(r'Stagnation Pressure (r/R=0.7, Optimised $C_T=0.75$)')
plt.ylabel('Stagnation Pressure [Pa]')
plt.xticks([-2, 0, 2], [r'$\infty$ Up', 'Rotor', r'$\infty$ Down'])
plt.grid(True, linestyle=':'); plt.tight_layout(); plt.show()

r_edges_sp = np.linspace(R_root, R, N_OPT + 1)
r_mid_sp   = (r_edges_sp[:-1] + r_edges_sp[1:]) / 2.0
rR_sp      = r_mid_sp / R

P_inf      = 101325
P_dyn_inf  = 0.5 * rho * U0**2
P_tot_inf  = P_inf + P_dyn_inf         

Cp_sp, Ct_sp, span_sp = solve_bem_spanwise(8, opt_offsets, n_elem=N_OPT)

a_arr      = np.array(span_sp['a'])
CT_local   = CTfunction(a_arr)                    
P_drop_arr = P_dyn_inf * CT_local                 

P_inf_up   = np.full_like(rR_sp, P_tot_inf)
P_rotor_up = np.full_like(rR_sp, P_tot_inf)
P_rotor_dn = P_tot_inf - P_drop_arr
P_inf_dn   = P_tot_inf - P_drop_arr

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(rR_sp, P_inf_up   - P_inf, 'b-',  linewidth=2,   label=r'$\infty$ upwind')
ax.plot(rR_sp, P_rotor_up - P_inf, 'g--', linewidth=2,   label='Rotor (upwind face)')
ax.plot(rR_sp, P_rotor_dn - P_inf, 'r-',  linewidth=2,   label='Rotor (downwind face)')
ax.plot(rR_sp, P_inf_dn   - P_inf, 'm:',  linewidth=2,   label=r'$\infty$ downwind')

ax.fill_between(rR_sp,
                P_rotor_up - P_inf,
                P_rotor_dn - P_inf,
                alpha=0.15, color='red', label='Pressure drop across disk')

ax.set_xlabel('r/R')
ax.set_ylabel(r'$\Delta p_{\mathrm{stag}}$ relative to $p_\infty$ [Pa]')
ax.set_title(r'Radial distribution of stagnation pressure at four streamtube locations'
             '\n' r'(Optimised rotor, $\lambda=8$, $C_T=0.75$)')
ax.set_xlim(0.2, 1.0)
ax.legend(loc='lower left')
ax.grid(True, linestyle=':')
plt.tight_layout()
plt.show()

print(f"\nMean stagnation pressure drop across disk: {np.mean(P_drop_arr):.2f} Pa")
print(f"Peak pressure drop (mid-span):             {np.max(P_drop_arr):.2f} Pa")
print(f"Tip pressure drop (r/R=1.0):               {P_drop_arr[-1]:.2f} Pa")

plt.figure(figsize=(10, 5))
plt.plot(baseline_results[8]['r_R'], baseline_results[8]['a'], 'r--', label=r'Baseline $a$ ($\lambda=8$)')
plt.plot(opt_span['r_R'],            opt_span['a'],            'b-',  label=r'Optimised $a$ ($C_T=0.75$)')
plt.axhline(0.25, color='k', linestyle=':', label=r'Theoretical $a=0.25$')
plt.title('Axial Induction: Baseline vs Optimised')
plt.xlabel('r/R'); plt.ylabel('a'); plt.xlim(0.2, 1.0)
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

d8 = baseline_results[8]
plt.figure(figsize=(10, 5))
plt.plot(d8['r_R'], d8['F'],      'r-',  label=r'Total $F$')
plt.plot(d8['r_R'], d8['F_tip'],  'g.',  label=r'$F_{tip}$')
plt.plot(d8['r_R'], d8['F_root'], 'b.',  label=r'$F_{root}$')
plt.xlabel('r/R'); plt.ylabel('Correction Factor'); plt.xlim(0.2, 1.0)
plt.title('Prandtl Correction Factors (TSR=8)')
plt.legend(loc='lower right'); plt.grid(True); plt.tight_layout(); plt.show()

n_range  = [5, 10, 20, 50, 100, 200]
thrust_n = [solve_bem(8, n_elem=n)[2] / 1000 for n in n_range]
plt.figure(figsize=(8, 5))
plt.plot(n_range, thrust_n, 'o-')
plt.xlabel('Number of Annuli'); plt.ylabel('Total Thrust [kN]')
plt.title('Influence of Annuli Count on Total Thrust (TSR=8)')
plt.grid(True); plt.tight_layout(); plt.show()

_, _, _, _, data_const  = solve_bem(8, n_elem=20, spacing='constant')
_, _, _, _, data_cosine = solve_bem(8, n_elem=20, spacing='cosine')
plt.figure(figsize=(8, 5))
plt.plot(data_const['r_R'],  data_const['pN'],  'r-o', label='Constant')
plt.plot(data_cosine['r_R'], data_cosine['pN'], 'b-s', label='Cosine')
plt.xlabel('r/R'); plt.ylabel(r'$p_N$ [N/m]'); plt.xlim(0.2, 1.0)
plt.title(r'Effect of Spacing on Load Distribution ($N=20$)')
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

conv_hist = solve_bem(8, n_elem=50, track_conv=True)
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(conv_hist) + 1), np.array(conv_hist) / 1000, 'g-')
plt.xlabel('Iteration'); plt.ylabel('Total Thrust [kN]')
plt.title(r'Convergence History (TSR=8, $N=50$)')
plt.grid(True); plt.tight_layout(); plt.show()

alpha_col = df_polar.iloc[:, 0]
cl_col    = df_polar.iloc[:, 1]
cd_col    = df_polar.iloc[:, 2]

plt.figure(figsize=(8, 5))
plt.plot(alpha_col, cl_col, 'b-', linewidth=2)
plt.plot(8.73,1.168, 'ro', label='Design Point (α=8.73°)')
plt.xlabel(r'$\alpha$ [deg]'); plt.ylabel(r'$C_l$')
plt.title(r'$C_l$ vs. $\alpha$'); plt.xlim(-5, 20); plt.ylim(-0.5, 1.25)
plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 5))
plt.plot(cd_col, cl_col, 'r-', linewidth=2)
plt.plot(0.0099,1.168, 'ro', label='Design Point (α=8.73°)')
plt.xlabel(r'$C_d$'); plt.ylabel(r'$C_l$')
plt.title(r'Drag Polar: $C_l$ vs. $C_d$')
plt.xlim(0, 0.1); plt.grid(True, linestyle=':', alpha=0.7); plt.tight_layout(); plt.show()

ld       = cl_col / (cd_col + 1e-9)
best_idx = ld.idxmax()
print(f"\nMax L/D: {ld[best_idx]:.2f} at α={alpha_col[best_idx]:.2f}°, "
      f"Cl={cl_col[best_idx]:.4f}, Cd={cd_col[best_idx]:.4f}")

r_mid_plot = ((np.linspace(R_root, R, N_OPT + 1)[:-1] +
               np.linspace(R_root, R, N_OPT + 1)[1:]) / 2.0) / R

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(r_mid_plot, opt_offsets, 'g-o', markersize=4)
axes[0, 0].axhline(0, color='k', linestyle='--', linewidth=0.8)
axes[0, 0].set_xlabel('r/R'); axes[0, 0].set_ylabel(r'$\Delta\beta(r)$ [deg]')
axes[0, 0].set_title('Optimal Spanwise Pitch Offset')
axes[0, 0].set_xlim(0.2, 1.0); axes[0, 0].grid(True)

axes[0, 1].plot(span_ref['r_R'], span_ref['a'], 'r--', label='Reference')
axes[0, 1].plot(span_sw['r_R'],  span_sw['a'],  'b-',  label='Spanwise optimised')
axes[0, 1].axhline(1/3, color='k', linestyle=':', label='Betz $a=1/3$')
axes[0, 1].set_xlabel('r/R'); axes[0, 1].set_ylabel('$a$ [-]')
axes[0, 1].set_title('Axial Induction')
axes[0, 1].set_xlim(0.2, 1.0); axes[0, 1].legend(); axes[0, 1].grid(True)

axes[1, 0].plot(span_ref['r_R'], span_ref['pN'], 'r--', label='Reference')
axes[1, 0].plot(span_sw['r_R'],  span_sw['pN'],  'b-',  label='Spanwise optimised')
axes[1, 0].set_xlabel('r/R'); axes[1, 0].set_ylabel(r'$p_N$ [N/m]')
axes[1, 0].set_title('Normal Load Distribution')
axes[1, 0].set_xlim(0.2, 1.0); axes[1, 0].legend(); axes[1, 0].grid(True)

axes[1, 1].plot(span_ref['r_R'], span_ref['alpha'], 'r--', label='Reference')
axes[1, 1].plot(span_sw['r_R'],  span_sw['alpha'],  'b-',  label='Spanwise optimised')
axes[1, 1].axhline(alpha_design, color='k', linestyle=':',
                   label=rf'$\alpha^*={alpha_design:.1f}°$')
axes[1, 1].set_xlabel('r/R'); axes[1, 1].set_ylabel(r'$\alpha$ [deg]')
axes[1, 1].set_title('Angle of Attack')
axes[1, 1].set_xlim(0.2, 1.0); axes[1, 1].legend(); axes[1, 1].grid(True)

plt.suptitle(
    rf'Spanwise Pitch Optimisation ($\lambda=8$, $C_T={Ct_sw:.3f}$)  —  '
    rf'$C_P$: {Cp_ref:.4f} $\rightarrow$ {Cp_sw:.4f}',
    fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

tsr_case  = 8
span_case = baseline_results[tsr_case]

rR_base    = np.array(span_case['r_R'])
Cl_base    = np.array(span_case['Cl'])
chord_base = np.array([get_chord_baseline(r) for r in rR_base * R])
Clc_base   = Cl_base * chord_base

rR_opt    = np.array(span_sw['r_R'])
Cl_opt    = np.array(span_sw['Cl'])
chord_opt = chord_base
Clc_opt   = Cl_opt * chord_opt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(rR_base, Cl_base,   'b-',  linewidth=2, label=r'$C_l$ baseline')
ax0r = axes[0].twinx()
ax0r.plot(rR_base, chord_base,   'r', linewidth=2, label='Chord [m]')
axes[0].plot(rR_opt, Cl_opt,     'b--', linewidth=2, label=r'$C_l$ optimised')
axes[0].set_xlabel('r/R')
axes[0].set_ylabel(r'$C_l$ [-]', color='b')
ax0r.set_ylabel('Chord [m]', color='r')
axes[0].set_xlim(0.2, 1.0)
axes[0].set_title(r'Spanwise $C_l$ and chord')
axes[0].legend(loc='upper left', fontsize=8)
ax0r.legend(loc='center right', fontsize=8)
axes[0].grid(True, linestyle=':')

axes[1].plot(rR_base, Clc_base, 'b-',  linewidth=2, label='Baseline')
axes[1].plot(rR_opt,  Clc_opt,  'r-',  linewidth=2, label='Optimised')
axes[1].set_xlabel('r/R')
axes[1].set_ylabel(r'$C_l \cdot c$ [m]')
axes[1].set_title(r'$C_l \cdot c$ distribution')
axes[1].set_xlim(0.2, 1.0)
axes[1].legend()
axes[1].grid(True, linestyle=':')

Clc_diff = np.interp(rR_base, rR_opt, Clc_opt) - Clc_base
axes[2].plot(rR_base, Clc_diff, 'g-', linewidth=2)
axes[2].axhline(0, color='k', linestyle='--', linewidth=0.8)
axes[2].fill_between(rR_base, Clc_diff, 0,
                     where=(Clc_diff >= 0), alpha=0.2, color='green', label='Gain')
axes[2].fill_between(rR_base, Clc_diff, 0,
                     where=(Clc_diff < 0),  alpha=0.2, color='red',   label='Loss')
axes[2].set_xlabel('r/R')
axes[2].set_ylabel(r'$\Delta(C_l \cdot c)$ [m]')
axes[2].set_title(r'Change in $C_l \cdot c$: optimised $-$ baseline')
axes[2].set_xlim(0.2, 1.0)
axes[2].legend()
axes[2].grid(True, linestyle=':')

plt.suptitle(
    r'Lift distribution: baseline vs optimised ($\lambda=8$, $C_T=0.75$)',
    fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\nBaseline  — max Cl*c: {np.max(Clc_base):.4f} m at r/R={rR_base[np.argmax(Clc_base)]:.2f}")
print(f"Optimised — max Cl*c: {np.max(Clc_opt):.4f}  m at r/R={rR_opt[np.argmax(Clc_opt)]:.2f}")