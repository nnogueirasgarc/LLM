import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d, test
import matplotlib.pyplot as plt
from ass1 import rR, baseline_results, summary


class Airfoil:
    def __init__(self, filepath, name=""):
        self.name = name
        self.alpha = []
        self.cl = []
        self.cd = []
        
        self._load_polar(filepath)
        
        self.get_cl = interp1d(self.alpha, self.cl, kind='linear', fill_value="extrapolate")
        self.get_cd = interp1d(self.alpha, self.cd, kind='linear', fill_value="extrapolate")

    def _load_polar(self, filepath):
        """Reads the polar file and extracts alpha, cl, cd."""
        if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            df = pd.read_excel(filepath, skiprows=3)
            self.alpha = np.radians(df['Alfa'].values)
            self.cl = df['Cl'].values
            self.cd = df['Cd'].values
            
        elif filepath.endswith('.txt'):
            df = pd.read_csv(filepath, delim_whitespace=True, skiprows=2, names=['AoA', 'cl', 'cd', 'cm'])
            self.alpha = np.radians(df['AoA'].values)
            self.cl = df['cl'].values
            self.cd = df['cd'].values
        else:
            raise ValueError("Unsupported file format. Use .xlsx, .xls,  or .txt")

class RotorGeometry:
    def __init__(self, R, R_root, N_segments, blade_pitch_deg, chord_func, twist_func, spacing='uniform'):
        self.R = R
        self.R_root = R_root
        self.N = N_segments
        self.blade_pitch = np.radians(blade_pitch_deg)
        
        self.r_boundaries, self.r_control_points = self._discretize_blade(spacing)
        
        self.r_over_R = self.r_control_points / self.R
        
        self.chord = chord_func(self.r_over_R, self.R)
        self.twist = twist_func(self.r_over_R)
        
        self.pitch = self.twist + self.blade_pitch

    def _discretize_blade(self, spacing):
        """Creates segment boundaries and control point locations."""
        if spacing == 'uniform':
            boundaries = np.linspace(self.R_root, self.R, self.N + 1)
        
        elif spacing == 'cosine':
            theta = np.linspace(0, np.pi, self.N + 1)
            boundaries = self.R_root + 0.5 * (self.R - self.R_root) * (1 - np.cos(theta))
        else:
            raise ValueError("Spacing must be 'uniform' or 'cosine'")
            
        control_points = (boundaries[:-1] + boundaries[1:]) / 2.0
        
        return boundaries, control_points

class WakeGeometry:
    def __init__(self, r_boundaries, U_inf, a_w, Omega, n_revolutions, n_wake_segments_per_rev):
        self.r_boundaries = r_boundaries
        self.U_wake = U_inf * (1 - a_w)
        self.Omega = Omega
        self.n_revolutions = n_revolutions
        self.n_wake_segments_per_rev = n_wake_segments_per_rev
        
        total_time = (n_revolutions * 2 * np.pi) / Omega
        total_nodes = int(n_revolutions * n_wake_segments_per_rev) + 1
        self.t_array = np.linspace(0, total_time, total_nodes)
        
        self.nodes = self._generate_wake_nodes()
        
    def _generate_wake_nodes(self):
        """
        Creates a 3D array of shape (N_boundaries, N_time_steps, 3).
        nodes[i, j] returns the [x, y, z] coordinate of the i-th boundary's 
        trailing vortex at the j-th time step.
        """
        nodes = np.zeros((len(self.r_boundaries), len(self.t_array), 3))
        
        for i, r in enumerate(self.r_boundaries):
            nodes[i, :, 0] = self.t_array * self.U_wake
            nodes[i, :, 1] = r * np.sin(self.Omega * self.t_array)
            nodes[i, :, 2] = r * np.cos(self.Omega * self.t_array)
            
        return nodes

class LiftingLineSolver:

    def __init__(self, geometry, wake, core_radius=0.01):
        self.geometry = geometry
        self.wake = wake
        self.core = core_radius
        self.N = geometry.N
        
        self.MatrixU = np.zeros((self.N, self.N))
        self.MatrixV = np.zeros((self.N, self.N))
        self.MatrixW = np.zeros((self.N, self.N))
        
        print("Assembling influence matrices...")
        self._build_influence_matrices()
        print("Matrices assembled successfully!")
        
    def _build_influence_matrices(self):
        """Calculates the velocity induced by every horseshoe j on every control point i."""
        for i in range(self.N):
            icp_coord = np.array([0.0, 0.0, self.geometry.r_control_points[i]])
            
            for j in range(self.N):
                u_ind, v_ind, w_ind = 0.0, 0.0, 0.0
                
                p1 = self.wake.nodes[j, 0, :]   
                p2 = self.wake.nodes[j+1, 0, :] 
                vel = velocity_3D_from_vortex_filament(1.0, p1, p2, icp_coord, self.core)
                u_ind += vel[0]; v_ind += vel[1]; w_ind += vel[2]
                
                n_steps = self.wake.nodes.shape[1]
                for k in range(n_steps - 1):
                    p_start_root = self.wake.nodes[j, k+1, :]
                    p_end_root = self.wake.nodes[j, k, :]
                    vel = velocity_3D_from_vortex_filament(1.0, p_start_root, p_end_root, icp_coord, self.core)
                    u_ind += vel[0]; v_ind += vel[1]; w_ind += vel[2]
                    
                    p_start_tip = self.wake.nodes[j+1, k, :]
                    p_end_tip = self.wake.nodes[j+1, k+1, :]
                    vel = velocity_3D_from_vortex_filament(1.0, p_start_tip, p_end_tip, icp_coord, self.core)
                    u_ind += vel[0]; v_ind += vel[1]; w_ind += vel[2]
                    
                self.MatrixU[i, j] = u_ind
                self.MatrixV[i, j] = v_ind
                self.MatrixW[i, j] = w_ind

    def solve(self, U_inf, Omega, airfoil, max_iters=2000, tol=1e-4, relaxation=0.1, rotor_type='turbine'):
        """Runs the iterative loop to find the converged circulation distribution."""
        print(f"Starting iterative solver for {rotor_type}...")
        
        Gamma = np.zeros(self.N)
        Gamma_new = np.zeros(self.N)
        phi_array = np.zeros(self.N)
        alpha_eff_array = np.zeros(self.N)
        
        for iteration in range(max_iters):
            u_ind = np.dot(self.MatrixU, Gamma)
            v_ind = np.dot(self.MatrixV, Gamma)
            
            for i in range(self.N):
                r = self.geometry.r_control_points[i]
                c = self.geometry.chord[i]
                pitch = self.geometry.pitch[i]
                
                if rotor_type == 'turbine':
                    V_axial = U_inf + u_ind[i]
                    V_azim = (Omega * r) + v_ind[i] 
                    phi_array[i] = np.arctan2(V_axial, V_azim)
                    alpha_eff_array[i] = phi_array[i] - pitch
                
                V_rel = np.sqrt(V_axial**2 + V_azim**2)
                Cl = airfoil.get_cl(alpha_eff_array[i])
                Gamma_new[i] = 0.5 * V_rel * c * Cl
                
            error = np.max(np.abs(Gamma_new - Gamma))
            ref_error = max(np.max(np.abs(Gamma_new)), 0.001) 
            rel_error = error / ref_error
            
            if rel_error < tol:
                print(f"Converged successfully after {iteration} iterations!")
                self.Gamma = Gamma_new
                self.u_ind = u_ind
                self.v_ind = v_ind
                self.phi = phi_array.copy()
                self.alpha_eff = alpha_eff_array.copy()
                return self.Gamma
                
            Gamma = Gamma * (1 - relaxation) + Gamma_new * relaxation
            
        print("Warning: Solver reached maximum iterations without fully converging.")
        self.Gamma = Gamma
        self.phi = phi_array.copy()
        self.alpha_eff = alpha_eff_array.copy()
        return Gamma

    def calculate_performance(self, U_inf, Omega, airfoil, num_blades, rho=1.225, rotor_type='turbine'):
        """Calculates loads and non-dimensional coefficients after convergence."""
        self.a = -self.u_ind / U_inf
        self.a_prime = self.v_ind / (Omega * self.geometry.r_control_points)
        
        self.F_norm = np.zeros(self.N)
        self.F_tan = np.zeros(self.N)
        
        for i in range(self.N):
            r = self.geometry.r_control_points[i]
            c = self.geometry.chord[i]
            
            if rotor_type == 'turbine':
                V_axial = U_inf + self.u_ind[i]
                V_azim = (Omega * r) + self.v_ind[i]
            else:
                V_axial = U_inf - self.u_ind[i]
                V_azim = (Omega * r) - self.v_ind[i]
                
            V_rel = np.sqrt(V_axial**2 + V_azim**2)
            
            L_prime = rho * V_rel * self.Gamma[i]
            Cd = airfoil.get_cd(self.alpha_eff[i])
            D_prime = 0.5 * rho * V_rel**2 * c * Cd
            
            phi = self.phi[i]
            
            if rotor_type == 'turbine':
                self.F_norm[i] = L_prime * np.cos(phi) + D_prime * np.sin(phi)
                self.F_tan[i] = L_prime * np.sin(phi) - D_prime * np.cos(phi)
            
            
        dr = np.diff(self.geometry.r_boundaries) 
        Total_Thrust = np.sum(self.F_norm * dr) * num_blades
        Total_Torque = np.sum(self.F_tan * self.geometry.r_control_points * dr) * num_blades
        Total_Power = Total_Torque * Omega
        
        if rotor_type == 'turbine':
            A = np.pi * self.geometry.R**2
            self.CT = Total_Thrust / (0.5 * rho * U_inf**2 * A)
            self.CP = Total_Power / (0.5 * rho * U_inf**3 * A)
        
        print(f"--- Global Performance ({num_blades} Blades) ---")
        print(f"Thrust Coefficient (CT): {self.CT:.4f}")
        print(f"Power Coefficient (CP):  {self.CP:.4f}")


def velocity_3D_from_vortex_filament(gamma, xv1, xv2, xvp, core):
    """
    Calculates the 3D velocity induced by a straight vortex filament.
    
    Parameters:
    gamma (float): Circulation strength of the vortex.
    xv1 (array): [x, y, z] start coordinates of the filament.
    xv2 (array): [x, y, z] end coordinates of the filament.
    xvp (array): [x, y, z] target control point where velocity is calculated.
    core (float): Vortex core radius to prevent singularity.
    
    Returns:
    numpy array: [u, v, w] induced velocity components.
    """
    x1, y1, z1 = np.array(xv1)
    x2, y2, z2 = np.array(xv2)
    xp, yp, zp = np.array(xvp)
    
    r1_vec = np.array([xp - x1, yp - y1, zp - z1])
    r2_vec = np.array([xp - x2, yp - y2, zp - z2])
    
    r1_dist = np.linalg.norm(r1_vec)
    r2_dist = np.linalg.norm(r2_vec)
    
    r1_x_r2 = np.cross(r1_vec, r2_vec)
    r1_x_r2_sqr = np.sum(r1_x_r2**2)
    
    r0_vec = np.array([x2 - x1, y2 - y1, z2 - z1])
    
    r0_dot_r1 = np.dot(r0_vec, r1_vec)
    r0_dot_r2 = np.dot(r0_vec, r2_vec)
    
    if r1_x_r2_sqr < core**2:
        r1_x_r2_sqr = core**2
    if r1_dist < core:
        r1_dist = core
    if r2_dist < core:
        r2_dist = core
        
    K = (gamma / (4 * np.pi * r1_x_r2_sqr)) * ((r0_dot_r1 / r1_dist) - (r0_dot_r2 / r2_dist))
    
    return K * r1_x_r2

def plot_assignment_results(solver):
    """Generates the required radial distribution plots."""
    r_R = solver.geometry.r_over_R
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Lifting Line Model Results', fontsize=16)
    
    axs[0, 0].plot(r_R, np.degrees(solver.phi), 'b-o', label='Inflow Angle ($\phi$)')
    axs[0, 0].plot(r_R, np.degrees(solver.alpha_eff), 'r-s', label='Angle of Attack ($\\alpha$)')
    axs[0, 0].set_title('Angles vs. Radial Position')
    axs[0, 0].set_ylabel('Angle [degrees]')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    axs[0, 1].plot(r_R, solver.a, 'g-o', label='Axial Induction ($a$)')
    axs[0, 1].plot(r_R, solver.a_prime, 'm-s', label='Tangential Induction ($a\'$)')
    axs[0, 1].set_title('Induction Factors vs. Radial Position')
    axs[0, 1].set_ylabel('Induction Factor [-]')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(r_R, solver.F_norm, 'k-o', label='Normal/Axial Load ($F_{norm}$)')
    axs[1, 0].plot(r_R, solver.F_tan, 'c-s', label='Tangential Load ($F_{tan}$)')
    axs[1, 0].set_title('Blade Loading vs. Radial Position')
    axs[1, 0].set_ylabel('Load [N/m]')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(r_R, solver.Gamma, 'purple', marker='D')
    axs[1, 1].set_title('Bound Circulation ($\Gamma$) vs. Radial Position')
    axs[1, 1].set_ylabel('Circulation [m$^2$/s]')
    axs[1, 1].grid(True)
    
    axs[2, 0].axis('off')
    axs[2, 1].axis('off')
    
    text_str = f"$C_T$ = {solver.CT:.4f}\n$C_P$ = {solver.CP:.4f}"
    fig.text(0.5, 0.2, text_str, fontsize=14, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.25, 1, 0.95])
    plt.show()

def run_turbine_case():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        turbine_file = os.path.join(current_dir, 'polar DU95W180.xlsx') 
        
        turbine_airfoil = Airfoil(turbine_file, name='DU95W180')
        
        print("--- Testing Step 1 (Airfoils) ---")
        print(f"Turbine Cl at 5 deg: {turbine_airfoil.get_cl(np.radians(5.0)):.4f}")
        
        print("\n--- Testing Step 2 (Geometry) ---")
        
        turbine_twist = lambda r_R: np.radians(14 * (1 - r_R))
        turbine_chord = lambda r_R, R: 3 * (1 - r_R) + 1

        turbine = RotorGeometry(
            R = 50.0, 
            R_root = 0.2 * 50.0, 
            N_segments = 10,
            blade_pitch_deg = -2.0, 
            chord_func = turbine_chord,
            twist_func = turbine_twist,
            spacing = 'uniform'
        )
        print("Turbine Control Points (r):")
        print(np.round(turbine.r_control_points, 2))

        print("\n--- Testing Step 3 (Biot-Savart Engine) ---")
        
        start_pt = [0, -1, 0]
        end_pt = [0, 1, 0]
        target_pt = [1, 0, 0]
        
        ind_vel = velocity_3D_from_vortex_filament(gamma=1.0, xv1=start_pt, xv2=end_pt, xvp=target_pt, core=0.01)
        
        print("Vortex from [0,-1,0] to [0,1,0]")
        print("Target Point: [1,0,0]")
        print(f"Induced Velocity [u, v, w]: {np.round(ind_vel, 4)}")

        print("\n--- Testing Step 4 (Wake Geometry) ---")
        
        U0_turbine = 10.0
        TSR = 6.0
        Omega_turbine = (TSR * U0_turbine) / turbine.R
        
        turbine_wake = WakeGeometry(
            r_boundaries = turbine.r_boundaries,
            U_inf = U0_turbine,
            a_w = 0.20,
            Omega = Omega_turbine,
            n_revolutions = 3,
            n_wake_segments_per_rev = 12
        )
        
        print(f"Wake Matrix Shape: {turbine_wake.nodes.shape}")
        print("Coordinates of the root vortex (r=10m) for the first 3 time steps:")
        print(np.round(turbine_wake.nodes[0, 0:3, :], 2))

        print("\n--- Testing Step 5 (Influence Matrices) ---")
        
        solver = LiftingLineSolver(turbine, turbine_wake, core_radius=0.1)
        
        print(f"Matrix U shape: {solver.MatrixU.shape}")
        print(f"Influence of Horseshoe 0 on Control Point 0:")
        print(f"u: {solver.MatrixU[0,0]:.4f}, v: {solver.MatrixV[0,0]:.4f}, w: {solver.MatrixW[0,0]:.4f}")

        print("\n--- Testing Step 6 (Iterative Solver) ---")
        
        converged_gamma = solver.solve(
            U_inf = U0_turbine, 
            Omega = Omega_turbine, 
            airfoil = turbine_airfoil,
            relaxation = 0.1
        )
        
        print("\nFinal Converged Circulation Distribution (Gamma):")
        print(np.round(converged_gamma, 2))

        solver.calculate_performance(U_inf=U0_turbine, Omega=Omega_turbine, airfoil=turbine_airfoil, num_blades=3)
        
        plot_assignment_results(solver)
        plot_combined_comparison(solver, rR, baseline_results, summary)


def run_convection_speed_sensitivity():
    print("\n--- SENSITIVITY: CONVECTION SPEED (a_w) ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    airfoil = Airfoil(os.path.join(current_dir, 'polar DU95W180.xlsx'), name='DU95W180')
    turbine = RotorGeometry(R=50.0, R_root=10.0, N_segments=15, blade_pitch_deg=-2.0, 
                            chord_func=lambda r, R: 3*(1-r)+1, twist_func=lambda r: np.radians(14*(1-r)), spacing='uniform')
    
    U0 = 10.0; Omega = (8.0 * U0) / turbine.R
    aw_values = [0.1, 0.2, 0.33]
    
    plt.figure(figsize=(8, 5))
    for aw in aw_values:
        wake = WakeGeometry(turbine.r_boundaries, U0, aw, Omega, 4, 12)
        solver = LiftingLineSolver(turbine, wake, core_radius=0.1)
        gamma = solver.solve(U0, Omega, airfoil, relaxation=0.1)
        plt.plot(turbine.r_over_R, gamma, marker='o', label=f'$a_w$ = {aw}')
        
    plt.title('Sensitivity to Assumed Convection Speed ($a_w$)')
    plt.xlabel('Radial Position r/R'); plt.ylabel('Circulation $\Gamma$ [m$^2$/s]')
    plt.grid(True); plt.legend(); plt.show()

def run_spacing_sensitivity():
    print("\n--- SENSITIVITY: SPACING (Uniform vs Cosine) ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    airfoil = Airfoil(os.path.join(current_dir, 'polar DU95W180.xlsx'), name='DU95W180')
    U0 = 10.0; Omega = (8.0 * U0) / 50.0
    
    plt.figure(figsize=(8, 5))
    for space_type in ['uniform', 'cosine']:
        turbine = RotorGeometry(R=50.0, R_root=10.0, N_segments=15, blade_pitch_deg=-2.0, 
                                chord_func=lambda r, R: 3*(1-r)+1, twist_func=lambda r: np.radians(14*(1-r)), spacing=space_type)
        wake = WakeGeometry(turbine.r_boundaries, U0, 0.20, Omega, 4, 12)
        solver = LiftingLineSolver(turbine, wake, core_radius=0.1)
        gamma = solver.solve(U0, Omega, airfoil, relaxation=0.1)
        plt.plot(turbine.r_over_R, gamma, marker='o', label=f'Spacing: {space_type}')
        
    plt.title('Sensitivity to Discretization Spacing')
    plt.xlabel('Radial Position r/R'); plt.ylabel('Circulation $\Gamma$ [m$^2$/s]')
    plt.grid(True); plt.legend(); plt.show()

def run_azimuthal_sensitivity():
    print("\n--- SENSITIVITY: AZIMUTHAL STEPS ---")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    airfoil = Airfoil(os.path.join(current_dir, 'polar DU95W180.xlsx'), name='DU95W180')
    turbine = RotorGeometry(R=50.0, R_root=10.0, N_segments=15, blade_pitch_deg=-2.0, 
                            chord_func=lambda r, R: 3*(1-r)+1, twist_func=lambda r: np.radians(14*(1-r)), spacing='uniform')
    
    U0 = 10.0; Omega = (8.0 * U0) / turbine.R
    steps_per_rev = [6, 12, 24]
    
    plt.figure(figsize=(8, 5))
    for steps in steps_per_rev:
        wake = WakeGeometry(turbine.r_boundaries, U0, 0.20, Omega, 3, steps)
        solver = LiftingLineSolver(turbine, wake, core_radius=0.1)
        gamma = solver.solve(U0, Omega, airfoil, relaxation=0.1)
        plt.plot(turbine.r_over_R, gamma, marker='o', label=f'{steps} steps/rev')
        
    plt.title('Sensitivity to Azimuthal Discretization')
    plt.xlabel('Radial Position r/R'); plt.ylabel('Circulation $\Gamma$ [m$^2$/s]')
    plt.grid(True); plt.legend(); plt.show()

def run_wake_length_sensitivity():
    print("\n" + "="*40)
    print(" EXECUTING SENSITIVITY: WAKE LENGTH CONVERGENCE")
    print("="*40)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    turbine_airfoil = Airfoil(os.path.join(current_dir, 'polar DU95W180.xlsx'), name='DU95W180')
    
    turbine_twist = lambda r_R: np.radians(14 * (1 - r_R))
    turbine_chord = lambda r_R, R: 3 * (1 - r_R) + 1
    
    turbine = RotorGeometry(R=50.0, R_root=10.0, N_segments=15, blade_pitch_deg=-2.0, 
                            chord_func=turbine_chord, twist_func=turbine_twist, spacing='uniform')
    
    U0 = 10.0; TSR = 8.0; Omega = (TSR * U0) / turbine.R
    
    wake_lengths_to_test = [1, 2, 3, 5, 8]
    
    CT_results = []
    CP_results = []
    
    for revs in wake_lengths_to_test:
        print(f"Calculating for Wake Length: {revs} Revolutions...")
        
        wake = WakeGeometry(turbine.r_boundaries, U_inf=U0, a_w=0.20, Omega=Omega, 
                            n_revolutions=revs, n_wake_segments_per_rev=12)
        
        solver = LiftingLineSolver(turbine, wake, core_radius=0.1)
        solver.solve(U_inf=U0, Omega=Omega, airfoil=turbine_airfoil, relaxation=0.1)
        solver.calculate_performance(U_inf=U0, Omega=Omega, airfoil=turbine_airfoil, num_blades=3)
        
        CT_results.append(solver.CT)
        CP_results.append(solver.CP)
        
    plt.figure(figsize=(10, 5))
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.set_xlabel('Wake Length (Number of Rotations)')
    ax1.set_ylabel('Thrust Coefficient ($C_T$)', color='tab:blue')
    ax1.plot(wake_lengths_to_test, CT_results, 'b-o', label='$C_T$ (Thrust)')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Power Coefficient ($C_P$)', color='tab:red')  
    ax2.plot(wake_lengths_to_test, CP_results, 'r-s', label='$C_P$ (Power)')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    plt.title('Convergence of Global Coefficients vs. Wake Length')
    fig.tight_layout() 
    plt.show()



#THIS SECTION NEEDS TO BE FINISHED!!!

#BASICLY FINISH IMPORTING VARIABLES FROM ASS1 TO COMPARE BEM AND LLM

def plot_combined_comparison(solver, rR, baseline_results, summary):
    r_R = solver.geometry.r_over_R
    
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Lifting Line Model Results', fontsize=16)
    
    axs[0, 0].plot(r_R, np.degrees(solver.phi), 'b-o', label='Lifting Line $\phi$')
    axs[0, 0].plot(rR[6], baseline_results[6]["phi"], 'b-', label='BEM $\phi$')
    axs[0, 0].plot(r_R, np.degrees(solver.alpha_eff), 'r-s', label='Lifting Line $\alpha$')
    axs[0, 0].plot(rR[6], baseline_results[6]["alpha"], 'r-', label='BEM $\\alpha$')
    axs[0, 0].set_title('Angles vs. Radial Position')
    axs[0, 0].set_ylabel('Angle [degrees]')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    plt.show()

    axs[0, 1].plot(r_R, solver.a, 'g-o', label='Lifting Line $a$')
    axs[0, 1].plot(rR, bem_res.a, 'g-', label='BEM $a$')
    axs[0, 1].plot(r_R, solver.a_prime, 'm-s', label='Lifting Line $a\'$')
    axs[0, 1].plot(rR, bem_res.ap, 'm-', label='BEM $a\'$')
    axs[0, 1].set_title('Induction Factors vs. Radial Position')
    axs[0, 1].set_ylabel('Induction Factor [-]')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    axs[1, 0].plot(r_R, solver.F_norm, 'k-o', label='Lifting Line $F_{norm}$')
    axs[1, 0].plot(rR, bem_res.F_norm, 'k-', label='BEM $F_{norm}$')
    axs[1, 0].plot(r_R, solver.F_tan, 'c-s', label='Lifting Line $F_{tan}$')
    axs[1, 0].plot(rR, bem_res.CT_load, 'c-', label='BEM $F_{tan}$')
    axs[1, 0].set_title('Blade Loading vs. Radial Position')
    axs[1, 0].set_ylabel('Load [N/m]')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    axs[1, 1].plot(r_R, solver.Gamma, 'purple', marker='D', linestyle='--', label='Lifting Line $\Gamma$')
    axs[1, 1].plot(rR, bem_res.Gamma, 'purple', linestyle='--', label='BEM $\Gamma$')
    axs[1, 1].set_title('Bound Circulation ($\Gamma$) vs. Radial Position')
    axs[1, 1].set_ylabel('Circulation [m$^2$/s]')
    axs[1, 1].grid(True)
    
    axs[2, 0].axis('off')
    axs[2, 1].axis('off')
    
    text_str = f"$C_T$ = {solver.CT:.4f}\n$C_P$ = {solver.CP:.4f}"
    fig.text(0.5, 0.2, text_str, fontsize=14, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.25, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    run_turbine_case()
    run_convection_speed_sensitivity()
    run_spacing_sensitivity()
    run_azimuthal_sensitivity()
    run_wake_length_sensitivity()
