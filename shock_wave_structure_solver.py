import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator
from scipy.integrate import solve_bvp
from scipy.optimize import root
import os
import sys



SAVE_DIR = 'results'         
DATA_DIR = 'data'            

# --- FIGURE FORMAT AND SAVE FUNCTION ---
def save(fig, name):
        base_path = os.path.join(SAVE_DIR, name)
        os.makedirs(SAVE_DIR, exist_ok=True)
        fig.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
        print(f"Saved: {name}")

def save_data_csv(x_data, y_data, name):
    base_path = os.path.join(SAVE_DIR, name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    datos = {
        'x_axis': x_data,
        'y_axis': y_data
    }
    df = pd.DataFrame(datos)
    file_path = os.path.join(SAVE_DIR, f"{name}.csv")
    df.to_csv(file_path, index=False)
    print(f"Data saved in: {file_path}")


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'axes.grid': False, 
    })


# --- DOMAIN PARAMETERS ---
num_points = 10000       # Number of points in the solution
xi = 100                 # Number of points for the objective function evaluation
xi_up = -xi
xi_down = xi
amplitude_input = 2     # Amplitude for the initial guess
tol_rel = 1e-5          # Relative tolerance for BVP solver


# --- A. PHYSICAL CONSTANTS ---
R_UNIV = 8314.46 
M_CO2 = 44.01
R_SPECIFIC = R_UNIV / M_CO2
d_CO2 = 3.941e-10    # Effective Diameter of CO2 molecule [m]
kb = 1.380649e-23    # Boltzmann Constant [J/K]
P_c = 73.773*1e5     # Critical Pressure [Pa]
T_c = 304.1282       # Critical Temperature [K]
rho_c = 467.6        # Critical Density [kg/m3]


# --- B. UPSTREAM CONDITIONS ---
T_us = 300            # Temperature [K]
P_us = 100*1e5        # Pressure [Pa]
Ma_us = 1.5           # Mach Number
initial_guess_factor_u_ds = 0.9  # Initial guess factor for u_ds in RH solver
initial_guess_factor_T_ds = 1.1  # Initial guess factor for T_ds in RH solver


# =================================================================
#                      I. NIST DATA LOADING
# =================================================================             

P_us_bar = P_us / 1e5  # Pressure in bar
P_c_bar = P_c / 1e5    # Critical Pressure in bar


_DATA_CACHE = {}

def load_nist_data():
    if 'df_full' in _DATA_CACHE: return _DATA_CACHE['df_full']

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Directory not found: {DATA_DIR}")
        return None
    
    all_data = []
    print("INFO: Loading NIST tables...")
    for f in os.listdir(DATA_DIR):
        if f.startswith("CO2_") and f.endswith(".txt"):
            try:
                df = pd.read_csv(os.path.join(DATA_DIR, f), sep='\t', decimal='.', na_values=['nan', 'NaN'])
                all_data.append(df)
            except Exception as e: pass

    if not all_data: return None
    df = pd.concat(all_data, ignore_index=True)

    # Cleaning
    df.dropna(inplace=True)
    df.drop_duplicates(subset=['Temperature (K)', 'Pressure (bar)'], inplace=True)
    df['Enthalpy (J/kg)'] = df['Enthalpy (kJ/kg)'] * 1000.0
    df['Cp (J/kg*K)'] = df['Cp (J/g*K)'] * 1000.0
    df['Cv (J/kg*K)'] = df['Cv (J/g*K)'] * 1000.0

    # Calculate Z
    df['Z'] = (df['Pressure (bar)'] * 1e5) / (df['Density (kg/m3)'] * R_SPECIFIC * df['Temperature (K)'])

    _DATA_CACHE['df_full'] = df
    return df

def create_interpolators():
    df = load_nist_data()
    if df is None: return None
    points = df[['Temperature (K)', 'Pressure (bar)']].values
    interps = {}

    print("INFO: Creating Interpolators...")
    for col in ['Density (kg/m3)', 'Enthalpy (J/kg)', 'Viscosity (Pa*s)', 
                'Therm. Cond. (W/m*K)', 'Sound Spd. (m/s)', 'Cp (J/kg*K)', 'Cv (J/kg*K)', 'Z']:
        try:
            interps[col] = CloughTocher2DInterpolator(points, df[col].values)
        except:
            print(f"WARNING: Cubic failed for {col}, falling back to Linear.")
            interps[col] = LinearNDInterpolator(points, df[col].values)

    return interps

NIST = create_interpolators()

# =================================================================
#              II. THERMODYNAMIC HELPER FUNCTIONS
# =================================================================

def get_prop_exact(prop, T, P_bar):
    if NIST is None: return None
    val = NIST[prop](T, P_bar)
    # Handle NaN from interpolation bounds
    if np.isnan(val): return None
    return float(val) 

def get_constants_S0():
    rho_us = get_prop_exact('Density (kg/m3)', T_us, P_us_bar)
    a_us = get_prop_exact('Sound Spd. (m/s)', T_us, P_us_bar)
    mu_us = get_prop_exact('Viscosity (Pa*s)', T_us, P_us_bar)
    k_us = get_prop_exact('Therm. Cond. (W/m*K)', T_us, P_us_bar)
    Cp_us = get_prop_exact('Cp (J/kg*K)', T_us, P_us_bar)
    Cv_us = get_prop_exact('Cv (J/kg*K)', T_us, P_us_bar)
    h_us = get_prop_exact('Enthalpy (J/kg)', T_us, P_us_bar)
    Z_us = get_prop_exact('Z', T_us, P_us_bar)

    if rho_us is None: return None

    u_us = Ma_us * a_us
    lambda_us = kb * T_us * Z_us / (np.sqrt(2) * np.pi * d_CO2**2 * P_us)  # Mean Free Path [m]
    Re_us = (rho_us * u_us * lambda_us) / (4/3 * mu_us)
    Pr_us = (4/3 * mu_us * Cp_us) / k_us
    Pe_us = Re_us * Pr_us
    Ec_us = u_us**2 / (Cp_us * T_us)
    Br_us = Ec_us * Pr_us

    print(f"rho_us: {rho_us:.2f}, Re_us: {Re_us:.5f}, Pr_us: {Pr_us:.2f}, Pe_us: {Pe_us:.2f}, Ec_us: {Ec_us:.4f}, Br_us: {Br_us:.4f} u_us: {u_us:.2f} m/s")

    gamma_us = Cp_us / Cv_us

    return {
        'u_us': u_us, 'rho_us': rho_us, 'h_us': h_us, 'mu_us': mu_us, 'k_us': k_us, 'Cp_us': Cp_us,
        'Cv_us':Cv_us, 'Re_us': Re_us, 'Pr_us': Pr_us, 'Pe_us': Pe_us, 'Ec_us': Ec_us, 
        'Z_us': Z_us, 'gamma_us': gamma_us, 'P_us': P_us, 'lambda_us': lambda_us
    }

CONST = get_constants_S0()


# =================================================================
#                  III. NUMERICAL DERIVATIVES
# =================================================================
def get_nist_partials(T, P_bar):
    """
    Calculates (d_rho/d_T)|P and (d_rho/d_P)|T using finite differences 
    on the NIST table interpolators.
    """
    eps_T = 1e-4 * T
    eps_P = 1e-4 * P_bar
    
    # Base density
    rho0 = NIST['Density (kg/m3)'](T, P_bar)
    
    # Perturb T
    rho_dT_val = NIST['Density (kg/m3)'](T + eps_T, P_bar)
    drho_dT = (rho_dT_val - rho0) / eps_T
    
    # Perturb P
    rho_dP_val = NIST['Density (kg/m3)'](T, P_bar + eps_P)
    drho_dP_bar = (rho_dP_val - rho0) / eps_P # This is d(rho)/d(P_bar)
    
    return drho_dT, drho_dP_bar


# =================================================================
#                    IV. RANKINE-HUGONIOT
# =================================================================

def solve_RH():
    print("INFO: Solving Rankine-Hugoniot...")
    flux_mass = CONST['rho_us'] * CONST['u_us']
    flux_mom  = CONST['P_us'] + CONST['rho_us'] * CONST['u_us']**2
    flux_h    = CONST['h_us'] + 0.5 * CONST['u_us']**2

    def resid(vars):
        u_ds, T_ds = vars
        if u_ds > CONST['u_us'] or u_ds < 0: return [1e9, 1e9]

        rho_ds = flux_mass / u_ds
        P_ds = flux_mom - rho_ds * u_ds**2
        P_ds_bar = P_ds / 1e5 

        rho_nist = get_prop_exact('Density (kg/m3)', T_ds, P_ds_bar)
        h_nist = get_prop_exact('Enthalpy (J/kg)', T_ds, P_ds_bar)

        if rho_nist is None: return [1e9, 1e9]

        err_rho = (rho_nist - rho_ds)/(rho_ds+1e-12)

        err_h = ((h_nist + 0.5 * u_ds**2) - flux_h)/(flux_h+1e-12)
        
        return [err_rho, err_h]
    
    guess = [CONST['u_us']*initial_guess_factor_u_ds, T_us*initial_guess_factor_T_ds]
    sol = root(resid, guess, method='hybr')
    print(f"INFO: Root-finding status: {sol.success}")
    print(f"INFO: Number of function evaluations: {sol.nfev}") 
    print(f"INFO: Final residual error: {np.linalg.norm(sol.fun):.2e}")

    if sol.success:
        u_ds, T_ds = sol.x
        rho_ds = flux_mass / u_ds
        P_ds_bar = (flux_mom - rho_ds*u_ds**2)/1e5
        Ma_ds = u_ds / get_prop_exact('Sound Spd. (m/s)', T_ds, P_ds_bar)
        print(f"   Downstream: u_ds={u_ds:.2f} m/s, Ma_ds = {Ma_ds:.2f} T_ds={T_ds:.2f} K, P_ds={P_ds_bar:.2f} bar")
        return {'Ma_ds': Ma_ds, 'u_ds': u_ds, 'T_ds': T_ds, 'P_ds': P_ds_bar}
    else:
        print("ERROR: Rankine-Hugoniot failed.")
        return None
    
S1 = solve_RH()


# --- C. DERIVED UPSTREAM CONDITIONS ---
u_us = CONST['u_us']                            # Velocity [m/s]
rho_us = CONST['rho_us']                        # Density [kg/m3]
h_us_star = CONST['h_us'] / (CONST['u_us']**2)  # Normalised Enthalpy at Upstream


# --- D. UPSTREAM BOUNDARY CONDITIONS ---
u_us_star = u_us/u_us            # Normalised Velocity at Upstream
T_us_star = T_us/T_us            # Normalised Temperature at Upstream
P_us_star = P_us/(rho_us*u_us*u_us)             # Normalised Pressure at Upstream


# --- E. CALCULATED DOWNSTREAM CONDITIONS ---
u_ds = S1['u_ds']          # Velocity at Downstream [m/s]
T_ds = S1['T_ds']          # Temperature at Downstream [K]
P_ds = S1['P_ds'] * 1e5    # Pressure at Downstream [Pa]


# --- F. Independent variables ---
u_star = np.ones(num_points)*u_us_star  # Normalised Velocity [u/u0]
T_star = np.ones(num_points)*T_us_star  # Normalised Temperature [T/T0]
P_star = np.ones(num_points)*P_us_star  # Normalised Pressure [P/P0]


# =================================================================
#             SAFETY HELPER (Prevents Crash)
# =================================================================
def get_prop_safe(prop, T, P_bar):
    """
    Wraps get_prop_exact to handle NoneType errors if NIST fails.
    """
    val = get_prop_exact(prop, T, P_bar)
    if val is None or np.isnan(val):
        return 1.0  # Dummy value to prevent solver crash
    return float(val)


# --- G. SYSTEM OF EQUATIONS ---
def fun(x, Y, p):
    u_star = Y[0]
    T_star = Y[1]
    P_star = Y[2]
        
    correction = (1.0 + p[0])
    correction_P = P_us_star * (1.0 + p[0])

    du_star_dxi = np.zeros(len(u_star))
    dT_star_dxi = np.zeros(len(T_star))
    drho_star_dxi = np.zeros(len(T_star))
    dP_star_dxi = np.zeros(len(P_star))

    for i in range(len(P_star)):
        P_val_bar = P_star[i] * (rho_us*u_us*u_us) / 1e5  # Convert to bar
        T_val = T_star[i] * T_us
        u_val = u_star[i] * u_us

        # --- CLAMPING LOGIC ---
        T_clamp = np.clip(T_val, T_us*0.9, T_ds*1.1)
        P_clamp = np.clip(P_val_bar, P_us_bar*0.9, (P_ds/1e5)*1.1)
        # --------------------------

        # Use safe lookup with clamped values
        kappa_star = get_prop_safe('Therm. Cond. (W/m*K)', T_clamp, P_clamp) / CONST['k_us']
        h_star = get_prop_safe('Enthalpy (J/kg)', T_clamp, P_clamp) / (CONST['u_us']**2)
        rho_star = get_prop_safe('Density (kg/m3)', T_clamp, P_clamp) / rho_us
        mu_star = get_prop_exact('Viscosity (Pa*s)', T_clamp, P_clamp) / CONST['mu_us']

        drho_dT_SI, drho_dP_bar = get_nist_partials(T_clamp, P_clamp)
        drho_dP_SI = drho_dP_bar / 1e5
        
        coeff_P = drho_dP_SI * u_us * u_us
        if abs(coeff_P) < 1e-12: coeff_P = 1e-12 # Prevent div by zero

        du_star_dxi[i] = CONST['Re_us']/mu_star * ( (u_star[i]-correction) + (P_star[i]-correction_P) )
        dT_star_dxi[i] = ((CONST['Pe_us'] * CONST['Ec_us'])/kappa_star) * ( (h_star-h_us_star) -0.5*((u_star[i]-correction)**2) - u_star[i]*(P_star[i]-correction_P) ) 
        
        # Protect against zero velocity for density equation
        u_safe = u_star[i] if abs(u_star[i]) > 1e-6 else 1e-6
        drho_star_dxi[i] = -(rho_star/u_safe) * du_star_dxi[i]   
        
        dP_star_dxi[i] = (drho_star_dxi[i] - ((drho_dT_SI * T_us / rho_us) * dT_star_dxi[i])) / coeff_P

    return [du_star_dxi, dT_star_dxi, dP_star_dxi]


# --- H. BOUNDARY CONDITIONS ---
def bc(Y_up, Y_down, p):
    
    res_u_up = Y_up[0] - u_us_star
    res_T_up = Y_up[1] - T_us_star
    res_P_up = Y_up[2] - P_us_star
    
    res_u_down = Y_down[0] - (u_ds / u_us)
    res_T_down = Y_down[1] - (T_ds / T_us)
    res_P_down = Y_down[2] - (P_ds / (rho_us*u_us*u_us))
  
    return [res_T_up, res_P_up, res_T_down, res_P_down] 


# --- I. INITIAL GUESS ---
xi = np.linspace(xi_up, xi_down, num_points)  # mesh points
Y_guess = np.zeros((3, xi.size))
Amplitude = amplitude_input
Y_guess[0] = u_us_star + (u_ds/u_us - u_us_star) * (np.tanh(xi/Amplitude) + 1)/2
Y_guess[1] = T_us_star + (T_ds/T_us - T_us_star) * (np.tanh(xi/Amplitude) + 1)/2
Y_guess[2] = P_us_star + (P_ds/(rho_us*u_us*u_us) - P_us_star) * (np.tanh(xi/Amplitude) + 1)/2


# --- J. PLOT INITIAL GUESS ---
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))
# axs[0, 0].plot(xi, Y_guess[2], 'r-', lw=2)
# axs[0, 0].set_ylabel(r'$P^*$')
# axs[0, 1].plot(xi, Y_guess[1], 'g-', lw=2)
# axs[0, 1].set_ylabel(r'$T^*$')
# axs[1, 0].plot(xi, Y_guess[0], 'b-', lw=2)
# axs[1, 0].set_ylabel(r'$u^*$')
# axs[1, 1].axis('off') 
# for ax in axs.flat:
#     if ax.axison:
#         ax.set_xlabel(r'$\xi$')
#         ax.grid(True, linestyle=':', alpha=0.6)

# plt.tight_layout()
# save(fig, f'Ma_us={Ma_us:.1f}_all_guesses_combined')
# plt.close(fig)

# --- K. SOLVE BVP ---
sol = solve_bvp(fun, bc, xi, Y_guess, p=[0.0], verbose=2, tol=tol_rel)


# --- L. CALCULATE VARIABLES TO PLOT ---
u_star_sol = sol.y[0]             # Normalised Velocity
T_star_sol = sol.y[1]             # Normalised Temperature
P_star_sol = sol.y[2]             # Normalised Pressure

u = u_star_sol * u_us              # Velocity [m/s]
T = T_star_sol * T_us              # Temperature [K]
P_bar = P_star_sol * ((rho_us*u_us*u_us) / 1e5)  # Pressure [bar]
rho = rho_us/u_star_sol            # Density [kg/m3]

a_list = []
for i in range(len(T)):
    val = get_prop_exact('Sound Spd. (m/s)', T[i], P_bar[i])
    a_list.append(val if val is not None else np.nan)
a = np.array(a_list)
Ma = u / a                        # Mach Number


# --- M. CALCULATE THICKNESS ---
def calculate_thickness(xi, rho):
    """
    Calculate shock thickness based on density gradient.
    """
    drho_dxi = np.gradient(rho, xi)
    max_slope = np.max(np.abs(drho_dxi))
    rho_us_val = rho[0]
    rho_ds_val = rho[-1]
    thickness_xi = (rho_ds_val-rho_us_val) / (max_slope + 1e-12)
    return thickness_xi

delta_xi = calculate_thickness(sol.x, rho)


# --- N. PLOT FINAL SOLUTIONS ---
# Velocity Ma
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(sol.x, Ma, 'b-', lw=2, label='Velocity')
ax1.set_xlabel(r'$\xi^\star$ ($x/\lambda_{us}$)')
ax1.set_ylabel('\textrm{Ma}')
ax1.legend(loc='best', fontsize=10, frameon=False)
save(fig1, f'Ma_us={Ma_us:.1f}_velocity_solution_P_us={P_us_bar:.1f}bar_T_us={T_us:.0f}K')
plt.close(fig1)
save_data_csv(sol.x, Ma, f'Ma_us={Ma_us:.1f}_velocity_solution_data_P_us={P_us_bar:.1f}bar_T_us={T_us:.0f}K')

# Temperature
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.plot(sol.x, T/T_c, 'g-', lw=2, label='Temperature')
ax2.set_xlabel(r'$\xi^\star$ ($x/\lambda_{us}$)')
ax2.set_ylabel(r'$T/T_c$')
ax2.legend(loc='best', fontsize=10, frameon=False)
save(fig2, f'Ma_us={Ma_us:.1f}_temperature_solution_P_us={P_us_bar:.1f}bar_T_us={T_us:.0f}K')
plt.close(fig2)
save_data_csv(sol.x, T/T_c, f'Ma_us={Ma_us:.1f}_temperature_solution_data_P_us={P_us_bar:.1f}bar_T_us={T_us:.0f}K')

# Pressure
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.plot(sol.x, P_bar/P_c_bar, 'r-', lw=2, label='Pressure')
ax3.set_xlabel(r'$\xi^\star$ ($x/\lambda_{us}$)')
ax3.set_ylabel(r'$P/P_c$')
ax3.legend(loc='best', fontsize=10, frameon=False)
save(fig3, f'Ma_us={Ma_us:.1f}_pressure_solution_P_us={P_us_bar:.1f}bar_T_us={T_us:.0f}K')
plt.close(fig3)
save_data_csv(sol.x, P_bar/P_c_bar, f'Ma_us={Ma_us:.1f}_pressure_solution_data_P_us={P_us_bar:.1f}bar_T_us={T_us:.0f}K')

# Density
fig4, ax4 = plt.subplots(figsize=(6, 4))
ax4.plot(sol.x, rho/rho_c, 'm-', lw=2, label='Density')
ax4.set_xlabel(r'$\xi^\star$ ($x/\lambda_{us}$)')
ax4.set_ylabel(r'$\rho/\rho_c$')
ax4.legend(loc='best', fontsize=10, frameon=False)
save(fig4, f'Ma_us={Ma_us:.1f}_density_solution_P_us={P_us_bar:.1f}bar_T_us={T_us:.0f}K')
plt.close(fig4)
save_data_csv(sol.x, rho/rho_c, f'Ma_us={Ma_us:.1f}_density_solution_data_P_us={P_us_bar:.1f}bar_T_us={T_us:.0f}K')

# Combined Plots
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0, 0].plot(sol.x, Ma, 'b-', lw=2)
axs[0, 0].set_ylabel(r'\textrm{Ma}')
axs[0, 0].text(0.95, 0.95, rf'$\delta = {delta_xi:.2f} \lambda_{{us}}$', transform=axs[0, 0].transAxes, ha='right', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

axs[0, 1].plot(sol.x, T/T_c, 'g-', lw=2)
axs[0, 1].set_ylabel(r'$T/T_c$')

axs[1, 0].plot(sol.x, P_bar/P_c_bar, 'r-', lw=2)
axs[1, 0].set_ylabel(r'$P/P_c$')

axs[1, 1].plot(sol.x, rho/rho_c, 'm-', lw=2)
axs[1, 1].set_ylabel(r'$\rho/\rho_c$')
for ax in axs.flat:
    if ax.axison:
        ax.set_xlabel(r'$\xi^\star$ ($x/\lambda_{us}$)')
        ax.grid(False)
        ax.legend(loc='best', fontsize=10, frameon=False)
plt.tight_layout()
save(fig, f'Ma_us={Ma_us:.1f}_all_solutions_combined_P_us={P_us_bar:.1f}bar_T_us={T_us:.0f}K')
plt.show()
# plt.close(fig)









