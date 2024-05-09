import numpy as np
import matplotlib.pyplot as plt

# Constants definition
kT = 0.026  # Thermal voltage at room temperature in eV
q = 1.6e-19  # Elementary charge in Coulombs
NA = 1e15  # Acceptor concentration for p-type in cm^-3
ND = 1e15  # Donor concentration for n-type in cm^-3
ni = 1e10  # Intrinsic carrier concentration in cm^-3
epsilon0 = 8.85e-14  # Permittivity of free space in F/cm
epsilonr = 11.7  # Relative permittivity for Silicon
epsilon = epsilon0 * epsilonr  # Total permittivity of Silicon in F/cm
Vt = kT * q  # Thermal voltage in Joules
phi_b_p = Vt * np.log(NA / ni) / q  # Built-in potential for p-type in Volts
phi_b_n = Vt * np.log(ND / ni) / q  # Built-in potential for n-type in Volts

# Charge density calculation for p-type silicon
def rho_p(V):
    p0 = NA
    px = p0 * np.exp(-V / kT)
    return q * ((px - p0) + ((ni**2) / NA) * (1 - np.exp(V / kT)))

# Charge density calculation for n-type silicon
def rho_n(V):
    n0 = ND
    nx = n0 * np.exp(V / kT)
    return q * ((nx - n0) + ((ni**2) / ND) * (1 - np.exp(-V / kT)))

# Voltage range setup for p-type and n-type
v_range_p = np.linspace(-0.2, 1, 500)
v_range_n = np.linspace(-1, 0.2, 500)

# Compute charge densities for both types
rho_p_range = rho_p(v_range_p)
rho_n_range = rho_n(v_range_n)

# Plotting for p-type silicon
plt.figure(1)
plt.semilogy(v_range_p, abs(rho_p_range), label='p-type', color='blue')
plt.ylim(1e-6, 1e3)
plt.xlabel('Surface Potential $Ψ_S$ (V)', fontsize=12)
plt.ylabel('Charge Density $|ρ|$ (C/cm³)', fontsize=12)
plt.title('Charge Density vs. Surface Potential for p-type Silicon', fontsize=14)

# Vertical lines to indicate critical potentials
plt.axvline(x=0, color='black', linestyle='--', label='Flat band')
plt.axvline(x=2 * phi_b_p, color='green', linestyle='--', label='Built-in potential (2$Ψ_B$)')
plt.axvline(x=phi_b_p, color='pink', linestyle='--', label='Built-in potential ($Ψ_B$)')

# Annotations for different regions
plt.annotate(
    '', 
    xy=(0, 1e-5), xycoords='data',
    xytext=(phi_b_p, 1e-5), textcoords='data',
    arrowprops=dict(arrowstyle='<->', lw=1.5)
)
plt.text(
    0.15, 1.2e-5, 'Depletion', 
    ha='center', va='bottom', 
    fontsize=10
)
plt.annotate(
    '', 
    xy=(phi_b_p, 1e-5), xycoords='data',
    xytext=(2 * phi_b_p, 1e-5), textcoords='data',
    arrowprops=dict(arrowstyle='<->', lw=1.5)
)
plt.text(
    phi_b_p + 0.15, 1.2e-5, 'Weak inversion', 
    ha='center', va='bottom', 
    fontsize=10
)
plt.annotate(
    '', 
    xy=(2 * phi_b_p, 1e-5), xycoords='data',
    xytext=(1, 1e-5), textcoords='data',
    arrowprops=dict(arrowstyle='<->', lw=1.5)
)
plt.text(
    2 * phi_b_p + 0.2, 1.2e-5, 'Strong inversion', 
    ha='center', va='bottom', 
    fontsize=10
)
plt.annotate(
    '', 
    xy=(-0.25, 1e-5), xycoords='data',
    xytext=(0, 1e-5), textcoords='data',
    arrowprops=dict(arrowstyle='<->', lw=1.5)
)
plt.text(
    -0.13, 1.2e-5, 'Accumulation', 
    ha='center', va='bottom', 
    fontsize=10
)
plt.annotate(
    '', 
    xy=(0, 1e-1), xycoords='data',
    xytext=(2 * phi_b_p, 1e-1), textcoords='data',
    arrowprops=dict(arrowstyle='<->', lw=1.5)
)
plt.text(
    0.25, 1.2e-1, '$2\Psi_B$', 
    ha='center', va='bottom', 
    fontsize=10
)

plt.annotate(
    '', 
    xy=(0, 1e-2), xycoords='data',
    xytext=(phi_b_p, 1e-2), textcoords='data',
    arrowprops=dict(arrowstyle='<->', lw=1.5)
)
plt.text(
    0.18, 1.2e-2, '$\Psi_B$', 
    ha='center', va='bottom', 
    fontsize=10
)


plt.text(-0.13, 3e-1, r'$\sim \exp\left(\frac{q \times |\Psi_S|}{2kT}\right)$', ha='center', va='bottom', fontsize=12)
plt.text(0.4, 2e-4, r'$\sim \sqrt{\Psi_S}$', ha='center', va='bottom', fontsize=12)
plt.text(0.9, 3e-1, r'$\sim \exp\left(\frac{q \times \Psi_S}{2kT}\right)$', ha='center', va='bottom', fontsize=12)

plt.text(0.1, 1.9e2, 'p-type Si (300 k)', ha='center', va='bottom', fontsize=10)
plt.text(0.1, 1e2, '$N_a = 1 \\times 10^{15} cm^{-3}$', ha='center', va='bottom', fontsize=10)


plt.legend()
plt.grid(True)
plt.show()



# Plotting for n-type silicon
plt.figure(2)
plt.semilogy(v_range_n, abs(rho_n_range), label='n-type', color='red')
plt.ylim(1e-6, 1e3)
plt.xlabel('Surface Potential $Ψ_S$ (V)', fontsize=12)
plt.ylabel('Charge Density $|ρ|$ (C/cm³)', fontsize=12)
plt.title('Charge Density vs. Surface Potential for n-type Silicon', fontsize=14)

# Vertical lines to indicate critical potentials
plt.axvline(x=0, color='black', linestyle='--', label='Flat band')
plt.axvline(x=-2 * phi_b_n, color='green', linestyle='--', label='Built-in potential (2$Ψ_B$)')
plt.axvline(x=-phi_b_n, color='pink', linestyle='--', label='Built-in potential ($Ψ_B$)')

# Annotations for different regions
plt.annotate(
    '', 
    xy=(0, 1e-5), xycoords='data',
    xytext=(-phi_b_n, 1e-5), textcoords='data',
    arrowprops=dict(arrowstyle='<->', lw=1.5)
)
plt.text(
    -0.15, 1.2e-5, 'Depletion', 
    ha='center', va='bottom', 
    fontsize=10
)
plt.annotate(
    '', 
    xy=(-phi_b_n, 1e-5), xycoords='data',
    xytext=(-2 * phi_b_n, 1e-5), textcoords='data',
    arrowprops=dict(arrowstyle='<->', lw=1.5)
)
plt.text(
    -phi_b_n - 0.15, 1.2e-5, 'Weak inversion', 
    ha='center', va='bottom', 
    fontsize=10
)
plt.annotate(
    '', 
    xy=(-2 * phi_b_n, 1e-5), xycoords='data',
    xytext=(-1, 1e-5), textcoords='data',
    arrowprops=dict(arrowstyle='<->', lw=1.5)
)
plt.text(
    -2 * phi_b_p - 0.2, 1.2e-5, 'Strong inversion', 
    ha='center', va='bottom', 
    fontsize=10
)
plt.annotate(
    '', 
    xy=(0.25, 1e-5), xycoords='data',
    xytext=(0, 1e-5), textcoords='data',
    arrowprops=dict(arrowstyle='<->', lw=1.5)
)
plt.text(
    0.13, 1.2e-5, 'Accumulation', 
    ha='center', va='bottom', 
    fontsize=10
)
plt.annotate(
    '', 
    xy=(0, 1e-1), xycoords='data',
    xytext=(-2 * phi_b_p, 1e-1), textcoords='data',
    arrowprops=dict(arrowstyle='<->', lw=1.5)
)
plt.text(
    -0.25, 1.2e-1, '$2\Psi_B$', 
    ha='center', va='bottom', 
    fontsize=10
)

plt.annotate(
    '', 
    xy=(0, 1e-2), xycoords='data',
    xytext=(-phi_b_p, 1e-2), textcoords='data',
    arrowprops=dict(arrowstyle='<->', lw=1.5)
)
plt.text(
    -0.18, 1.2e-2, '$\Psi_B$', 
    ha='center', va='bottom', 
    fontsize=10
)

plt.text(0.13, 3e-1, r'$\sim \exp\left(\frac{q \times |\Psi_S|}{2kT}\right)$', ha='center', va='bottom', fontsize=12)
plt.text(-0.4, 2e-4, r'$\sim \sqrt{\Psi_S}$', ha='center', va='bottom', fontsize=12)
plt.text(-0.9, 3e-1, r'$\sim \exp\left(\frac{q \times \Psi_S}{2kT}\right)$', ha='center', va='bottom', fontsize=12)

plt.text(-0.1, 1.9e2, 'n-type Si (300 k)', ha='center', va='bottom', fontsize=10)
plt.text(-0.1, 1e2, '$N_d = 1 \\times 10^{15} cm^{-3}$', ha='center', va='bottom', fontsize=10)

plt.legend()
plt.grid(True)
plt.show()
