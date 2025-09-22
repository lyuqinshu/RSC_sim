import numpy as np
import scipy.constants as cts
import scipy.stats as stats
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from scipy.special import hermite, factorial

from scipy.special import genlaguerre

# Meta parameters
amu = 1.66053906660e-27  # kg
mass = 59 * amu
trap_freq = np.array([75e3 * 2 * np.pi, 65e3 * 2 * np.pi, 13.6e3 * 2 * np.pi])  # trap frequency for x, y, z in rad/s
k_vec = 2 * np.pi / 531e-9  # wavevector of 531 nm light
decay_ratio = [1/3, 1/3, 1/3]  # branching ratio for mN = -1, 0, 1
branch_ratio = 0.0064 # barnching ratio of going to a different spin manifold

# angle [theta, phi] of the optical pumping light
angle_pump_sigma=[np.pi, 0.] 
angle_pump_pi=[np.pi/2, -np.pi/4]
LD_raman=[0.57, 0.61, 0.62]
# LD_raman=[0.6, 0.6, 0.6]
n_basis = np.arange(0, 40)

import importlib.resources as resources
pkg = "RSC_sim"  # your top-level package name
rel = "data/M_FACTOR_TABLE.npy"

with resources.files(pkg).joinpath(rel).open("rb") as f:
    M_FACTOR_TABLE = np.load(f, allow_pickle=False)



# Experiment parameter for RSC
# Use experiment data and expand 3 quanta for radial, 4 for axial

amp_matrix = {
    "0": [0.92],
    "X": [0.3, 0.65, 0.65, 0.7, 0.7, 0.85],
    "Y": [0.3, 0.65, 0.65, 0.7, 0.7, 0.85],
    "Z": [0.14, 0.14, 0.14, 0.28, 0.28, 0.35, 0.35, 0.4, 0.4]
}


duration_matrix = {
    "OP": [8e-5],
    "CO": [1e-4],
    "X": [5e-5, 7e-5, 7e-5, 9e-5, 9e-5, 11e-5],
    "Y": [5e-5, 7e-5, 7e-5, 9e-5, 9e-5, 11e-5],
    "Z": [2e-4, 2e-4, 2e-4, 5e-5, 5e-5, 7e-5, 7e-5, 9e-5, 9e-5]
}


def generalized_laguerre(alpha, n, x):
    L = genlaguerre(n, alpha)
    return L(x)

def M_factor_lookup(n_initial, n_final, ld):
    ld_index = int(round(np.abs(ld) / 0.0001))
    ld_index = min(ld_index, M_FACTOR_TABLE.shape[2] - 1)  # Clamp to max index
    return M_FACTOR_TABLE[n_initial, n_final, ld_index]


def M_factor(n1, n2, ita=0.57):
    """
    Calculate the M factor for the Rabi frequency of the Raman transition between states n1 and n2
    with Lamb-Dicke parameter ita.

    Parameters:
    - n1 (int): Initial quantum number
    - n2 (int): Final quantum number
    - ita (float): Lamb-Dicke parameter

    Returns:
    - M (float): The M factor for the transition
    """
    if n2 >= n1:
        delta_n = n2 - n1
        prefactor = np.sqrt(factorial(n1) / factorial(n2)) * ita**delta_n
        laguerre_poly = generalized_laguerre(delta_n, n1, ita**2)
    else:
        delta_n = n1 - n2
        prefactor = np.sqrt(factorial(n2) / factorial(n1)) * ita**delta_n
        laguerre_poly = generalized_laguerre(delta_n, n2, ita**2)

    M = prefactor * np.exp(-ita**2 / 2) * laguerre_poly
    return M

def LD_par_angle(LD0, angle_pump, theta_scatter):
    """
    Compute the effective Lamb-Dicke parameter for a given spontaneous emission angle.

    Parameters:
    - LD0 (float): Base Lamb-Dicke parameter (|Δk| * z0)
    - angle_pump (float): Angle of incoming photon (theta_pump) in radians
    - theta_scatter (float or ndarray): Polar angle of scattered photon (radians)
    - phi_scatter: (unused, included for interface compatibility)

    Returns:
    - eta_eff (float or ndarray): Effective Lamb-Dicke parameter
    """
    delta_kz = np.abs(np.cos(angle_pump) - np.cos(theta_scatter))
    return LD0 * delta_kz


def Delta_k(angle_pump, angle_scatter):
    """
    Compute 3D vector Δk = k_pump - k_scatter.

    Parameters:
    - angle_pump (list): [theta_pump, phi_pump] in radians
    - angle_scatter (list): [theta_scatter, phi_scatter] in radians

    Returns:
    - [Δk_x, Δk_y, Δk_z] (list of floats): Δk along x, y, z
    """
    theta_p, phi_p = angle_pump
    theta_s, phi_s = angle_scatter

    # Unit vectors for pump and scatter directions
    k_pump = np.array([
        np.sin(theta_p) * np.cos(phi_p),
        np.sin(theta_p) * np.sin(phi_p),
        np.cos(theta_p)
    ])
    k_scatter = np.array([
        np.sin(theta_s) * np.cos(phi_s),
        np.sin(theta_s) * np.sin(phi_s),
        np.cos(theta_s)
    ])

    delta_k = k_pump - k_scatter

    return (k_vec * delta_k).tolist()



def OP_prob(n1, n2, LD0=0.57, angle_pump=np.pi/4, N_theta=100):
    """
    Calculate the optical pumping transition probability between states n1 and n2,
    averaging over spontaneous emission polar angle θ only (assuming azimuthal symmetry).

    Parameters:
    - n1, n2 (int): Quantum numbers
    - LD0 (float): Base Lamb-Dicke parameter (Delta_k * z0)
    - angle_pump (float): Pump beam polar angle (radians)
    - N_theta (int): Number of θ points to sample

    Returns:
    - P (float): Solid-angle averaged transition probability
    """
    theta = np.linspace(0, np.pi, N_theta)
    dtheta = theta[1] - theta[0]

    P_total = 0.0
    for t in theta:
        LD = LD_par_angle(LD0, angle_pump, t)
        P = M_factor(n1, n2, LD)**2
        weight = np.sin(t) * dtheta * 2 * np.pi  # integrate φ analytically
        P_total += P * weight

    return P_total / (4 * np.pi)

def convert_to_LD(dK, trap_f):
    """
    Convert trap frequency to Lamb-Dicke parameter.
    ita = delta_k * x0

    Parameters:
    - dK (float): Delta k in SI
    - trap_f (float): Trap frequencies rad/s

    Returns:
    - ita (float): Lamb-Dicke parameter
    """
    hbar = 1.054571817e-34  # J·s
    x0 = np.sqrt(hbar / (2 * mass * trap_f))
    ita = x0 * dK
    return ita


class molecules:
    def __init__(self, state=1, n=np.array([10, 10, 20]), spin=0, branch_ratio=branch_ratio):
        """
        Initialize the molecules class.

        Parameters:
        - state (int): Initial mN state (-1, 0, 1)
        - n (list): Initial quantum numbers for x, y, z axis
        - spin (int): Initial spin manifold, 0 for (mS, mI) = (-1/2, -1/2), 1 for other
        """
        self.state = state
        self.n = n
        self.spin = spin
        self.branch_ratio = branch_ratio

    def Raman_transition(self, axis=0, delta_n=-1, time=1., print_report=True):
        """
        Perform a Raman transition along a specified axis.

        Parameters:
        - axis (int): Axis index (0 for x, 1 for y, 2 for z)
        - delta_n (int): Change in quantum number (e.g., -1 for cooling, +1 for heating)
        - time (float): Duration of the Raman pulse (in the unite of 1/Ω0)

        Returns:
        - success (bool): True if the transition was successful, False otherwise
        """

        if print_report:
            print(f"Before cooling, motinoal state: {self.n}, internal state {self.state}")
        n_initial = self.n[axis]
        n_final = n_initial + delta_n

        # Fail if n_final is smaller than 0
        if n_final < 0:
            return 2
        
        # Fail if not start in mN = 1 state
        if self.state != 1:
            return 3
        
        # Fail if not in the right spin manofild
        if self.spin != 0:
            return 4
        
        # Calculate the probability of the Raman transition
        # prob = np.sin(M_factor(n_initial, n_final, LD_raman[axis])*time/2)**2
        prob = np.sin(M_factor_lookup(n_initial, n_final, LD_raman[axis])*time/2)**2

        # Randomly determine if the transition is successful
        success = random.random() < prob

        if success:
            self.n[axis] = n_final
            self.state = -1
            if print_report:
                print(f"Cooling success, motinoal state: {self.n}, internal state {self.state}")
            return 0
        if print_report:
                print(f"Cooling fail, motinoal state: {self.n}, internal state {self.state}")
        return 1
    
    def Optical_pumping(self, print_report=True):
        """
        Perform optical pumping to return the molecule to mN = 1 state.

        Parameters:

        Returns:
        - None
        """
        pump_cycle = 0
        if self.state == 1:
            return pump_cycle
        
        while self.state != 1:
            if self.spin != 0:
                break
            # Choose random angle for spontaneous emission [theta, phi]
            scatter_angle = [np.pi*np.random.random(), 2*np.pi*np.random.random()]
            if self.state == -1:
                pump_angle = angle_pump_sigma
            else:
                pump_angle = angle_pump_pi
            dK = Delta_k(pump_angle, scatter_angle)

            # Check heating in all axis
            for axis in [0, 1, 2]:
                n_initial = self.n[axis]
                # Calculate transition probabilities for all possible n_final, replace this later by a lookup table
                probs = []
                ld = convert_to_LD(dK[axis], trap_freq[axis])
                # print(f'LD par at axis {axis} = {ld:.3f}')
                for n_final in n_basis:
                    # Probability of the transition is propotional to rabi_freq**2
                    rabi_freq = M_factor_lookup(n_initial, n_final, ld)
                    # rabi_freq = M_factor(n_initial, n_final, ld)
                    prob = rabi_freq**2
                    probs.append(prob)
                probs = np.array(probs)
                probs /= probs.sum()
                # formatted_probs = [f"{prob:.3f}" for prob in probs]
                # print(f'Transition probabilities: {formatted_probs}')
                n_final = np.random.choice(list(n_basis), p=probs)
                self.n[axis] = n_final
            # Randomly set mN state according to decay_ratio
            self.state = np.random.choice([-1, 0, 1], p=decay_ratio)
            # Randomly set spin manifold according to branch_ratio
            if random.random() < self.branch_ratio:
                self.spin = 1
            pump_cycle += 1
            if print_report:
                formatted_angle = [f"{angle:.3f}" for angle in scatter_angle]
                formatted_dk = [f"{dk/k_vec:.3f}" for dk in dK]
                dk_norm = np.linalg.norm(np.array(dK)) / k_vec

                print(
                    f"After OP # {pump_cycle}, "
                    f"photon scatter at {formatted_angle}, "
                    f"|Δk|/k = {dk_norm:.3f}, "
                    f"Δk/k = {formatted_dk}, "
                    f"motional quanta {self.n}, "
                    f"pump to state {self.state}"
                )

        if print_report:
            print(f"Success after {pump_cycle} OP cycles")
        return pump_cycle
        

import numpy as np

def cost_function(mol_list):
    """
    Compute statistics of vibrational states for surviving molecules.

    Parameters:
    - mol_list (list): List of molecule objects with .state, .spin, and .n attributes.

    Returns:
    - total_num (int): Number of surviving molecules (state=1, spin=0).
    - n_bar (list of float): Average n along [x, y, z] axes.
    - sem_n (list of float): Standard error of the mean for n along [x, y, z] axes.
    """
    surviving_n = []

    for mol in mol_list:
        if getattr(mol, 'state', None) == 1 and getattr(mol, 'spin', None) == 0:
            surviving_n.append(mol.n)

    surviving_n = np.array(surviving_n)  # shape: (num_survivors, 3)
    total_num = len(surviving_n)

    if total_num > 0:
        n_bar = np.mean(surviving_n, axis=0).tolist()
        sem_n = (np.std(surviving_n, axis=0, ddof=1) / np.sqrt(total_num)).tolist()
    else:
        n_bar = [np.nan, np.nan, np.nan]
        sem_n = [np.nan, np.nan, np.nan]

    return total_num, n_bar, sem_n



    

def initialize_thermal(temp, n, n_max=max(n_basis), branch_ratio=branch_ratio):
    """
    Initialize a list of molecules with motional quantum states sampled from
    a Boltzmann distribution at temperature `temp`.

    Parameters:
    - temp (list): List of temperatures in Kelvin at three axes.
    - n (int): Number of molecules to initialize.
    - n_max (int): Maximum quantum number to consider in the distribution.

    Returns:
    - mol_list (list of molecules): List of initialized molecule objects.
    """
    k_B = cts.k
    hbar = cts.hbar

    mol_list = []
    ns = np.arange(n_max)

    for _ in range(n):
        n_thermal = []
        for i, omega in enumerate(trap_freq):
            energies = (ns + 0.5) * hbar * omega
            probs = np.exp(-energies / (k_B * temp[i]))
            probs /= probs.sum()  # normalize
            sampled_n = np.random.choice(ns, p=probs)
            n_thermal.append(sampled_n)
        mol = molecules(state=1, n=n_thermal, branch_ratio=branch_ratio)
        mol_list.append(mol)

    return mol_list



def apply_raman_sequence(mol_list, pulse_sequence, optical_pumping=True, print_report=False):
    """
    Apply a sequence of Raman transitions followed by optical pumping to a list of molecules,
    one pulse at a time. After each pulse, count how many molecules are in motional ground state [0,0,0].

    Parameters:
    - mol_list (list of molecules): List of molecule instances.
    - pulse_sequence (list of [axis, delta_n, t]): Raman pulse sequence.
    - print_report (bool): Whether to print output during transitions.

    Returns:
    - n_bars (list of list): Average motional number at three axis.
    - num_survive (list of int): Total number of molecules survived after each pulse.
    - ground_state_counts (list of int): Number of molecules in motional ground state after each pulse.
    - cost (float): Return the cost function which evaluates the cooling.
    """
    num_survive = []
    ground_state_counts = []
    sems = []
    n_bars = []

    # Append for initial molecules
    ground_count = np.sum([mol.n == [0, 0, 0] for mol in mol_list if mol.state==1 and mol.spin==0])
    ground_state_counts.append(ground_count)
    n, n_bar, sem = cost_function(mol_list)
    sems.append(sem)
    num_survive.append(n)
    n_bars.append(n_bar)

    for pulse_index, (axis, delta_n, t) in tqdm(enumerate(pulse_sequence), total=len(pulse_sequence), desc="Applying pulses"):
        for i, mol in enumerate(mol_list):
            mol.Raman_transition(axis=axis, delta_n=delta_n, time=t, print_report=print_report)
            if optical_pumping:
                mol.Optical_pumping(print_report=print_report)

        # Count molecules in ground state after this pulse
        ground_count = np.sum([mol.n == [0, 0, 0] for mol in mol_list if mol.state==1 and mol.spin==0])
        ground_state_counts.append(ground_count)
        n, n_bar, sem = cost_function(mol_list)
        sems.append(sem)
        num_survive.append(n)
        n_bars.append(n_bar)

    return n_bars, num_survive, ground_state_counts, sems


def readout_molecule_properties(mol_list, trap_freq=trap_freq, n_max_fit=max(n_basis)):
    """
    Analyze motional states from a list of molecules that survived and fit effective temperatures.

    Parameters:
    -----------
    mol_list : list
        List of molecule objects to analyze. Each molecule must have attributes:
        - n : list or array-like of [nx, ny, nz]
        - spin : integer
    trap_freq : ndarray
        Trap frequencies [ω_x, ω_y, ω_z] in rad/s.
    n_max_fit : int
        Maximum quantum number to include in fit (currently unused in this function).

    Returns:
    --------
    states_x, states_y, states_z : ndarray
        Arrays of motional quantum numbers along x, y, and z for spin=0 molecules.
    avg_n : list of float
        Average motional quantum number along [x, y, z] axes.
    grd_n : int
        Number of molecules in the motional ground state (n=[0,0,0], spin=0).
    """
    kB = cts.k
    hbar = cts.hbar

    # Filter spin=0 molecules
    mol_spin0 = [mol for mol in mol_list if mol.spin == 0]

    # Extract motional quantum numbers
    states_x = np.array([mol.n[0] for mol in mol_spin0])
    states_y = np.array([mol.n[1] for mol in mol_spin0])
    states_z = np.array([mol.n[2] for mol in mol_spin0])

    # Average motional excitation per axis
    avg_n = [np.mean(states_x), np.mean(states_y), np.mean(states_z)]

    # Count motional ground state molecules (n = [0,0,0])
    grd_n = sum(1 for mol in mol_spin0 if mol.n[0] == 0 and mol.n[1] == 0 and mol.n[2] == 0)

    return states_x, states_y, states_z, avg_n, grd_n




# RSC pulse duration

scaling_x = np.pi/0.3/(amp_matrix['X'][-2]*duration_matrix['X'][-2])
scaling_y = np.pi/0.3/(amp_matrix['Y'][-2]*duration_matrix['Y'][-2])
scaling_z = np.pi/0.3/(amp_matrix['Z'][-2]*duration_matrix['Z'][-2])

def pulse_time(axis, delta_n):
    if axis==0:
        return scaling_x*amp_matrix['X'][-delta_n-1]*duration_matrix['X'][-delta_n-1]
    if axis==1:
        return scaling_y*amp_matrix['Y'][-delta_n-1]*duration_matrix['Y'][-delta_n-1]
    else:
        return scaling_z*amp_matrix['Z'][-delta_n-1]*duration_matrix['Z'][-delta_n-1]




from collections import Counter


import matplotlib.pyplot as plt
from collections import Counter

def plot_n_distribution(mol_list):
    """
    Plot histograms showing the distribution of vibrational quantum numbers (n)
    separately for the x, y, and z axes in the given list of molecules.

    Parameters:
    -----------
    mol_list : list of molecule objects
        Each object must have a .n attribute, which is a list or tuple [nx, ny, nz].
    title : str, optional
        Title of the entire figure (default is "Histogram of n Distribution by Axis").
    """
    mol_num = 0
    # Split n values by axis
    n_x, n_y, n_z = [], [], []
    for mol in mol_list:
        if mol.spin == 0:
            if mol.state == 1:
                n_vals = mol.n
                n_x.append(n_vals[0])
                n_y.append(n_vals[1])
                n_z.append(n_vals[2])
                mol_num += 1

    # Count frequencies
    counts_x = Counter(n_x)
    counts_y = Counter(n_y)
    counts_z = Counter(n_z)

    # Determine global x-axis range
    all_n = n_x + n_y + n_z
    n_min, n_max = min(all_n), max(all_n)

    # Plot side-by-side histograms
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    axes[0].bar(counts_x.keys(), counts_x.values(), color='salmon', edgecolor='black')
    axes[0].set_title("n Distribution (X axis)")
    axes[1].bar(counts_y.keys(), counts_y.values(), color='mediumseagreen', edgecolor='black')
    axes[1].set_title("n Distribution (Y axis)")
    axes[2].bar(counts_z.keys(), counts_z.values(), color='cornflowerblue', edgecolor='black')
    axes[2].set_title("n Distribution (Z axis)")

    for ax in axes:
        ax.set_xlabel("n")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xticks(range(n_min, n_max + 1, 5))

    axes[0].set_ylabel("Count")
    fig.suptitle(f'{mol_num} molecules survived')
    plt.tight_layout()
    plt.show()

    return counts_x, counts_y, counts_z

def plot_time_sequence_data(n_bar, num_survive, ground_state_count, sem):

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))

    # Plot 1: Ground state count
    axs[0].plot(range(len(ground_state_count)), ground_state_count, marker='o')
    axs[0].set_title("3D Ground State Count")
    axs[0].set_xlabel("Pulse #")
    axs[0].set_ylabel("# in [0,0,0]")
    axs[0].grid(True)

    # Plot 2: Standard error
    for i in [0, 1, 2]:
        axs[1].plot(range(len(sem)), np.array(sem)[:, i], marker='o', label=f'axis {i}')
    axs[1].set_title("Standard error")
    axs[1].set_xlabel("Pulse #")
    axs[1].set_ylabel("Standard error")
    axs[1].grid(True)

    # Plot 3: Molecules Survived
    axs[2].plot(range(len(num_survive)), num_survive, marker='o')
    axs[2].set_title("Surviving Molecules")
    axs[2].set_xlabel("Pulse #")
    axs[2].set_ylabel("Survivors")
    axs[2].grid(True)

    # Plot 4: Average n per axis
    for i in [0, 1, 2]:
        axs[3].plot(range(len(n_bar)), np.array(n_bar)[:, i], marker='o', label=f'axis {i}')
    axs[3].set_title("Avg. Motional n")
    axs[3].set_xlabel("Pulse #")
    axs[3].set_ylabel("⟨n⟩")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()
