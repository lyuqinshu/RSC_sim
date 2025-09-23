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




def apply_raman_sequence(
    mol_list,
    pulse_sequence,
    optical_pumping=True,
    print_report=False,
    rng=None,
    record_all=False
):
    """
    Apply a Raman sequence (with optional optical pumping) to a list of molecules.
    Tracks, at each step (including the initial pre-pulse state):
      • rate_survive: fraction of the initial eligible cohort with state==1 and spin==0
      • ground_state_rate: fraction of the initial eligible cohort in [0,0,0] AND eligible
      • n_bar: mean motional quanta per axis (x,y,z) over the initial eligible cohort
      • Bootstrap SEs for the two rates and for n_bar (per axis)

    Cohort:
      - The denominator for all rates and for n̄ is the FIXED initial eligible set:
        molecules with state==1 and spin==0 before any pulses.

    Parameters
    ----------
    mol_list : list[molecules]
        Each item exposes .n (np.array([nx,ny,nz])), .state (int), .spin (int),
        and methods Raman_transition(...) and Optical_pumping(...).
    pulse_sequence : list[[axis, delta_n, t]]
        Sequence of Raman pulses to apply.
    optical_pumping : bool
        Whether to call Optical_pumping() after each Raman pulse.
    print_report : bool
        Pass-through verbosity to molecule methods.
    rng : np.random.Generator | None
        Random generator for reproducibility.

    Returns
    -------
    n_bars : list[np.ndarray(shape=(3,))]
        Mean motional quanta (x,y,z) over the fixed cohort at each step.
    rate_survive : list[float]
        Survival fraction at each step.
    ground_state_rate : list[float]
        Ground-state fraction at each step.
    se_survive : list[float]
        Bootstrap SE of rate_survive at each step.
    se_ground : list[float]
        Bootstrap SE of ground_state_rate at each step.
    se_nbar : list[np.ndarray(shape=(3,))]
        Bootstrap SEs of n̄ per axis at each step.
    """
    n_boot = len(mol_list)  # number of bootstrap samples
    if rng is None:
        rng = np.random.default_rng()

    # Build fixed initial eligible cohort (indices into mol_list)
    initial_idxs = [i for i, mol in enumerate(mol_list) if mol.state == 1 and mol.spin == 0]
    N0 = len(initial_idxs)

    # -------- helpers over current mol_list restricted to initial cohort --------
    def current_arrays_over_cohort():
        """Return (surv_vec[0/1], gnd_vec[0/1], n_matrix[N0,3]) for current mol_list over initial cohort."""
        if N0 == 0:
            return np.array([], dtype=int), np.array([], dtype=int), np.empty((0, 3), dtype=float)

        surv = np.fromiter(
            (1 if (mol_list[i].state == 1 and mol_list[i].spin == 0) else 0 for i in initial_idxs),
            dtype=int, count=N0
        )
        gnd = np.fromiter(
            (1 if (mol_list[i].state == 1 and mol_list[i].spin == 0 and np.array_equal(mol_list[i].n, np.array([0,0,0]))) else 0
             for i in initial_idxs),
            dtype=int, count=N0
        )
        n_mat = np.vstack([mol_list[i].n for i in initial_idxs]).astype(float)  # shape (N0, 3)
        return surv, gnd, n_mat

    def bootstrap_rate_se(ind_vec):
        """Bootstrap SE of the mean of a 0/1 indicator vector over the cohort."""
        if N0 <= 1:
            return 0.0
        means = np.empty(n_boot, dtype=float)
        for b in range(n_boot):
            sample = rng.choice(ind_vec, size=N0, replace=True)
            means[b] = sample.mean()
        return float(means.std(ddof=1))

    def bootstrap_nbar_se(n_mat):
        """
        Bootstrap SE of mean per axis from n_mat shape (N0,3).
        Returns array of shape (3,) with SEs for x,y,z.
        """
        if N0 <= 1:
            return np.zeros(3, dtype=float)

        means = np.empty((n_boot, 3), dtype=float)
        for b in range(n_boot):
            idxs = rng.integers(0, N0, size=N0)
            sample = n_mat[idxs, :]        # (N0, 3)
            means[b, :] = sample.mean(axis=0)
        return means.std(axis=0, ddof=1)

    # ------------------------ storage ------------------------
    survive_rate = []
    ground_state_rate = []
    se_survive = []
    se_ground = []
    n_bars = []
    se_nbar = []

    # ------------------------ initial snapshot ------------------------
    surv_vec, gnd_vec, n_mat = current_arrays_over_cohort()

    if N0 == 0:
        # No eligible molecules initially: define all as zeros
        survive_rate.append(0.0)
        ground_state_rate.append(0.0)
        se_survive.append(0.0)
        se_ground.append(0.0)
        n_bars.append(np.array([0.0, 0.0, 0.0]))
        se_nbar.append(np.array([0.0, 0.0, 0.0]))
    else:
        survive_rate.append(float(surv_vec.mean()))
        ground_state_rate.append(float(gnd_vec.mean()))
        se_survive.append(bootstrap_rate_se(surv_vec))
        se_ground.append(bootstrap_rate_se(gnd_vec))
        n_bars.append(n_mat.mean(axis=0))
        se_nbar.append(bootstrap_nbar_se(n_mat))

    # ------------------------ apply pulses ------------------------
    i = 0
    for axis, delta_n, t in tqdm(pulse_sequence, desc="Applying pulses", total=len(pulse_sequence)):
        for mol in mol_list:
            mol.Raman_transition(axis=axis, delta_n=delta_n, time=t, print_report=print_report)
            if optical_pumping:
                mol.Optical_pumping(print_report=print_report)

        # Recompute stats over the same fixed cohort
        if record_all or i == len(pulse_sequence) - 1:
            surv_vec, gnd_vec, n_mat = current_arrays_over_cohort()
            if N0 == 0:
                survive_rate.append(0.0)
                ground_state_rate.append(0.0)
                se_survive.append(0.0)
                se_ground.append(0.0)
                n_bars.append(np.array([0.0, 0.0, 0.0]))
                se_nbar.append(np.array([0.0, 0.0, 0.0]))
            else:
                survive_rate.append(float(surv_vec.mean()))
                ground_state_rate.append(float(gnd_vec.mean()))
                se_survive.append(bootstrap_rate_se(surv_vec))
                se_ground.append(bootstrap_rate_se(gnd_vec))
                n_bars.append(n_mat.mean(axis=0))
                se_nbar.append(bootstrap_nbar_se(n_mat))
        i += 1

    return np.array(n_bars), np.array(survive_rate), np.array(ground_state_rate), np.array(se_nbar), np.array(se_survive), np.array(se_ground)



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





from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots in some setups

def get_n_distribution(mol_list, plot=True, scatter=True):
    """
    Get the distribution of vibrational quantum numbers (n) separately for the x, y, and z axes,
    and optionally plot histograms and a 3D scatter plot of the motional states.

    Parameters
    ----------
    mol_list : list of molecule objects
        Each object must have a .n attribute, which is a list or tuple [nx, ny, nz].
    plot : bool, optional
        If True, plot histograms of the n distributions (default: True).
    scatter : bool, optional
        If True, also plot a 3D scatter plot of (nx, ny, nz) for each molecule (default: True).

    Returns
    -------
    counts_x, counts_y, counts_z : Counter
        Frequency counts of n values for x, y, and z axes.
    """
    mol_num = 0
    # Collect n values
    n_x, n_y, n_z = [], [], []
    states = []
    for mol in mol_list:
        if mol.spin == 0 and mol.state == 1:
            n_vals = mol.n
            n_x.append(n_vals[0])
            n_y.append(n_vals[1])
            n_z.append(n_vals[2])
            states.append(n_vals)
            mol_num += 1

    # Count frequencies
    counts_x = Counter(n_x)
    counts_y = Counter(n_y)
    counts_z = Counter(n_z)

    if plot:
        # Histograms
        all_n = n_x + n_y + n_z
        n_min, n_max = min(all_n), max(all_n)

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

    if scatter and states:
        # 3D scatter plot
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        xs, ys, zs = zip(*states)

        ax.scatter(xs, ys, zs, c='purple', alpha=0.7, edgecolor='k')
        ax.set_xlabel("n_x")
        ax.set_ylabel("n_y")
        ax.set_zlabel("n_z")
        ax.set_title(f"3D Scatter of Motional States ({mol_num} molecules)")
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
