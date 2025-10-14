import numpy as np
import scipy.constants as cts
import scipy.stats as stats
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from scipy.special import hermite, factorial
import json
from scipy.special import genlaguerre
from .generate_M import precompute_M_factors_parallel
from importlib.resources import files, as_file
from pathlib import Path
from multiprocessing import current_process

_pkg = "RSC_sim"
_rel = "data/M_FACTOR_TABLE.npy"

# ---- Load simulation parameters from config.json ----
with open(files(_pkg).joinpath("config.json"), "r") as f:
    _cfg = json.load(f)

amu = 1.66053906660e-27  # kg
mass = float(_cfg["mass"]*amu)
trap_freq = np.array(_cfg["trap_freq"], dtype=float)*2*np.pi # trap frequency for x, y, z in rad/s     
k_vec = float(2 * np.pi / _cfg["lambda"]) # wavevector of 531 nm light                        
decay_ratio = list(map(float, _cfg["decay_ratio"])) # branching ratio for mN = -1, 0, 1
branch_ratio = float(_cfg["branch_ratio"]) # barnching ratio of going to a different spin manifold
trap_depth = float(_cfg["trap_depth"] * cts.k) # trap depth in J
max_n = list(map(int, _cfg["max_n"])) # max n to consider for each axis
LD_RES = float(_cfg["LD_RES"]) # resolution of LD parameter in the lookup table

# angle [theta, phi] of the optical pumping light
angle_pump_sigma = list(map(float, _cfg["angle_pump_sigma"]))
angle_pump_pi    = list(map(float, _cfg["angle_pump_pi"]))

LD_raman = list(map(float, _cfg["LD_raman"]))

n_basis = [np.arange(0, n) for n in max_n]
# trap_freq is in rad/s; convert to Hz via /(2π) for the formula
n_limit = [int(trap_depth / (cts.h * f / (2 * np.pi))) for f in trap_freq]



def _compute_m_table(target_path: Path):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        precompute_M_factors_parallel(workers=1)


def _load_m_table() -> np.ndarray:
    """Load the M-factor table, computing it if it doesn't exist yet."""
    tgt = files(_pkg).joinpath(_rel)
    try:
        with tgt.open("rb") as f:
            return np.load(f, allow_pickle=False)
    except FileNotFoundError:
        print("M_FACTOR_TABLE.npy not found; computing it now...")
        with as_file(tgt) as p:
            p = Path(p)
            _compute_m_table(p)
            with p.open("rb") as f:
                return np.load(f, allow_pickle=False)

# Public: the lookup table
M_FACTOR_TABLE = _load_m_table()

def generalized_laguerre(alpha, n, x):
    L = genlaguerre(n, alpha)
    return L(x)

def M_factor_lookup(n_initial, n_final, ld):
    ld_index = int(round(np.abs(ld) / LD_RES))
    ld_index = min(ld_index, M_FACTOR_TABLE.shape[2] - 1)  # Clamp to max index
    return M_FACTOR_TABLE[n_initial, n_final, ld_index]


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
        - branch_ratio (float): Branching ratio to a different spin manifold during optical pumping
        - islost (bool): Whether the molecule is lost (default: False)
        """
        self.state = state
        self.n = n
        self.spin = spin
        self.branch_ratio = branch_ratio
        self.islost = False

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
        
        # Fail if the molecule is lost
        if self.islost:
            return 5
        
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
        if self.islost:
                return pump_cycle
        
        while self.state != 1:
            if self.spin != 0:
                break
            if self.islost:
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
                for n_final in n_basis[axis]:
                    # Probability of the transition is propotional to rabi_freq**2
                    rabi_freq = M_factor_lookup(n_initial, n_final, ld)
                    # rabi_freq = M_factor(n_initial, n_final, ld)
                    prob = rabi_freq**2
                    probs.append(prob)
                probs = np.array(probs)
                probs /= probs.sum()
                # formatted_probs = [f"{prob:.3f}" for prob in probs]
                # print(f'Transition probabilities: {formatted_probs}')
                n_final = np.random.choice(list(n_basis[axis]), p=probs)
                self.n[axis] = n_final

                # check if the molecule is lost
                if n_final >= n_limit[axis]:
                    self.islost = True
                    if print_report:
                        print(f"Molecule lost after OP, motional state {self.n}, internal state {self.state}")

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
        

    

def initialize_thermal(temp, n, branch_ratio=branch_ratio):
    """
    Initialize a list of molecules with motional quantum states sampled from
    a Boltzmann distribution at temperature `temp`.

    Parameters:
    - temp (list): List of temperatures in Kelvin at three axes.
    - n (int): Number of molecules to initialize.

    Returns:
    - mol_list (list of molecules): List of initialized molecule objects.
    """
    k_B = cts.k
    hbar = cts.hbar

    mol_list = []
    ns = []
    for i, n_m in enumerate(trap_freq):
        ns.append(np.arange(n_limit[i]+1))

    for _ in range(n):
        n_thermal = []
        for i, omega in enumerate(trap_freq):
            energies = (ns[i] + 0.5) * hbar * omega
            probs = np.exp(-energies / (k_B * temp[i]))
            probs /= probs.sum()  # normalize
            sampled_n = np.random.choice(ns[i], p=probs)
            n_thermal.append(sampled_n)
        mol = molecules(state=1, n=n_thermal, branch_ratio=branch_ratio)
        mol_list.append(mol)

    return mol_list




import os, time, random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

# ---------- worker init & helpers ----------

def _init_worker(base_seed=None):
    pid = os.getpid()
    seed = (int(base_seed) if base_seed is not None else int(time.time())) ^ (pid & 0xFFFFFFFF)
    np.random.seed(seed); random.seed(seed)

def _snapshot_for_molecule(mol):
    eligible = (mol.state == 1) and (mol.spin == 0) and (not getattr(mol, "islost", False))
    surv = 1 if eligible else 0
    gnd  = 1 if (eligible and np.array_equal(np.array(mol.n), np.array([0,0,0]))) else 0
    n_vec = np.array(mol.n, dtype=float)
    return surv, gnd, n_vec

def _run_one_molecule_all_pulses_with_index(
    idx, mol, pulse_sequence, optical_pumping,
    print_report_in_worker, include_initial_snapshot, record_all
):
    """Run all pulses for one molecule and return (idx, surv(T,), gnd(T,), n_mat(T,3), UPDATED mol)."""
    surv_list, gnd_list, n_list = [], [], []

    # Initial snapshot
    if include_initial_snapshot:
        s, g, n = _snapshot_for_molecule(mol)
        surv_list.append(s); gnd_list.append(g); n_list.append(n)

    # Run through all pulses
    for pi, (axis, delta_n, t) in enumerate(pulse_sequence):
        mol.Raman_transition(axis=int(axis), delta_n=int(delta_n), time=float(t),
                             print_report=print_report_in_worker)
        if optical_pumping:
            mol.Optical_pumping(print_report=print_report_in_worker)

        if record_all or (pi == len(pulse_sequence) - 1):
            # Record either every pulse or just the last one
            s, g, n = _snapshot_for_molecule(mol)
            surv_list.append(s); gnd_list.append(g); n_list.append(n)

    return (
        idx,
        np.asarray(surv_list, dtype=float),
        np.asarray(gnd_list, dtype=float),
        np.vstack(n_list).astype(float),
        mol,   # return the updated molecule
    )

# ---------- parallel bootstrap helpers (per timestep mapping) ----------

def _rate_se_for_column(args):
    """Compute bootstrap SE for a single time column of a 0/1 indicator vector."""
    col, n_boot, seed = args  # col shape (N,)
    N = col.shape[0]
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, N, size=(n_boot, N))
    means = col[idxs].mean(axis=1)
    return float(means.std(ddof=1))

def _nbar_se_for_time(args):
    """Compute bootstrap SE (x,y,z) for a single timestep n-matrix (N,3)."""
    n_mat_t, n_boot, seed = args  # n_mat_t shape (N,3)
    N = n_mat_t.shape[0]
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, N, size=(n_boot, N))
    # sample -> (n_boot, N, 3) -> means (n_boot,3)
    means = n_mat_t[idxs, :].mean(axis=1)
    se = means.std(axis=0, ddof=1)
    return se.astype(float)

def _bootstrap_rate_se_over_time_parallel(ind_matrix: np.ndarray,
                                          base_seed: Optional[int],
                                          max_workers: Optional[int]):
    """
    ind_matrix: (N, T) of 0/1 indicators.
    Returns SE over time: (T,)
    """
    N, T = ind_matrix.shape
    if N <= 1:
        return np.zeros(T, dtype=float)
    n_boot = N
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    # Prepare args per timestep (avoid reusing the same seed)
    args = [(ind_matrix[:, t], n_boot, (None if base_seed is None else base_seed + t)) for t in range(T)]
    ses = np.empty(T, dtype=float)
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(base_seed,)) as ex:
        for t, se_t in enumerate(ex.map(_rate_se_for_column, args, chunksize=max(1, T // (max_workers*2)))):
            ses[t] = se_t
    return ses

def _bootstrap_nbar_se_over_time_parallel(n_tensor: np.ndarray,
                                          base_seed: Optional[int],
                                          max_workers: Optional[int]):
    """
    n_tensor: (N, T, 3)
    Returns SE over time: (T, 3)
    """
    N, T, _ = n_tensor.shape
    if N <= 1:
        return np.zeros((T, 3), dtype=float)
    n_boot = N
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    args = [(n_tensor[:, t, :], n_boot, (None if base_seed is None else base_seed + 10_000 + t)) for t in range(T)]
    se = np.empty((T, 3), dtype=float)
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(base_seed,)) as ex:
        for t, se_t in enumerate(ex.map(_nbar_se_for_time, args, chunksize=max(1, T // (max_workers*2)))):
            se[t, :] = se_t
    return se

# ---------- main API (updates mol_list in-place) ----------

def apply_raman_sequence(
    mol_list,
    pulse_sequence,
    optical_pumping=True,
    rng=None,
    max_workers=None,
    worker_seed=None,
    include_initial_snapshot=True,
    record_all=False,  
    boot_max_workers=None,
    boot_worker_seed=None,
    get_sem=False
):
    """
    - Each worker runs one molecule through all pulses.
    - If record_all=True: stats after every pulse (+ initial snapshot if enabled).
    - If record_all=False: only stats at t=0 and final state.
    - Updated molecules are written back into mol_list.
    - Returns aggregated arrays over the fixed cohort.
    """
    if rng is None:
        rng = np.random.default_rng()

    N0 = len(mol_list)
    P = len(pulse_sequence)
    if record_all:
        T = (1 + P) if include_initial_snapshot else P
    else:
        # only initial and final
        T = 2 if include_initial_snapshot else 1

    if N0 == 0:
        zeros_T = np.zeros(T, dtype=float)
        zeros_T3 = np.zeros((T,3), dtype=float)
        return zeros_T3, zeros_T, zeros_T, zeros_T3, zeros_T, zeros_T

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    surv = np.empty((N0, T), dtype=float)
    gnd  = np.empty((N0, T), dtype=float)
    n_ts = np.empty((N0, T, 3), dtype=float)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(worker_seed,)
    ) as ex:
        futures = [
            ex.submit(
                _run_one_molecule_all_pulses_with_index,
                i, mol, pulse_sequence, bool(optical_pumping),
                False, include_initial_snapshot, record_all
            )
            for i, mol in enumerate(mol_list)
        ]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Applying pulses to molecules"):
            idx, surv_i, gnd_i, n_i, mol_updated = fut.result()
            surv[idx, :] = surv_i
            gnd[idx,  :] = gnd_i
            n_ts[idx, :, :] = n_i
            mol_list[idx] = mol_updated   # update in-place

    survive_rate      = surv.mean(axis=0)
    ground_state_rate = gnd.mean(axis=0)
    n_bars            = n_ts.mean(axis=0)

    # -------- Parallelized bootstrap SEs over time --------
    # Use rng to derive a base seed for reproducibility if provided.
    # (If rng is random, seeds vary; set boot_worker_seed for fixed results.)
    if get_sem:
        
        base_seed = boot_worker_seed if (boot_worker_seed is not None) else int(rng.integers(0, 2**31 - 1))

        se_survive = _bootstrap_rate_se_over_time_parallel(surv, base_seed, boot_max_workers)
        se_ground  = _bootstrap_rate_se_over_time_parallel(gnd,  base_seed + 1 if base_seed is not None else None, boot_max_workers)
        se_nbar    = _bootstrap_nbar_se_over_time_parallel(n_ts, base_seed + 2 if base_seed is not None else None, boot_max_workers)
    else:
        se_survive = np.zeros(T, dtype=float)
        se_ground  = np.zeros(T, dtype=float)
        se_nbar    = np.zeros((T, 3), dtype=float)

    return (
        n_bars.astype(float),
        survive_rate.astype(float),
        ground_state_rate.astype(float),
        se_nbar.astype(float),
        se_survive.astype(float),
        se_ground.astype(float),
    )

def apply_raman_pulses_serial(
    mol,
    pulse_sequence,
    optical_pumping: bool = True,
    rng=None,
):
    """
    Apply a sequence of Raman pulses to a single molecule in serial.
    """
    import numpy as np, random

    if rng is not None:
        # Sync rng into numpy and Python's random so molecule methods
        # using np.random/random will be reproducible
        np.random.set_state(rng.bit_generator.state)
        random.seed(rng.integers(0, 2**32 - 1))

    for axis, delta_n, t in pulse_sequence:
        
        mol.Raman_transition(axis=int(axis), delta_n=int(delta_n), time=float(t), print_report=False)
        if optical_pumping:
            mol.Optical_pumping(print_report=False)

    return mol




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
        if mol.spin == 0 and mol.state == 1 and not mol.islost:
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
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

        # X axis
        axes[0].bar(counts_x.keys(), counts_x.values(),
                    color='salmon', edgecolor='black')
        axes[0].set_title("n Distribution (X axis)")
        max_x = max(counts_x.keys()) if counts_x else 0
        axes[0].set_xticks(np.linspace(0, max_x, 10, dtype=int))

        # Y axis
        axes[1].bar(counts_y.keys(), counts_y.values(),
                    color='mediumseagreen', edgecolor='black')
        axes[1].set_title("n Distribution (Y axis)")
        max_y = max(counts_y.keys()) if counts_y else 0
        axes[1].set_xticks(np.linspace(0, max_y, 10, dtype=int))

        # Z axis
        axes[2].bar(counts_z.keys(), counts_z.values(),
                    color='cornflowerblue', edgecolor='black')
        axes[2].set_title("n Distribution (Z axis)")
        max_z = max(counts_z.keys()) if counts_z else 0
        axes[2].set_xticks(np.linspace(0, max_z, 10, dtype=int))

        for ax in axes:
            ax.set_xlabel("n")
            ax.grid(True, linestyle='--', alpha=0.5)

        axes[0].set_ylabel("Count")
        fig.suptitle(f"{mol_num} molecules survived")
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


def plot_time_sequence_data(n_bar, num_survive, ground_state_count, n_err, num_err, ground_err):

    fig, axs = plt.subplots(1, 3, figsize=(20, 4))

    # Plot 1: Ground state count
    axs[0].errorbar(range(len(ground_state_count)), ground_state_count, ground_err, marker='o')
    axs[0].set_title("3D Ground State Count")
    axs[0].set_xlabel("Pulse #")
    axs[0].set_ylabel("# in [0,0,0]")
    axs[0].grid(True)

    # Plot 2: N_bar
    for i in [0, 1, 2]:
        axs[1].errorbar(range(len(n_bar)), np.array(n_bar)[:, i], np.array(n_err)[:, i], marker='o', label=f'axis {i}')
    axs[1].set_title("N_bar")
    axs[1].set_xlabel("Pulse #")
    axs[1].set_ylabel("Standard error")
    axs[1].grid(True)
    axs[1].legend()

    # Plot 3: Molecules Survived
    axs[2].errorbar(range(len(num_survive)), num_survive, num_err, marker='o')
    axs[2].set_title("Surviving Molecules")
    axs[2].set_xlabel("Pulse #")
    axs[2].set_ylabel("Survivors")
    axs[2].grid(True)


    plt.tight_layout()
    plt.show()
