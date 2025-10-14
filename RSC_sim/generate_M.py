import numpy as np
import scipy.constants as cts
import scipy.stats as stats
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from scipy.special import hermite, factorial
from multiprocessing import cpu_count, current_process, get_context
import json
from importlib.resources import files

# ---- Load simulation parameters from config.json ----
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

import numpy as np
from scipy.special import factorial, eval_genlaguerre
from tqdm import tqdm
from pathos.multiprocessing import Pool, cpu_count

MAX_N = max(max_n)
LD_MIN = float(_cfg['LD_MIN'])
LD_MAX = float(_cfg['LD_MAX'])
LD_GRID = np.arange(LD_MIN, LD_MAX + LD_RES, LD_RES)  
LD_LEN = len(LD_GRID)

M_FACTOR_TABLE = np.zeros((MAX_N + 1, MAX_N + 1, LD_LEN), dtype=np.float64)

from scipy.special import eval_genlaguerre, gammaln

def M_factor(n1, n2, eta=0.57):
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
    if n1 < 0 or n2 < 0:
        raise ValueError("n1, n2 must be nonnegative")

    # exact η=0 case: carrier only
    if eta == 0.0:
        return 1.0 if n1 == n2 else 0.0

    # tiny-eta guard to avoid log(0) while keeping correct limit
    # (optional; you can omit if you never pass tiny positive values)
    if eta < 1e-300:
        return 1.0 if n1 == n2 else 0.0

    if n2 >= n1:
        delta = n2 - n1
        log_pref = 0.5*(gammaln(n1 + 1) - gammaln(n2 + 1)) + delta*np.log(eta)
        L = eval_genlaguerre(n1, delta, eta*eta)
    else:
        delta = n1 - n2
        log_pref = 0.5*(gammaln(n2 + 1) - gammaln(n1 + 1)) + delta*np.log(eta)
        L = eval_genlaguerre(n2, delta, eta*eta)

    return np.exp(-0.5*eta*eta + log_pref) * L


def _single_M_factor_task(args):
    """Worker function: computes M_factor for a given (n_i, n_f, ld_index)."""
    n_i, n_f, ld_index, ld = args
    val = M_factor(n_i, n_f, ld)
    return (n_i, n_f, ld_index, val)


def precompute_M_factors_parallel(workers=None):
    if workers is None:
        workers = cpu_count()

    print(f"Using {workers} workers to compute M_factor table...")

    tasks = [
        (n_i, n_f, ld_index, ld)
        for ld_index, ld in enumerate(LD_GRID)
        for n_i in range(MAX_N + 1)
        for n_f in range(MAX_N + 1)
    ]

    # Choose iterator: no Pool if not in MainProcess (e.g., import-time in a child)
    if current_process().name != "MainProcess" or workers <= 1:
        iterator = map(_single_M_factor_task, tasks)
        for n_i, n_f, ld_index, val in tqdm(iterator, total=len(tasks), desc="Computing M_factors"):
            M_FACTOR_TABLE[n_i, n_f, ld_index] = val
    else:
        # Safer spawn context across platforms; stream results as they come
        ctx = get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            # chunksize speeds up first results and tqdm movement
            for n_i, n_f, ld_index, val in tqdm(
                pool.imap_unordered(_single_M_factor_task, tasks, chunksize=256),
                total=len(tasks),
                desc="Computing M_factors"
            ):
                M_FACTOR_TABLE[n_i, n_f, ld_index] = val

    np.save("M_FACTOR_TABLE.npy", M_FACTOR_TABLE)
    print("M_factor precomputation complete.")

