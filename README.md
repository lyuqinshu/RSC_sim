# RSC_sim

This project simulates the Raman sideband cooling (RSC) of CaF molecules in a tweezer trap following the theoratical proposal from [Caldwell et al.](https://doi.org/10.1103/PhysRevResearch.2.013251) 
and experimantal implementation from [Bao et al.](https://doi.org/10.1103/PhysRevX.14.031002)
It can be easily tailored to other species and experiments by modifying config.json

---


## Configuration (`config.json`)

Example:
```python
{
  "mass": 123.0,                 # amu
  "trap_freq": [80e3, 80e3, 20e3], # Hz for x,y,z (converted internally to rad/s)
  "lambda": 531e-9,             # m (e.g., 531 nm)
  "decay_ratio": [0.25, 0.25, 0.5], # P(mN=-1,0,1) after OP
  "branch_ratio": 0.05,          # probability to switch spin manifold during OP
  "trap_depth": 200e-6,          # Kelvin
  "max_n": [200, 200, 200],      # n cutoff per axis for M table and OP, should be larger than the trap depth
  "LD_RES": 0.01,                # η grid resolution
  "LD_MIN": 0.0,
  "LD_MAX": 3.0,
  "angle_pump_sigma": [1.57, 0.0], # [theta, phi] rad for σ-pump, referenced to the axial trap axis
  "angle_pump_pi":    [0.0, 0.0],  # [theta, phi] rad for π-pump, referenced to the axial trap axis
  "LD_raman": [0.5, 0.5, 0.3]    # |Δk|x0 per axis used for Raman pulses
}
```

---

## Quick start

### 1) Precompute M-factor table (first run only)

On first import, the package attempts to load `data/M_FACTOR_TABLE.npy`. If missing, it will compute and save it. You can also force generation:

```python
from RSC_sim.generate_M import precompute_M_factors_parallel

# Safer to run from a main guard (for Windows/macOS):
if __name__ == "__main__":
    precompute_M_factors_parallel(workers=4)  # choose worker count
```
The table shape is `(MAX_N+1, MAX_N+1, LD_LEN)` where `MAX_N = max(max_n)` and `LD_LEN = len(np.arange(LD_MIN, LD_MAX+LD_RES, LD_RES))`.

### 2) Initialize a thermal ensemble

```python
import RSC_sim as rsc

temps = [200e-6, 200e-6, 200e-6]    # Kelvin per axis
N = 1000                          # number of molecules
mol_list = rsc.initialize_thermal(temps, N)
```

### 3) Build a pulse sequence

Use helpers tied to experimental timings, here we use the sequence from [Bao et al.](https://doi.org/10.1103/PhysRevX.14.031002):
```python
from RSC_sim import get_original_sequence

seq_XY, seq_XYZ1, seq_XYZ2, seq_XYZ3, seq_XYZ4 = get_original_sequence()
pulse_sequence = seq_XY + seq_XYZ1 + seq_XYZ2 + seq_XYZ3 + seq_XYZ4  # list of [axis, delta_n, time]
```

Or from a text file:
```python
from RSC_sim import load_sequence, get_sequence_unit

base = load_sequence("best_sequence_same_length.txt") # [[axis, delta_n], ...]
pulse_sequence = [get_sequence_unit(ax, dn) for ax, dn in base]
```

### 4) Run the simulation (parallel cohort)

```python
from RSC_sim import apply_raman_sequence

n_bar, survive_rate, ground_state_rate, se_nbar, se_survive, se_ground = apply_raman_sequence(
    mol_list,
    pulse_sequence,
    optical_pumping=True,
    max_workers=8,
    worker_seed=12345,        # for reproducible worker RNG init
    include_initial_snapshot=True,
    record_all=True,          # collect stats after each pulse
    boot_max_workers=4,
    boot_worker_seed=999,     # for reproducible bootstrap SEs
    get_sem=True
)
```

### 5) Plot

```python
from RSC_sim import plot_time_sequence_data, get_n_distribution

plot_time_sequence_data(n_bar, survive_rate, ground_state_rate, se_nbar, se_survive, se_ground)
get_n_distribution(mol_list, plot=True, scatter=True)
```

---


## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

---

