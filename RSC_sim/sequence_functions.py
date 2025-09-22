import numpy as np
import scipy.constants as cts
import scipy.stats as stats
import gc


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
    

def get_sequence_unit(axis, delta_n, sm=None):
    """
    Get a single RSC pulse unit [axis, delta_n, scaled_time].

    Parameters:
    -----------
    axis : int
        Axis index (0 = x, 1 = y, 2 = z).
    delta_n : int
        Change in vibrational quantum number (typically negative for sideband cooling).
    sm : ndarray, optional
        Scaling matrix of shape (3, N), where sm[axis, abs(delta_n) - 1] is the scale factor.
        If None, defaults to ones.

    Returns:
    --------
    unit : list
        A single pulse unit as [axis, delta_n, scaled_time].
    """
    if sm is None:
        sm = np.ones((3, 5))  # Default scale matrix for |Δn| = 1 to 5

    index = abs(delta_n) - 1
    scaled_time = sm[axis, index] * pulse_time(axis, delta_n)

    return [axis, delta_n, scaled_time]

def load_sequence(filepath="best_sequence_same_length.txt"):
    '''
    Load sequence from .txt file, in the form of [[axis, delta_n], ...]
    Retuens a sequence that can be used in apply_raman_sequence
    '''
    with open(filepath, "r") as f:
        lines = f.readlines()
    sequence = [eval(line.strip()) for line in lines]  # [[axis, delta_n], ...]
    return sequence

def get_original_sequence(sm=None):
    '''
    Get the RSC pulse sequences from PRL (2024).

    Parameters 
    - sm (np.ndarray): scale matrix of shape (3, N), where:
        sm[axis, i] scales the duration of pulse on `axis` with Δn corresponding to pulse i
        If None, defaults to ones.

    Returns:
    - tuple of sequences (lists of [axis, delta_n, time])
    '''
    if sm is None:
        sm = np.ones((3, 5))  # default scale matrix (3 axes, up to Δn = 5)


    sequence_XY = [
        [0, -3, sm[0, 2] * pulse_time(0, -3)],
        [1, -3, sm[1, 2] * pulse_time(1, -3)],
        [0, -2, sm[0, 1] * pulse_time(0, -2)],
        [1, -2, sm[1, 1] * pulse_time(1, -2)],
    ]

    sequence_XYZ1 = [
        [2, -5, sm[2, 4] * pulse_time(2, -5)],
        [0, -2, sm[0, 1] * pulse_time(0, -2)],
        [2, -4, sm[2, 3] * pulse_time(2, -4)],
        [1, -2, sm[1, 1] * pulse_time(1, -2)],
        [2, -5, sm[2, 4] * pulse_time(2, -5)],
        [0, -1, sm[0, 0] * pulse_time(0, -1)],
        [2, -4, sm[2, 3] * pulse_time(2, -4)],
        [1, -1, sm[1, 0] * pulse_time(1, -1)],
    ]

    sequence_XYZ2 = [
        [2, -4, sm[2, 3] * pulse_time(2, -4)],
        [0, -2, sm[0, 1] * pulse_time(0, -2)],
        [2, -3, sm[2, 2] * pulse_time(2, -3)],
        [1, -2, sm[1, 1] * pulse_time(1, -2)],
        [2, -4, sm[2, 3] * pulse_time(2, -4)],
        [0, -1, sm[0, 0] * pulse_time(0, -1)],
        [2, -3, sm[2, 2] * pulse_time(2, -3)],
        [1, -1, sm[1, 0] * pulse_time(1, -1)],
    ]

    sequence_XYZ3 = [
        [2, -3, sm[2, 2] * pulse_time(2, -3)],
        [0, -2, sm[0, 1] * pulse_time(0, -2)],
        [2, -2, sm[2, 1] * pulse_time(2, -2)],
        [1, -2, sm[1, 1] * pulse_time(1, -2)],
        [2, -3, sm[2, 2] * pulse_time(2, -3)],
        [0, -1, sm[0, 0] * pulse_time(0, -1)],
        [2, -2, sm[2, 1] * pulse_time(2, -2)],
        [1, -1, sm[1, 0] * pulse_time(1, -1)],
    ]

    sequence_XYZ4 = [
        [2, -2, sm[2, 1] * pulse_time(2, -2)],
        [0, -2, sm[0, 1] * pulse_time(0, -2)],
        [2, -1, sm[2, 0] * pulse_time(2, -1)],
        [1, -2, sm[1, 1] * pulse_time(1, -2)],
        [2, -2, sm[2, 1] * pulse_time(2, -2)],
        [0, -1, sm[0, 0] * pulse_time(0, -1)],
        [2, -1, sm[2, 0] * pulse_time(2, -1)],
        [1, -1, sm[1, 0] * pulse_time(1, -1)],
    ]

    return sequence_XY, sequence_XYZ1, sequence_XYZ2, sequence_XYZ3, sequence_XYZ4