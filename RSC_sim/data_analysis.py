import os
import json
import numpy as np
import pandas as pd

def load_param_and_history(folder_name: str, par_name: str, summary: str = "last"):
    """
    Collects a specified parameter and corresponding history data from all subfolders
    under the given folder_name, and returns a DataFrame.

    Parameters
    ----------
    folder_name : str
        Path to the parent folder containing subfolders.
    par_name : str
        The name of the parameter to extract from each config.json file.
    summary : str, optional
        How to summarize the history ('last', 'max', 'min', or 'mean').
        Default is 'last'.

    Returns
    -------
    df : pandas.DataFrame
        Columns: ['folder', par_name, 'history', 'summary_value']
    """
    records = []

    for subdir in sorted(os.listdir(folder_name)):
        subpath = os.path.join(folder_name, subdir)
        if not os.path.isdir(subpath):
            continue

        config_path = os.path.join(subpath, "config.json")
        history_path = os.path.join(subpath, "history.txt")

        if not os.path.exists(config_path) or not os.path.exists(history_path):
            print(f"Skipping {subdir}: missing config.json or history.txt")
            continue

        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)
        param_value = config.get(par_name, None)

        # Load history
        with open(history_path, "r") as f:
            content = f.read()

        # Split the file by brackets to isolate each array
        chunks = content.split('[')
        all_histories = []

        for chunk in chunks:
            chunk = chunk.strip().strip(']')
            if not chunk:
                continue
            try:
                arr = np.fromstring(chunk, sep=' ')
                if len(arr) > 0:
                    all_histories.append(arr)
            except Exception as e:
                print(f"Warning: could not parse a history block in {subdir}: {e}")

        # Stack into 2D array if possible, else keep list
        try:
            history = np.vstack(all_histories)  # shape (n_runs, n_points)
        except ValueError:
            history = all_histories



        # Compute summary metric
        if summary == "last":
            summary_value = history[-1]
        elif summary == "max":
            summary_value = np.max(history)
        elif summary == "min":
            summary_value = np.min(history)
        elif summary == "mean":
            summary_value = np.mean(history)
        else:
            raise ValueError("summary must be one of: 'last', 'max', 'min', 'mean'")

        records.append({
            "folder": subdir,
            par_name: param_value,
            "history": history,
            "summary_value": summary_value
        })

    return pd.DataFrame(records)
