
import numpy as np
from numpy.typing import NDArray


def calculate_signal_discrepancy(signal: NDArray, cost_name: str, params: dict) -> NDArray:
    match cost_name.lower():
        case 'rbf':
            window_size = params.get('window_size', 50)
            gamma = params.get('gamma', 1.)
            discrepancy = _calculate_rbf_discrepancy(signal, window_size, gamma)
            return discrepancy
        case _:
            return np.zeros_like(signal)


def _calculate_rbf_discrepancy(signal: NDArray, window_size: int, gamma: float) -> NDArray:
    # diff between t vs s
    diff = signal[:, np.newaxis] - signal[np.newaxis, :]
    diff_exp = np.exp(-gamma * (diff**2))
    # calculate rbf on every window
    number = len(signal)
    window_size = window_size + (window_size % 2)  # to even
    half_window = window_size // 2
    # calculate discrepancy
    discrepancy_list = [0.] * half_window
    for i in range(half_window, number - half_window):
        # cost in a window
        all_cost = diff_exp[i-half_window:i+half_window, i-half_window:i+half_window].sum()
        left_cost = diff_exp[i-half_window:i, i-half_window:i].sum()
        right_cost = diff_exp[i:i+half_window, i:i+half_window].sum()
        # discrepancy
        discrepancy = (left_cost + right_cost) / half_window - all_cost / window_size
        discrepancy_list.append(discrepancy)
    discrepancy_list.extend([0.] * half_window)
    return np.array(discrepancy_list)
