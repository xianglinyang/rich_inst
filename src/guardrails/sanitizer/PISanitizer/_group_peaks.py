import numpy as np
from scipy.signal import find_peaks, peak_widths, savgol_filter


def _group_consecutive(peaks, max_gap):
    if len(peaks) == 0:
        return []
    groups, current = [], [peaks[0]]
    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i - 1] <= max_gap:
            current.append(peaks[i])
        else:
            groups.append(current)
            current = [peaks[i]]
    groups.append(current)
    return groups


def group_peaks(x, smooth_win=9, max_gap=10, threshold=0.01,
                prominence=0.0, distance=5, height=0.005, rel_height=0.95):
    smooth_x = list(savgol_filter(x, smooth_win, 2, mode="constant", cval=0.0)) \
        if len(x) >= smooth_win else x

    peaks, _ = find_peaks(smooth_x, prominence=prominence, distance=distance, height=height)
    peak_groups = _group_consecutive(peaks, max_gap)
    above = {i for i, v in enumerate(smooth_x) if v >= threshold}

    remove_list, top_values = [], []
    for group in peak_groups:
        if set(group) & above:
            widths = peak_widths(smooth_x, group, rel_height=rel_height)
            remove_list.append((int(widths[-2][0]), int(widths[-1][-1])))
            top_values.append(max(smooth_x[j] for j in group))

    if top_values:
        best = int(np.argmax(top_values))
        return smooth_x, [remove_list[best]]
    return smooth_x, []
