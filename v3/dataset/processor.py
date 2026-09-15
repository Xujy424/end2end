import numpy as np
import bottleneck as bn


def winsorize(
    x: np.ndarray,
    method: str = "mad",
    p: float = 0.01,
    n_sigma: float = 3.0
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    valid = np.isfinite(values)
    work = np.where(valid, values, np.nan)
    if method == "quantile":
        lower = np.nanquantile(work, p, axis=1, keepdims=True)
        upper = np.nanquantile(work, 1.0 - p, axis=1, keepdims=True)
    elif method == "sigma":
        mean = np.nanmean(work, axis=1, keepdims=True)
        std = np.nanstd(work, axis=1, keepdims=True)
        lower = mean - n_sigma * std
        upper = mean + n_sigma * std
    elif method == "mad":
        median = np.nanmedian(work, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(work - median), axis=1, keepdims=True)
        radius = n_sigma * 1.4826 * mad
        lower = median - radius
        upper = median + radius
    else:
        raise ValueError(f"unsupported winsorization method: {method}")
    return np.where(valid, np.clip(work, lower, upper), np.nan)


def cross_sectional_zscore(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    valid = np.isfinite(values)
    work = np.where(valid, values, np.nan)
    mean = np.nanmean(work, axis=1, keepdims=True)
    std = np.nanstd(work, axis=1, keepdims=True)
    out = np.divide(work - mean, std, out=np.full_like(work, np.nan), where=std > 1e-12)
    return np.where(valid, out, np.nan)


def calc_indmv_neutral_longshort(ind_signal, temp_mv):
    ix = ~(np.isnan(ind_signal) | np.isinf(ind_signal) | np.isnan(temp_mv) | np.isinf(temp_mv))
    ind_signal[~ix] = np.nan
    temp_mv[~ix] = np.nan

    mv_mean = bn.nanmean(temp_mv, axis=1)
    signal_mean = bn.nanmean(ind_signal, axis=1)
    m = (mv_mean * signal_mean - bn.nanmean(temp_mv * ind_signal, axis=1)) / (mv_mean**2 - bn.nanmean(temp_mv**2, axis=1) + 1e-6)
    b = signal_mean - m * mv_mean
    residual = (ind_signal.T - (temp_mv.T * m) - b).T
    ind_signal = (residual.T - bn.nanmean(residual, axis=1)) / (bn.nanstd(residual, axis=1) + 1e-6)
    return ind_signal.T


def indmv_neutral_longshort(alpha_vec, ind_arr, mv_arr):
    new_signal = np.full_like(alpha_vec, np.nan)   # [T,N]
    ln_mv = np.log(mv_arr)
    for i in range(31):
        ind_ix = ind_arr == i
        ind_select = ind_ix.any(axis=0)
        ind_ix_select = ind_ix[:, ind_select]
        ind_signal = alpha_vec[:, ind_select].copy()
        ind_signal[~ind_ix_select] = np.nan
        temp_mv = ln_mv[:, ind_select].copy()
        new_signal[ind_ix] = calc_indmv_neutral_longshort(ind_signal, temp_mv)[ind_ix_select]
    return new_signal


def pct_change(x: np.ndarray, periods: int = 1) -> np.ndarray:
    res = np.full(x.shape, np.nan)
    current = x[periods:]
    base = x[:-periods]
    x_diff = current - base
    valid = ~np.isnan(current) & ~np.isnan(base)
    nonzero_base = valid & (base != 0)
    zero_base = valid & (base == 0)
    res_part = np.full(x_diff.shape, np.nan)
    np.divide(
        x_diff,
        base,
        out=res_part,
        where=nonzero_base,
    )
    res_part[zero_base] = 0.0
    res[periods:] = res_part
    return res