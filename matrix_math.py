from __future__ import annotations

import numpy as np
import bottleneck as bn
import pandas as pd


EPS = 1e-12


def nearest_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, floor)
    repaired = (vectors * values) @ vectors.T
    return (repaired + repaired.T) / 2

def cov_to_vol_corr(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vol = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    denom = np.outer(vol, vol)
    corr = np.divide(covariance, denom, out=np.eye(covariance.shape[0]), where=denom > 1e-12)
    corr = np.clip(corr, -0.999999, 0.999999)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return vol, corr


# ---------------------------------------------------------------------------
# Cross-sectional preprocessing
# ---------------------------------------------------------------------------
def winsorize(
    x: np.ndarray,
    *,
    method: str = "mad",
    p: float = 0.01,
    n_sigma: float = 3.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
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


def cross_sectional_zscore(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    work = np.where(valid, values, np.nan)
    mean = np.nanmean(work, axis=1, keepdims=True)
    std = np.nanstd(work, axis=1, keepdims=True)
    out = np.divide(work - mean, std, out=np.full_like(work, np.nan), where=std > EPS)
    return np.where(valid, out, np.nan)


def robust_extreme_ratio_by_row(x: np.ndarray, *, mad_multiple: float = 5.0, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    work = np.where(valid, values, np.nan)
    median = np.nanmedian(work, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(work - median), axis=1, keepdims=True)
    fallback = np.abs(work - median) > EPS
    normal = np.abs(work - median) > mad_multiple * mad
    extreme = np.where((np.isfinite(mad) & (mad > EPS)), normal, fallback)
    counts = np.sum(valid, axis=1)
    hits = np.sum(extreme & valid, axis=1)
    return np.divide(hits, counts, out=np.full(values.shape[0], np.nan, dtype=float), where=counts > 0)


# ---------------------------------------------------------------------------
# Ranking, IC, and correlation
# ---------------------------------------------------------------------------
def corr(a, b, axis):
    b[np.isnan(a)] = np.nan
    a[np.isnan(b)] = np.nan
    arr = (
            (bn.nanmean(a * b, axis=axis) - bn.nanmean(a, axis=axis) * bn.nanmean(b, axis=axis))
            / (bn.nanstd(a, axis=axis) + 1e-6)
            / (bn.nanstd(b, axis=axis) + 1e-6)
    )
    bn.replace(arr, np.nan, 0)
    arr[np.isinf(arr)] = 0
    return arr

def IC(y_, y):
    ics = corr(y_.copy(), y.copy(), axis=-1)
    return ics

def rankIC(y_, y):
    rank_ics = corr(bn.nanrankdata(y_.copy(), axis=-1), bn.nanrankdata(y.copy(), axis=-1), axis=-1)
    return rank_ics


def factor_ic_history(
    factors: np.ndarray,
    labels: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    rank: bool = True,
    min_obs: int = 20,
) -> np.ndarray:
    x_all = np.asarray(factors, dtype=float)
    y_all = np.asarray(labels, dtype=float)
    if x_all.ndim != 3 or y_all.shape != x_all.shape[:2]:
        raise ValueError("factors must be T x N x K and labels must be T x N")
    valid_mask = np.ones(x_all.shape[:2], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    y = np.where(valid_mask, y_all, np.nan)
    out = np.full((x_all.shape[0], x_all.shape[2]), np.nan, dtype=float)
    corr_fn = rankIC if rank else IC
    for k in range(x_all.shape[2]):
        x = np.where(valid_mask, x_all[:, :, k], np.nan)
        value = corr_fn(x, y)
        valid_count = np.sum(np.isfinite(x) & np.isfinite(y), axis=1)
        out[:, k] = np.where(valid_count >= min_obs, value, np.nan)
    return out


def calc_icir(ic: np.ndarray) -> np.ndarray:
    values = np.asarray(ic, dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    return np.divide(mean, std, out=np.zeros_like(mean), where=std > EPS)


# ---------------------------------------------------------------------------
# Neutralization
# ---------------------------------------------------------------------------
def neutralize_by_exposures(
    y: np.ndarray,
    exposures: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    add_intercept: bool = True,
    min_obs: int | None = None,
    ridge: float = 1e-8,
) -> np.ndarray:
    target = np.asarray(y, dtype=float)
    x = np.asarray(exposures, dtype=float)
    if x.ndim != 3 or x.shape[:2] != target.shape:
        raise ValueError("exposures must be shaped T x N x K and aligned with y")

    valid = np.isfinite(target) & np.isfinite(x).all(axis=2)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if weights is not None:
        w_all = np.asarray(weights, dtype=float)
        valid &= np.isfinite(w_all) & (w_all > 0)
    threshold = min_obs or (x.shape[2] + int(add_intercept) + 5)
    out = np.full_like(target, np.nan, dtype=float)

    for t in range(target.shape[0]):
        ok = valid[t]
        if ok.sum() < threshold:
            continue
        design = x[t, ok]
        if add_intercept:
            design = np.column_stack([np.ones(ok.sum()), design])
        yy = target[t, ok]
        if weights is None:
            xtx = design.T @ design
            xty = design.T @ yy
        else:
            ww = w_all[t, ok]
            xtx = design.T @ (design * ww[:, None])
            xty = design.T @ (yy * ww)
        penalty = ridge * np.eye(xtx.shape[0], dtype=float)
        if add_intercept:
            penalty[0, 0] = 0.0
        try:
            beta = np.linalg.solve(xtx + penalty, xty)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(xtx + penalty) @ xty
        out[t, ok] = yy - design @ beta
    return out


def industry_size_neutralize(
    factor: np.ndarray,
    industry: np.ndarray,
    market_cap: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    standardize: bool = True,
) -> np.ndarray:
    industry_arr = np.asarray(industry, dtype=float)
    finite_codes = np.unique(industry_arr[np.isfinite(industry_arr)]).astype(int)
    dummies = [(industry_arr == code).astype(float) for code in finite_codes]
    log_size = np.log(np.clip(np.asarray(market_cap, dtype=float), EPS, None))
    exposures = np.stack([log_size, *dummies], axis=2)
    residual = neutralize_by_exposures(factor, exposures, mask=mask)
    return cross_sectional_zscore(residual, mask=mask) if standardize else residual


# ---------------------------------------------------------------------------
# Group, return, and drawdown diagnostics
# ---------------------------------------------------------------------------
def _as_matrix(values) -> np.ndarray:
    return values.values if isinstance(values, pd.DataFrame) else np.asarray(values, dtype=float)


def group_membership(alpha, num_group: int = 10) -> np.ndarray:
    alpha_values = _as_matrix(alpha)
    rank = bn.nanrankdata(alpha_values, axis=-1)
    num_signal = np.nanmax(rank, axis=-1)
    stock_each_group = num_signal // num_group
    groups = np.zeros((num_group, alpha_values.shape[0], alpha_values.shape[1]), dtype=bool)
    for i in range(num_group):
        if i == num_group - 1:
            groups[i] = (rank.T > stock_each_group * i).T & (rank <= num_signal[:, None])
        else:
            groups[i] = (rank > stock_each_group[:, None] * i) & (rank <= stock_each_group[:, None] * (i + 1))
    return groups


def calc_group_weights(alpha, *, num_group: int = 5, long_only: bool = True) -> np.ndarray:
    groups = group_membership(alpha, num_group=num_group)
    top = groups[-1]
    top_count = top.sum(axis=1, keepdims=True)
    weights = np.divide(top.astype(float), top_count, out=np.zeros_like(top, dtype=float), where=top_count > 0)
    if long_only:
        return weights
    bottom = groups[0]
    bottom_count = bottom.sum(axis=1, keepdims=True)
    short = np.divide(bottom.astype(float), bottom_count, out=np.zeros_like(bottom, dtype=float), where=bottom_count > 0)
    return 0.5 * weights - 0.5 * short


def calc_group_ret(alpha, label, num_group=10, *, demean: bool = True):
    alpha_values = _as_matrix(alpha)
    label_values = _as_matrix(label)
    groups = group_membership(alpha_values, num_group=num_group)
    group_ret = np.full((num_group, alpha_values.shape[0]), np.nan, dtype=float)
    for i in range(num_group):
        count = groups[i].sum(axis=1)
        total = np.nansum(np.where(groups[i], label_values, 0.0), axis=1)
        group_ret[i] = np.divide(total, count, out=np.full(alpha_values.shape[0], np.nan), where=count > 0)
    if demean:
        group_ret = group_ret - np.nanmean(group_ret, axis=0)
    if isinstance(alpha, pd.DataFrame):
        col_list = list(range(1, num_group + 1))[::-1]
        return pd.DataFrame(group_ret.T, columns=col_list, index=alpha.index)
    return group_ret.T


def calc_annret(ret_df):
    nav = np.nancumprod(1+ret_df.values)
    years = (ret_df.index[-1] - ret_df.index[0]).days / 242
    total_ret = nav[-1]/nav[0]-1
    annret = (1+total_ret)**(1/years) - 1
    return annret


def calc_annvol(ret_df):
    annvol = np.nanstd(ret_df.values) * np.sqrt(242)
    return annvol


def calc_sharpe(ret_df):
    annret = calc_annret(ret_df)
    annvol = calc_annvol(ret_df)
    sharpe = annret / annvol if annvol>0 else 0
    return sharpe


def calc_maxdrawdown(ret_df):
    nav = np.nancumprod(1+ret_df.values)
    return ((nav - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav)).min()


def calc_calmar(ret_df):
    annret = calc_annret(ret_df)
    max_dd = calc_maxdrawdown(ret_df)
    calmar = annret / abs(max_dd) if max_dd<0 else np.nan
    return calmar


def calc_weekly_bps(ret_df):
    weekly_rets = ret_df.resample('W').apply(lambda x: (1+x).prod()-1).dropna()
    weekly_avg_bps = weekly_rets.mean() * 10000
    return weekly_avg_bps


def calc_holdings(alpha: pd.DataFrame, num_group: int = 10) -> pd.DataFrame:
    rank = bn.nanrankdata(alpha.values, axis=-1)   # (n_dates, n_stocks)
    n_valid = np.nansum(~np.isnan(alpha.values), axis=-1)
    stock_each_group = n_valid // num_group
    topgroup_ix = (rank.T > stock_each_group * (num_group-1) ) & (rank.T <= n_valid) # (n_stocks, n_dates)
    bottomgroup_ix = (rank.T > 0) & (rank.T <= stock_each_group)
    long_holds = pd.DataFrame(topgroup_ix.T, index=alpha.index, columns=alpha.columns)
    short_holds= pd.DataFrame(bottomgroup_ix.T, index=alpha.index, columns=alpha.columns)
    holds = pd.DataFrame(0, index=alpha.index, columns=alpha.columns, dtype=int)
    holds[long_holds.values] = 1
    holds[short_holds.values] = -1
    return holds


def calc_turnover(holds,  freq: str = 'D') -> pd.Series:
    if freq == 'D':
        rebalance_dates = holds.index.sort_values()  
    elif freq == 'W':
        rebalance_dates = pd.Series(holds.index).groupby(holds.index.to_period('W-MON')).first().values
    elif freq == 'M':
        rebalance_dates = pd.Series(holds.index).groupby(holds.index.to_period('M')).first().values
    else:
        raise ValueError("freq must be: 'D', 'W', 'M'")
    
    #holdings = calc_top_holdings(alpha)
    curr = holds.loc[rebalance_dates].astype(int)
    prev = curr.shift(1).fillna(0).astype(int)
    
    change = (curr - prev).abs().sum(axis=1)          
    total = prev.abs().sum(axis=1) + curr.abs().sum(axis=1)
    turnover = change / total
    turnover = turnover.fillna(0.0)
    turnover.iloc[0] = np.nan
    return turnover.rename('turnover').to_frame(name='turnover')

# ---------------------------------------------------------------------------
# Portfolio weight helpers
# ---------------------------------------------------------------------------
def cap_and_renormalize(weights: np.ndarray, *, max_weight: float, iterations: int = 50) -> np.ndarray:
    w = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    active = w > EPS
    if not active.any():
        return w

    cap = max(max_weight, 1.0 / active.sum())
    w = w / w.sum()

    for _ in range(iterations):
        over = active & (w > cap)
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = active & (w < cap - EPS)
        free_sum = float(w[free].sum())
        if free_sum <= EPS:
            break
        w[free] *= (free_sum + excess) / free_sum
    return w


def normalize_long_only(score: np.ndarray, *, mask: np.ndarray | None = None, max_weight: float = 0.02) -> np.ndarray:
    raw = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None:
        raw = np.where(mask, raw, 0.0)
    raw = np.maximum(raw, 0.0)
    out = np.zeros_like(raw)
    for t in range(raw.shape[0]):
        row = raw[t]
        if row.sum() <= EPS:
            eligible = np.asarray(mask[t], dtype=bool) if mask is not None else np.ones_like(row, dtype=bool)
            row = eligible.astype(float)
        w = row / row.sum() if row.sum() > EPS else row
        out[t] = cap_and_renormalize(w, max_weight=max_weight) if max_weight > 0 else w
    return out


