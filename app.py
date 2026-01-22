# ============================================================
# DOE COPILOT — V9 FINAL (Streamlit, Python 3.13)
# "Minitab +": Assistant + garde-fous + contraintes + séquentiel
#
# Points clés:
# - Plans: Full 2-level, Fractionnaire (générateurs auto + évaluation alias),
#         Hadamard screening, Taguchi OA (2 niveaux), CCD, Box-Behnken, LHS
#         + D-optimal sous contraintes (candidate set + greedy)
# - Contraintes: combos interdits, règles conditionnelles (IF ... THEN ...),
#               facteurs discrets (pas), bornes
# - Exécution atelier: traçabilité (opérateur/machine/lot/timestamp/ambiance)
# - Analyse: OLS + ANOVA + diagnostics + Pareto + Henry + Cook + Box-Cox
# - S/N Taguchi + Robustesse (control vs noise) + Monte-Carlo mean/std
# - Puissance/MDE (approx) selon sigma, alpha, power
# - Ordonnancement setup-friendly (option) + avertissement
# - Autopilot: fold-over, passer en RSM, confirmation
# - Rapport PDF avec figures
# ============================================================

import io
import re
import json
import time
import math
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

from itertools import product, combinations
from scipy.linalg import hadamard
from scipy.stats import qmc, probplot, t as student_t
from scipy.stats import boxcox as scipy_boxcox
from scipy.stats import boxcox_normmax

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="DOE Copilot V9 FINAL", layout="wide")
st.title("DOE Copilot — V9 FINAL (Assistant + Contraintes + Analyse + Autopilot)")


# ============================================================
# Data models
# ============================================================
@dataclass
class Factor:
    name: str
    kind: str          # "quant" or "cat"
    role: str          # "control" or "noise"
    low: Optional[float] = None
    high: Optional[float] = None
    step: float = 0.0
    levels: Optional[List[str]] = None  # cat: 2 levels


@dataclass
class DesignConfig:
    design_type: str
    random_seed: int = 42
    replicates: int = 1
    center_points: int = 0
    n_blocks: int = 1
    randomize_within_block: bool = True
    randomize_global: bool = True

    # Fractionnaire auto (générateurs)
    frac_target_runs: int = 16
    frac_search_iters: int = 800
    frac_include_2fi: bool = True  # for resolution scoring

    # D-optimal
    dopt_runs: int = 16
    dopt_model: str = "main"  # main / main+2fi

    # CCD
    ccd_alpha: str = "rotatable"  # rotatable / face-centered

    # LHS
    lhs_samples: int = 20

    # Taguchi
    taguchi_array: str = "AUTO"  # AUTO/L4/L8/L16/L32

    # Run order
    runorder_mode: str = "random"  # random / setup_optimized


@dataclass
class ConstraintRule:
    kind: str  # "forbidden_combo" or "if_then"
    payload: dict


# ============================================================
# Session state
# ============================================================
def ensure_state():
    defaults = {
        "plan_id": 0,
        "factors": [],
        "constraints": [],
        "design_cfg": None,
        "doe_coded": None,
        "doe_real": None,
        "results": None,
        "y_cols": ["Y"],
        "analysis_cache": None,
        "analysis_cache_key": None,
        "autopilot_notes": [],
        "last_figures": {},  # store figs for PDF
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ensure_state()


# ============================================================
# Utils
# ============================================================
def sanitize_name(name: str, default: str = "X") -> str:
    name = (name or "").strip().replace(" ", "_")
    name = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    return name if name else default


def stable_hash_df(df: pd.DataFrame) -> str:
    b = df.to_csv(index=False).encode("utf-8")
    return hashlib.md5(b).hexdigest()


def add_runorder(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df.insert(0, "RunOrder", np.arange(1, len(df) + 1))
    return df


def repeat_df(df: pd.DataFrame, reps: int) -> pd.DataFrame:
    if reps <= 1:
        return df.reset_index(drop=True)
    return pd.concat([df] * reps, ignore_index=True)


def round_to_step(values: np.ndarray, step: float) -> np.ndarray:
    if step is None or step == 0:
        return values
    return np.round(values / step) * step


def coded_to_real_quant(coded: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + (coded + 1.0) * (high - low) / 2.0


def build_real_from_coded(df_coded: pd.DataFrame, factors: List[Factor]) -> pd.DataFrame:
    df_real = df_coded.copy()
    for f in factors:
        if f.kind == "quant":
            x = coded_to_real_quant(df_coded[f.name].astype(float).values, float(f.low), float(f.high))
            x = round_to_step(x, float(f.step))
            df_real[f.name] = x
        else:
            lv = f.levels or ["A", "B"]
            if len(lv) < 2:
                lv = lv + ["B"]
            df_real[f.name] = np.where(df_coded[f.name].astype(float).values >= 0, lv[1], lv[0])
    return df_real


def make_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="DOE")
    return buffer.getvalue()


def is_two_level_only(df_coded: pd.DataFrame, factor_cols: List[str]) -> bool:
    for c in factor_cols:
        vals = set(pd.Series(df_coded[c]).dropna().unique().tolist())
        if not vals.issubset({-1.0, 1.0}):
            return False
    return True


def apply_blocking(df_real: pd.DataFrame, df_coded: pd.DataFrame, n_blocks: int, seed: int, randomize_within_block: bool):
    df_real = df_real.copy()
    df_coded = df_coded.copy()
    if n_blocks <= 1:
        df_real["Block"] = 1
        df_coded["Block"] = 1
        return df_real, df_coded

    n = len(df_real)
    blocks = np.tile(np.arange(1, n_blocks + 1), int(np.ceil(n / n_blocks)))[:n]
    df_real["Block"] = blocks
    df_coded["Block"] = blocks

    if randomize_within_block:
        rng = np.random.RandomState(seed)
        idx = []
        for b in range(1, n_blocks + 1):
            idx_b = df_real.index[df_real["Block"] == b].to_list()
            rng.shuffle(idx_b)
            idx.extend(idx_b)
        df_real = df_real.loc[idx].reset_index(drop=True)
        df_coded = df_coded.loc[idx].reset_index(drop=True)

    return df_real, df_coded


# ============================================================
# Constraints engine
# ============================================================
def _eval_condition(row: dict, cond: dict) -> bool:
    """
    cond example:
    {"factor":"Temp","op":">=","value":190}
    {"factor":"Mat","op":"==","value":"A"}
    """
    f = cond["factor"]
    op = cond["op"]
    val = cond["value"]
    x = row.get(f)

    # robust cast
    if isinstance(x, (np.generic,)):
        x = x.item()

    if op == "==": return x == val
    if op == "!=": return x != val
    if op == ">=": return float(x) >= float(val)
    if op == "<=": return float(x) <= float(val)
    if op == ">": return float(x) > float(val)
    if op == "<": return float(x) < float(val)
    return False


def check_constraints_row(row: dict, constraints: List[ConstraintRule]) -> bool:
    """
    Return True if row is valid.
    - forbidden_combo: payload {"matches": {"X1": value, "X2": value}}
    - if_then: payload {"if":[cond,...],"then":[cond,...]} where then is required if if is true
    """
    for r in constraints:
        if r.kind == "forbidden_combo":
            matches = r.payload.get("matches", {})
            ok = True
            for k, v in matches.items():
                if row.get(k) != v:
                    ok = False
                    break
            if ok:
                return False

        elif r.kind == "if_then":
            if_conds = r.payload.get("if", [])
            then_conds = r.payload.get("then", [])
            if_true = all(_eval_condition(row, c) for c in if_conds) if if_conds else False
            if if_true:
                then_true = all(_eval_condition(row, c) for c in then_conds) if then_conds else True
                if not then_true:
                    return False

    return True


def filter_by_constraints(df_real: pd.DataFrame, constraints: List[ConstraintRule]) -> pd.DataFrame:
    if not constraints:
        return df_real
    mask = []
    for _, r in df_real.iterrows():
        mask.append(check_constraints_row(r.to_dict(), constraints))
    return df_real.loc[mask].reset_index(drop=True)


# ============================================================
# Run order optimization (setup-friendly)
# ============================================================
def _distance_run(a: pd.Series, b: pd.Series, factors: List[Factor]) -> float:
    """
    Distance in factor space:
    - quant: normalized abs diff
    - cat: 0/1
    """
    d = 0.0
    for f in factors:
        if f.kind == "quant":
            lo, hi = float(f.low), float(f.high)
            denom = (hi - lo) if (hi - lo) != 0 else 1.0
            d += abs(float(a[f.name]) - float(b[f.name])) / denom
        else:
            d += 0.0 if a[f.name] == b[f.name] else 1.0
    return d


def setup_optimized_order(df_real: pd.DataFrame, factors: List[Factor], seed: int = 42) -> pd.DataFrame:
    """
    Greedy nearest-neighbor ordering to reduce changes.
    Keeps blocks together (assumes Block exists).
    """
    df = df_real.copy()
    if "Block" not in df.columns:
        df["Block"] = 1

    out_parts = []
    rng = np.random.RandomState(seed)

    for b, part in df.groupby("Block", sort=True):
        part = part.reset_index(drop=True).copy()
        n = len(part)
        if n <= 2:
            out_parts.append(part)
            continue

        remaining = list(range(n))
        start = rng.choice(remaining)
        order = [start]
        remaining.remove(start)

        while remaining:
            last = order[-1]
            dists = [(i, _distance_run(part.loc[last], part.loc[i], factors)) for i in remaining]
            i_best = min(dists, key=lambda x: x[1])[0]
            order.append(i_best)
            remaining.remove(i_best)

        out_parts.append(part.loc[order].reset_index(drop=True))

    df_out = pd.concat(out_parts, ignore_index=True)
    return df_out


# ============================================================
# DOE generation (coded)
# ============================================================
def _ff2n(k: int) -> np.ndarray:
    return np.array(list(product([-1.0, 1.0], repeat=k)), dtype=float)


def _hadamard_screening(k: int) -> np.ndarray:
    N = 1
    while N < (k + 1):
        N *= 2
    H = hadamard(N).astype(float)
    return H[:, 1:k+1]


def _foldover(X: np.ndarray) -> np.ndarray:
    return np.vstack([X, -X])


def _ccd_design(k: int, center_points: int = 0, alpha_mode: str = "rotatable") -> np.ndarray:
    alpha = 1.0 if alpha_mode == "face-centered" else float(np.sqrt(k))
    factorial = _ff2n(k)
    axial = []
    for i in range(k):
        v = np.zeros(k, dtype=float)
        v[i] = alpha; axial.append(v.copy())
        v[i] = -alpha; axial.append(v.copy())
    axial = np.array(axial, dtype=float)
    center = np.zeros((int(center_points), k), dtype=float) if center_points > 0 else np.zeros((0, k), dtype=float)
    return np.vstack([factorial, axial, center])


def _bb_design(k: int, center_points: int = 0) -> np.ndarray:
    if k < 3:
        raise ValueError("Box–Behnken nécessite au moins 3 facteurs.")
    runs = []
    for i in range(k):
        for j in range(i + 1, k):
            for si in [-1.0, 1.0]:
                for sj in [-1.0, 1.0]:
                    v = np.zeros(k, dtype=float)
                    v[i] = si; v[j] = sj
                    runs.append(v)
    X = np.array(runs, dtype=float)
    if center_points > 0:
        X = np.vstack([X, np.zeros((int(center_points), k), dtype=float)])
    return X


def _lhs_design(k: int, samples: int = 20, seed: int = 42) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=k, seed=seed)
    X01 = sampler.random(n=int(samples))
    return (2 * X01 - 1).astype(float)


# Taguchi OA 2-level (using Hadamard patterns)
def _taguchi_L4() -> np.ndarray:
    return np.array([
        [-1, -1, -1],
        [-1, +1, +1],
        [+1, -1, +1],
        [+1, +1, -1],
    ], dtype=float)

def _taguchi_L8() -> np.ndarray:
    H = hadamard(8).astype(float)
    return H[:, 1:8]

def _taguchi_L16() -> np.ndarray:
    H = hadamard(16).astype(float)
    return H[:, 1:16]

def _taguchi_L32() -> np.ndarray:
    H = hadamard(32).astype(float)
    return H[:, 1:32]

def _taguchi_auto(k: int) -> Tuple[str, np.ndarray]:
    if k <= 3:
        return "L4", _taguchi_L4()[:, :k]
    if k <= 7:
        return "L8", _taguchi_L8()[:, :k]
    if k <= 15:
        return "L16", _taguchi_L16()[:, :k]
    if k <= 31:
        return "L32", _taguchi_L32()[:, :k]
    raise ValueError("Taguchi 2 niveaux supporte jusqu’à 31 facteurs (L32).")


# ============================================================
# Fractional factorial via generators (auto search)
# ============================================================
def _make_frac_design(k: int, p: int, gens: List[Tuple[int, Tuple[int, ...]]], seed: int = 42) -> np.ndarray:
    """
    Build 2^(k-p) fractional factorial:
    - base factors = k-p columns from full factorial
    - dependent factors defined as product of subset of base columns
    gens: list of (dep_index, tuple(base_indices_to_multiply)) using indices in base space
    """
    base_k = k - p
    Xb = _ff2n(base_k)  # 2^(k-p) runs
    X = np.zeros((Xb.shape[0], k), dtype=float)

    # place base columns first
    X[:, :base_k] = Xb

    # dependent columns
    for dep_i, subset in gens:
        v = np.ones(Xb.shape[0], dtype=float)
        for j in subset:
            v *= Xb[:, j]
        X[:, dep_i] = v
    return X


def _score_resolution_like(X: np.ndarray, include_2fi: bool = True) -> Tuple[int, int]:
    """
    Heuristic "resolution":
    - If any main effect is perfectly aliased with any 2FI -> res <= III
    - If main effects not aliased with 2FI but 2FI among themselves aliased -> res ~ IV
    - If even 2FI not aliased with main or other 2FI -> res ~ V+ (rare for small runs)
    We approximate by scanning correlations ±1 between effect columns.
    Returns (res_est, alias_count_main2fi)
    """
    n, k = X.shape
    cols = [X[:, i] for i in range(k)]
    # normalize columns (center)
    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        if d < 1e-12:
            return 0.0
        return float(np.dot(a, b) / d)

    alias_main_2fi = 0
    if include_2fi:
        for i in range(k):
            for j in range(i + 1, k):
                inter = cols[i] * cols[j]
                for m in range(k):
                    c = corr(inter, cols[m])
                    if abs(abs(c) - 1.0) < 1e-6:
                        alias_main_2fi += 1

    # res estimate
    if alias_main_2fi > 0:
        return 3, alias_main_2fi

    # check 2FI with 2FI
    alias_2fi_2fi = 0
    if include_2fi:
        inter_cols = []
        for i in range(k):
            for j in range(i + 1, k):
                inter_cols.append(cols[i] * cols[j])
        for a in range(len(inter_cols)):
            for b in range(a + 1, len(inter_cols)):
                c = corr(inter_cols[a], inter_cols[b])
                if abs(abs(c) - 1.0) < 1e-6:
                    alias_2fi_2fi += 1

    if alias_2fi_2fi > 0:
        return 4, 0
    return 5, 0


def auto_fractional_generators(k: int, target_runs: int, iters: int = 800, seed: int = 42, include_2fi: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Auto build fractional factorial using generators:
    - choose p so that 2^(k-p) close to target_runs (<= target_runs)
    - random search on generator subsets to maximize resolution heuristic
    NOTE: This is a best-effort heuristic (stable on cloud, no pyDOE2).
    """
    rng = np.random.RandomState(seed)

    # choose p such that runs = 2^(k-p) <= target_runs and >= k+1 (rough)
    best_p = None
    best_runs = None
    for p in range(1, k):  # at least one dependent
        runs = 2 ** (k - p)
        if runs <= target_runs:
            best_p = p
            best_runs = runs
            break
    if best_p is None:
        # fallback: screening hadamard
        X = _hadamard_screening(k)
        return X, [f"Fractionnaire AUTO impossible pour target_runs={target_runs}. Fallback: Hadamard {X.shape[0]} essais."]

    p = best_p
    base_k = k - p
    # dependent columns indices in final X: base_k..k-1
    dep_indices = list(range(base_k, k))

    best = None
    best_res = -1
    best_alias = 10**9
    best_gens = None

    # subsets from base indices, exclude empty
    base_indices = list(range(base_k))
    possible_subsets = []
    # limit subset size (avoid very long words)
    for r in range(2, min(4, base_k) + 1):
        possible_subsets.extend(list(combinations(base_indices, r)))
    if not possible_subsets:
        # base_k too small => just use pair products if possible
        possible_subsets = list(combinations(base_indices, 2)) if base_k >= 2 else [(0,)]

    for _ in range(int(iters)):
        gens = []
        used = set()
        for dep in dep_indices:
            subset = possible_subsets[rng.randint(0, len(possible_subsets))]
            # avoid duplicates a bit
            tries = 0
            while (subset in used) and tries < 10:
                subset = possible_subsets[rng.randint(0, len(possible_subsets))]
                tries += 1
            used.add(subset)
            gens.append((dep, tuple(subset)))

        X = _make_frac_design(k, p, gens, seed=seed)
        res_est, alias_main2fi = _score_resolution_like(X, include_2fi=include_2fi)

        # prefer higher resolution, then fewer alias main/2fi
        if (res_est > best_res) or (res_est == best_res and alias_main2fi < best_alias):
            best_res = res_est
            best_alias = alias_main2fi
            best = X
            best_gens = gens

    notes = [f"Fractionnaire AUTO générateurs: runs=2^(k-p) = {2**(k-p)} (k={k}, p={p}).",
             f"Résolution heuristique estimée ≈ {best_res} (alias main↔2FI={best_alias})."]
    if best_gens:
        # show generator summary using letters
        letters = [chr(ord('A') + i) for i in range(k)]
        gen_str = []
        for dep, subset in best_gens:
            left = letters[dep]
            right = "".join([letters[j] for j in subset])
            gen_str.append(f"{left}={right}")
        notes.append("Générateurs: " + ", ".join(gen_str))
    return best, notes


# ============================================================
# D-optimal under constraints (candidate set + greedy)
# ============================================================
def build_candidate_set(factors: List[Factor], constraints: List[ConstraintRule], max_candidates: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Candidate set in REAL units (already respecting steps):
    - For quant: {low, mid, high} or more points if step small (capped)
    - For cat: 2 levels
    Then filter constraints.
    If too large, subsample.
    """
    rng = np.random.RandomState(seed)

    grids = []
    names = []
    for f in factors:
        names.append(f.name)
        if f.kind == "quant":
            lo, hi = float(f.low), float(f.high)
            mid = (lo + hi) / 2.0
            pts = np.array([lo, mid, hi], dtype=float)
            pts = round_to_step(pts, float(f.step))
            pts = np.unique(pts)
            # if step defines many points, optionally enrich to 5 points
            if len(pts) < 3:
                pts = np.unique(np.array([lo, hi], dtype=float))
            if len(pts) < 5:
                extra = np.linspace(lo, hi, 5)
                extra = round_to_step(extra, float(f.step))
                pts = np.unique(np.concatenate([pts, extra]))
            grids.append(list(pts))
        else:
            lv = f.levels or ["A", "B"]
            grids.append([lv[0], lv[1]])

    # full product can explode
    approx_size = 1
    for g in grids:
        approx_size *= len(g)

    if approx_size <= max_candidates:
        rows = list(product(*grids))
        cand = pd.DataFrame(rows, columns=names)
    else:
        # random sample from continuous ranges/levels
        n = max_candidates
        rows = []
        for _ in range(n):
            row = {}
            for f in factors:
                if f.kind == "quant":
                    v = rng.uniform(float(f.low), float(f.high))
                    v = float(round_to_step(np.array([v]), float(f.step))[0])
                    row[f.name] = v
                else:
                    lv = f.levels or ["A", "B"]
                    row[f.name] = lv[rng.randint(0, 2)]
            rows.append(row)
        cand = pd.DataFrame(rows)

    cand = filter_by_constraints(cand, constraints)
    if len(cand) == 0:
        return cand

    # cap
    if len(cand) > max_candidates:
        idx = rng.choice(np.arange(len(cand)), size=max_candidates, replace=False)
        cand = cand.iloc[idx].reset_index(drop=True)

    return cand


def build_model_matrix(df: pd.DataFrame, factors: List[Factor], model: str) -> np.ndarray:
    """
    Build X matrix for D-optimal selection:
    - intercept + main
    - optionally 2FI among quantitative/control only (more stable)
    Categorical are encoded as 0/1 by equality to second level.
    """
    cols = []
    cols.append(np.ones(len(df), dtype=float))

    # main effects
    main_vectors = {}
    for f in factors:
        if f.kind == "quant":
            lo, hi = float(f.low), float(f.high)
            # coded to [-1,1] based on real
            x = 2.0 * (df[f.name].astype(float).values - lo) / (hi - lo) - 1.0 if (hi - lo) != 0 else 0.0
            main_vectors[f.name] = x
        else:
            lv = f.levels or ["A", "B"]
            x = (df[f.name].values == lv[1]).astype(float)
            main_vectors[f.name] = x

    for f in factors:
        cols.append(main_vectors[f.name])

    if model == "main+2fi":
        # 2FI only on quantitative controls (practical)
        q = [f.name for f in factors if f.kind == "quant" and f.role == "control"]
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                cols.append(main_vectors[q[i]] * main_vectors[q[j]])

    X = np.column_stack(cols).astype(float)
    return X


def greedy_d_optimal(df_cand: pd.DataFrame, factors: List[Factor], n_runs: int, model: str = "main") -> Tuple[pd.DataFrame, List[str]]:
    """
    Greedy D-optimal selection:
    - start with random point
    - iteratively add point that maximizes log det(X'X)
    Robust and simple; good enough for constrained selection.
    """
    if len(df_cand) == 0:
        return df_cand, ["Aucun candidat valide après contraintes."]

    n_runs = int(min(n_runs, len(df_cand)))
    rng = np.random.RandomState(42)

    # precompute model matrix for all candidates
    Xall = build_model_matrix(df_cand, factors, model=model)

    # start
    idx_sel = [int(rng.randint(0, len(df_cand)))]
    XtX = np.outer(Xall[idx_sel[0]], Xall[idx_sel[0]])

    def logdet(A):
        # stable logdet
        sign, ld = np.linalg.slogdet(A + 1e-12*np.eye(A.shape[0]))
        return ld if sign > 0 else -1e18

    for _ in range(n_runs - 1):
        best_i = None
        best_ld = -1e18
        for i in range(len(df_cand)):
            if i in idx_sel:
                continue
            x = Xall[i]
            A = XtX + np.outer(x, x)
            ld = logdet(A)
            if ld > best_ld:
                best_ld = ld
                best_i = i
        if best_i is None:
            break
        idx_sel.append(best_i)
        x = Xall[best_i]
        XtX = XtX + np.outer(x, x)

    df_sel = df_cand.iloc[idx_sel].reset_index(drop=True)
    notes = [
        f"D-optimal greedy sélectionné: {len(df_sel)} essais (modèle={model}).",
        "Remarque: D-optimal sous contraintes ≈ très pratique en industrie (DOE réaliste).",
    ]
    return df_sel, notes


# ============================================================
# Analysis helpers
# ============================================================
def sn_ratio(y: np.ndarray, mode: str) -> float:
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) == 0:
        return np.nan
    if mode == "Smaller-the-better":
        return float(-10.0 * np.log10(np.mean(y ** 2)))
    if mode == "Larger-the-better":
        eps = 1e-12
        return float(-10.0 * np.log10(np.mean(1.0 / ((y ** 2) + eps))))
    mu = float(np.mean(y))
    var = float(np.var(y, ddof=1)) if len(y) > 1 else 0.0
    if var <= 0:
        return np.nan
    return float(10.0 * np.log10((mu ** 2) / var))


def build_formula(y_col: str, factors: List[Factor], include_inter: bool, include_quad: bool, include_block: bool, include_noise: bool = True) -> str:
    """
    include_noise=False can fit only control factors (for some use cases),
    but default is include noise too (needed for robust prediction).
    """
    x_terms = []
    quant = []
    quant_control = []
    for f in factors:
        if (not include_noise) and (f.role == "noise"):
            continue
        if f.kind == "quant":
            x_terms.append(f.name)
            quant.append(f.name)
            if f.role == "control":
                quant_control.append(f.name)
        else:
            x_terms.append(f"C({f.name})")

    base = " + ".join(x_terms) if x_terms else "1"
    formula = f"{y_col} ~ {base}"

    # interactions among quant controls by default (stable)
    if include_inter and len(quant_control) >= 2:
        inter_terms = []
        for i in range(len(quant_control)):
            for j in range(i + 1, len(quant_control)):
                inter_terms.append(f"{quant_control[i]}:{quant_control[j]}")
        if inter_terms:
            formula += " + " + " + ".join(inter_terms)

    if include_quad and len(quant_control) >= 1:
        quad_terms = [f"I({q}**2)" for q in quant_control]
        formula += " + " + " + ".join(quad_terms)

    if include_block:
        formula += " + C(Block)"
    return formula


def compute_effects_two_level(df_coded: pd.DataFrame, y: pd.Series, include_interactions: bool = True) -> pd.DataFrame:
    tmp = df_coded.copy()
    tmp["_Y_"] = y.values
    cols = [c for c in df_coded.columns if c not in ["RunOrder", "Block"]]
    effects = []
    for col in cols:
        hi = tmp.loc[tmp[col] == 1, "_Y_"].mean()
        lo = tmp.loc[tmp[col] == -1, "_Y_"].mean()
        effects.append((col, hi - lo))
    if include_interactions and len(cols) >= 2:
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                name = f"{cols[i]}:{cols[j]}"
                inter = tmp[cols[i]] * tmp[cols[j]]
                hi = tmp.loc[inter == 1, "_Y_"].mean()
                lo = tmp.loc[inter == -1, "_Y_"].mean()
                effects.append((name, hi - lo))
    eff = pd.DataFrame(effects, columns=["Terme", "Effet"])
    eff["|Effet|"] = eff["Effet"].abs()
    return eff.sort_values("|Effet|", ascending=False).reset_index(drop=True)


def pareto_t(model) -> pd.DataFrame:
    tvals = model.tvalues.copy()
    tvals = tvals.drop(labels=["Intercept"], errors="ignore")
    df = pd.DataFrame({"Terme": tvals.index, "|t|": np.abs(tvals.values)})
    return df.sort_values("|t|", ascending=False).reset_index(drop=True)


def cooks_distance(model) -> np.ndarray:
    infl = model.get_influence()
    return infl.cooks_distance[0]


def propose_boxcox(y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    Suggest Box-Cox lambda (requires y>0). If not, return None.
    Returns (lambda, shift_applied).
    """
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 6:
        return None, None
    shift = 0.0
    if np.min(y) <= 0:
        shift = abs(np.min(y)) + 1e-6
    y2 = y + shift
    try:
        lam = float(boxcox_normmax(y2, method="mle"))
        return lam, shift
    except Exception:
        return None, None


# ============================================================
# Power / MDE (approx)
# ============================================================
def mde_for_beta(se_beta: float, alpha: float, power: float, df: int) -> float:
    """
    Approx minimal detectable effect on coefficient:
    MDE ≈ (t_{1-alpha/2} + t_{power}) * SE
    (common approximation used in planning)
    """
    t_alpha = float(student_t.ppf(1 - alpha / 2, df))
    t_power = float(student_t.ppf(power, df))
    return (t_alpha + t_power) * float(se_beta)


# ============================================================
# Robust optimization (mean + std over noise via Monte-Carlo)
# ============================================================
def sample_noise(factors: List[Factor], n: int, seed: int = 123) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    for _ in range(n):
        r = {}
        for f in factors:
            if f.role != "noise":
                continue
            if f.kind == "quant":
                v = rng.uniform(float(f.low), float(f.high))
                v = float(round_to_step(np.array([v]), float(f.step))[0])
                r[f.name] = v
            else:
                lv = f.levels or ["A", "B"]
                r[f.name] = lv[rng.randint(0, 2)]
        rows.append(r)
    return pd.DataFrame(rows)


def robust_predict_mean_std(model, factors: List[Factor], control_setting: dict, n_mc: int = 1000, seed: int = 123) -> Tuple[float, float]:
    noise = sample_noise(factors, n=n_mc, seed=seed)
    # build df with all factors
    base = {}
    for f in factors:
        if f.role == "control":
            if f.name in control_setting:
                base[f.name] = control_setting[f.name]
            else:
                # default center/level0
                if f.kind == "quant":
                    base[f.name] = (float(f.low) + float(f.high)) / 2
                else:
                    lv = f.levels or ["A", "B"]
                    base[f.name] = lv[0]
    # expand
    rows = []
    for i in range(len(noise)):
        r = dict(base)
        for col in noise.columns:
            r[col] = noise.iloc[i][col]
        rows.append(r)
    df = pd.DataFrame(rows)
    pred = model.predict(df).astype(float).values
    return float(np.mean(pred)), float(np.std(pred, ddof=1)) if len(pred) > 1 else 0.0


# ============================================================
# PDF report with figures
# ============================================================
def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def build_pdf(title: str, lines: List[str], fig_bytes_list: List[bytes]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4

    y = h - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, title)
    y -= 22

    c.setFont("Helvetica", 10)
    for line in lines:
        if y < 80:
            c.showPage()
            y = h - 40
            c.setFont("Helvetica", 10)
        c.drawString(40, y, line[:120])
        y -= 13

    # figures
    for fb in fig_bytes_list:
        img = ImageReader(io.BytesIO(fb))
        # new page for each figure for readability
        c.showPage()
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, h - 40, "Figure")
        # fit image
        img_w = w - 80
        img_h = h - 120
        c.drawImage(img, 40, 60, width=img_w, height=img_h, preserveAspectRatio=True, anchor="c")

    c.save()
    return buf.getvalue()


# ============================================================
# Reset for new plan
# ============================================================
def reset_for_new_plan(df_real: pd.DataFrame, df_coded: pd.DataFrame, cfg: DesignConfig, factors: List[Factor], notes: List[str]):
    st.session_state.plan_id += 1
    st.session_state.design_cfg = cfg
    st.session_state.factors = factors
    st.session_state.doe_real = df_real
    st.session_state.doe_coded = df_coded
    st.session_state.autopilot_notes = notes

    # results table (atelier)
    res = df_real.copy()
    res["Done"] = False
    res["Comment"] = ""
    res["Operator"] = ""
    res["Machine"] = ""
    res["MaterialLot"] = ""
    res["AmbientTemp"] = np.nan
    res["AmbientHum"] = np.nan
    res["Timestamp"] = ""
    for y in st.session_state.y_cols:
        if y not in res.columns:
            res[y] = np.nan
    st.session_state.results = res

    st.session_state.analysis_cache = None
    st.session_state.analysis_cache_key = None
    st.session_state.last_figures = {}


# ============================================================
# UI NAV
# ============================================================
st.sidebar.markdown("## Navigation")
step = st.sidebar.radio(
    "Étapes",
    [
        "Assistant",
        "Projet",
        "Facteurs",
        "Contraintes",
        "Plan",
        "Exécution",
        "Analyse",
        "Puissance/MDE",
        "Robustesse",
        "Autopilot",
        "Optimisation",
        "Rapport",
    ],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.caption("Nouveau plan ⇒ reset Exécution/Analyse automatique.")


# ============================================================
# ASSISTANT
# ============================================================
if step == "Assistant":
    st.subheader("Assistant — recommandations intelligentes")
    c1, c2 = st.columns([1, 1])
    with c1:
        objectif = st.selectbox(
            "Objectif",
            [
                "Screening (trouver facteurs importants)",
                "Modélisation (effets + interactions)",
                "Optimisation (RSM / surfaces)",
                "Robustesse (contrôle vs bruit)",
                "Contraintes fortes (combos interdits)",
                "Explorer une zone (sans modèle)",
            ],
            index=0
        )
        k = st.number_input("Nombre de facteurs (estimé)", 1, 31, 6, 1)
        max_runs = st.number_input("Essais max (contrainte)", 4, 500, 16, 1)

    with c2:
        has_noise = st.checkbox("J’ai du bruit (ambiance/lot/opérateur)", value=True)
        suspect_curvature = st.checkbox("Je suspecte de la courbure", value=False)
        need_blocks = st.checkbox("Je dois bloquer (jour/opérateur/machine)", value=False)

    recos = []
    if objectif.startswith("Contraintes"):
        recos.append("➡️ **D-optimal sous contraintes** (tu décris règles/combinations interdites).")
        recos.append("   Ensuite: confirmation + éventuellement RSM sur facteurs clés.")
    elif objectif.startswith("Robustesse"):
        recos.append("➡️ **Taguchi (2 niveaux)** + S/N + séparation contrôle/bruit.")
        recos.append("   Puis: **CCD** sur facteurs contrôle importants.")
    elif objectif.startswith("Optimisation") or suspect_curvature:
        recos.append("➡️ **RSM** : CCD (rotatable) ou Box–Behnken (k≥3).")
        recos.append("   Astuce: commence par screening si k>6.")
    elif objectif.startswith("Modélisation"):
        if (2 ** k) <= max_runs and k <= 8:
            recos.append("➡️ **Factoriel complet 2 niveaux** + réplicats/blocs.")
        else:
            recos.append("➡️ **Fractionnaire auto (générateurs)** proche de la limite essais.")
            recos.append("   Si aliasing: fold-over.")
    elif objectif.startswith("Screening"):
        if k <= 7 and (2 ** k) <= max_runs:
            recos.append("➡️ **Factoriel complet 2 niveaux** (screening propre).")
        else:
            recos.append("➡️ **Hadamard** ou **Taguchi** (peu d’essais).")
            recos.append("   Puis: fold-over / RSM sur top facteurs.")
    else:
        recos.append("➡️ **LHS** pour explorer, puis modèle si besoin.")

    if need_blocks:
        recos.append("⚙️ Active les **blocs** (Block) pour éviter un faux effet process.")
    if has_noise:
        recos.append("🛡️ Déclare des facteurs **noise** (bruit) pour activer la robustesse avancée.")
    st.info("\n".join(recos))
    st.caption("Ensuite: Facteurs → Contraintes → Plan.")


# ============================================================
# PROJET (save/load)
# ============================================================
if step == "Projet":
    st.subheader("Projet — sauvegarde / chargement")
    if st.session_state.doe_real is not None and st.session_state.results is not None:
        project = {
            "factors": [asdict(f) for f in st.session_state.factors],
            "constraints": [asdict(r) for r in st.session_state.constraints],
            "design_cfg": asdict(st.session_state.design_cfg) if st.session_state.design_cfg else None,
            "doe_real": st.session_state.doe_real.to_dict(orient="list"),
            "doe_coded": st.session_state.doe_coded.to_dict(orient="list") if st.session_state.doe_coded is not None else None,
            "results": st.session_state.results.to_dict(orient="list"),
            "y_cols": st.session_state.y_cols,
        }
        b = json.dumps(project, ensure_ascii=False).encode("utf-8")
        st.download_button("Télécharger projet (.json)", b, file_name="doe_project_v9.json", mime="application/json")
    else:
        st.info("Génère un plan pour sauvegarder.")

    st.markdown("---")
    up = st.file_uploader("Charger projet (.json)", type=["json"])
    if up is not None:
        try:
            proj = json.loads(up.read().decode("utf-8"))
            st.session_state.factors = [Factor(**f) for f in proj.get("factors", [])]
            st.session_state.constraints = [ConstraintRule(**r) for r in proj.get("constraints", [])]
            dc = proj.get("design_cfg")
            st.session_state.design_cfg = DesignConfig(**dc) if dc else None
            st.session_state.doe_real = pd.DataFrame(proj.get("doe_real"))
            st.session_state.doe_coded = pd.DataFrame(proj.get("doe_coded")) if proj.get("doe_coded") else None
            st.session_state.results = pd.DataFrame(proj.get("results"))
            st.session_state.y_cols = proj.get("y_cols", ["Y"])
            st.session_state.analysis_cache = None
            st.session_state.analysis_cache_key = None
            st.success("Projet chargé ✅")
        except Exception as e:
            st.error(f"Erreur chargement: {e}")


# ============================================================
# FACTEURS
# ============================================================
if step == "Facteurs":
    st.subheader("Facteurs (Contrôle vs Bruit)")
    n = st.number_input("Nombre de facteurs", 1, 31, 6, 1)
    factors: List[Factor] = []
    any_issue = False

    for i in range(int(n)):
        st.markdown(f"### Facteur {i+1}")
        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        with c1:
            name = sanitize_name(st.text_input("Nom", value=f"X{i+1}", key=f"fx_name_{i}"), default=f"X{i+1}")
        with c2:
            kind = st.selectbox("Type", ["quant", "cat"], index=0, key=f"fx_kind_{i}")
        with c3:
            role = st.selectbox("Rôle", ["control", "noise"], index=0, key=f"fx_role_{i}")
        with c4:
            stepv = float(st.number_input("Pas/arrondi (quant)", value=0.0, min_value=0.0, key=f"fx_step_{i}"))

        if kind == "quant":
            c5, c6 = st.columns([1, 1])
            with c5:
                low = float(st.number_input("Bas", value=0.0, key=f"fx_low_{i}"))
            with c6:
                high = float(st.number_input("Haut", value=1.0, key=f"fx_high_{i}"))
            if high == low:
                any_issue = True
                st.warning("⚠️ Bas = Haut.")
            factors.append(Factor(name=name, kind="quant", role=role, low=low, high=high, step=stepv))
        else:
            lv1 = st.text_input("Niveau bas", value="A", key=f"fx_lv1_{i}")
            lv2 = st.text_input("Niveau haut", value="B", key=f"fx_lv2_{i}")
            factors.append(Factor(name=name, kind="cat", role=role, levels=[lv1, lv2]))

    if any_issue:
        st.error("Corrige les facteurs quantitatifs où Bas = Haut.")
    else:
        st.session_state.factors = factors
        st.success("Facteurs enregistrés ✅")


# ============================================================
# CONTRAINTES
# ============================================================
if step == "Contraintes":
    st.subheader("Contraintes (combos interdits + règles conditionnelles)")
    if not st.session_state.factors:
        st.warning("Définis d’abord les facteurs.")
        st.stop()

    factors = st.session_state.factors
    f_names = [f.name for f in factors]

    st.markdown("### 1) Combinaisons interdites")
    st.caption("Ex: Interdire (Mat=B ET Temp=220).")
    with st.expander("Ajouter une combinaison interdite", expanded=False):
        pick = st.multiselect("Quels facteurs entrent dans la règle ?", f_names, default=f_names[:2] if len(f_names) >= 2 else f_names)
        matches = {}
        for fn in pick:
            f = next(x for x in factors if x.name == fn)
            if f.kind == "quant":
                matches[fn] = float(st.number_input(f"Valeur interdite pour {fn}", value=float(f.low), key=f"forb_{fn}"))
            else:
                lv = f.levels or ["A", "B"]
                matches[fn] = st.selectbox(f"Valeur interdite pour {fn}", lv, key=f"forbcat_{fn}")
        if st.button("➕ Ajouter combo interdite"):
            st.session_state.constraints.append(ConstraintRule(kind="forbidden_combo", payload={"matches": matches}))
            st.success("Ajouté ✅")

    st.markdown("### 2) Règles IF ... THEN ...")
    st.caption("Ex: IF Vitesse > 70 THEN Température >= 190")
    with st.expander("Ajouter une règle conditionnelle", expanded=False):
        colA, colB = st.columns([1, 1])
        with colA:
            if_factor = st.selectbox("IF facteur", f_names, key="if_factor")
            if_op = st.selectbox("IF opérateur", ["==", "!=", ">", ">=", "<", "<="], key="if_op")
            if_val = st.text_input("IF valeur (nombre ou texte)", value="0", key="if_val")
        with colB:
            then_factor = st.selectbox("THEN facteur", f_names, key="then_factor")
            then_op = st.selectbox("THEN opérateur", ["==", "!=", ">", ">=", "<", "<="], key="then_op")
            then_val = st.text_input("THEN valeur (nombre ou texte)", value="0", key="then_val")

        # cast value according to factor kind
        def cast_val(fname: str, txt: str):
            f = next(x for x in factors if x.name == fname)
            if f.kind == "quant":
                return float(txt)
            return txt

        if st.button("➕ Ajouter IF/THEN"):
            rule = {
                "if": [{"factor": if_factor, "op": if_op, "value": cast_val(if_factor, if_val)}],
                "then": [{"factor": then_factor, "op": then_op, "value": cast_val(then_factor, then_val)}],
            }
            st.session_state.constraints.append(ConstraintRule(kind="if_then", payload=rule))
            st.success("Ajouté ✅")

    st.markdown("---")
    st.markdown("### Contraintes actuelles")
    if not st.session_state.constraints:
        st.info("Aucune contrainte.")
    else:
        for idx, r in enumerate(st.session_state.constraints):
            st.write(f"**#{idx+1}** — {r.kind} — {r.payload}")
        if st.button("🗑️ Tout supprimer"):
            st.session_state.constraints = []
            st.success("Contraintes supprimées ✅")


# ============================================================
# PLAN
# ============================================================
if step == "Plan":
    st.subheader("Génération du plan")
    if not st.session_state.factors:
        st.warning("Définis d’abord les facteurs.")
        st.stop()

    factors = st.session_state.factors
    names = [f.name for f in factors]
    k = len(names)

    left, right = st.columns([1, 2])
    with left:
        design_type = st.selectbox(
            "Type de plan",
            [
                "Factoriel complet (2 niveaux)",
                "Fractionnaire auto (générateurs)",
                "Screening (Hadamard)",
                "Taguchi (2 niveaux)",
                "CCD (RSM)",
                "Box–Behnken (RSM)",
                "LHS (exploration)",
                "D-optimal sous contraintes",
            ],
            index=1
        )

        seed = st.number_input("Graine (random)", 0, 10000, 42, 1)

        st.markdown("### Exécution")
        replicates = st.number_input("Réplicats", 1, 50, 1, 1)
        center_points = st.number_input("Points centraux", 0, 50, 0, 1)
        n_blocks = st.number_input("Blocs", 1, 30, 1, 1)
        randomize_within_block = st.checkbox("Randomiser dans blocs", value=True)
        randomize_global = st.checkbox("Randomiser global (si 1 bloc)", value=True)

        st.markdown("### Ordre d’essais")
        runorder_mode = st.selectbox("Run order", ["random", "setup_optimized"], index=0)
        if runorder_mode == "setup_optimized":
            st.warning("⚠️ L’ordre 'setup_optimized' améliore l’atelier, mais réduit la randomisation (validité stats).")

        # per-design settings
        frac_target_runs = 16
        frac_search_iters = 800
        frac_include_2fi = True
        taguchi_array = "AUTO"
        ccd_alpha = "rotatable"
        lhs_samples = 20
        dopt_runs = 16
        dopt_model = "main"

        if design_type == "Fractionnaire auto (générateurs)":
            frac_target_runs = st.number_input("Essais cible (puissance de 2)", 4, 256, 16, 1)
            frac_search_iters = st.number_input("Itérations recherche (qualité)", 100, 5000, 800, 100)
            frac_include_2fi = st.checkbox("Score résolution via 2FI", value=True)

        if design_type == "Taguchi (2 niveaux)":
            taguchi_array = st.selectbox("OA", ["AUTO", "L4", "L8", "L16", "L32"], index=0)

        if design_type == "CCD (RSM)":
            ccd_alpha = st.selectbox("Alpha", ["rotatable", "face-centered"], index=0)

        if design_type == "LHS (exploration)":
            lhs_samples = st.number_input("Nb points LHS", 5, 500, 20, 1)

        if design_type == "D-optimal sous contraintes":
            dopt_runs = st.number_input("Nb essais D-optimal", 4, 200, 16, 1)
            dopt_model = st.selectbox("Modèle cible", ["main", "main+2fi"], index=0)

    with right:
        cfg = DesignConfig(
            design_type=design_type,
            random_seed=int(seed),
            replicates=int(replicates),
            center_points=int(center_points),
            n_blocks=int(n_blocks),
            randomize_within_block=bool(randomize_within_block),
            randomize_global=bool(randomize_global),
            frac_target_runs=int(frac_target_runs),
            frac_search_iters=int(frac_search_iters),
            frac_include_2fi=bool(frac_include_2fi),
            ccd_alpha=ccd_alpha,
            lhs_samples=int(lhs_samples),
            taguchi_array=taguchi_array,
            dopt_runs=int(dopt_runs),
            dopt_model=dopt_model,
            runorder_mode=runorder_mode
        )

        if st.button("✅ Générer le plan (reset)"):
            notes = []
            constraints = st.session_state.constraints

            try:
                # build coded design for classical plans; for D-optimal generate in real then map coded
                if design_type == "Factoriel complet (2 niveaux)":
                    X = _ff2n(k)
                    df_coded = pd.DataFrame(X, columns=names)
                    notes.append(f"Factoriel complet: {len(df_coded)} essais.")

                elif design_type == "Fractionnaire auto (générateurs)":
                    X, n1 = auto_fractional_generators(
                        k=k,
                        target_runs=cfg.frac_target_runs,
                        iters=cfg.frac_search_iters,
                        seed=cfg.random_seed,
                        include_2fi=cfg.frac_include_2fi
                    )
                    df_coded = pd.DataFrame(X, columns=names)
                    notes.extend(n1)

                elif design_type == "Screening (Hadamard)":
                    X = _hadamard_screening(k)
                    df_coded = pd.DataFrame(X, columns=names)
                    notes.append(f"Hadamard screening: {len(df_coded)} essais.")

                elif design_type == "Taguchi (2 niveaux)":
                    if cfg.taguchi_array == "AUTO":
                        arr, X = _taguchi_auto(k)
                        notes.append(f"Taguchi AUTO: {arr} ({X.shape[0]} essais).")
                    else:
                        arr = cfg.taguchi_array.upper()
                        if arr == "L4": X = _taguchi_L4()[:, :k]
                        elif arr == "L8": X = _taguchi_L8()[:, :k]
                        elif arr == "L16": X = _taguchi_L16()[:, :k]
                        elif arr == "L32": X = _taguchi_L32()[:, :k]
                        else: raise ValueError("OA Taguchi invalide.")
                        notes.append(f"Taguchi: {arr} ({X.shape[0]} essais).")
                    df_coded = pd.DataFrame(X, columns=names)

                elif design_type == "CCD (RSM)":
                    X = _ccd_design(k, center_points=cfg.center_points, alpha_mode=cfg.ccd_alpha)
                    df_coded = pd.DataFrame(X, columns=names)
                    notes.append(f"CCD: {len(df_coded)} essais (centres inclus).")

                elif design_type == "Box–Behnken (RSM)":
                    X = _bb_design(k, center_points=cfg.center_points)
                    df_coded = pd.DataFrame(X, columns=names)
                    notes.append(f"Box–Behnken: {len(df_coded)} essais (centres inclus).")

                elif design_type == "LHS (exploration)":
                    X = _lhs_design(k, samples=cfg.lhs_samples, seed=cfg.random_seed)
                    df_coded = pd.DataFrame(X, columns=names)
                    notes.append(f"LHS: {len(df_coded)} essais.")

                elif design_type == "D-optimal sous contraintes":
                    # generate candidate set in REAL units, filter constraints, then select D-optimal
                    cand = build_candidate_set(factors, constraints, max_candidates=5000, seed=cfg.random_seed)
                    if len(cand) == 0:
                        st.error("Aucun candidat valide après contraintes.")
                        st.stop()

                    df_real_sel, n1 = greedy_d_optimal(cand, factors, n_runs=cfg.dopt_runs, model=cfg.dopt_model)
                    notes.extend(n1)

                    # convert real->coded for quant (needed for some diagnostics)
                    df_coded = pd.DataFrame({f.name: 0.0 for f in factors})
                    for f in factors:
                        if f.kind == "quant":
                            lo, hi = float(f.low), float(f.high)
                            x = df_real_sel[f.name].astype(float).values
                            coded = 2.0 * (x - lo) / (hi - lo) - 1.0 if (hi - lo) != 0 else 0.0
                            df_coded[f.name] = coded
                        else:
                            lv = f.levels or ["A", "B"]
                            df_coded[f.name] = np.where(df_real_sel[f.name].values == lv[1], 1.0, -1.0)
                    # for D-optimal we already have real
                    df_real = df_real_sel.copy()

                else:
                    raise ValueError("Plan non supporté.")
            except Exception as e:
                st.error(f"Erreur génération: {e}")
                st.stop()

            # For non D-optimal, convert coded->real then apply constraints
            if design_type != "D-optimal sous contraintes":
                # add extra centers if requested and not already included structurally
                if design_type in [
                    "Factoriel complet (2 niveaux)",
                    "Fractionnaire auto (générateurs)",
                    "Screening (Hadamard)",
                    "Taguchi (2 niveaux)",
                    "LHS (exploration)",
                ] and cfg.center_points > 0:
                    centers = pd.DataFrame({n: 0.0 for n in names}, index=range(cfg.center_points))
                    df_coded = pd.concat([df_coded, centers], ignore_index=True)

                # replicates
                df_coded = repeat_df(df_coded, cfg.replicates)
                df_real = build_real_from_coded(df_coded, factors)

                # apply constraints in REAL space (important)
                before = len(df_real)
                df_real = filter_by_constraints(df_real, constraints)
                if len(df_real) < before:
                    notes.append(f"Contraintes: {before - len(df_real)} essais supprimés (invalides).")
                if len(df_real) == 0:
                    st.error("Toutes les lignes ont été éliminées par les contraintes.")
                    st.stop()

                # align coded with filtered real by recomputing coded from filtered real
                # (simple and robust)
                df_coded2 = pd.DataFrame()
                for f in factors:
                    if f.kind == "quant":
                        lo, hi = float(f.low), float(f.high)
                        x = df_real[f.name].astype(float).values
                        coded = 2.0 * (x - lo) / (hi - lo) - 1.0 if (hi - lo) != 0 else 0.0
                        df_coded2[f.name] = coded
                    else:
                        lv = f.levels or ["A", "B"]
                        df_coded2[f.name] = np.where(df_real[f.name].values == lv[1], 1.0, -1.0)
                df_coded = df_coded2

            # blocking
            df_real, df_coded = apply_blocking(df_real, df_coded, cfg.n_blocks, cfg.random_seed, cfg.randomize_within_block)

            # run order
            if cfg.n_blocks <= 1 and cfg.randomize_global:
                df_real = df_real.sample(frac=1, random_state=cfg.random_seed).reset_index(drop=True)
                df_coded = df_coded.loc[df_real.index].reset_index(drop=True)

            if cfg.runorder_mode == "setup_optimized":
                df_real = setup_optimized_order(df_real, factors, seed=cfg.random_seed)
                # rebuild coded from reordered real for consistency
                df_coded = pd.DataFrame()
                for f in factors:
                    if f.kind == "quant":
                        lo, hi = float(f.low), float(f.high)
                        x = df_real[f.name].astype(float).values
                        df_coded[f.name] = 2.0 * (x - lo) / (hi - lo) - 1.0 if (hi - lo) != 0 else 0.0
                    else:
                        lv = f.levels or ["A", "B"]
                        df_coded[f.name] = np.where(df_real[f.name].values == lv[1], 1.0, -1.0)

            # add runorder columns
            df_real = add_runorder(df_real)
            df_coded = add_runorder(df_coded)

            reset_for_new_plan(df_real, df_coded, cfg, factors, notes)

            st.success("Plan généré ✅ (Exécution/Analyse réinitialisées)")
            if notes:
                st.info("\n".join(notes))
            st.dataframe(st.session_state.doe_real, use_container_width=True)

            st.download_button("Télécharger CSV", st.session_state.doe_real.to_csv(index=False).encode("utf-8"),
                               file_name="plan.csv", mime="text/csv")
            st.download_button("Télécharger Excel", make_excel_bytes(st.session_state.doe_real),
                               file_name="plan.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ============================================================
# EXÉCUTION (atelier)
# ============================================================
if step == "Exécution":
    st.subheader("Exécution — mode atelier + traçabilité")
    if st.session_state.results is None:
        st.warning("Génère un plan d’abord.")
        st.stop()

    c1, c2 = st.columns([2, 1])
    with c2:
        st.markdown("### Réponses (Y)")
        new_y = st.text_input("Ajouter une réponse", placeholder="Ex: Retrait, Poids, Y2", value="")
        if st.button("➕ Ajouter Y"):
            ny = sanitize_name(new_y, default="")
            if ny and ny not in st.session_state.y_cols:
                st.session_state.y_cols.append(ny)
                st.session_state.results[ny] = np.nan
                st.success(f"Ajouté: {ny}")
            else:
                st.info("Nom vide ou déjà présent.")

        st.markdown("---")
        st.markdown("### Pré-remplir traçabilité (optionnel)")
        op = st.text_input("Opérateur (défaut)", value="")
        mc = st.text_input("Machine (défaut)", value="")
        lot = st.text_input("Lot matière (défaut)", value="")
        if st.button("Appliquer aux lignes vides"):
            df = st.session_state.results.copy()
            df.loc[df["Operator"] == "", "Operator"] = op
            df.loc[df["Machine"] == "", "Machine"] = mc
            df.loc[df["MaterialLot"] == "", "MaterialLot"] = lot
            st.session_state.results = df
            st.success("Appliqué ✅")

        st.markdown("---")
        if st.button("🕒 Timestamp NOW sur lignes Done"):
            df = st.session_state.results.copy()
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            df.loc[(df["Done"] == True) & (df["Timestamp"] == ""), "Timestamp"] = now
            st.session_state.results = df
            st.success("OK ✅")

        st.markdown("---")
        st.caption("Remplissage:")
        for y in st.session_state.y_cols:
            st.write(f"- {y}: {int(st.session_state.results[y].notna().sum())}/{len(st.session_state.results)}")

    with c1:
        st.session_state.results = st.data_editor(
            st.session_state.results,
            use_container_width=True,
            num_rows="fixed",
            key=f"exec_editor_{st.session_state.plan_id}",
        )


# ============================================================
# ANALYSE
# ============================================================
if step == "Analyse":
    st.subheader("Analyse — aide avancée (ANOVA, Henry, Cook, Box-Cox, S/N)")
    if st.session_state.results is None or st.session_state.doe_coded is None:
        st.warning("Génère un plan et saisis des données.")
        st.stop()

    factors = st.session_state.factors
    df_res = st.session_state.results.copy()
    df_coded = st.session_state.doe_coded.copy()

    y_candidates = [y for y in st.session_state.y_cols if y in df_res.columns]
    y_col = st.selectbox("Réponse Y", options=y_candidates)

    include_inter = st.checkbox("Interactions (entre contrôles quant)", value=True)
    include_quad = st.checkbox("Quadratique (RSM sur contrôles quant)", value=False)
    include_block = st.checkbox("Inclure Block", value=True)
    include_noise = st.checkbox("Inclure facteurs de bruit dans le modèle", value=True)

    sn_mode = st.selectbox("S/N (Taguchi)", ["Smaller-the-better", "Larger-the-better", "Nominal-the-best"], index=0)

    mask = df_res[y_col].notna()
    n_obs = int(mask.sum())
    if n_obs < 6:
        st.warning("Pas assez de points (>=6 conseillé).")
        st.stop()

    df_model = df_res.loc[mask].copy()
    if "Block" not in df_model.columns:
        df_model["Block"] = 1

    formula = build_formula(y_col, factors, include_inter, include_quad, include_block, include_noise=include_noise)
    st.code(formula)

    # guardrail by quick fit
    try:
        tmp = smf.ols(formula=formula, data=df_model).fit()
        p = int(tmp.df_model) + 1
    except Exception:
        p = 12
    if n_obs <= p:
        st.error(f"Modèle trop riche: N={n_obs} <= paramètres~{p}. Réduis le modèle / ajoute essais.")
        st.stop()

    cache_key = stable_hash_df(df_model[[c for c in df_model.columns if c in ["RunOrder", "Block", y_col] + [f.name for f in factors]]]) + f"|{formula}|{sn_mode}|plan={st.session_state.plan_id}"

    if st.button("🚀 Lancer l’analyse"):
        try:
            model = smf.ols(formula=formula, data=df_model).fit()
            try:
                anova_tbl = anova_lm(model, typ=2)
            except Exception:
                anova_tbl = None

            coded_cols = [c for c in df_coded.columns if c not in ["RunOrder", "Block"]]
            df_coded_used = df_coded.loc[mask.values, coded_cols].reset_index(drop=True)
            y_used = df_res.loc[mask, y_col].reset_index(drop=True)

            two_level = is_two_level_only(df_coded_used, coded_cols)
            eff = compute_effects_two_level(df_coded_used, y_used, include_interactions=True) if two_level else None
            pt = pareto_t(model)
            sn = sn_ratio(y_used.values, sn_mode)
            cooks = cooks_distance(model)

            lam, shift = propose_boxcox(y_used.values)

            st.session_state.analysis_cache = {
                "model": model,
                "anova": anova_tbl,
                "two_level": two_level,
                "effects": eff,
                "pareto_t": pt,
                "sn": sn,
                "cooks": cooks,
                "boxcox_lambda": lam,
                "boxcox_shift": shift,
                "df_model": df_model,
                "y_col": y_col,
                "formula": formula
            }
            st.session_state.analysis_cache_key = cache_key
            st.success("Analyse calculée ✅")
        except Exception as e:
            st.error(f"Erreur analyse: {e}")

    if st.session_state.analysis_cache is None:
        st.info("Clique sur **Lancer l’analyse**.")
        st.stop()

    A = st.session_state.analysis_cache
    model = A["model"]
    anova_tbl = A["anova"]
    two_level = A["two_level"]
    eff = A["effects"]
    pt = A["pareto_t"]
    sn = A["sn"]
    cooks = A["cooks"]
    lam = A["boxcox_lambda"]
    shift = A["boxcox_shift"]
    df_model_used = A["df_model"]

    # Summary
    st.markdown("## Résumé")
    fit = pd.DataFrame({
        "R²": [model.rsquared],
        "R² ajusté": [model.rsquared_adj],
        "AIC": [model.aic],
        "BIC": [model.bic],
        "N": [int(model.nobs)],
        "S/N": [sn],
    })
    st.dataframe(fit, use_container_width=True)

    st.markdown("## Coefficients")
    coef = pd.DataFrame({"coef": model.params, "std_err": model.bse, "t": model.tvalues, "p_value": model.pvalues})
    st.dataframe(coef, use_container_width=True)

    st.markdown("## ANOVA")
    if anova_tbl is None:
        st.warning("ANOVA indisponible.")
    else:
        st.dataframe(anova_tbl, use_container_width=True)

    # Pareto
    st.markdown("## Pareto")
    figs = {}

    if two_level and eff is not None:
        st.caption("Design 2 niveaux → Pareto des effets.")
        st.dataframe(eff, use_container_width=True)
        figp = plt.figure()
        plt.bar(range(len(eff)), eff["|Effet|"].values)
        plt.xticks(range(len(eff)), eff["Terme"].values, rotation=90)
        plt.ylabel("|Effet|")
        plt.title("Pareto des effets")
        plt.tight_layout()
        st.pyplot(figp)
        figs["Pareto"] = fig_to_png_bytes(figp)
    else:
        st.caption("Design non 2 niveaux → Pareto |t|.")
        st.dataframe(pt, use_container_width=True)
        figp = plt.figure()
        plt.bar(range(len(pt)), pt["|t|"].values)
        plt.xticks(range(len(pt)), pt["Terme"].values, rotation=90)
        plt.ylabel("|t|")
        plt.title("Pareto |t|")
        plt.tight_layout()
        st.pyplot(figp)
        figs["Pareto"] = fig_to_png_bytes(figp)

    # Diagnostics
    st.markdown("## Diagnostics")
    resid = model.resid
    fitted = model.fittedvalues

    f1 = plt.figure()
    plt.scatter(fitted, resid)
    plt.axhline(0)
    plt.xlabel("Ajustés")
    plt.ylabel("Résidus")
    plt.title("Résidus vs ajustés")
    st.pyplot(f1)
    figs["Resid_vs_Fit"] = fig_to_png_bytes(f1)

    f2 = plt.figure()
    (osm, osr), (slope, intercept, r) = probplot(resid, dist="norm")
    plt.scatter(osm, osr)
    xline = np.array([np.min(osm), np.max(osm)])
    plt.plot(xline, slope * xline + intercept)
    plt.title(f"Droite de Henry (r={r:.3f})")
    plt.xlabel("Quantiles théoriques")
    plt.ylabel("Résidus ordonnés")
    st.pyplot(f2)
    figs["Henry"] = fig_to_png_bytes(f2)

    f3 = plt.figure()
    plt.scatter(df_model_used["RunOrder"], df_model_used[A["y_col"]])
    plt.xlabel("RunOrder")
    plt.ylabel(A["y_col"])
    plt.title("Réponse vs ordre (dérive?)")
    st.pyplot(f3)
    figs["Y_vs_Order"] = fig_to_png_bytes(f3)

    st.markdown("## Points influents (Cook)")
    df_cook = pd.DataFrame({"RunIndex": np.arange(1, len(cooks) + 1), "CookD": cooks})
    st.dataframe(df_cook.sort_values("CookD", ascending=False).head(10), use_container_width=True)
    f4 = plt.figure()
    plt.stem(df_cook["RunIndex"], df_cook["CookD"], basefmt=" ")
    plt.xlabel("Run (dans l'analyse)")
    plt.ylabel("Cook's D")
    plt.title("Influence (Cook)")
    plt.tight_layout()
    st.pyplot(f4)
    figs["Cook"] = fig_to_png_bytes(f4)

    st.markdown("## Box-Cox (proposition)")
    if lam is None:
        st.info("Box-Cox non proposé (pas assez de points ou données trop difficiles).")
    else:
        if shift and shift > 0:
            st.info(f"Proposition: appliquer Box-Cox avec λ≈{lam:.3f} après décalage +{shift:.6g} (car Y≤0).")
        else:
            st.info(f"Proposition: appliquer Box-Cox avec λ≈{lam:.3f} (Y>0).")

    # store figs for PDF
    st.session_state.last_figures = figs


# ============================================================
# PUISSANCE / MDE
# ============================================================
if step == "Puissance/MDE":
    st.subheader("Puissance / Effet minimal détectable (MDE)")
    st.caption("Approximation utile pour planifier: MDE ≈ (t_alpha + t_power) * SE_beta")

    if st.session_state.analysis_cache is None:
        st.warning("Fais d’abord une analyse (onglet Analyse).")
        st.stop()

    A = st.session_state.analysis_cache
    model = A["model"]
    coef = model.params
    bse = model.bse
    df = int(model.df_resid)

    alpha = st.slider("Alpha (risque de type I)", 0.001, 0.20, 0.05, 0.001)
    power = st.slider("Puissance cible", 0.50, 0.99, 0.80, 0.01)

    out = []
    for name in bse.index:
        if name == "Intercept":
            continue
        mde = mde_for_beta(float(bse[name]), alpha=alpha, power=power, df=df)
        out.append((name, float(bse[name]), float(mde)))

    df_mde = pd.DataFrame(out, columns=["Terme", "SE(beta)", "MDE(beta)"])
    st.dataframe(df_mde.sort_values("MDE(beta)"), use_container_width=True)

    st.caption("Interprétation: si ton effet réel < MDE, tu risques de ne pas le détecter avec ce plan.")


# ============================================================
# ROBUSTESSE (avancée)
# ============================================================
if step == "Robustesse":
    st.subheader("Robustesse avancée — mean + std via Monte-Carlo sur facteurs bruit")
    if st.session_state.analysis_cache is None:
        st.warning("Fais une analyse (modèle) d’abord.")
        st.stop()

    factors = st.session_state.factors
    noise = [f for f in factors if f.role == "noise"]
    control_q = [f for f in factors if f.role == "control" and f.kind == "quant"]
    if len(noise) == 0:
        st.info("Aucun facteur noise déclaré. (Facteurs → rôle=noise)")
        st.stop()
    if len(control_q) == 0:
        st.info("Aucun facteur control quantitatif à optimiser.")
        st.stop()

    A = st.session_state.analysis_cache
    model = A["model"]
    y_col = A["y_col"]

    st.caption("L’idée: on fixe les contrôles, on tire au hasard les bruits, on estime mean/std(Y).")
    n_mc = st.slider("Nb simulations Monte-Carlo", 200, 10000, 1000, 200)

    st.markdown("### Définir un réglage contrôle (à tester)")
    control_setting = {}
    for f in control_q:
        control_setting[f.name] = float(st.number_input(f"{f.name}", value=(float(f.low)+float(f.high))/2.0))

    if st.button("🧪 Estimer robustesse (mean/std)"):
        mu, sd = robust_predict_mean_std(model, factors, control_setting, n_mc=n_mc, seed=123)
        st.success(f"Résultat: mean={mu:.4g} ; std={sd:.4g} (sur bruit simulé)")

    st.markdown("---")
    st.markdown("### Optimisation robuste (chercher contrôle qui maximise mean et minimise std)")
    w_mean = st.slider("Poids moyenne", 0.0, 5.0, 1.0, 0.1)
    w_std = st.slider("Poids écart-type (à minimiser)", 0.0, 5.0, 1.0, 0.1)
    n_search = st.slider("Nb essais (recherche)", 200, 20000, 3000, 200)

    if st.button("🎯 Optimiser robuste"):
        rng = np.random.RandomState(42)
        best = None
        best_score = -1e18
        best_mu = None
        best_sd = None

        lows = np.array([float(f.low) for f in control_q])
        highs = np.array([float(f.high) for f in control_q])

        for _ in range(int(n_search)):
            x = rng.uniform(lows, highs)
            setting = {control_q[i].name: float(x[i]) for i in range(len(control_q))}
            mu, sd = robust_predict_mean_std(model, factors, setting, n_mc=n_mc, seed=rng.randint(0, 10**9))
            # score: maximize mean, minimize std
            score = w_mean * mu - w_std * sd
            if score > best_score:
                best_score = score
                best = setting
                best_mu, best_sd = mu, sd

        st.success(f"Meilleur score={best_score:.4g} ; mean={best_mu:.4g} ; std={best_sd:.4g}")
        st.dataframe(pd.DataFrame([best]), use_container_width=True)


# ============================================================
# AUTOPILOT
# ============================================================
if step == "Autopilot":
    st.subheader("Autopilot — recommandations + plan suivant")
    if st.session_state.doe_real is None:
        st.info("Génère un plan d’abord.")
        st.stop()
    if st.session_state.analysis_cache is None:
        st.warning("Fais une analyse d’abord.")
        st.stop()

    cfg = st.session_state.design_cfg
    factors = st.session_state.factors
    A = st.session_state.analysis_cache
    model = A["model"]
    pt = pareto_t(model)

    recos = []
    if cfg and cfg.design_type in ["Screening (Hadamard)", "Taguchi (2 niveaux)", "Fractionnaire auto (générateurs)"]:
        recos.append("Tu es en screening/fractionnaire → étape suivante: fold-over ou RSM sur top facteurs.")
    if model.rsquared_adj < 0.6:
        recos.append("R² ajusté modéré → vérifie outliers/Box-Cox, ou ajoute essais, ou passe en RSM si courbure.")
    if A.get("boxcox_lambda") is not None:
        recos.append("Box-Cox proposé → possible amélioration des hypothèses (normalité / variance).")
    recos.append("Toujours: plan de confirmation (2–4 essais) autour du réglage recommandé.")

    st.markdown("### Recommandations")
    st.write("\n".join([f"- {r}" for r in recos]))

    st.markdown("---")
    st.markdown("### Plan suivant (génération automatique)")
    action = st.selectbox(
        "Action",
        ["Fold-over (clarifier alias)", "Passer en RSM (CCD) sur top contrôles", "Plan de confirmation (répéter meilleur run)"],
        index=1
    )

    # detect top control quant factors from t
    control_q = [f.name for f in factors if f.role == "control" and f.kind == "quant"]
    ranked = [t for t in pt["Terme"].tolist() if t in control_q]
    default_top = ranked[:3] if len(ranked) >= 3 else control_q[:min(3, len(control_q))]

    if action == "Fold-over (clarifier alias)":
        st.caption("Fold-over disponible pour designs 2 niveaux (-1/+1).")
        reps = st.number_input("Réplicats", 1, 10, 1, 1)
        centers = st.number_input("Centres ajoutés", 0, 20, 0, 1)
        if st.button("🚀 Générer fold-over"):
            df_coded = st.session_state.doe_coded.copy()
            cols = [c for c in df_coded.columns if c not in ["RunOrder", "Block"]]
            if not is_two_level_only(df_coded, cols):
                st.error("Fold-over nécessite un design 2 niveaux (-1/+1).")
                st.stop()
            X = df_coded[cols].astype(float).values
            X2 = _foldover(X)
            df_coded2 = pd.DataFrame(X2, columns=cols)
            if centers > 0:
                df_coded2 = pd.concat([df_coded2, pd.DataFrame({c: 0.0 for c in cols}, index=range(int(centers)))], ignore_index=True)
            df_coded2 = repeat_df(df_coded2, int(reps))

            df_real2 = build_real_from_coded(df_coded2, factors)
            df_real2["Block"] = 1
            df_coded2["Block"] = 1
            df_real2 = add_runorder(df_real2)
            df_coded2 = add_runorder(df_coded2)

            cfg2 = DesignConfig(design_type="Fold-over", random_seed=42)
            notes = ["Autopilot: fold-over (clarification alias)."]
            reset_for_new_plan(df_real2, df_coded2, cfg2, factors, notes)
            st.success("Nouveau plan fold-over généré ✅")

    elif action == "Passer en RSM (CCD) sur top contrôles":
        chosen = st.multiselect("Facteurs contrôle (quant) pour CCD", control_q, default=default_top)
        centers = st.number_input("Centres CCD", 0, 20, 4, 1)
        alpha = st.selectbox("Alpha CCD", ["rotatable", "face-centered"], index=0)
        if st.button("🚀 Générer CCD (RSM)"):
            if len(chosen) < 2:
                st.error("Choisis au moins 2 facteurs.")
                st.stop()
            # Build new coded for chosen, keep others fixed at center/level0
            chosen_factors = [f for f in factors if f.name in chosen]
            k2 = len(chosen_factors)
            X = _ccd_design(k2, center_points=int(centers), alpha_mode=alpha)
            df_coded_small = pd.DataFrame(X, columns=[f.name for f in chosen_factors])

            df_coded2 = pd.DataFrame()
            for f in factors:
                if f.name in df_coded_small.columns:
                    df_coded2[f.name] = df_coded_small[f.name]
                else:
                    df_coded2[f.name] = 0.0 if f.kind == "quant" else -1.0

            df_real2 = build_real_from_coded(df_coded2, factors)
            df_real2["Block"] = 1
            df_coded2["Block"] = 1
            df_real2 = add_runorder(df_real2)
            df_coded2 = add_runorder(df_coded2)

            cfg2 = DesignConfig(design_type="CCD (RSM)", random_seed=42, center_points=int(centers), ccd_alpha=alpha)
            notes = [f"Autopilot: CCD sur {', '.join(chosen)} (autres facteurs fixés)."]
            reset_for_new_plan(df_real2, df_coded2, cfg2, factors, notes)
            st.success("Nouveau plan CCD généré ✅")

    else:
        st.caption("Répète le meilleur run mesuré (confirmation).")
        if st.session_state.results is None:
            st.stop()
        y = A["y_col"]
        df = st.session_state.results.copy()
        if df[y].notna().sum() == 0:
            st.warning("Pas de Y mesuré.")
            st.stop()
        best_idx = df[y].astype(float).idxmax()
        best_row = df.loc[best_idx].copy()
        nrep = st.number_input("Nb répétitions", 2, 50, 4, 1)
        if st.button("🚀 Générer plan confirmation"):
            # take factor columns from doe_real
            fac_cols = [f.name for f in factors]
            rows = []
            for _ in range(int(nrep)):
                rows.append({c: best_row[c] for c in fac_cols})
            df_real2 = pd.DataFrame(rows)
            df_real2["Block"] = 1
            df_real2 = add_runorder(df_real2)

            # coded approx
            df_coded2 = pd.DataFrame()
            for f in factors:
                if f.kind == "quant":
                    lo, hi = float(f.low), float(f.high)
                    x = df_real2[f.name].astype(float).values
                    df_coded2[f.name] = 2.0 * (x - lo) / (hi - lo) - 1.0 if (hi - lo) != 0 else 0.0
                else:
                    lv = f.levels or ["A", "B"]
                    df_coded2[f.name] = np.where(df_real2[f.name].values == lv[1], 1.0, -1.0)
            df_coded2["Block"] = 1
            df_coded2 = add_runorder(df_coded2)

            cfg2 = DesignConfig(design_type="Confirmation", random_seed=42)
            notes = ["Autopilot: plan de confirmation sur meilleur run."]
            reset_for_new_plan(df_real2, df_coded2, cfg2, factors, notes)
            st.success("Plan confirmation généré ✅")


# ============================================================
# OPTIMISATION (désirabilité classique)
# ============================================================
if step == "Optimisation":
    st.subheader("Optimisation multi-réponses (désirabilité)")
    if st.session_state.results is None:
        st.warning("Génère un plan et saisis des résultats.")
        st.stop()
    factors = st.session_state.factors
    df_res = st.session_state.results.copy()

    y_candidates = [y for y in st.session_state.y_cols if y in df_res.columns]
    selected_y = st.multiselect("Réponses à optimiser", y_candidates, default=y_candidates[:1] if y_candidates else [])
    if not selected_y:
        st.info("Choisis au moins une réponse.")
        st.stop()

    include_inter = st.checkbox("Interactions (contrôles quant)", value=True)
    include_quad = st.checkbox("Quadratique (contrôles quant)", value=False)
    include_block = st.checkbox("Inclure Block", value=True)
    include_noise = st.checkbox("Inclure bruit dans modèles", value=True)

    goals = {}
    st.markdown("### Objectifs")
    for y in selected_y:
        st.markdown(f"**{y}**")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            goal = st.selectbox(f"Objectif ({y})", ["Maximiser", "Minimiser", "Cibler"], key=f"g_{y}")
        with c2:
            low = float(st.number_input(f"Seuil bas {y}", value=0.0, key=f"lo_{y}"))
        with c3:
            high = float(st.number_input(f"Seuil haut {y}", value=1.0, key=f"hi_{y}"))
        with c4:
            weight = float(st.number_input(f"Poids {y}", value=1.0, min_value=1.0, key=f"w_{y}"))
        target = None
        if goal == "Cibler":
            target = float(st.number_input(f"Cible {y}", value=(low + high)/2, key=f"t_{y}"))
        goals[y] = {"goal": goal, "low": low, "high": high, "target": target, "weight": weight}

    # fit models
    models = {}
    for y in selected_y:
        m = df_res[y].notna()
        if int(m.sum()) < 6:
            st.error(f"Pas assez de points pour {y}.")
            st.stop()
        dfm = df_res.loc[m].copy()
        if "Block" not in dfm.columns:
            dfm["Block"] = 1
        formula = build_formula(y, factors, include_inter, include_quad, include_block, include_noise=include_noise)
        models[y] = smf.ols(formula=formula, data=dfm).fit()

    def desirability_single(y_pred: float, goal: str, low: float, high: float, target: Optional[float]=None) -> float:
        if goal == "Maximiser":
            if y_pred <= low: return 0.0
            if y_pred >= high: return 1.0
            return (y_pred - low) / (high - low)
        if goal == "Minimiser":
            if y_pred >= high: return 0.0
            if y_pred <= low: return 1.0
            return (high - y_pred) / (high - low)
        if target is None:
            target = (low + high) / 2
        if y_pred <= low or y_pred >= high: return 0.0
        if y_pred == target: return 1.0
        if y_pred < target:
            return (y_pred - low) / (target - low)
        return (high - y_pred) / (high - target)

    control_q = [f for f in factors if f.role == "control" and f.kind == "quant"]
    if len(control_q) == 0:
        st.error("Aucun facteur contrôle quantitatif à optimiser.")
        st.stop()

    n_samples = st.slider("Recherche (essais virtuels)", 500, 50000, 8000, 500)
    if st.button("🎯 Optimiser"):
        rng = np.random.RandomState(42)
        lows = np.array([float(f.low) for f in control_q])
        highs = np.array([float(f.high) for f in control_q])

        bestD = -1.0
        best_x = None
        best_preds = None

        for _ in range(int(n_samples)):
            x = rng.uniform(lows, highs)
            setting = {control_q[i].name: float(x[i]) for i in range(len(control_q))}
            preds = {}
            d_list = []
            for y, mod in models.items():
                yp = float(mod.predict(pd.DataFrame([setting]))[0])
                preds[y] = yp
                g = goals[y]
                d = desirability_single(yp, g["goal"], g["low"], g["high"], g.get("target"))
                d_list.append(d ** g.get("weight", 1.0))
            D = float(np.prod(d_list) ** (1.0 / max(1, len(d_list))))
            if D > bestD:
                bestD = D
                best_x = setting
                best_preds = preds

        st.success(f"Désirabilité globale = {bestD:.3f}")
        st.dataframe(pd.DataFrame([best_x]), use_container_width=True)
        st.dataframe(pd.DataFrame([best_preds]), use_container_width=True)


# ============================================================
# RAPPORT
# ============================================================
if step == "Rapport":
    st.subheader("Rapport PDF (avec figures) + Markdown")
    if st.session_state.doe_real is None:
        st.info("Génère un plan d’abord.")
        st.stop()

    cfg = st.session_state.design_cfg
    factors = st.session_state.factors
    lines = []
    lines.append(f"Plan ID: {st.session_state.plan_id}")
    if cfg:
        lines.append(f"Type: {cfg.design_type}")
        lines.append(f"Réplicats: {cfg.replicates}")
        lines.append(f"Centres: {cfg.center_points}")
        lines.append(f"Blocs: {cfg.n_blocks}")
        lines.append(f"RunOrder: {cfg.runorder_mode}")

    if st.session_state.autopilot_notes:
        lines.append("---- Notes ----")
        lines.extend(st.session_state.autopilot_notes)

    if st.session_state.constraints:
        lines.append("---- Contraintes ----")
        for r in st.session_state.constraints:
            lines.append(f"{r.kind}: {r.payload}")

    lines.append("---- Facteurs ----")
    for f in factors:
        if f.kind == "quant":
            lines.append(f"{f.name} ({f.role}): [{f.low}, {f.high}] pas={f.step}")
        else:
            lines.append(f"{f.name} ({f.role}): cat {f.levels}")

    if st.session_state.results is not None:
        lines.append("---- Remplissage ----")
        for y in st.session_state.y_cols:
            if y in st.session_state.results.columns:
                lines.append(f"{y}: {int(st.session_state.results[y].notna().sum())}/{len(st.session_state.results)}")

    fig_bytes_list = []
    if st.session_state.last_figures:
        # attach in a stable order
        for key in ["Pareto", "Resid_vs_Fit", "Henry", "Y_vs_Order", "Cook"]:
            if key in st.session_state.last_figures:
                fig_bytes_list.append(st.session_state.last_figures[key])

    pdf = build_pdf("Rapport DOE Copilot V9 FINAL", lines, fig_bytes_list)
    st.download_button("Télécharger PDF", pdf, file_name="rapport_doe_v9_final.pdf", mime="application/pdf")

    md = ["# Rapport DOE Copilot V9 FINAL\n"] + [f"- {l}" for l in lines]
    md.append("\n## Aperçu plan (10 lignes)\n")
    md.append(st.session_state.doe_real.head(10).to_markdown(index=False))
    st.download_button("Télécharger Markdown", "\n".join(md).encode("utf-8"),
                       file_name="rapport_doe_v9_final.md", mime="text/markdown")

    st.markdown("### Aperçu du plan")
    st.dataframe(st.session_state.doe_real.head(30), use_container_width=True)
