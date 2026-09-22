"""
SmartZ-EDU — bản một file (single-file) để deploy lên Streamlit Community Cloud.
Chuẩn hoá điểm số khi phổ điểm không thuần nhất bằng GMM thích ứng và Z-score lượng tử hoá Z_q.
"""
from __future__ import annotations

import io
import re
import unicodedata


# ===========================================================================
# PHẦN 1 — LÕI TÍNH TOÁN
# ===========================================================================
"""
Lõi tính toán của SmartZ-EDU.

Toàn bộ công thức bám sát mục 3.2 – 3.4 của báo cáo:
  - Cơ chế thích ứng: GMM k = 1..4, chọn k có BIC nhỏ nhất, diễn giải ΔBIC theo thang Raftery.
  - Kiểm định tỉ số hợp lý bằng bootstrap tham số (k = 1 so với k = 2).
  - Ba chỉ số: Z truyền thống, Z* (GMM mềm), Z_q (lượng tử hoá theo hỗn hợp).
  - Kiểm tra tính đơn điệu của Z* trên lưới bước 0.01.
  - γ_cao, ngưỡng điểm x* tương đương, ngoại lệ sư phạm.
Cấu hình tái lập: random_state = 42, n_init = 20, covariance_type = 'full'.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import brentq
from scipy.signal import argrelextrema
from sklearn.mixture import GaussianMixture

RANDOM_STATE = 42
N_INIT = 20
COV_TYPE = "full"
K_MAX = 4
GRID_STEP = 0.01


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def raftery_label(delta_bic: float) -> str:
    """Diễn giải chênh lệch BIC theo thang Raftery (1995)."""
    d = abs(delta_bic)
    if d <= 2:
        return "Không phân biệt được"
    if d <= 6:
        return "Bằng chứng yếu"
    if d <= 10:
        return "Bằng chứng mạnh"
    return "Bằng chứng rất mạnh"


def fit_gmm(x: np.ndarray, k: int, n_init: int = N_INIT,
            random_state: int = RANDOM_STATE, cov_type: str = COV_TYPE) -> GaussianMixture:
    g = GaussianMixture(n_components=k, covariance_type=cov_type, n_init=n_init,
                        random_state=random_state)
    g.fit(np.asarray(x, dtype=float).reshape(-1, 1))
    return g


@dataclass
class MixParams:
    """Tham số hỗn hợp 1 chiều, sắp theo trung bình tăng dần."""
    weights: np.ndarray
    means: np.ndarray
    sds: np.ndarray

    @property
    def k(self) -> int:
        return len(self.means)

    @classmethod
    def from_gmm(cls, g: GaussianMixture) -> "MixParams":
        w = g.weights_.ravel()
        m = g.means_.ravel()
        cov = g.covariances_
        if g.covariance_type == "full":
            v = cov.reshape(len(w), -1)[:, 0]
        elif g.covariance_type == "tied":
            v = np.full(len(w), np.ravel(cov)[0])
        elif g.covariance_type == "diag":
            v = cov.ravel()
        else:  # spherical
            v = cov.ravel()
        order = np.argsort(m)
        return cls(w[order], m[order], np.sqrt(v[order]))

    # --- các đại lượng của hỗn hợp -------------------------------------
    def comp_pdf(self, x):
        x = np.asarray(x, dtype=float)[:, None]
        return self.weights * stats.norm.pdf(x, self.means, self.sds)

    def pdf(self, x):
        return self.comp_pdf(x).sum(axis=1)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)[:, None]
        return (self.weights * stats.norm.cdf(x, self.means, self.sds)).sum(axis=1)

    def gamma(self, x):
        """γ_j(x) — xác suất hậu nghiệm thuộc thành phần j (cột j)."""
        p = self.comp_pdf(x)
        s = p.sum(axis=1, keepdims=True)
        s[s == 0] = np.finfo(float).tiny
        return p / s

    def gamma_high(self, x):
        """γ_cao(x): xác suất thuộc thành phần có trung bình cao nhất."""
        return self.gamma(x)[:, -1]

    def z_soft(self, x):
        """Z* = Σ γ_j(x) · (x − μ_j)/σ_j."""
        x = np.asarray(x, dtype=float)
        zj = (x[:, None] - self.means) / self.sds
        return (self.gamma(x) * zj).sum(axis=1)

    def z_quantile(self, x):
        """Z_q = Φ⁻¹(F_mix(x)) — luôn đơn điệu tăng nghiêm ngặt (Định lý 2)."""
        F = np.clip(self.cdf(x), 1e-12, 1 - 1e-12)
        return stats.norm.ppf(F)


# ---------------------------------------------------------------------------
# Bước 1 – Thống kê mô tả
# ---------------------------------------------------------------------------
def describe_by_group(x: np.ndarray, groups: np.ndarray | None):
    rows = []

    def _row(name, v):
        v = np.asarray(v, dtype=float)
        sw_p = stats.shapiro(v).pvalue if 3 <= len(v) <= 5000 else np.nan
        return {
            "Nhóm": name, "N": len(v), "Trung bình": v.mean(),
            "Độ lệch chuẩn": v.std(ddof=1) if len(v) > 1 else np.nan,
            "Trung vị": np.median(v), "Độ lệch (skew)": stats.skew(v) if len(v) > 2 else np.nan,
            "Shapiro–Wilk p": sw_p,
        }

    if groups is not None:
        for gname in sorted(pd_unique(groups)):
            rows.append(_row(str(gname), x[groups == gname]))
    rows.append(_row("Toàn khối", x))
    return rows


def pd_unique(a):
    seen, out = set(), []
    for v in a:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ---------------------------------------------------------------------------
# Bước 2 – Cơ chế thích ứng + bootstrap LRT
# ---------------------------------------------------------------------------
def adaptive_selection(x: np.ndarray, k_max: int = K_MAX, n_init: int = N_INIT,
                       random_state: int = RANDOM_STATE):
    x = np.asarray(x, dtype=float)
    k_max = int(min(k_max, max(1, len(np.unique(x)) - 1)))
    models, bics = {}, {}
    for k in range(1, k_max + 1):
        g = fit_gmm(x, k, n_init=n_init, random_state=random_state)
        models[k] = g
        bics[k] = g.bic(x.reshape(-1, 1))
    k_best = min(bics, key=bics.get)
    sorted_bic = sorted(bics.values())
    d_runner = sorted_bic[1] - sorted_bic[0] if len(sorted_bic) > 1 else np.nan
    d_1_to_2 = bics[1] - bics[2] if 2 in bics else np.nan
    return {
        "models": models, "bics": bics, "k_best": k_best,
        "delta_runner_up": d_runner, "delta_1_to_2": d_1_to_2,
    }


def bootstrap_lrt(x: np.ndarray, B: int = 200, n_init_boot: int = 5,
                  random_state: int = RANDOM_STATE, progress=None):
    """Kiểm định tỉ số hợp lý H0: k = 1 so với H1: k = 2 bằng bootstrap tham số."""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    n = len(x)
    g1 = fit_gmm(x, 1)
    g2 = fit_gmm(x, 2)
    lrt_obs = 2 * n * (g2.score(x) - g1.score(x))
    mu, sd = g1.means_.ravel()[0], np.sqrt(g1.covariances_.ravel()[0])
    rng = np.random.default_rng(random_state)
    null = np.empty(B)
    for b in range(B):
        xb = rng.normal(mu, sd, size=n).reshape(-1, 1)
        h1 = fit_gmm(xb, 1, n_init=1)
        h2 = fit_gmm(xb, 2, n_init=n_init_boot, random_state=random_state + b)
        null[b] = max(0.0, 2 * n * (h2.score(xb) - h1.score(xb)))
        if progress is not None:
            progress((b + 1) / B)
    p = (1 + np.sum(null >= lrt_obs)) / (B + 1)
    return {"lrt_obs": lrt_obs, "crit95": float(np.quantile(null, 0.95)),
            "p_value": float(p), "B": B, "null": null}


def count_modes(mix: MixParams, lo: float, hi: float) -> tuple[int, np.ndarray]:
    grid = np.arange(lo, hi + GRID_STEP / 2, GRID_STEP)
    dens = mix.pdf(grid)
    idx = argrelextrema(dens, np.greater)[0]
    return len(idx), grid[idx]


# ---------------------------------------------------------------------------
# Bước 4 – Kiểm tra tính đơn điệu của Z*
# ---------------------------------------------------------------------------
def monotonicity_check(mix: MixParams, lo: float, hi: float):
    """Quét lưới bước 0.01, trả về các khoảng mà Z* giảm."""
    grid = np.round(np.arange(lo, hi + GRID_STEP / 2, GRID_STEP), 4)
    z = mix.z_soft(grid)
    dec = np.diff(z) < -1e-12
    intervals = []
    i = 0
    while i < len(dec):
        if dec[i]:
            j = i
            while j + 1 < len(dec) and dec[j + 1]:
                j += 1
            intervals.append((float(grid[i]), float(grid[j + 1])))
            i = j + 1
        else:
            i += 1
    # Điều kiện Δ ≤ 2σ của Định lý 1 (dùng σ gộp, chỉ để tham khảo khi k = 2)
    theorem = None
    if mix.k == 2:
        delta = mix.means[1] - mix.means[0]
        sigma_pool = float(np.sqrt(np.sum(mix.weights * mix.sds ** 2)))
        theorem = {"delta": float(delta), "sigma_pool": sigma_pool,
                   "ratio": float(delta / sigma_pool),
                   "predict_monotone": bool(delta <= 2 * sigma_pool)}
    return {"monotone": len(intervals) == 0, "intervals": intervals, "theorem": theorem}


# ---------------------------------------------------------------------------
# Bước 5 – ngưỡng x* tương đương
# ---------------------------------------------------------------------------
def _gamma_crossing(mix: MixParams, threshold: float, lo: float, hi: float) -> float | None:
    """Nghiệm đầu tiên (tính từ điểm thấp) của γ_cao(x) = threshold, làm tròn 2 chữ số."""
    grid = np.arange(lo, hi + GRID_STEP / 2, GRID_STEP)
    f = mix.gamma_high(grid) - threshold
    idx = np.where((f[:-1] < 0) & (f[1:] >= 0))[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    root = brentq(lambda t: mix.gamma_high(np.array([t]))[0] - threshold, grid[i], grid[i + 1])
    return round(float(root), 2)


def x_star_high(mix: MixParams, threshold: float, lo: float, hi: float) -> float | None:
    """Ngưỡng điểm x* tương đương với γ_cao = threshold (ngoại lệ chiều lên: điểm ≥ x*)."""
    return _gamma_crossing(mix, threshold, lo, hi)


def x_star_low(mix: MixParams, threshold: float, lo: float, hi: float) -> float | None:
    """Ngưỡng điểm x* tương đương với γ_cao = threshold (ngoại lệ chiều xuống: điểm ≤ x*)."""
    return _gamma_crossing(mix, threshold, lo, hi)


# ---------------------------------------------------------------------------
# Tổng hợp cho một cột điểm
# ---------------------------------------------------------------------------
@dataclass
class ColumnResult:
    name: str
    x: np.ndarray
    selection: dict
    k: int
    mix: MixParams
    lo: float
    hi: float
    mono: dict | None
    n_modes: int
    mode_locs: np.ndarray
    scores: dict = field(default_factory=dict)  # tên chỉ số -> mảng giá trị


def analyse_column(name: str, x: np.ndarray, groups: np.ndarray | None,
                   k_override: int | None = None, scale=(0.0, 10.0)) -> ColumnResult:
    x = np.asarray(x, dtype=float)
    sel = adaptive_selection(x)
    k = k_override if k_override else sel["k_best"]
    k = min(k, max(sel["models"]))
    mix = MixParams.from_gmm(sel["models"][k])
    lo = float(min(scale[0], x.min()))
    hi = float(max(scale[1], x.max()))
    n_modes, mode_locs = count_modes(mix, lo, hi)

    scores = {}
    z_trad = (x - x.mean()) / x.std(ddof=0)
    scores["Z truyền thống"] = z_trad
    if groups is not None:
        zg = np.full_like(x, np.nan)
        for gname in pd_unique(groups):
            m = groups == gname
            sd = x[m].std(ddof=0)
            zg[m] = (x[m] - x[m].mean()) / sd if sd > 0 else 0.0
        scores["Z theo nhóm hành chính"] = zg

    mono = None
    if k >= 2:
        scores["Z* (GMM mềm)"] = mix.z_soft(x)
        scores["Z_q (lượng tử hoá)"] = mix.z_quantile(x)
        scores["γ_cao"] = mix.gamma_high(x)
        mono = monotonicity_check(mix, lo, hi)
    else:
        # k = 1: Z_q trùng Z truyền thống (Φ⁻¹(Φ(z)) = z)
        scores["Z_q (lượng tử hoá)"] = z_trad

    return ColumnResult(name, x, sel, k, mix, lo, hi, mono, n_modes, mode_locs, scores)


def official_index_name(res: ColumnResult) -> str:
    return "Z_q (lượng tử hoá)"


# ===========================================================================
# PHẦN 2 — ĐỌC FILE
# ===========================================================================
"""Đọc file điểm Excel/CSV, tự nhận diện dòng tiêu đề và đoán các cột."""




def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", str(s)).lower().strip()
    return re.sub(r"\s+", " ", s)


def _is_texty(v) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, (int, float, np.number)):
        return False
    try:
        float(str(v).replace(",", "."))
        return False
    except ValueError:
        return True


def detect_header_row(raw: pd.DataFrame, max_scan: int = 20) -> int:
    """Chọn dòng đầu tiên mà phần lớn ô là chữ và dòng ngay dưới có số."""
    best, best_score = 0, -1.0
    for i in range(min(max_scan, len(raw) - 1)):
        row = raw.iloc[i]
        nonnull = row.notna().sum()
        if nonnull < 2:
            continue
        text_ratio = sum(_is_texty(v) for v in row) / nonnull
        nxt = raw.iloc[i + 1]
        num_next = sum((not _is_texty(v)) and pd.notna(v) for v in nxt)
        score = text_ratio * nonnull + 0.5 * num_next
        if text_ratio >= 0.6 and score > best_score:
            best, best_score = i, score
            if text_ratio == 1.0 and num_next >= 1:
                break
    return best


def list_sheets(file_bytes: bytes, filename: str) -> list[str]:
    if filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names
    return ["(CSV)"]


def read_raw(file_bytes: bytes, filename: str, sheet: str | None) -> pd.DataFrame:
    if filename.lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, header=None)
    text = file_bytes.decode("utf-8-sig", errors="replace")
    sep = ";" if text.count(";") > text.count(",") else ","
    return pd.read_csv(io.StringIO(text), header=None, sep=sep)


def apply_header(raw: pd.DataFrame, header_row: int) -> pd.DataFrame:
    cols, seen = [], {}
    for j, c in enumerate(raw.iloc[header_row]):
        name = str(c).strip() if pd.notna(c) else f"Cột {j + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 0
        cols.append(name)
    df = raw.iloc[header_row + 1:].copy()
    df.columns = cols
    df = df.dropna(how="all").reset_index(drop=True)
    for c in df.columns:
        conv = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        if conv.notna().sum() >= 0.8 * df[c].notna().sum() and df[c].notna().sum() > 0:
            df[c] = conv
    return df


def guess_group_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if "loại hình" in _norm(c) or "loai hinh" in _norm(c):
            return c
    cands = [c for c in df.columns
             if not pd.api.types.is_numeric_dtype(df[c]) and 2 <= df[c].nunique(dropna=True) <= 5]
    return cands[0] if cands else None


def guess_score_cols(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        v = df[c].dropna()
        if len(v) < 10 or v.nunique() < 5:
            continue
        if v.min() >= 0 and v.max() <= 10:
            out.append(c)
    pri = [c for c in out if "điểm" in _norm(c) or "diem" in _norm(c)]
    return pri or out


def guess_id_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        n = _norm(c)
        if any(k in n for k in ["mã", "ma hs", "id", "stt", "họ tên", "ho ten"]):
            return c
    return None


# ===========================================================================
# PHẦN 3 — BIỂU ĐỒ
# ===========================================================================
"""Biểu đồ matplotlib cho SmartZ-EDU."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.dpi": 110,
})

PALETTE = ["#2563eb", "#f97316", "#16a34a", "#9333ea", "#dc2626"]


def plot_spectrum(res: ColumnResult, groups=None, x_hi=None, x_lo=None):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = res.x
    bins = np.arange(np.floor(res.lo), np.ceil(res.hi) + 0.5, 0.5)
    if groups is not None:
        labels = pd_unique(groups)
        data = [x[groups == g] for g in labels]
        ax.hist(data, bins=bins, stacked=True, density=True, alpha=0.45,
                color=PALETTE[: len(labels)], label=[str(l) for l in labels], edgecolor="white")
    else:
        ax.hist(x, bins=bins, density=True, alpha=0.45, color="#94a3b8", edgecolor="white",
                label="Học sinh")
    grid = np.linspace(res.lo, res.hi, 600)
    ax.plot(grid, res.mix.pdf(grid), color="black", lw=2.2, label=f"Mật độ hỗn hợp (k = {res.k})")
    if res.k >= 2:
        comps = res.mix.comp_pdf(grid)
        for j in range(res.k):
            ax.plot(grid, comps[:, j], ls="--", lw=1.4, color=PALETTE[j % 5],
                    label=f"Thành phần {j + 1}: μ={res.mix.means[j]:.2f}, σ={res.mix.sds[j]:.2f}")
    for m in res.mode_locs:
        ax.axvline(m, color="gray", lw=0.8, ls=":")
    if x_hi is not None:
        ax.axvline(x_hi, color="#dc2626", lw=1.5, label=f"x* cụm cao = {x_hi:.2f}")
    if x_lo is not None:
        ax.axvline(x_lo, color="#0891b2", lw=1.5, label=f"x* cụm thấp = {x_lo:.2f}")
    ax.set_xlabel("Điểm")
    ax.set_ylabel("Mật độ")
    ax.set_title(f"Phổ điểm – {res.name}")
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    fig.tight_layout()
    return fig


def plot_indices(res: ColumnResult):
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    grid = np.round(np.arange(res.lo, res.hi + 0.005, 0.01), 4)
    mu, sd = res.x.mean(), res.x.std(ddof=0)
    ax.plot(grid, (grid - mu) / sd, color="#64748b", lw=1.5, label="Z truyền thống")
    if res.k >= 2:
        ax.plot(grid, res.mix.z_soft(grid), color="#f97316", lw=2, label="Z* (GMM mềm)")
        ax.plot(grid, res.mix.z_quantile(grid), color="#2563eb", lw=2.2, label="Z_q (lượng tử hoá)")
        if res.mono and not res.mono["monotone"]:
            for a, b in res.mono["intervals"]:
                ax.axvspan(a, b, color="#fecaca", alpha=0.6)
            ax.plot([], [], color="#fecaca", lw=8, label="Vùng Z* giảm (không đơn điệu)")
    ax.set_xlabel("Điểm x")
    ax.set_ylabel("Giá trị chỉ số")
    ax.set_title("So sánh các chỉ số chuẩn hoá theo điểm")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_gamma(res: ColumnResult, th_hi: float, th_lo: float, x_hi=None, x_lo=None):
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    grid = np.linspace(res.lo, res.hi, 600)
    ax.plot(grid, res.mix.gamma_high(grid), color="#9333ea", lw=2, label="γ_cao(x)")
    ax.axhline(th_hi, color="#dc2626", ls="--", lw=1, label=f"Ngưỡng cao = {th_hi:.2f}")
    ax.axhline(th_lo, color="#0891b2", ls="--", lw=1, label=f"Ngưỡng thấp = {th_lo:.2f}")
    if x_hi is not None:
        ax.axvline(x_hi, color="#dc2626", lw=1)
    if x_lo is not None:
        ax.axvline(x_lo, color="#0891b2", lw=1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Điểm x")
    ax.set_ylabel("γ_cao")
    ax.set_title("Xác suất thuộc cụm điểm cao theo điểm số")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def plot_progress(dz, groups=None, label="ΔZ_q"):
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    if groups is not None:
        labels = pd_unique(groups)
        ax.boxplot([dz[groups == g] for g in labels], vert=False, widths=0.5)
        ax.set_yticks(range(1, len(labels) + 1), [str(l) for l in labels])
    else:
        ax.hist(dz, bins=30, color="#2563eb", alpha=0.7)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel(label)
    ax.set_title(f"Phân bố tiến bộ tương đối ({label})")
    fig.tight_layout()
    return fig


# ===========================================================================
# PHẦN 4 — GIAO DIỆN
# ===========================================================================
"""
SmartZ-EDU — Chuẩn hoá điểm số khi phổ điểm không thuần nhất
bằng Mô hình Hỗn hợp Gauss (GMM) và Z-score lượng tử hoá Z_q.

Chạy cục bộ:   streamlit run app.py
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="SmartZ-EDU", page_icon="📊", layout="wide")

_HERE = Path(__file__).parent
SAMPLE_PATH = next((p for p in [_HERE / "data" / "Diem_GK_CK_An_Danh_4Khoi.xlsx",
                     _HERE / "Diem_GK_CK_An_Danh_4Khoi.xlsx"] if p.exists()),
                    _HERE / "data" / "Diem_GK_CK_An_Danh_4Khoi.xlsx")
ZQ = "Z_q (lượng tử hoá)"
ZS = "Z* (GMM mềm)"

DISCLAIMER = (
    "**Khuyến cáo sử dụng.** Kết quả, đặc biệt là danh sách *ngoại lệ sư phạm*, "
    "**không được và không nên** dùng làm căn cứ duy nhất cho bất kỳ quyết định hành chính nào "
    "đối với học sinh. Đây là kết quả tính toán trên một môn học, ở một số mốc đánh giá nhất định; "
    "cần đối chiếu với nhận định của giáo viên chủ nhiệm, giáo viên bộ môn và hoàn cảnh cụ thể của từng em."
)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_raw(file_bytes: bytes, filename: str, sheet: str | None):
    return read_raw(file_bytes, filename, sheet)


@st.cache_data(show_spinner=False)
def get_sheets(file_bytes: bytes, filename: str):
    return list_sheets(file_bytes, filename)


@st.cache_data(show_spinner=False)
def run_analysis(name: str, x: np.ndarray, groups, k_override):
    return analyse_column(name, x, groups, k_override=k_override)


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{nd}f}"


# ---------------------------------------------------------------------------
# Thanh bên: nhập dữ liệu
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 SmartZ-EDU")
    st.caption("Chuẩn hoá điểm số bằng GMM thích ứng và Z-score lượng tử hoá")

    source = st.radio("Nguồn dữ liệu", ["Tải file lên", "Dùng dữ liệu mẫu"], horizontal=True)
    file_bytes, filename = None, None
    if source == "Tải file lên":
        up = st.file_uploader("File điểm (Excel hoặc CSV)", type=["xlsx", "xls", "csv"])
        if up is not None:
            file_bytes, filename = up.getvalue(), up.name
    elif SAMPLE_PATH.exists():
        file_bytes, filename = SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name
        st.caption("Dữ liệu mẫu: điểm Toán ẩn danh 4 khối, 2 học kỳ.")
    else:
        st.warning("Không tìm thấy file dữ liệu mẫu trong thư mục `data/`.")

    df = None
    if file_bytes is not None:
        sheets = get_sheets(file_bytes, filename)
        default_sheet = sheets.index("K9_HK2") if "K9_HK2" in sheets else 0
        sheet = st.selectbox("Trang tính", sheets, index=default_sheet) if len(sheets) > 1 else sheets[0]
        raw = load_raw(file_bytes, filename, None if sheet == "(CSV)" else sheet)
        auto_h = detect_header_row(raw)
        h = st.number_input("Dòng tiêu đề (tự nhận diện)", min_value=1, max_value=max(1, len(raw)),
                            value=auto_h + 1, step=1,
                            help="Hệ thống tự tìm dòng chứa tên cột. Sửa lại nếu nhận diện sai.")
        df = apply_header(raw, int(h) - 1)

        cols = list(df.columns)
        g_guess = guess_group_col(df)
        group_col = st.selectbox("Cột loại hình lớp", ["(Không có)"] + cols,
                                 index=(cols.index(g_guess) + 1) if g_guess in cols else 0)
        group_col = None if group_col == "(Không có)" else group_col

        s_guess = [c for c in guess_score_cols(df) if c != group_col]
        score_cols = st.multiselect("Cột điểm cần phân tích", cols, default=s_guess)

        id_guess = guess_id_col(df)
        id_col = st.selectbox("Cột mã học sinh (để hiển thị)", ["(Không có)"] + cols,
                              index=(cols.index(id_guess) + 1) if id_guess in cols else 0)
        id_col = None if id_col == "(Không có)" else id_col

        with st.expander("Tuỳ chọn nâng cao"):
            k_mode = st.selectbox("Số thành phần k", ["Tự động (BIC)", "1", "2", "3", "4"],
                                  help="Mặc định để dữ liệu tự quyết định bằng BIC.")
            k_override = None if k_mode.startswith("Tự động") else int(k_mode)
            st.caption("Cấu hình tái lập: random_state = 42, n_init = 20, covariance_type = 'full'.")
    st.divider()
    st.caption("Dự án nghiên cứu khoa học kỹ thuật — dữ liệu được xử lý trong phiên làm việc, "
               "không lưu trữ trên máy chủ.")


# ---------------------------------------------------------------------------
# Trang chào
# ---------------------------------------------------------------------------
st.title("SmartZ-EDU")
st.markdown("##### Chuẩn hoá điểm số khi phổ điểm không thuần nhất — GMM thích ứng & Z-score lượng tử hoá")

if df is None:
    st.info("👈 Tải file điểm lên ở thanh bên, hoặc chọn **Dùng dữ liệu mẫu** để xem thử.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Tải dữ liệu**\n\nFile Excel/CSV, mỗi dòng một học sinh; có cột loại hình lớp "
                "và một hoặc nhiều cột điểm (thang 0–10).")
    c2.markdown("**2. Hệ thống tự phân tích**\n\nChọn số thành phần bằng BIC, kiểm tra tính đơn điệu "
                "của Z*, tự chuyển sang Z_q an toàn.")
    c3.markdown("**3. Đọc kết quả**\n\nPhổ điểm, ngoại lệ sư phạm kèm ngưỡng điểm x*, tiến bộ ΔZ_q, "
                "tải kết quả về Excel.")
    st.stop()

if not score_cols:
    st.warning("Hãy chọn ít nhất một cột điểm ở thanh bên.")
    st.stop()

# ---------------------------------------------------------------------------
# Phân tích từng cột điểm
# ---------------------------------------------------------------------------
results, subsets = {}, {}
with st.spinner("Đang ước lượng mô hình hỗn hợp Gauss…"):
    for c in score_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            st.error(f"Cột **{c}** không phải cột số.")
            st.stop()
        mask = df[c].notna()
        if group_col:
            mask &= df[group_col].notna()
        sub = df.loc[mask].copy()
        if len(sub) < 20:
            st.error(f"Cột **{c}** chỉ có {len(sub)} giá trị hợp lệ — cần tối thiểu 20.")
            st.stop()
        groups = sub[group_col].astype(str).values if group_col else None
        res = run_analysis(c, sub[c].to_numpy(dtype=float), groups, k_override)
        for name, vals in res.scores.items():
            sub[name] = vals
        results[c], subsets[c] = res, sub

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Phân tích & phổ điểm", "🎯 Ngoại lệ sư phạm", "📈 Tiến bộ", "📥 Kết quả & tải về", "📘 Phương pháp"])

# ---------------------------------------------------------------------------
# Tab 1: Phân tích
# ---------------------------------------------------------------------------
with tab1:
    col_tabs = st.tabs(score_cols) if len(score_cols) > 1 else [st.container()]
    for c, ct in zip(score_cols, col_tabs):
        res, sub = results[c], subsets[c]
        groups = sub[group_col].astype(str).values if group_col else None
        with ct:
            sel = res.selection
            d12 = sel["delta_1_to_2"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Số học sinh", len(res.x))
            m2.metric("Số thành phần k", res.k,
                      help="Chọn tự động theo BIC nhỏ nhất" if k_override is None else "Do người dùng cố định")
            m3.metric("ΔBIC (k=1 → k=2)", fmt(d12), raftery_label(d12) if not np.isnan(d12) else None,
                      delta_color="off")
            m4.metric("Số đỉnh của phổ điểm", res.n_modes,
                      help="Số cực đại của mật độ hỗn hợp đã ước lượng")

            # --- Cảnh báo đơn điệu ---
            if res.k == 1:
                st.info("Cơ chế thích ứng chọn **k = 1**: phổ điểm được mô tả tốt bởi một phân phối chuẩn. "
                        "Hệ thống dùng **Z-score truyền thống** (khi k = 1, Z_q trùng Z).")
            elif not res.mono["monotone"]:
                iv = "; ".join(f"[{a:.2f}; {b:.2f}]" for a, b in res.mono["intervals"])
                st.error(f"⚠️ **Phát hiện vùng không đơn điệu của Z\\***: trên khoảng điểm {iv}, "
                         "học sinh có điểm cao hơn lại nhận Z\\* thấp hơn. "
                         "Hệ thống **đã tự động chuyển sang Z_q** làm chỉ số chính thức.")
            else:
                st.success("✅ Z\\* đơn điệu trên toàn thang điểm của dữ liệu này. "
                           "Chỉ số chính thức vẫn là **Z_q** (đơn điệu nghiêm ngặt với mọi tham số — Định lý 2).")
            th = res.mono["theorem"] if res.mono else None
            if th:
                st.caption(f"Đối chiếu Định lý 1 (k = 2): Δ = {th['delta']:.2f}, σ gộp = {th['sigma_pool']:.2f}, "
                           f"Δ/σ = {th['ratio']:.2f} → dự báo "
                           f"{'đơn điệu' if th['predict_monotone'] else 'không đơn điệu'} (ngưỡng Δ/σ = 2).")

            left, right = st.columns([1.35, 1])
            with left:
                st.pyplot(plot_spectrum(res, groups), clear_figure=True)
                st.pyplot(plot_indices(res), clear_figure=True)
            with right:
                st.markdown("**Cơ chế thích ứng tự động (BIC)**")
                best = min(sel["bics"].values())
                bic_df = pd.DataFrame({
                    "k": list(sel["bics"].keys()),
                    "BIC": list(sel["bics"].values()),
                })
                bic_df["ΔBIC so với tốt nhất"] = bic_df["BIC"] - best
                bic_df["Mức bằng chứng (Raftery)"] = [
                    "✔ Được chọn" if d == 0 else raftery_label(d) for d in bic_df["ΔBIC so với tốt nhất"]]
                st.dataframe(bic_df.style.format({"BIC": "{:.2f}", "ΔBIC so với tốt nhất": "{:.2f}"}),
                             hide_index=True, width="stretch")
                st.caption("Thang Raftery: 0–2 không phân biệt được · 2–6 yếu · 6–10 mạnh · >10 rất mạnh.")

                st.markdown("**Tham số mô hình hỗn hợp**")
                st.dataframe(pd.DataFrame({
                    "Thành phần": [f"{j + 1}" for j in range(res.k)],
                    "Tỉ trọng π": res.mix.weights, "Trung bình μ": res.mix.means, "Độ lệch chuẩn σ": res.mix.sds,
                }).style.format({"Tỉ trọng π": "{:.3f}", "Trung bình μ": "{:.2f}", "Độ lệch chuẩn σ": "{:.2f}"}),
                    hide_index=True, width="stretch")

                st.markdown("**Kiểm định tỉ số hợp lý – bootstrap tham số (k=1 so với k=2)**")
                key = f"lrt::{filename}::{c}::{len(res.x)}::{float(res.x.sum()):.4f}"
                bcol1, bcol2 = st.columns([1, 1])
                B = bcol1.selectbox("Số mẫu bootstrap B", [199, 500, 999], index=1, key=f"B_{c}")
                if bcol2.button("Chạy kiểm định", key=f"run_{c}", width="stretch"):
                    bar = st.progress(0.0, text="Đang tạo mẫu bootstrap…")
                    st.session_state[key] = bootstrap_lrt(res.x, B=B, progress=lambda p: bar.progress(p))
                    bar.empty()
                if key in st.session_state:
                    r = st.session_state[key]
                    p_txt = f"< {1 / (r['B'] + 1):.3f}" if r["p_value"] <= 1 / (r["B"] + 1) + 1e-12 else f"{r['p_value']:.3f}"
                    st.dataframe(pd.DataFrame([{
                        "LRT quan sát": r["lrt_obs"], "Ngưỡng 95%": r["crit95"],
                        f"p (B = {r['B']})": p_txt,
                        "Kết luận": "Bác bỏ mô hình 1 thành phần" if r["p_value"] < 0.05 else "Chưa bác bỏ",
                    }]).style.format({"LRT quan sát": "{:.2f}", "Ngưỡng 95%": "{:.2f}"}),
                        hide_index=True, width="stretch")
                    st.caption("Kiểm định chỉ khẳng định phổ điểm không phải một Gauss đơn; "
                               "nó không chứng minh phổ điểm có hai đỉnh.")
                else:
                    st.caption("Bấm *Chạy kiểm định* (B = 500 mất khoảng 10 giây).")

                st.markdown("**Thống kê mô tả**")
                st.dataframe(pd.DataFrame(describe_by_group(res.x, groups)).style.format(
                    {"Trung bình": "{:.2f}", "Độ lệch chuẩn": "{:.2f}", "Trung vị": "{:.2f}",
                     "Độ lệch (skew)": "{:.2f}", "Shapiro–Wilk p": "{:.4f}"}),
                    hide_index=True, width="stretch")
    st.caption(DISCLAIMER)

# ---------------------------------------------------------------------------
# Tab 2: Ngoại lệ sư phạm
# ---------------------------------------------------------------------------
outlier_tables = {}
with tab2:
    c = st.selectbox("Cột điểm", score_cols, key="out_col")
    res, sub = results[c], subsets[c]
    if res.k < 2:
        st.warning("Cột điểm này có k = 1 (không có cấu trúc hỗn hợp) nên không xác định được cụm điểm cao/thấp.")
    elif not group_col:
        st.warning("Cần chọn **cột loại hình lớp** ở thanh bên để xác định ngoại lệ sư phạm.")
    else:
        glabels = pd_unique(sub[group_col].astype(str).values)
        means = {g: sub.loc[sub[group_col].astype(str) == g, c].mean() for g in glabels}
        g_low_default = min(means, key=means.get)
        g_high_default = max(means, key=means.get)
        a, b = st.columns(2)
        g_low = a.selectbox("Nhóm có mặt bằng thấp hơn (vd. Lớp hai buổi)", glabels,
                            index=glabels.index(g_low_default))
        g_high = b.selectbox("Nhóm có mặt bằng cao hơn (vd. Lớp TC)", glabels,
                             index=glabels.index(g_high_default))
        th_hi = a.slider("Ngưỡng γ_cao cho nhóm thấp (γ_cao >)", 0.50, 0.95, 0.70, 0.05)
        th_lo = b.slider("Ngưỡng γ_cao cho nhóm cao (γ_cao <)", 0.05, 0.50, 0.30, 0.05)

        xh = x_star_high(res.mix, th_hi, res.lo, res.hi)
        xl = x_star_low(res.mix, th_lo, res.lo, res.hi)
        gcol = sub[group_col].astype(str)
        up = sub[(gcol == g_low) & (sub["γ_cao"] > th_hi)].sort_values(c, ascending=False)
        down = sub[(gcol == g_high) & (sub["γ_cao"] < th_lo)].sort_values(c)
        outlier_tables[c] = (up, down)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{g_low} thuộc cụm điểm cao", len(up))
        m2.metric("Ngưỡng điểm x* (cụm cao)", fmt(xh))
        m3.metric(f"{g_high} thuộc cụm điểm thấp", len(down))
        m4.metric("Ngưỡng điểm x* (cụm thấp)", fmt(xl))
        if xh is not None and xl is not None:
            st.info(f"Diễn đạt tương đương: **{len(up)} học sinh {g_low}** đạt từ **{xh:.2f} điểm** trở lên — "
                    f"mức điểm mà theo cấu trúc phổ điểm của khối, đặc trưng cho cụm điểm cao; "
                    f"**{len(down)} học sinh {g_high}** có điểm từ **{xl:.2f}** trở xuống — đặc trưng cho cụm điểm thấp. "
                    "Ngưỡng x* do dữ liệu tự xác định.")

        grid = np.linspace(res.lo, res.hi, 1001)
        gh = res.mix.gamma_high(grid)
        if np.any(np.diff(gh) < -1e-9) and res.k == 2:
            imax = int(np.argmax(gh))
            st.caption(f"Ghi chú kỹ thuật: vì σ hai cụm khác nhau, γ_cao đạt cực đại {gh[imax]:.3f} tại x ≈ {grid[imax]:.2f} "
                       f"rồi giảm nhẹ về {gh[-1]:.3f} ở x = {grid[-1]:.1f}.")

        p1, p2 = st.columns(2)
        p1.pyplot(plot_spectrum(res, sub[group_col].astype(str).values, x_hi=xh, x_lo=xl), clear_figure=True)
        p2.pyplot(plot_gamma(res, th_hi, th_lo, xh, xl), clear_figure=True)

        show_cols = [x for x in [id_col] if x] + [k for k in sub.columns if k.lower().strip() == "lớp"] + \
                    [group_col, c, "γ_cao", ZQ]
        show_cols = list(dict.fromkeys(show_cols))
        fmt_map = {c: "{:.2f}", "γ_cao": "{:.3f}", ZQ: "{:.2f}"}
        t1, t2 = st.columns(2)
        with t1:
            st.markdown(f"**{g_low} có γ_cao > {th_hi:.2f}** ({len(up)} học sinh)")
            st.dataframe(up[show_cols].style.format(fmt_map), hide_index=True, width="stretch", height=360)
        with t2:
            st.markdown(f"**{g_high} có γ_cao < {th_lo:.2f}** ({len(down)} học sinh)")
            st.dataframe(down[show_cols].style.format(fmt_map), hide_index=True, width="stretch", height=360)
    st.warning(DISCLAIMER)

# ---------------------------------------------------------------------------
# Tab 3: Tiến bộ
# ---------------------------------------------------------------------------
progress_df = None
with tab3:
    if len(score_cols) < 2:
        st.info("Chọn ít nhất **hai cột điểm** (ví dụ Giữa kỳ và Cuối kỳ) để so sánh tiến bộ.")
    else:
        a, b = st.columns(2)
        c1 = a.selectbox("Mốc trước", score_cols, index=0)
        c2 = b.selectbox("Mốc sau", score_cols, index=1)
        if c1 == c2:
            st.warning("Hãy chọn hai cột điểm khác nhau.")
        else:
            r1, r2 = results[c1], results[c2]
            s1, s2 = subsets[c1], subsets[c2]
            idx = s1.index.intersection(s2.index)
            prog = df.loc[idx, [x for x in [id_col, group_col] if x]].copy()
            prog[c1], prog[c2] = df.loc[idx, c1], df.loc[idx, c2]
            prog[f"Z_q {c1}"], prog[f"Z_q {c2}"] = s1.loc[idx, ZQ], s2.loc[idx, ZQ]
            prog["ΔZ_q"] = prog[f"Z_q {c2}"] - prog[f"Z_q {c1}"]
            has_zs = ZS in s1.columns and ZS in s2.columns
            if has_zs:
                prog["ΔZ* (đối chứng)"] = s2.loc[idx, ZS] - s1.loc[idx, ZS]
            progress_df = prog

            if r1.k != r2.k:
                st.warning(f"Hai mốc có số thành phần khác nhau (k = {r1.k} và k = {r2.k}): "
                           "ΔZ\\* không so sánh được trực tiếp. Chỉ dùng ΔZ_q.")
            if any(r.mono and not r.mono["monotone"] for r in (r1, r2)):
                st.warning("Z\\* không đơn điệu ở ít nhất một mốc: ΔZ\\* chỉ mang tính đối chứng.")
            st.success("Tiến bộ tương đối được đo bằng **ΔZ_q = Z_q(mốc sau) − Z_q(mốc trước)**. "
                       "ΔZ_q > 0: vị trí tương đối của học sinh trong khối được cải thiện.")

            m1, m2, m3 = st.columns(3)
            m1.metric("Số học sinh có đủ hai mốc", len(prog))
            m2.metric("Tiến bộ (ΔZ_q > 0.5)", int((prog["ΔZ_q"] > 0.5).sum()))
            m3.metric("Sụt giảm (ΔZ_q < −0.5)", int((prog["ΔZ_q"] < -0.5).sum()))

            p1, p2 = st.columns([1.2, 1])
            p1.pyplot(plot_progress(prog["ΔZ_q"].to_numpy(),
                                    prog[group_col].astype(str).values if group_col else None), clear_figure=True)
            if group_col:
                agg = {"ΔZ_q": ["count", "mean", "std"]}
                if has_zs:
                    agg["ΔZ* (đối chứng)"] = ["mean"]
                g = prog.groupby(group_col).agg(agg)
                g.columns = ["N", "ΔZ_q trung bình", "ΔZ_q độ lệch chuẩn"] + (["ΔZ* trung bình"] if has_zs else [])
                p2.markdown("**Theo loại hình lớp**")
                p2.dataframe(g.style.format("{:.3f}", subset=[x for x in g.columns if x != "N"]),
                             width="stretch")

            fm = {k: "{:.2f}" for k in prog.columns if k not in (id_col, group_col)}
            t1, t2 = st.columns(2)
            t1.markdown("**10 học sinh tiến bộ nhiều nhất**")
            t1.dataframe(prog.nlargest(10, "ΔZ_q").style.format(fm), hide_index=True, width="stretch")
            t2.markdown("**10 học sinh sụt giảm nhiều nhất**")
            t2.dataframe(prog.nsmallest(10, "ΔZ_q").style.format(fm), hide_index=True, width="stretch")
    st.caption(DISCLAIMER)

# ---------------------------------------------------------------------------
# Tab 4: Kết quả & tải về
# ---------------------------------------------------------------------------
with tab4:
    out = df.copy()
    summary_rows, bic_rows = [], []
    for c in score_cols:
        res, sub = results[c], subsets[c]
        for name in ["Z truyền thống", "Z theo nhóm hành chính", ZS, ZQ, "γ_cao"]:
            if name in sub.columns:
                out.loc[sub.index, f"{name} [{c}]"] = sub[name]
        mono_txt = "—" if res.k == 1 else ("Có" if res.mono["monotone"] else
                                           "Không: " + "; ".join(f"[{a:.2f};{b:.2f}]" for a, b in res.mono["intervals"]))
        summary_rows.append({
            "Cột điểm": c, "N": len(res.x), "k được chọn": res.k,
            "ΔBIC (1→2)": res.selection["delta_1_to_2"], "Mức bằng chứng": raftery_label(res.selection["delta_1_to_2"]),
            "Số đỉnh": res.n_modes, "Z* đơn điệu": mono_txt,
            "x* cụm cao (γ>0.7)": x_star_high(res.mix, 0.7, res.lo, res.hi) if res.k >= 2 else None,
            "x* cụm thấp (γ<0.3)": x_star_low(res.mix, 0.3, res.lo, res.hi) if res.k >= 2 else None,
            "Chỉ số chính thức": "Z truyền thống" if res.k == 1 else "Z_q",
            "Tham số (π; μ; σ)": " | ".join(f"{w:.3f}; {m:.2f}; {s:.2f}"
                                            for w, m, s in zip(res.mix.weights, res.mix.means, res.mix.sds)),
        })
        for k, v in res.selection["bics"].items():
            bic_rows.append({"Cột điểm": c, "k": k, "BIC": v})
    if progress_df is not None:
        out.loc[progress_df.index, "ΔZ_q"] = progress_df["ΔZ_q"]

    summary = pd.DataFrame(summary_rows)
    st.markdown("**Tóm tắt mô hình**")
    st.dataframe(summary, hide_index=True, width="stretch")
    st.markdown("**Bảng kết quả chi tiết**")
    st.dataframe(out, hide_index=True, width="stretch", height=420)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="Ket_qua", index=False)
        summary.to_excel(xw, sheet_name="Tom_tat_mo_hinh", index=False)
        pd.DataFrame(bic_rows).to_excel(xw, sheet_name="BIC", index=False)
        for c, (up, down) in outlier_tables.items():
            up.to_excel(xw, sheet_name="Ngoai_le_cum_cao"[:31], index=False)
            down.to_excel(xw, sheet_name="Ngoai_le_cum_thap"[:31], index=False)
        if progress_df is not None:
            progress_df.to_excel(xw, sheet_name="Tien_bo", index=False)
        pd.DataFrame({"Khuyến cáo": [DISCLAIMER.replace("*", "")]}).to_excel(xw, sheet_name="Khuyen_cao", index=False)
    st.download_button("⬇️ Tải kết quả (Excel)", buf.getvalue(), file_name="SmartZ-EDU_ket_qua.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       type="primary")
    st.caption(DISCLAIMER)

# ---------------------------------------------------------------------------
# Tab 5: Phương pháp
# ---------------------------------------------------------------------------
with tab5:
    st.markdown("### Quy trình 5 bước (áp dụng riêng cho từng cột điểm)")
    st.markdown(
        "1. **Thống kê mô tả & kiểm tra cấu trúc hỗn hợp** — Shapiro–Wilk sàng lọc, kiểm định tỉ số hợp lý "
        "bằng bootstrap tham số (k = 1 so với k = 2).\n"
        "2. **Cơ chế thích ứng tự động** — ước lượng GMM với k = 1…4, chọn k có BIC nhỏ nhất, diễn giải ΔBIC "
        "theo thang Raftery. Nếu k = 1 dùng Z truyền thống.\n"
        "3. **Xây dựng chỉ số** — Z truyền thống, Z\\* (GMM mềm), Z_q (lượng tử hoá), kèm Z theo nhóm hành chính.\n"
        "4. **Kiểm tra tính đơn điệu** — quét lưới bước 0.01; nếu Z\\* có đoạn giảm thì cảnh báo và dùng Z_q.\n"
        "5. **Ngoại lệ sư phạm & tiến bộ** — γ_cao, ngưỡng điểm x\\* tương đương, ΔZ_q.")
    st.markdown("### Công thức")
    st.latex(r"z = \frac{x-\mu}{\sigma}")
    st.latex(r"\gamma_j(x) = \frac{\pi_j\,\mathcal N(x;\mu_j,\sigma_j)}{\sum_{i=1}^{K}\pi_i\,\mathcal N(x;\mu_i,\sigma_i)}")
    st.latex(r"Z^* = \sum_{j=1}^{K}\gamma_j(x)\,z_j,\qquad z_j=\frac{x-\mu_j}{\sigma_j}")
    st.latex(r"Z_q = \Phi^{-1}\!\big(F_{mix}(x)\big),\qquad F_{mix}(x)=\sum_{j=1}^{K}\pi_j\,\Phi\!\Big(\frac{x-\mu_j}{\sigma_j}\Big)")
    st.markdown(
        "**Định lý 1.** Với hai thành phần cùng phương sai σ, Z\\* đơn điệu tăng trên toàn trục số "
        "khi và chỉ khi Δ = |μ₂ − μ₁| ≤ 2σ.\n\n"
        "**Định lý 2.** Với mọi tham số (π_j > 0, σ_j > 0), Z_q đơn điệu tăng nghiêm ngặt — "
        "không bao giờ đảo ngược thứ tự điểm số.\n\n"
        "**Lưu ý về phạm vi ý nghĩa.** Vì Z_q là phép biến đổi đơn điệu của điểm, thứ hạng theo Z_q trùng "
        "thứ hạng theo điểm thô trong cùng một đợt. Giá trị của Z_q nằm ở thang đo có ý nghĩa xác suất đúng, "
        "khả năng so sánh giữa các đợt đánh giá, và các đại lượng phụ trợ như γ.")
    st.markdown("### Tái lập")
    st.code("GaussianMixture(n_components=k, covariance_type='full', n_init=20, random_state=42)", language="python")
    st.caption(DISCLAIMER)
