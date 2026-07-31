"""
SmartZ-EDU — Hệ thống Z-score thích ứng bằng Mô hình Hỗn hợp Gauss Mềm (Soft GMM)
hỗ trợ đánh giá công bằng và ra quyết định quản lý giáo dục THCS.

Phương pháp chính: Z-score GMM MỀM (soft/posterior-weighted) — mỗi học sinh được
gán một trọng số xác suất (gamma) thuộc về từng "cụm năng lực", thay vì bị ép buộc
phân loại cứng vào một nhóm duy nhất.

Cơ chế thích ứng tự động: so sánh BIC giữa mô hình 1 thành phần (đơn đỉnh) và
2 thành phần (đa đỉnh) để quyết định dùng GMM hay Z-score truyền thống.

Chạy local:  streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

st.set_page_config(
    page_title="SmartZ-EDU — Z-score thích ứng bằng Soft GMM",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# GIAO DIỆN: CSS TUỲ CHỈNH
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main > div { padding-top: 1.2rem; }

    .smartz-hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5f8a 45%, #3a8fb7 100%);
        border-radius: 18px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.4rem;
        color: white;
    }
    .smartz-hero h1 {
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        color: white;
    }
    .smartz-hero p {
        font-size: 1rem;
        opacity: 0.92;
        margin: 0;
        max-width: 850px;
        line-height: 1.5;
    }
    .smartz-badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 999px;
        padding: 0.15rem 0.75rem;
        font-size: 0.78rem;
        margin-right: 0.4rem;
        margin-top: 0.7rem;
    }

    .smartz-card {
        background: var(--background-color, #ffffff);
        border: 1px solid rgba(120,120,120,0.18);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
    }
    .smartz-section-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .smartz-decision-gmm {
        background: linear-gradient(90deg, #e8f8ee 0%, #d6f2e0 100%);
        border-left: 5px solid #2e9e5b;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        font-size: 0.95rem;
    }
    .smartz-decision-naive {
        background: linear-gradient(90deg, #eaf2fb 0%, #dcebf9 100%);
        border-left: 5px solid #3a7ec9;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        font-size: 0.95rem;
    }
    .smartz-flag-up {
        background: #fff4e5;
        border-left: 5px solid #e6912c;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
    }
    .smartz-flag-down {
        background: #fdeaea;
        border-left: 5px solid #d9534f;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(120,120,120,0.06);
        border-radius: 12px;
        padding: 0.7rem 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# LÕI THUẬT TOÁN
# ----------------------------------------------------------------------

def fit_gmm_components(x: np.ndarray):
    """Fit GMM 2 thành phần, sắp xếp theo trung bình tăng dần (cụm 1 = điểm thấp hơn)."""
    gmm2 = GaussianMixture(n_components=2, n_init=20, random_state=42).fit(x.reshape(-1, 1))
    means = gmm2.means_.flatten()
    stds = np.sqrt(gmm2.covariances_).flatten()
    weights = gmm2.weights_.flatten()
    order = np.argsort(means)
    mu1, s1, pi1 = means[order[0]], stds[order[0]], weights[order[0]]
    mu2, s2, pi2 = means[order[1]], stds[order[1]], weights[order[1]]
    return (mu1, s1, pi1), (mu2, s2, pi2)


def adaptive_check(x: np.ndarray):
    """So sánh BIC(k=1) và BIC(k=2) để quyết định có nên dùng GMM hay không.
    Trả về (dung_gmm: bool, bic1, bic2, delta_bic)."""
    X = x.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=42).fit(X)
    gmm2 = GaussianMixture(n_components=2, n_init=20, random_state=42).fit(X)
    bic1, bic2 = gmm1.bic(X), gmm2.bic(X)
    delta = bic1 - bic2  # dương nghĩa là k=2 tốt hơn
    return delta > 0, bic1, bic2, delta


def z_naive(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std(ddof=1)


def z_soft_gmm(x: np.ndarray, comp1, comp2):
    """Z-score GMM MỀM: trung bình có trọng số xác suất hậu nghiệm (gamma) giữa
    Z-score tính theo từng thành phần Gauss. Đây là phương pháp chính của hệ thống."""
    mu1, s1, pi1 = comp1
    mu2, s2, pi2 = comp2
    f1 = stats.norm.pdf(x, mu1, s1)
    f2 = stats.norm.pdf(x, mu2, s2)
    denom = pi1 * f1 + pi2 * f2
    denom = np.where(denom <= 0, 1e-300, denom)
    gamma1 = (pi1 * f1) / denom
    gamma2 = 1.0 - gamma1
    z1 = (x - mu1) / s1
    z2 = (x - mu2) / s2
    z_soft = gamma1 * z1 + gamma2 * z2
    return z_soft, gamma1, gamma2


def z_hard_gmm(x: np.ndarray, comp1, comp2):
    """Z-score GMM CỨNG (tham khảo/so sánh): gán cứng vào cụm có gamma cao nhất."""
    _, gamma1, _ = z_soft_gmm(x, comp1, comp2)
    mu1, s1, _ = comp1
    mu2, s2, _ = comp2
    hard1 = gamma1 >= 0.5
    return np.where(hard1, (x - mu1) / s1, (x - mu2) / s2)


def analyze_column(raw: pd.DataFrame, group_col: str, score_col: str):
    """Xử lý một cột điểm: làm sạch dữ liệu, kiểm tra thích ứng, tính các loại Z-score.
    Giữ nguyên chỉ số (index) gốc của `raw` để có thể ghép nối chính xác giữa nhiều
    cột điểm khác nhau (vd so sánh tiến bộ GK -> CK) mà không bị lệch dòng."""
    data = raw[[group_col, score_col]].copy()
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data = data.dropna()  # giữ nguyên index gốc, KHÔNG reset_index
    if data.empty:
        return None

    x = data[score_col].values
    use_gmm, bic1, bic2, delta = adaptive_check(x)

    result = data.copy()
    result["Z_truyen_thong"] = z_naive(x).round(3)

    comp1 = comp2 = None
    if use_gmm:
        comp1, comp2 = fit_gmm_components(x)
        z_s, gamma1, gamma2 = z_soft_gmm(x, comp1, comp2)
        z_h = z_hard_gmm(x, comp1, comp2)
        result["gamma_cum_thap"] = gamma1.round(4)
        result["gamma_cum_cao"] = gamma2.round(4)
        result["Z_GMM_mem"] = z_s.round(3)
        result["Z_GMM_cung"] = z_h.round(3)

    return {
        "score_col": score_col, "data": data, "x": x,
        "use_gmm": use_gmm, "bic1": bic1, "bic2": bic2, "delta": delta,
        "comp1": comp1, "comp2": comp2, "result": result,
    }


def render_distribution_chart(score_col: str, x: np.ndarray, comp1, comp2, use_gmm: bool):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    lo, hi = float(np.floor(x.min())), float(np.ceil(x.max()))
    bins = np.arange(max(0, lo - 0.5), hi + 1.0, 0.5)
    ax.hist(x, bins=bins, density=True, color="#8ecae6", edgecolor="white", alpha=0.85,
            label="Phổ điểm thực tế")
    xs = np.linspace(bins[0], bins[-1], 400)
    if use_gmm and comp1 is not None:
        mu1, s1, pi1 = comp1
        mu2, s2, pi2 = comp2
        ax.plot(xs, pi1 * stats.norm.pdf(xs, mu1, s1), color="#e76f51", ls="--", lw=2,
                label=f"Cụm điểm thấp hơn (μ≈{mu1:.1f})")
        ax.plot(xs, pi2 * stats.norm.pdf(xs, mu2, s2), color="#2a9d8f", ls="--", lw=2,
                label=f"Cụm điểm cao hơn (μ≈{mu2:.1f})")
        ax.plot(xs, pi1 * stats.norm.pdf(xs, mu1, s1) + pi2 * stats.norm.pdf(xs, mu2, s2),
                color="#1d3557", lw=2.4, label="Tổng hợp GMM 2 đỉnh")
    else:
        mu, sd = x.mean(), x.std()
        ax.plot(xs, stats.norm.pdf(xs, mu, sd), color="#1d3557", lw=2.4,
                label=f"Phân phối chuẩn (μ≈{mu:.1f})")
    ax.set_title(f"Phổ điểm {score_col} — bước nhảy 0.5 điểm", fontsize=12, fontweight="bold")
    ax.set_xlabel("Điểm số"); ax.set_ylabel("Mật độ xác suất")
    ax.set_xticks(np.arange(bins[0], bins[-1] + 0.5, 0.5))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, ls="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def render_analysis(raw: pd.DataFrame, group_col: str, r: dict, exception_threshold: float):
    """Hiển thị toàn bộ kết quả phân tích cho MỘT cột điểm (dùng chung cho cả 2 chế độ)."""
    score_col, x, result = r["score_col"], r["x"], r["result"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số học sinh hợp lệ", f"{len(x)}")
    m2.metric("BIC (k = 1)", f"{r['bic1']:.1f}")
    m3.metric("BIC (k = 2)", f"{r['bic2']:.1f}")
    m4.metric("ΔBIC (k1 − k2)", f"{r['delta']:+.1f}")

    if r["use_gmm"]:
        st.markdown(
            '<div class="smartz-decision-gmm">✅ &nbsp;<b>Phổ điểm có bằng chứng đa đỉnh</b> '
            '(ΔBIC &gt; 0) → hệ thống tự động áp dụng <b>Z-score GMM Mềm</b>.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="smartz-decision-naive">ℹ️ &nbsp;<b>Không đủ bằng chứng đa đỉnh</b> '
            '(ΔBIC ≤ 0) → hệ thống tự động dùng <b>Z-score truyền thống</b>, '
            'không cưỡng ép mô hình 2 cụm.</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    tab_names = ["📈 Biểu đồ phổ điểm", "🗂️ Bảng kết quả"]
    if r["use_gmm"]:
        tab_names[1:1] = ["🚩 Ngoại lệ sư phạm", "📋 Tham số mô hình"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        fig = render_distribution_chart(score_col, x, r["comp1"], r["comp2"], r["use_gmm"])
        st.pyplot(fig, use_container_width=True)

    if r["use_gmm"]:
        with tabs[1]:
            st.caption(
                f"Học sinh được gắn cờ khi xác suất γ thuộc **cụm khác với nhóm hành chính hiện "
                f"tại** vượt ngưỡng **{exception_threshold:.2f}** — gợi ý tham mưu chuyển lớp / phụ đạo."
            )
            groups = result[group_col].unique().tolist()
            if len(groups) == 2:
                means_by_group = result.groupby(group_col)[score_col].mean()
                low_group, high_group = means_by_group.idxmin(), means_by_group.idxmax()
                exc_up = result[(result[group_col] == low_group) & (result["gamma_cum_cao"] > exception_threshold)]
                exc_down = result[(result[group_col] == high_group) & (result["gamma_cum_cao"] < 1 - exception_threshold)]

                cA, cB = st.columns(2)
                with cA:
                    st.markdown(
                        f'<div class="smartz-flag-up">🔺 <b>{len(exc_up)} học sinh</b> thuộc '
                        f'<i>{low_group}</i> có năng lực gần với cụm điểm cao (γ &gt; '
                        f'{exception_threshold:.2f}) — cân nhắc bồi dưỡng / đề xuất chuyển lớp.</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(exc_up, use_container_width=True, height=220)
                with cB:
                    st.markdown(
                        f'<div class="smartz-flag-down">🔻 <b>{len(exc_down)} học sinh</b> thuộc '
                        f'<i>{high_group}</i> có năng lực gần với cụm điểm thấp (γ &lt; '
                        f'{1 - exception_threshold:.2f}) — cân nhắc phụ đạo thêm.</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(exc_down, use_container_width=True, height=220)
            else:
                st.info("Cần đúng 2 nhóm trong cột loại hình lớp để phát hiện ngoại lệ sư phạm.")

        with tabs[2]:
            mu1, s1, pi1 = r["comp1"]
            mu2, s2, pi2 = r["comp2"]
            cA, cB = st.columns(2)
            cA.markdown(f"**Cụm điểm thấp hơn**  \nμ = {mu1:.3f}  \nσ = {s1:.3f}  \ntrọng số = {pi1:.3f}")
            cB.markdown(f"**Cụm điểm cao hơn**  \nμ = {mu2:.3f}  \nσ = {s2:.3f}  \ntrọng số = {pi2:.3f}")
            st.write("**Z-score trung bình theo nhóm (truyền thống vs GMM Mềm):**")
            st.dataframe(result.groupby(group_col)[["Z_truyen_thong", "Z_GMM_mem"]].mean().round(3),
                         use_container_width=True)

    with tabs[-1]:
        st.dataframe(result, use_container_width=True)
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            result.to_excel(writer, index=False, sheet_name="KetQua")
        st.download_button(
            f"⬇️ Tải kết quả ({score_col}) — Excel", data=buf.getvalue(),
            file_name=f"ketqua_{score_col}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_{score_col}",
        )


def render_progress_comparison(raw: pd.DataFrame, group_col: str, results_by_col: dict, col_from: str, col_to: str):
    """So sánh tiến bộ Z-score GMM Mềm giữa 2 cột điểm, ghép đúng theo học sinh (index gốc)."""
    r_from, r_to = results_by_col[col_from], results_by_col[col_to]
    if not (r_from["use_gmm"] and r_to["use_gmm"]):
        st.info(
            "Chỉ có thể so sánh tiến bộ khi CẢ HAI cột điểm đều dùng Z-score GMM Mềm "
            "(cột dùng Z-score truyền thống chưa có Z* để so sánh)."
        )
        return

    merged = pd.concat(
        [raw[[group_col]], r_from["result"]["Z_GMM_mem"].rename(f"Z_GMM_mem_{col_from}"),
         r_to["result"]["Z_GMM_mem"].rename(f"Z_GMM_mem_{col_to}")],
        axis=1, join="inner",
    ).dropna()
    merged["Tien_bo"] = merged[f"Z_GMM_mem_{col_to}"] - merged[f"Z_GMM_mem_{col_from}"]

    st.markdown(
        f"Chênh lệch Z-score GMM Mềm từ **{col_from}** sang **{col_to}** — dương nghĩa là tiến bộ, "
        f"âm là sa sút, theo đúng nhóm năng lực tham chiếu của từng học sinh "
        f"(đã ghép đúng theo từng học sinh, n = {len(merged)})."
    )
    summary = merged.groupby(group_col)["Tien_bo"].agg(["mean", "std", "count"]).round(3)
    summary.columns = ["Tiến bộ TB", "Độ lệch chuẩn", "Số học sinh"]
    st.dataframe(summary, use_container_width=True)
    st.dataframe(merged, use_container_width=True)


# ----------------------------------------------------------------------
# GIAO DIỆN CHÍNH
# ----------------------------------------------------------------------

st.markdown(
    """
    <div class="smartz-hero">
        <h1>🎓 SmartZ-EDU</h1>
        <p>Hệ thống Z-score thích ứng bằng Mô hình Hỗn hợp Gauss Mềm (Soft GMM) —
        hỗ trợ đánh giá công bằng và ra quyết định quản lý giáo dục khi trường có
        nhiều loại hình lớp khiến phổ điểm không còn phân phối chuẩn đơn đỉnh.</p>
        <span class="smartz-badge">🧠 Soft GMM</span>
        <span class="smartz-badge">🔄 Thích ứng tự động (BIC)</span>
        <span class="smartz-badge">🚩 Phát hiện ngoại lệ sư phạm</span>
        <span class="smartz-badge">📈 Theo dõi tiến bộ</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("ℹ️ Về phương pháp Soft GMM và cơ chế thích ứng tự động"):
    st.markdown(
        """
- **Z-score GMM Mềm (phương pháp chính):** mỗi học sinh được gán trọng số xác suất
  γ (gamma) thuộc về từng cụm năng lực, thay vì bị ép buộc phân loại cứng. Học sinh
  ở vùng ranh giới giữa hai cụm sẽ có Z-score biến thiên mượt mà, tránh bị đánh giá
  cực đoan một cách bất công.
- **Cơ chế thích ứng tự động:** hệ thống so sánh BIC giữa mô hình 1 thành phần và
  2 thành phần. Nếu phổ điểm không đủ bằng chứng đa đỉnh, hệ thống **tự động dùng
  Z-score truyền thống** thay vì cưỡng ép dữ liệu vào khuôn mẫu 2 cụm không phù hợp.
        """
    )

with st.sidebar:
    st.markdown("### 📂 1. Tải dữ liệu")
    uploaded = st.file_uploader("Chọn file Excel/CSV điểm số", type=["xlsx", "xls", "csv"])
    st.caption(
        "File cần có tối thiểu 2 cột: **loại hình lớp** (vd Lớp TC / Lớp hai buổi) và "
        "**điểm số**. Có thể có nhiều cột điểm (Giữa kỳ, Cuối kỳ...)."
    )
    st.markdown("---")
    st.markdown("### ⚙️ 2. Tuỳ chọn phân tích")
    exception_threshold = st.slider(
        "Ngưỡng γ — 'ngoại lệ sư phạm'", 0.5, 0.9, 0.7, 0.05,
        help="Học sinh có xác suất γ thuộc cụm khác vượt ngưỡng này sẽ được gắn cờ.",
    )

if uploaded is None:
    st.info("👈 Hãy tải lên file dữ liệu điểm số ở thanh bên trái để bắt đầu.")
    st.stop()

# --- Đọc file, tự nhận diện dòng tiêu đề ---
try:
    if uploaded.name.endswith(".csv"):
        raw_no_header = pd.read_csv(uploaded, header=None)
    else:
        raw_no_header = pd.read_excel(uploaded, header=None)
except Exception as e:
    st.error(f"Không đọc được file: {e}")
    st.stop()

preview_rows = min(10, len(raw_no_header))
non_null_counts = raw_no_header.head(preview_rows).notna().sum(axis=1)
guess_header_row = int(non_null_counts.idxmax())

with st.expander("📄 3. Xem trước dữ liệu thô & chọn dòng tiêu đề", expanded=False):
    st.dataframe(raw_no_header.head(preview_rows), use_container_width=True)
    header_row = st.number_input(
        "Dòng nào là DÒNG TIÊU ĐỀ? (đánh số từ 0)",
        min_value=0, max_value=preview_rows - 1, value=guess_header_row, step=1,
    )

raw = raw_no_header.iloc[header_row + 1:].copy()
raw.columns = raw_no_header.iloc[header_row].values
raw = raw.dropna(axis=1, how="all")
raw = raw.loc[:, [c for c in raw.columns if pd.notna(c)]]
raw.index = range(len(raw))  # chỉ số gốc dùng để ghép nối chính xác giữa các cột điểm

cols = raw.columns.tolist()
st.markdown("### 🎯 4. Chọn cột dữ liệu & chế độ phân tích")

c_group, c_mode = st.columns([1, 1])
with c_group:
    group_col = st.selectbox("Cột chứa LOẠI HÌNH LỚP", cols, index=0)
with c_mode:
    mode = st.radio(
        "Chế độ phân tích", ["🔎 Một cột điểm", "🧮 Nhiều cột điểm (so sánh)"],
        horizontal=True,
    )

available_score_cols = [c for c in cols if c != group_col]

if mode == "🔎 Một cột điểm":
    score_cols = [st.selectbox("Chọn cột ĐIỂM SỐ cần phân tích", available_score_cols)]
else:
    score_cols = st.multiselect(
        "Chọn các cột ĐIỂM SỐ cần phân tích (vd Giữa kỳ và Cuối kỳ để so sánh tiến bộ)",
        available_score_cols, default=available_score_cols[:2],
    )

if not score_cols:
    st.warning("Vui lòng chọn ít nhất một cột điểm số.")
    st.stop()

st.markdown("---")

# --- Phân tích từng cột đã chọn ---
results_by_col = {}
for score_col in score_cols:
    r = analyze_column(raw, group_col, score_col)
    if r is None:
        st.warning(f"Không có dữ liệu hợp lệ cho cột `{score_col}`.")
        continue
    results_by_col[score_col] = r

if not results_by_col:
    st.stop()

if len(results_by_col) == 1:
    only_col = list(results_by_col.keys())[0]
    st.markdown(f"## 📊 Kết quả phân tích: `{only_col}`")
    render_analysis(raw, group_col, results_by_col[only_col], exception_threshold)
else:
    tab_labels = [f"📊 {c}" for c in results_by_col] + ["📈 So sánh tiến bộ"]
    outer_tabs = st.tabs(tab_labels)
    for tab, score_col in zip(outer_tabs[:-1], results_by_col):
        with tab:
            render_analysis(raw, group_col, results_by_col[score_col], exception_threshold)
    with outer_tabs[-1]:
        keys = list(results_by_col.keys())
        cA, cB = st.columns(2)
        with cA:
            col_from = st.selectbox("Từ cột", keys, index=0)
        with cB:
            col_to = st.selectbox("Sang cột", keys, index=min(1, len(keys) - 1))
        if col_from == col_to:
            st.info("Hãy chọn hai cột điểm khác nhau để so sánh tiến bộ.")
        else:
            render_progress_comparison(raw, group_col, results_by_col, col_from, col_to)

st.markdown("---")
st.caption(
    "Phương pháp: Z-score GMM Mềm chuẩn hoá điểm theo trung bình có trọng số xác suất hậu "
    "nghiệm (γ) giữa các thành phần Gauss, tránh phân loại cứng gây bất công ở vùng ranh giới. "
    "Hệ thống tự động chuyển về Z-score truyền thống khi không đủ bằng chứng đa đỉnh (so sánh BIC)."
)
