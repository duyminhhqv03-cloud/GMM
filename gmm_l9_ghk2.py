"""
SmartZ-EDU — Hệ thống Z-score thích ứng bằng Mô hình Hỗn hợp Gauss Mềm (Soft GMM)
hỗ trợ đánh giá công bằng và ra quyết định quản lý giáo dục THCS.

Phương pháp chính: Z-score GMM MỀM (soft/posterior-weighted) — mỗi học sinh được
gán một trọng số xác suất (gamma) thuộc về từng "cụm năng lực", thay vì bị ép buộc
phân loại cứng vào một nhóm duy nhất. Điều này giúp học sinh ở vùng ranh giới giữa
hai cụm không bị đánh giá cực đoan.

Cơ chế thích ứng tự động: so sánh BIC giữa mô hình 1 thành phần (đơn đỉnh) và
2 thành phần (đa đỉnh). Nếu phổ điểm thực chất chỉ có 1 đỉnh (không đủ bằng chứng
đa phương thức), hệ thống tự động dùng Z-score truyền thống thay vì "cưỡng ép"
dữ liệu vào một mô hình 2 cụm không phù hợp.

Chạy local:  streamlit run app.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.mixture import GaussianMixture

st.set_page_config(page_title="SmartZ-EDU — Z-score thích ứng bằng Soft GMM", layout="wide")

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
    """So sánh BIC(k=1) vs BIC(k=2) để quyết định có nên dùng GMM hay không.
    Trả về (dung_gmm: bool, bic1, bic2, delta_bic)."""
    X = x.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=42).fit(X)
    gmm2 = GaussianMixture(n_components=2, n_init=20, random_state=42).fit(X)
    bic1, bic2 = gmm1.bic(X), gmm2.bic(X)
    delta = bic1 - bic2  # dương = k=2 tốt hơn
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
    z_soft, gamma1, gamma2 = z_soft_gmm(x, comp1, comp2)
    mu1, s1, _ = comp1
    mu2, s2, _ = comp2
    hard1 = gamma1 >= 0.5
    z_hard = np.where(hard1, (x - mu1) / s1, (x - mu2) / s2)
    return z_hard


# ----------------------------------------------------------------------
# GIAO DIỆN
# ----------------------------------------------------------------------

st.title("🎓 SmartZ-EDU — Z-score thích ứng bằng Mô hình Hỗn hợp Gauss Mềm")
st.caption(
    "Hệ thống hỗ trợ đánh giá công bằng và ra quyết định quản lý giáo dục khi trường có "
    "nhiều loại hình lớp (vd: Lớp Tăng cường Tiếng Anh và Lớp hai buổi) khiến phổ điểm "
    "không còn phân phối chuẩn đơn đỉnh."
)
with st.expander("ℹ️ Về phương pháp Soft GMM và cơ chế thích ứng tự động"):
    st.markdown(
        """
- **Z-score GMM Mềm (phương pháp chính):** mỗi học sinh được gán trọng số xác suất
  γ (gamma) thuộc về từng cụm năng lực, thay vì bị ép buộc phân loại cứng. Học sinh
  ở vùng ranh giới giữa hai cụm sẽ có Z-score biến thiên mượt mà, tránh bị đánh giá
  cực đoan một cách bất công.
- **Cơ chế thích ứng tự động:** hệ thống so sánh BIC giữa mô hình 1 thành phần và
  2 thành phần. Nếu phổ điểm không đủ bằng chứng đa đỉnh (BIC không cải thiện),
  hệ thống **tự động dùng Z-score truyền thống** thay vì cưỡng ép dữ liệu vào
  khuôn mẫu 2 cụm không phù hợp.
"""
    )

with st.sidebar:
    st.header("1. Tải dữ liệu")
    uploaded = st.file_uploader("Chọn file Excel/CSV điểm số", type=["xlsx", "xls", "csv"])
    st.markdown("---")
    st.markdown(
        "**Định dạng file cần có tối thiểu 2 cột:**\n"
        "- Cột **loại hình lớp** (ví dụ: Lớp TC / Lớp hai buổi)\n"
        "- Cột **điểm số** cần chuẩn hoá — khuyến nghị chạy **riêng** cho từng cột "
        "(Giữa kỳ, Cuối kỳ...) để theo dõi tiến bộ, thay vì gộp chung."
    )
    exception_threshold = st.slider(
        "Ngưỡng γ để xác định 'ngoại lệ sư phạm'", 0.5, 0.9, 0.7, 0.05,
        help="Học sinh có γ thuộc cụm khác > ngưỡng này sẽ được gắn cờ như một trường hợp"
             " đáng chú ý cho công tác quản lý (chuyển lớp / phụ đạo).",
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

st.subheader("2. Xem trước dữ liệu & chọn dòng tiêu đề")
st.dataframe(raw_no_header.head(preview_rows), use_container_width=True)
header_row = st.number_input(
    "Dòng nào là DÒNG TIÊU ĐỀ? (đánh số từ 0)",
    min_value=0, max_value=preview_rows - 1, value=guess_header_row, step=1,
)
raw = raw_no_header.iloc[header_row + 1:].copy()
raw.columns = raw_no_header.iloc[header_row].values
raw = raw.dropna(axis=1, how="all")
raw = raw.loc[:, [c for c in raw.columns if pd.notna(c)]].reset_index(drop=True)
st.dataframe(raw.head(10), use_container_width=True)

cols = raw.columns.tolist()
col_a, col_b = st.columns(2)
with col_a:
    group_col = st.selectbox("Cột chứa LOẠI HÌNH LỚP", cols, index=0)
with col_b:
    score_cols = st.multiselect(
        "Cột/các cột ĐIỂM SỐ (chọn riêng từng cột, vd Giữa kỳ và Cuối kỳ, để so sánh tiến bộ)",
        [c for c in cols if c != group_col],
        default=[c for c in cols if c != group_col][:2],
    )

if not score_cols:
    st.warning("Vui lòng chọn ít nhất một cột điểm số.")
    st.stop()

soft_results = {}  # lưu để so sánh tiến bộ giữa các cột điểm (vd GK -> CK)

for score_col in score_cols:
    st.markdown("---")
    st.header(f"📊 Phân tích: `{score_col}`")

    data = raw[[group_col, score_col]].dropna().copy()
    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data = data.dropna().reset_index(drop=True)
    if data.empty:
        st.warning(f"Không có dữ liệu hợp lệ cho cột {score_col}.")
        continue

    x = data[score_col].values

    # --- Cơ chế thích ứng tự động ---
    use_gmm, bic1, bic2, delta = adaptive_check(x)
    c1, c2, c3 = st.columns(3)
    c1.metric("Số học sinh", len(x))
    c2.metric("BIC (k=1 / k=2)", f"{bic1:.1f} / {bic2:.1f}")
    c3.metric("ΔBIC (k1 − k2)", f"{delta:.1f}")

    if use_gmm:
        st.success(
            "✅ Phổ điểm có bằng chứng đa đỉnh (ΔBIC > 0) → hệ thống áp dụng **Z-score GMM Mềm**."
        )
        comp1, comp2 = fit_gmm_components(x)
        z_s, gamma1, gamma2 = z_soft_gmm(x, comp1, comp2)
        z_h = z_hard_gmm(x, comp1, comp2)
        zn = z_naive(x)

        result = data.copy()
        result["Z_truyen_thong"] = zn.round(3)
        result["gamma_cum_thap"] = gamma1.round(4)
        result["gamma_cum_cao"] = gamma2.round(4)
        result["Z_GMM_mem"] = z_s.round(3)
        result["Z_GMM_cung"] = z_h.round(3)
        soft_results[score_col] = result.set_index(result.index)[["Z_GMM_mem"]].rename(
            columns={"Z_GMM_mem": f"Z_GMM_mem_{score_col}"}
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 Biểu đồ", "🚩 Ngoại lệ sư phạm", "📋 Tham số mô hình", "🗂️ Bảng kết quả"]
        )

        with tab1:
            mu1, s1, pi1 = comp1
            mu2, s2, pi2 = comp2
            fig, ax = plt.subplots(figsize=(9, 5))
            bins = np.arange(0, 10.5, 0.5)
            ax.hist(x, bins=bins, density=True, color="#87CEEB", edgecolor="black", alpha=0.65,
                    label="Phổ điểm thực tế")
            xs = np.linspace(0, 10, 400)
            ax.plot(xs, pi1 * stats.norm.pdf(xs, mu1, s1), "r--", lw=2, label=f"Cụm điểm thấp hơn (μ≈{mu1:.1f})")
            ax.plot(xs, pi2 * stats.norm.pdf(xs, mu2, s2), "g--", lw=2, label=f"Cụm điểm cao hơn (μ≈{mu2:.1f})")
            ax.plot(xs, pi1 * stats.norm.pdf(xs, mu1, s1) + pi2 * stats.norm.pdf(xs, mu2, s2),
                    "b-", lw=2.3, label="Tổng hợp GMM 2 đỉnh")
            ax.set_title(f"Phổ điểm {score_col} — bước nhảy 0.5 điểm")
            ax.set_xlabel("Điểm số"); ax.set_ylabel("Mật độ xác suất")
            ax.set_xticks(np.arange(0, 10.5, 0.5)); ax.tick_params(axis="x", rotation=45)
            ax.legend(fontsize=8); ax.grid(True, ls="--", alpha=0.4)
            plt.tight_layout()
            st.pyplot(fig)

        with tab2:
            st.markdown(
                f"Học sinh được gắn cờ khi xác suất γ thuộc **cụm khác với nhóm hành chính hiện tại** "
                f"vượt ngưỡng **{exception_threshold}** — gợi ý tham mưu chuyển lớp / phụ đạo."
            )
            groups = data[group_col].unique().tolist()
            if len(groups) == 2:
                means_by_group = data.groupby(group_col)[score_col].mean()
                low_group = means_by_group.idxmin()
                high_group = means_by_group.idxmax()
                exc1 = result[(result[group_col] == low_group) & (result["gamma_cum_cao"] > exception_threshold)]
                exc2 = result[(result[group_col] == high_group) & (result["gamma_cum_cao"] < 1 - exception_threshold)]
                st.write(f"🔺 **{len(exc1)} học sinh** thuộc *{low_group}* có năng lực thể hiện gần với cụm điểm cao (γ > {exception_threshold}) — cân nhắc bồi dưỡng/đề xuất chuyển lớp.")
                st.dataframe(exc1, use_container_width=True)
                st.write(f"🔻 **{len(exc2)} học sinh** thuộc *{high_group}* có năng lực thể hiện gần với cụm điểm thấp (γ < {1-exception_threshold:.1f}) — cân nhắc phụ đạo thêm.")
                st.dataframe(exc2, use_container_width=True)
            else:
                st.info("Cần đúng 2 nhóm trong cột loại hình lớp để phát hiện ngoại lệ sư phạm.")

        with tab3:
            st.write(f"**Cụm điểm thấp hơn:** μ = {mu1:.3f}, σ = {s1:.3f}, trọng số = {pi1:.3f}")
            st.write(f"**Cụm điểm cao hơn:** μ = {mu2:.3f}, σ = {s2:.3f}, trọng số = {pi2:.3f}")
            st.dataframe(result.groupby(group_col)[["Z_truyen_thong", "Z_GMM_mem"]].mean())

        with tab4:
            st.dataframe(result, use_container_width=True)
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                result.to_excel(writer, index=False, sheet_name="KetQua")
            st.download_button(
                f"⬇️ Tải kết quả ({score_col})", data=buf.getvalue(),
                file_name=f"ketqua_{score_col}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    else:
        st.info(
            "ℹ️ Không đủ bằng chứng đa đỉnh (ΔBIC ≤ 0) → hệ thống **tự động dùng Z-score "
            "truyền thống** cho cột điểm này, thay vì cưỡng ép mô hình 2 cụm không phù hợp."
        )
        zn = z_naive(x)
        result = data.copy()
        result["Z_truyen_thong"] = zn.round(3)
        st.dataframe(result, use_container_width=True)

# --- So sánh tiến bộ giữa các cột điểm đã chọn (vd GK -> CK) ---
if len(soft_results) >= 2:
    st.markdown("---")
    st.header("📈 So sánh tiến bộ giữa các cột điểm (Z-score GMM Mềm)")
    keys = list(soft_results.keys())
    merged = pd.concat([raw[[group_col]]] + [soft_results[k] for k in keys], axis=1).dropna()
    c1, c2 = keys[0], keys[1]
    merged["Tien_bo"] = merged[f"Z_GMM_mem_{c2}"] - merged[f"Z_GMM_mem_{c1}"]
    st.write(f"Chênh lệch Z-score GMM Mềm từ **{c1}** sang **{c2}** — dương nghĩa là tiến bộ, âm là sa sút, theo đúng nhóm tham chiếu năng lực của từng học sinh.")
    st.dataframe(merged.groupby(group_col)["Tien_bo"].agg(["mean", "std", "count"]))
    st.dataframe(merged, use_container_width=True)

st.markdown("---")
st.caption(
    "Phương pháp: Z-score GMM Mềm chuẩn hoá điểm theo trung bình có trọng số xác suất hậu "
    "nghiệm (γ) giữa các thành phần Gauss, tránh phân loại cứng gây bất công ở vùng ranh giới. "
    "Hệ thống tự động chuyển về Z-score truyền thống khi không đủ bằng chứng đa đỉnh (so sánh BIC)."
)
