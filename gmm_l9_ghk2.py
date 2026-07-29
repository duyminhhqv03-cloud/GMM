# -*- coding: utf-8 -*-
"""
HỆ THỐNG TÍNH Z-SCORE VÀ VẼ ĐỒ THỊ BƯỚC NHẢY 0.5 ĐIỂM BẰNG MÔ HÌNH GMM
Triển khai trên giao diện Streamlit Web App
"""

import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# --- CẤU HÌNH TRANG STREAMLIT ---
st.set_page_config(
    page_title="Phân Tích Phổ Điểm GMM", page_icon="📊", layout="wide"
)

st.title("📊 PHÂN TÍCH PHỔ ĐIỂM GIỮA KỲ BẰNG MÔ HÌNH GMM")
st.write(
    "Ứng dụng tính Z-Score đa đỉnh GMM, Z-Score truyền thống và trực quan hóa phổ điểm với bước nhảy 0.5 điểm."
)

# --- THANH BÊN (SIDEBAR) ---
st.sidebar.header("📁 Tải dữ liệu & Cấu hình")

# 1. Tải file Excel
uploaded_file = st.sidebar.file_uploader(
    "Tải lên file Excel điểm số (.xlsx)", type=["xlsx"]
)

if uploaded_file is not None:
  try:
    # Đọc file Excel
    df = pd.read_excel(uploaded_file, header=0)

    st.success("✅ Đã tải file thành công!")

    # Chọn cột điểm
    all_columns = df.columns.tolist()
    default_col_idx = 0
    if "GK" in all_columns:
      default_col_idx = all_columns.index("GK")
    elif "CK" in all_columns:
      default_col_idx = all_columns.index("CK")

    col_diem = st.sidebar.selectbox(
        "Chọn cột điểm cần phân tích:",
        options=all_columns,
        index=default_col_idx,
    )

    # 3. Làm sạch dữ liệu và tạo df_clean
    df_clean = df.dropna(subset=[col_diem]).copy()
    X = df_clean[[col_diem]].values

    # 4. Tính Z-score truyền thống (Đơn đỉnh)
    mu_gk = df_clean[col_diem].mean()
    sigma_gk = df_clean[col_diem].std()
    df_clean['Z_DonDinh_TruyenThong'] = np.round(
        (df_clean[col_diem] - mu_gk) / sigma_gk, 4
    )

    # 5. Huấn luyện GMM và tính Z-score hiệu chỉnh (Đa đỉnh)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_).flatten()
    weights = gmm.weights_.flatten()

    idx_sort = np.argsort(means)
    mu1, sigma1, pi1 = (
        means[idx_sort[0]],
        stds[idx_sort[0]],
        weights[idx_sort[0]],
    )
    mu2, sigma2, pi2 = (
        means[idx_sort[1]],
        stds[idx_sort[1]],
        weights[idx_sort[1]],
    )

    f1 = norm.pdf(df_clean[col_diem], loc=mu1, scale=sigma1)
    f2 = norm.pdf(df_clean[col_diem], loc=mu2, scale=sigma2)

    gamma1 = (pi1 * f1) / (pi1 * f1 + pi2 * f2)
    gamma2 = 1.0 - gamma1

    z_cum1 = (df_clean[col_diem] - mu1) / sigma1
    z_cum2 = (df_clean[col_diem] - mu2) / sigma2

    df_clean['GK_XacSuat_Cum1'] = np.round(gamma1, 4)
    df_clean['GK_XacSuat_Cum2'] = np.round(gamma2, 4)
    df_clean['Z_DaDinh_GMM'] = np.round(gamma1 * z_cum1 + gamma2 * z_cum2, 4)

    # Hiển thị thông số mô hình
    st.subheader("📌 Thông số mô hình GMM 2 đỉnh")
    col1, col2 = st.columns(2)
    with col1:
      st.metric(
          label="Cụm Đại Trà (Cụm 1)",
          value=f"μ1 = {mu1:.2f}",
          delta=f"σ1 = {sigma1:.2f} | Tỷ trọng = {pi1*100:.1f}%",
      )
    with col2:
      st.metric(
          label="Cụm Tăng Cường (Cụm 2)",
          value=f"μ2 = {mu2:.2f}",
          delta=f"σ2 = {sigma2:.2f} | Tỷ trọng = {pi2*100:.1f}%",
      )

    # 6. HIỂN THỊ ĐỒ THỊ TÍCH HỢP BƯỚC NHẢY 0.5 ĐIỂM
    st.subheader("📈 Đồ thị phân tích phổ điểm")
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.histplot(
        df_clean[col_diem],
        binwidth=0.5,  # Độ rộng mỗi cột = 0.5 điểm
        binrange=(0, 10),  # Dải điểm từ 0 đến 10
        stat="density",
        color="skyblue",
        edgecolor="black",
        alpha=0.6,
        label="Phổ điểm thực tế",
        ax=ax,
    )

    x_axis = np.linspace(0, 10, 500)
    ax.plot(
        x_axis,
        pi1 * norm.pdf(x_axis, mu1, sigma1),
        "r--",
        linewidth=2,
        label=f"Cụm Đại Trà (μ={mu1:.1f})",
    )
    ax.plot(
        x_axis,
        pi2 * norm.pdf(x_axis, mu2, sigma2),
        "g--",
        linewidth=2,
        label=f"Cụm Tăng Cường (μ={mu2:.1f})",
    )
    ax.plot(
        x_axis,
        pi1 * norm.pdf(x_axis, mu1, sigma1)
        + pi2 * norm.pdf(x_axis, mu2, sigma2),
        "b-",
        linewidth=2.5,
        label="Tổng hợp GMM 2 đỉnh",
    )

    ax.set_title(
        "PHÂN TÍCH PHỔ ĐIỂM GIỮA KỲ BẰNG MÔ HÌNH GMM (BƯỚC NHẢY 0.5 ĐIỂM)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Điểm số giữa kỳ", fontsize=11)
    ax.set_ylabel("Mật độ xác suất", fontsize=11)
    ax.set_xticks(np.arange(0, 10.5, 0.5))
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlim(0, 10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    st.pyplot(fig)

    # 7. Hiển thị bảng dữ liệu & Xuất file kết quả
    st.subheader("📋 Xem trước dữ liệu kết quả")
    st.dataframe(df_clean.head(10))

    # Tạo buffer xuất file Excel
    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
      df_clean.to_excel(writer, index=False)

    st.download_button(
        label="📥 Tải xuống file kết quả Excel (Ket_Qua_ZScore_GiuaKy.xlsx)",
        data=output_buffer.getvalue(),
        file_name="Ket_Qua_ZScore_GiuaKy.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

  except Exception as e:
    st.error(f"Xảy ra lỗi trong quá trình xử lý file: {e}")

else:
  st.info(
      "👈 Vui lòng tải lên file Excel (.xlsx) ở thanh bên trái để bắt đầu phân tích."
  )
