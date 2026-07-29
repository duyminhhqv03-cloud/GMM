# -*- coding: utf-8 -*-
"""
HỆ THỐNG TÍNH Z-SCORE VÀ VẼ ĐỒ THỊ BƯỚC NHẢY 0.5 ĐIỂM BẰNG MÔ HÌNH GMM
Hỗ trợ chọn cột điểm linh hoạt theo yêu cầu
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
    page_title="Phân Tích Phổ Điểm GMM Linh Hoạt", page_icon="📊", layout="wide"
)

st.title("📊 PHÂN TÍCH PHỔ ĐIỂM BẰNG MÔ HÌNH GMM (TÙY CHỌN CỘT)")
st.write(
    "Ứng dụng tự động phân tích phổ điểm, tính Z-Score đơn đỉnh & đa đỉnh GMM theo **cột điểm tùy chọn** trong file Excel."
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
    st.sidebar.success("✅ Tải file thành công!")

    # 2. Chọn cột điểm cần phân tích
    all_columns = df.columns.tolist()

    # Tìm vị trí mặc định gợi ý (nếu có cột chứa từ GK, CK, Diem...)
    default_idx = 0
    for idx, col in enumerate(all_columns):
      col_str = str(col).upper()
      if any(k in col_str for k in ["GK", "GIUA KY", "GIỮA KỲ", "CK", "CUOI KY", "CUỐI KỲ", "DIEM", "ĐIỂM"]):
        default_idx = idx
        break

    col_diem = st.sidebar.selectbox(
        "🎯 Chọn cột điểm cần phân tích:",
        options=all_columns,
        index=default_idx,
        help="Hãy chọn cột chứa dữ liệu điểm số thực tế.",
    )

    # 3. Làm sạch dữ liệu và chuyển đổi cột điểm sang dạng số
    # Ép kiểu dữ liệu sang dạng số (chuyển các ô chữ/lỗi thành NaN)
    df_clean = df.copy()
    df_clean[col_diem] = pd.to_numeric(df_clean[col_diem], errors="coerce")

    # Loại bỏ các ô trống (NaN) ở cột điểm đã chọn
    df_clean = df_clean.dropna(subset=[col_diem]).copy()

    # Kiểm tra số lượng dòng hợp lệ
    if len(df_clean) < 2:
      st.error(
          f"❌ Cột **'{col_diem}'** không có đủ dữ liệu số để phân tích! "
          f"(Chỉ tìm thấy {len(df_clean)} dòng dữ liệu hợp lệ, cần tối thiểu 2 dòng số). "
          f"Vui lòng chọn lại cột điểm khác ở thanh bên trái."
      )
    else:
      X = df_clean[[col_diem]].values

      # 4. Tính Z-score truyền thống (Đơn đỉnh)
      mu_gk = df_clean[col_diem].mean()
      sigma_gk = df_clean[col_diem].std()
      df_clean[f"Z_DonDinh_{col_diem}"] = np.round(
          (df_clean[col_diem] - mu_gk) / sigma_gk, 4
      )

      # 5. Huấn luyện GMM và tính Z-score hiệu chỉnh (Đa đỉnh)
      gmm = GaussianMixture(n_components=2, random_state=42)
      gmm.fit(X)

      means = gmm.means_.flatten()
      stds = np.sqrt(gmm.covariances_).flatten()
      weights = gmm.weights_.flatten()

      # Sắp xếp cụm theo điểm trung bình tăng dần
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

      # Tránh lỗi chia cho 0 nếu mật độ bằng 0
      denom = pi1 * f1 + pi2 * f2
      denom[denom == 0] = 1e-10

      gamma1 = (pi1 * f1) / denom
      gamma2 = 1.0 - gamma1

      z_cum1 = (df_clean[col_diem] - mu1) / sigma1
      z_cum2 = (df_clean[col_diem] - mu2) / sigma2

      df_clean[f"{col_diem}_XacSuat_Cum1"] = np.round(gamma1, 4)
      df_clean[f"{col_diem}_XacSuat_Cum2"] = np.round(gamma2, 4)
      df_clean[f"Z_DaDinh_GMM_{col_diem}"] = np.round(
          gamma1 * z_cum1 + gamma2 * z_cum2, 4
      )

      # Hiển thị thông số mô hình
      st.subheader(f"📌 Thông số mô hình GMM 2 đỉnh cho cột: `{col_diem}`")
      col1, col2, col3 = st.columns(3)
      with col1:
        st.metric(
            label="Tổng số mẫu hợp lệ", value=f"{len(df_clean)} học sinh/SV"
        )
      with col2:
        st.metric(
            label="Cụm 1 (Đại Trà / Thấp)",
            value=f"μ1 = {mu1:.2f}",
            delta=f"σ1 = {sigma1:.2f} | Tỷ trọng = {pi1*100:.1f}%",
        )
      with col3:
        st.metric(
            label="Cụm 2 (Tăng Cường / Cao)",
            value=f"μ2 = {mu2:.2f}",
            delta=f"σ2 = {sigma2:.2f} | Tỷ trọng = {pi2*100:.1f}%",
        )

      # 6. VẼ ĐỒ THỊ TÍCH HỢP BƯỚC NHẢY 0.5 ĐIỂM
      st.subheader(f"📈 Đồ thị phân tích phổ điểm (`{col_diem}`)")
      fig, ax = plt.subplots(figsize=(12, 6))

      # Biểu đồ Histogram thực tế
      sns.histplot(
          df_clean[col_diem],
          binwidth=0.5,
          binrange=(0, 10),
          stat="density",
          color="skyblue",
          edgecolor="black",
          alpha=0.6,
          label="Phổ điểm thực tế",
          ax=ax,
      )

      # Đường cong GMM
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
          f"PHÂN TÍCH PHỔ ĐIỂM [{col_diem.upper()}] BẰNG MÔ HÌNH GMM (BƯỚC NHẢY 0.5 ĐIỂM)",
          fontsize=13,
          fontweight="bold",
      )
      ax.set_xlabel(f"Điểm số ({col_diem})", fontsize=11)
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
      output_filename = f"Ket_Qua_ZScore_{col_diem}.xlsx"
      output_buffer = io.BytesIO()
      with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
        df_clean.to_excel(writer, index=False)

      st.download_button(
          label=f"📥 Tải xuống file Excel kết quả ({output_filename})",
          data=output_buffer.getvalue(),
          file_name=output_filename,
          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      )

  except Exception as e:
    st.error(f"Xảy ra lỗi trong quá trình đọc file Excel: {e}")

else:
  st.info(
      "👈 Vui lòng tải lên file Excel (.xlsx) ở thanh bên trái để bắt đầu phân tích."
  )
