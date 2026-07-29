# -*- coding: utf-8 -*-
"""
HỆ THỐNG PHÂN TÍCH PHỔ ĐIỂM GMM - PHIÊN BẢN DASHBOARD HIỆN ĐẠI
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# --- CẤU HÌNH TRANG STREAMLIT (Nên để đầu tiên) ---
st.set_page_config(
    page_title="GMM Analytics Dashboard", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS TÙY CHỈNH CHO GIAO DIỆN HIỆN ĐẠI ---
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; }
    h1 { color: #1E3A8A; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { font-size: 1.1rem; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ DASHBOARD PHÂN TÍCH PHỔ ĐIỂM GMM TỰ ĐỘNG")
st.markdown("Hệ thống tự động nhận diện phân phối điểm số, phân cụm học sinh và tính toán Z-Score đa đỉnh.")

# --- THANH BÊN (SIDEBAR) ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2000/2000431.png", width=60) # Icon trang trí
st.sidebar.header("📁 Tải dữ liệu & Cấu hình")

uploaded_file = st.sidebar.file_uploader("Tải lên file Excel (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Đọc file Excel
        df = pd.read_excel(uploaded_file, header=0)
        st.sidebar.success("✅ Tải file thành công!")

        # Chọn cột điểm
        all_columns = df.columns.tolist()
        default_idx = 0
        for idx, col in enumerate(all_columns):
            if any(k in str(col).upper() for k in ["GK", "CK", "DIEM", "ĐIỂM", "SCORE"]):
                default_idx = idx
                break

        col_diem = st.sidebar.selectbox(
            "🎯 Chọn cột điểm cần phân tích:",
            options=all_columns,
            index=default_idx
        )
        
        # Tiện ích xem trước dữ liệu thô (Giúp sửa lỗi cột rỗng)
        with st.sidebar.expander("🛠️ Xem trước dữ liệu thô cột này"):
            st.dataframe(df[[col_diem]].head(10))

        # --- LÀM SẠCH DỮ LIỆU ---
        df_clean = df.copy()
        df_clean[col_diem] = pd.to_numeric(df_clean[col_diem], errors="coerce")
        df_clean = df_clean.dropna(subset=[col_diem]).copy()

        if len(df_clean) < 2:
            st.error(
                f"❌ Cột **'{col_diem}'** bị trống hoặc không chứa dữ liệu số hợp lệ! "
                f"(Chỉ tìm thấy {len(df_clean)} dòng hợp lệ). Vui lòng chọn cột khác."
            )
        else:
            X = df_clean[[col_diem]].values

            # --- TÍNH TOÁN GMM ---
            mu_gk = df_clean[col_diem].mean()
            sigma_gk = df_clean[col_diem].std()
            df_clean[f"Z_DonDinh_{col_diem}"] = np.round((df_clean[col_diem] - mu_gk) / sigma_gk, 4)

            gmm = GaussianMixture(n_components=2, random_state=42)
            gmm.fit(X)

            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_).flatten()
            weights = gmm.weights_.flatten()

            # Sắp xếp cụm (Cụm 1: Điểm thấp, Cụm 2: Điểm cao)
            idx_sort = np.argsort(means)
            mu1, sigma1, pi1 = means[idx_sort[0]], stds[idx_sort[0]], weights[idx_sort[0]]
            mu2, sigma2, pi2 = means[idx_sort[1]], stds[idx_sort[1]], weights[idx_sort[1]]

            f1 = norm.pdf(df_clean[col_diem], loc=mu1, scale=sigma1)
            f2 = norm.pdf(df_clean[col_diem], loc=mu2, scale=sigma2)

            denom = pi1 * f1 + pi2 * f2
            denom[denom == 0] = 1e-10

            gamma1 = (pi1 * f1) / denom
            gamma2 = 1.0 - gamma1

            z_cum1 = (df_clean[col_diem] - mu1) / sigma1
            z_cum2 = (df_clean[col_diem] - mu2) / sigma2

            df_clean[f"XacSuat_Cum1"] = np.round(gamma1, 4)
            df_clean[f"XacSuat_Cum2"] = np.round(gamma2, 4)
            df_clean[f"Z_DaDinh_{col_diem}"] = np.round(gamma1 * z_cum1 + gamma2 * z_cum2, 4)
            
            # Tính năng mới: Phân loại học sinh tự động
            df_clean["Phan_Loai_Cum"] = np.where(gamma2 > gamma1, "Cụm 2 (Tăng Cường)", "Cụm 1 (Đại Trà)")

            # ==========================================
            # TRÌNH BÀY GIAO DIỆN CHÍNH DẠNG TABS
            # ==========================================
            tab1, tab2, tab3 = st.tabs(["📊 Tổng quan & Đồ thị", "📈 Thống kê & Đánh giá", "📋 Dữ liệu & Xuất File"])

            # --- TAB 1: ĐỒ THỊ TƯƠNG TÁC PLOTLY ---
            with tab1:
                st.subheader("📌 Chỉ số cốt lõi")
                m1, m2, m3 = st.columns(3)
                m1.metric("Tổng số học sinh hợp lệ", f"{len(df_clean)} HS")
                m2.metric("Điểm Trung Bình (Toàn rạp)", f"{mu_gk:.2f} điểm")
                m3.metric("Độ lệch chuẩn chung", f"{sigma_gk:.2f}")

                st.markdown("---")
                st.subheader(f"📈 Phân tích Mật độ Xác suất (Plotly Interactive)")
                
                # Tạo dải X để vẽ đường cong
                x_axis = np.linspace(0, 10, 500)
                y1 = pi1 * norm.pdf(x_axis, mu1, sigma1)
                y2 = pi2 * norm.pdf(x_axis, mu2, sigma2)
                y_sum = y1 + y2

                fig = go.Figure()

                # Vẽ Histogram (Phổ điểm thực tế)
                fig.add_trace(go.Histogram(
                    x=df_clean[col_diem], 
                    histnorm='probability density',
                    xbins=dict(start=0, end=10, size=0.5),
                    marker_color='lightblue',
                    marker_line_color='black',
                    marker_line_width=1,
                    opacity=0.7,
                    name='Phổ điểm thực tế',
                    hovertemplate='Điểm: %{x}<br>Mật độ: %{y:.4f}<extra></extra>'
                ))

                # Đường Cụm 1
                fig.add_trace(go.Scatter(
                    x=x_axis, y=y1, mode='lines', 
                    line=dict(color='red', width=2, dash='dash'),
                    name=f'Cụm Đại Trà (μ={mu1:.2f})',
                    hovertemplate='Điểm: %{x:.1f}<br>Mật độ: %{y:.4f}'
                ))

                # Đường Cụm 2
                fig.add_trace(go.Scatter(
                    x=x_axis, y=y2, mode='lines', 
                    line=dict(color='green', width=2, dash='dash'),
                    name=f'Cụm Tăng Cường (μ={mu2:.2f})'
                ))

                # Đường Tổng hợp
                fig.add_trace(go.Scatter(
                    x=x_axis, y=y_sum, mode='lines', 
                    line=dict(color='blue', width=3),
                    name='Tổng hợp GMM 2 đỉnh'
                ))

                fig.update_layout(
                    title=dict(text=f"PHÂN TÍCH PHỔ ĐIỂM [{col_diem.upper()}] (BƯỚC NHẢY 0.5)", font=dict(size=18)),
                    xaxis_title="Điểm số",
                    yaxis_title="Mật độ xác suất",
                    xaxis=dict(tickmode='linear', tick0=0, dtick=0.5, range=[0, 10]),
                    hovermode="x unified",
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"),
                    margin=dict(l=40, r=40, t=60, b=40)
                )

                st.plotly_chart(fig, use_container_width=True)

            # --- TAB 2: THỐNG KÊ CHI TIẾT ---
            with tab2:
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    st.markdown("### 🔍 Thống kê mô tả")
                    stats_df = df_clean[col_diem].describe().reset_index()
                    stats_df.columns = ["Chỉ số", "Giá trị"]
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                with col_stat2:
                    st.markdown("### ⚙️ Thông số Mô hình GMM")
                    st.info(f"**Mô hình hội tụ sau:** {gmm.n_iter_} vòng lặp")
                    
                    gmm_data = {
                        "Cụm": ["Cụm 1 (Đại Trà)", "Cụm 2 (Tăng Cường)"],
                        "Trung bình (μ)": [np.round(mu1, 3), np.round(mu2, 3)],
                        "Độ lệch chuẩn (σ)": [np.round(sigma1, 3), np.round(sigma2, 3)],
                        "Tỷ trọng (%)": [f"{pi1*100:.1f}%", f"{pi2*100:.1f}%"]
                    }
                    st.table(pd.DataFrame(gmm_data))

            # --- TAB 3: DỮ LIỆU & XUẤT FILE ---
            with tab3:
                st.markdown("### 📋 Dữ liệu kết quả chi tiết")
                
                # Bộ lọc dữ liệu nhanh
                filter_cum = st.radio("Lọc theo nhóm học sinh:", ["Tất cả", "Cụm 1 (Đại Trà)", "Cụm 2 (Tăng Cường)"], horizontal=True)
                
                df_display = df_clean.copy()
                if filter_cum != "Tất cả":
                    df_display = df_display[df_display["Phan_Loai_Cum"] == filter_cum]

                st.dataframe(df_display, use_container_width=True)

                # Nút tải file Excel
                output_filename = f"Ket_Qua_GMM_{col_diem}.xlsx"
                output_buffer = io.BytesIO()
                with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
                    df_clean.to_excel(writer, index=False)

                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    label="📥 TẢI XUỐNG FILE EXCEL TOÀN BỘ KẾT QUẢ",
                    data=output_buffer.getvalue(),
                    file_name=output_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="primary"
                )

    except Exception as e:
        st.error(f"Xảy ra lỗi trong quá trình xử lý: {e}")

else:
    # Màn hình chờ khi chưa tải file
    st.info("👈 Vui lòng tải lên file Excel (.xlsx) ở thanh bên trái để xem Dashboard.")
    st.markdown("---")
    st.markdown("### Tính năng hỗ trợ:")
    st.markdown("- Tự động nhận diện cột điểm.")
    st.markdown("- Lọc, loại bỏ dữ liệu nhiễu/khoảng trắng tự động.")
    st.markdown("- Giao diện đồ thị tương tác cao.")
