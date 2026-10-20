# uts.py
# -*- coding: utf-8 -*-
"""
============================================================
UTS BUSINESS INTELLIGENCE - UNIVERSITAS TIMOR
Versi Web (Streamlit) — Tampilan Modern — REVISI FINAL
============================================================
Jalankan: py -m streamlit run uts.py
============================================================
"""

import os
import io
import base64
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="UTS Business Intelligence — Universitas Timor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"   # tidak ada sidebar kiri
)


# ============================================================
# HELPER — GAMBAR KE BASE64
# ============================================================
def get_base64_image(image_path):
    """Mengubah gambar lokal menjadi string base64 untuk disisipkan ke HTML/CSS."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None

# Load gambar dari folder assets
img_kampus = get_base64_image("assets/kampus.jpg")
img_profil = get_base64_image("assets/foto_profil.jpg")
img_logo = get_base64_image("assets/logo_unimor.jpg")


# ============================================================
# HELPER — FORMAT ANGKA (defensive)
# ============================================================
def _int_fmt(x):
    if x is None:
        return "-"
    try:
        if isinstance(x, bool):
            return str(x)
        if isinstance(x, (int, np.integer)):
            return f"{int(x):,}".replace(",", ".")
        if isinstance(x, (float, np.floating)):
            if not np.isfinite(x):
                return "-"
            return f"{int(round(float(x))):,}".replace(",", ".")
        f = float(x)
        if not np.isfinite(f):
            return str(x)
        return f"{int(round(f)):,}".replace(",", ".")
    except (ValueError, TypeError, OverflowError):
        return str(x)
    except Exception:
        return str(x)


def rp(x):
    if x is None:
        return "-"
    try:
        if isinstance(x, bool):
            return str(x)
        if isinstance(x, (int, np.integer)):
            return "Rp " + f"{int(x):,}".replace(",", ".")
        if isinstance(x, (float, np.floating)):
            if not np.isfinite(x):
                return "-"
            return "Rp " + f"{int(round(float(x))):,}".replace(",", ".")
        f = float(x)
        if not np.isfinite(f):
            return str(x)
        return "Rp " + f"{int(round(f)):,}".replace(",", ".")
    except (ValueError, TypeError, OverflowError):
        return str(x)
    except Exception:
        return str(x)


def _display_df(df, int_cols=None, rp_cols=None):
    if df is None or len(df.columns) == 0:
        return df
    d = df.copy()
    if int_cols is None:
        int_cols = [c for c in d.columns
                    if pd.api.types.is_integer_dtype(d[c])]
    rp_cols = rp_cols or []
    for c in int_cols:
        if c in d.columns:
            try: d[c] = d[c].apply(_int_fmt).astype(str)
            except Exception: pass
    for c in rp_cols:
        if c in d.columns:
            try: d[c] = d[c].apply(rp).astype(str)
            except Exception: pass
    return d


# ============================================================
# HELPER — VISUAL
# ============================================================
def _sec_title(icon, text):
    return (f'<div class="section-title">'
            f'<span class="icon">{icon}</span><span>{text}</span></div>')


def _kpi_row(items):
    cards = []
    for it in items:
        sub_html = (f'<div class="kpi-sub">{it["sub"]}</div>'
                    if it.get("sub") else "")
        cards.append(
            f'<div class="kpi-card kpi-{it.get("color","green")}">'
            f'<div class="kpi-icon">{it["icon"]}</div>'
            f'<div class="kpi-body">'
            f'<div class="kpi-label">{it["label"]}</div>'
            f'<div class="kpi-value">{it["value"]}</div>'
            f'{sub_html}</div></div>'
        )
    return f'<div class="kpi-grid">{"".join(cards)}</div>'


def _hl(kind, icon, title, text):
    return (f'<div class="hl-box hl-{kind}">'
            f'<div class="hl-icon">{icon}</div>'
            f'<div class="hl-body">'
            f'<div class="hl-title">{title}</div>'
            f'<div class="hl-text">{text}</div>'
            f'</div></div>')


def _info_card(icon, title, lines, color="blue", img_b64=None, img_style="width:100px; margin-bottom:15px; border-radius:8px;"):
    body = "".join([f'<div class="info-line">{l}</div>' for l in lines])
    img_html = f'<div style="text-align:center;"><img src="data:image/png;base64,{img_b64}" style="{img_style}"></div>' if img_b64 else f'<div class="info-icon">{icon}</div>'
    
    return (f'<div class="info-card info-{color}">'
            f'{img_html}'
            f'<div class="info-title">{title}</div>'
            f'<div class="info-body">{body}</div>'
            f'</div>')


# ============================================================
# CSS — MODERN & COLORFUL (Dinamis menyesuaikan background)
# ============================================================
hero_bg_css = "background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 45%, #f59e0b 100%);"
if img_kampus:
    hero_bg_css = f"background: linear-gradient(135deg, rgba(30, 58, 138, 0.85) 0%, rgba(37, 99, 235, 0.85) 45%, rgba(245, 158, 11, 0.85) 100%), url('data:image/jpeg;base64,{img_kampus}') center/cover;"

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        -webkit-font-smoothing: antialiased;
    }}
    .stApp {{
        background:
            radial-gradient(1200px 600px at 100% -10%, #eef4ff, transparent 60%),
            radial-gradient(900px 500px at -10% 10%, #fff8e1, transparent 55%),
            #f8fafc;
    }}
    .main .block-container {{
        padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1400px;
    }}
    h1, h2, h3, h4, h5 {{
        color: #0a4522; font-weight: 700; letter-spacing: -0.02em;
    }}

    /* ---------- HERO HEADER ---------- */
    .hero {{
        position: relative;
        {hero_bg_css}
        padding: 32px 36px; border-radius: 20px; color: #ffffff;
        margin-bottom: 22px; overflow: hidden;
        box-shadow: 0 16px 40px -12px rgba(30, 58, 138, 0.45);
    }}
    .hero::before {{
        content: ""; position: absolute; inset: 0;
        background:
            radial-gradient(600px 200px at 92% -10%, rgba(255,255,255,0.25), transparent 65%),
            radial-gradient(400px 200px at 0% 110%, rgba(0,0,0,0.10), transparent 70%);
        pointer-events: none;
    }}
    .hero h1 {{
        position: relative; margin: 0; font-size: 30px;
        font-weight: 800; letter-spacing: -0.6px; color: #ffffff;
    }}
    .hero p {{
        position: relative; margin: 6px 0 0 0;
        opacity: 0.95; font-size: 15px; color: #e0e7ff; font-weight: 400;
    }}
    .hero .tag {{
        display: inline-block; background: rgba(255,255,255,0.20);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255,255,255,0.30);
        color: #ffffff; padding: 5px 14px; border-radius: 999px;
        font-size: 12px; font-weight: 700; margin-right: 6px;
        margin-top: 12px; letter-spacing: 0.4px; text-transform: uppercase;
    }}

    /* ---------- KPI CARDS ---------- */
    .kpi-grid {{
        display: grid; grid-template-columns: repeat(4, 1fr);
        gap: 14px; margin: 4px 0 18px 0;
    }}
    @media (max-width: 900px) {{ .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
    @media (max-width: 520px) {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}
    .kpi-card {{
        position: relative; background: #ffffff; border: 1px solid #e8ecf0;
        border-radius: 16px; padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04); overflow: hidden;
        transition: transform .22s cubic-bezier(.4,0,.2,1), box-shadow .22s cubic-bezier(.4,0,.2,1);
        display: flex; align-items: flex-start; gap: 12px;
    }}
    .kpi-card:hover {{ transform: translateY(-3px); box-shadow: 0 14px 30px -12px rgba(15,23,42,0.20); }}
    .kpi-card::before {{
        content: ""; position: absolute; left: 0; top: 0; bottom: 0;
        width: 4px; border-radius: 4px 0 0 4px;
    }}
    .kpi-green::before  {{ background: linear-gradient(180deg, #0d5c2c, #1fa353); }}
    .kpi-blue::before   {{ background: linear-gradient(180deg, #1e40af, #3b82f6); }}
    .kpi-amber::before  {{ background: linear-gradient(180deg, #b45309, #f59e0b); }}
    .kpi-red::before    {{ background: linear-gradient(180deg, #991b1b, #ef4444); }}
    .kpi-purple::before {{ background: linear-gradient(180deg, #6d28d9, #8b5cf6); }}
    .kpi-cyan::before   {{ background: linear-gradient(180deg, #0e7490, #06b6d4); }}
    .kpi-icon {{
        width: 42px; height: 42px; border-radius: 12px;
        display: grid; place-items: center; font-size: 20px;
        flex-shrink: 0; box-shadow: 0 4px 10px -4px rgba(0,0,0,0.12);
    }}
    .kpi-green .kpi-icon  {{ background: linear-gradient(135deg,#d1fae5,#a7f3d0); color:#065f46; }}
    .kpi-blue .kpi-icon   {{ background: linear-gradient(135deg,#dbeafe,#bfdbfe); color:#1e3a8a; }}
    .kpi-amber .kpi-icon  {{ background: linear-gradient(135deg,#fef3c7,#fde68a); color:#78350f; }}
    .kpi-red .kpi-icon    {{ background: linear-gradient(135deg,#fee2e2,#fecaca); color:#7f1d1d; }}
    .kpi-purple .kpi-icon {{ background: linear-gradient(135deg,#ede9fe,#ddd6fe); color:#4c1d95; }}
    .kpi-cyan .kpi-icon   {{ background: linear-gradient(135deg,#cffafe,#a5f3fc); color:#155e75; }}
    .kpi-body {{ min-width: 0; flex: 1; }}
    .kpi-label {{
        font-size: 10.5px; text-transform: uppercase; letter-spacing: 1.1px;
        color: #64748b; font-weight: 700; margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 22px; font-weight: 800; color: #0f172a;
        letter-spacing: -0.5px; line-height: 1.1; word-break: break-word;
    }}
    .kpi-sub {{ font-size: 11px; color: #94a3b8; font-weight: 600; margin-top: 5px; }}

    /* ---------- INFO CARD (IDENTITAS) ---------- */
    .info-card {{
        background: #ffffff; border: 1px solid #e8ecf0;
        border-radius: 16px; padding: 24px 22px;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        transition: transform .2s, box-shadow .2s; height: 100%;
        margin-bottom: 10px; text-align: center;
    }}
    .info-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 24px -12px rgba(15,23,42,0.18);
    }}
    .info-icon {{
        width: 52px; height: 52px; border-radius: 14px;
        display: grid; place-items: center; font-size: 26px;
        margin: 0 auto 15px auto;
        box-shadow: 0 4px 10px -4px rgba(0,0,0,0.15);
    }}
    .info-blue .info-icon   {{ background: linear-gradient(135deg,#dbeafe,#bfdbfe); color:#1e3a8a; }}
    .info-green .info-icon  {{ background: linear-gradient(135deg,#d1fae5,#a7f3d0); color:#065f46; }}
    .info-amber .info-icon  {{ background: linear-gradient(135deg,#fef3c7,#fde68a); color:#78350f; }}
    .info-purple .info-icon {{ background: linear-gradient(135deg,#ede9fe,#ddd6fe); color:#4c1d95; }}
    .info-title {{
        font-size: 12px; text-transform: uppercase; letter-spacing: 1.5px;
        color: #64748b; font-weight: 700; margin-bottom: 14px;
        border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;
    }}
    .info-line {{
        font-size: 14.5px; color: #0f172a; margin: 8px 0; line-height: 1.5;
    }}
    .info-line b {{ color: #1e40af; }}
    .info-green .info-line b  {{ color: #065f46; }}
    .info-amber .info-line b  {{ color: #92400e; }}
    .info-purple .info-line b {{ color: #4c1d95; }}

    /* ---------- SECTION TITLE ---------- */
    .section-title {{
        display: flex; align-items: center; gap: 10px;
        margin: 10px 0 12px 0; padding-bottom: 10px;
        border-bottom: 2px solid #eef7f0;
        font-size: 1.25rem; font-weight: 800; color: #0a4522;
        letter-spacing: -0.3px;
    }}
    .section-title .icon {{
        width: 34px; height: 34px; display: grid; place-items: center;
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        color: #fff; border-radius: 10px; font-size: 17px;
        box-shadow: 0 4px 10px -3px rgba(30,64,175,0.5);
    }}

    /* ---------- HIGHLIGHT BOXES ---------- */
    .hl-box {{
        display: flex; align-items: flex-start; gap: 14px;
        padding: 16px 20px; border-radius: 14px; border: 1px solid;
        margin: 10px 0 16px 0;
        animation: fadeIn .4s cubic-bezier(.4,0,.2,1);
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(6px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .hl-success {{ background: linear-gradient(135deg,#ecfdf5,#d1fae5); border-color:#6ee7b7; }}
    .hl-info    {{ background: linear-gradient(135deg,#eff6ff,#dbeafe); border-color:#93c5fd; }}
    .hl-warn    {{ background: linear-gradient(135deg,#fffbeb,#fef3c7); border-color:#fcd34d; }}
    .hl-danger  {{ background: linear-gradient(135deg,#fef2f2,#fee2e2); border-color:#fca5a5; }}
    .hl-icon {{
        width: 42px; height: 42px; border-radius: 12px;
        background: rgba(255,255,255,0.85); display: grid; place-items: center;
        font-size: 20px; flex-shrink: 0;
        box-shadow: 0 2px 8px -2px rgba(0,0,0,0.08);
    }}
    .hl-body {{ flex: 1; }}
    .hl-title {{
        font-size: 12.5px; font-weight: 800; text-transform: uppercase;
        letter-spacing: .6px; margin-bottom: 5px; color: #0f172a;
    }}
    .hl-text {{ font-size: 13.5px; color: #334155; line-height: 1.6; font-weight: 500; }}
    .hl-text b {{ color: #0d5c2c; }}
    .hl-info .hl-text b {{ color: #1e40af; }}
    .hl-warn .hl-text b {{ color: #92400e; }}

    /* ---------- TABS ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px; background: #f8fafc; padding: 6px;
        border-radius: 12px; border: 1px solid #e8ecf0;
        flex-wrap: wrap; overflow-x: auto;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: auto; border-radius: 8px; padding: 9px 16px;
        font-weight: 600; font-size: 13.5px; color: #475569;
        background: transparent; transition: all .18s ease;
        white-space: nowrap;
    }}
    .stTabs [data-baseweb="tab"]:hover {{ background: #eef4ff; color: #1e40af; }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #1e40af, #3b82f6) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px -4px rgba(30,64,175,0.45);
    }}
    .stTabs [data-baseweb="tab-highlight"] {{ display: none; }}
    .stTabs [data-baseweb="tab-border"]    {{ display: none; }}
    .stTabs [data-baseweb="tab-panel"] {{
        padding-top: 18px; animation: fadeIn .35s ease;
    }}

    /* ---------- BUTTONS ---------- */
    .stButton > button, .stDownloadButton > button {{
        border-radius: 10px !important; font-weight: 600 !important;
        font-size: 13.5px !important; padding: 8px 16px !important;
        border: 1px solid #2563eb !important; background: #ffffff !important;
        color: #1e40af !important; transition: all .18s ease !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.05);
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background: #eef4ff !important; transform: translateY(-1px);
        box-shadow: 0 8px 18px -6px rgba(30,64,175,0.35);
    }}

    /* ---------- TABLE ---------- */
    [data-testid="stDataFrame"] {{
        border: 1px solid #e8ecf0 !important; border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.04);
    }}
    [data-testid="stDataFrame"] thead tr th {{
        background: #eef4ff !important; color: #1e40af !important;
        font-weight: 700 !important; font-size: 12.5px !important;
    }}

    /* ---------- EXPANDER ---------- */
    div[data-testid="stExpander"] {{
        border: 1px solid #e8ecf0 !important; border-radius: 12px !important;
        background: #ffffff; box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        transition: box-shadow .2s;
    }}
    div[data-testid="stExpander"]:hover {{
        box-shadow: 0 8px 22px -12px rgba(15,23,42,0.18);
    }}

    /* ---------- SLIDER ---------- */
    .stSlider [data-baseweb="slider"] [role="slider"] {{
        background: #2563eb !important; border-color: #2563eb !important;
        box-shadow: 0 0 0 6px rgba(37,99,235,0.15) !important;
    }}

    /* ---------- FILE UPLOADER ---------- */
    [data-testid="stFileUploader"] section {{
        border-radius: 12px !important;
        border: 2px dashed #bfdbfe !important;
        background: #f8fafc !important;
        padding: 16px !important;
    }}
    [data-testid="stFileUploader"] section:hover {{
        border-color: #2563eb !important;
        background: #eef4ff !important;
    }}

    /* ---------- SCROLLBAR ---------- */
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: #f1f5f9; }}
    ::-webkit-scrollbar-thumb {{
        background: #cbd5e1; border-radius: 999px; border: 2px solid #f1f5f9;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: #3b82f6; }}

    /* ---------- FOOTER ---------- */
    .app-footer {{
        text-align: center; color: #94a3b8; font-size: 12px;
        padding: 22px 0 12px 0; margin-top: 40px;
        border-top: 1px solid #e8ecf0;
    }}
    .app-footer strong {{ color: #1e40af; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>📊 Business Intelligence Dashboard</h1>
    <p>Ujian Tengah Semester — Universitas Timor · Fakultas Pertanian Sains & Kesehatan · Prodi Teknologi Informasi</p>
    <span class="tag">TIF46212</span>
    <span class="tag">Semester VII B</span>
    <span class="tag">Business Intelligence</span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA (cached)
# ============================================================
@st.cache_data(show_spinner=False)
def load_uploaded(file_bytes, filename, sheet):
    if filename.lower().endswith('.csv'):
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)


# ============================================================
# FUNGSI TAHAPAN
# ============================================================
def compute_eda(df):
    info = {}
    if 'Transaction_Date' in df.columns:
        df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'], errors='coerce')
        info['min_date'] = df['Transaction_Date'].min()
        info['max_date'] = df['Transaction_Date'].max()
    if 'Transaction_ID' in df.columns:
        info['n_trx'] = df['Transaction_ID'].nunique()
    if 'Customer_ID' in df.columns:
        info['n_cust'] = df['Customer_ID'].nunique()
    if 'Product_ID' in df.columns:
        info['n_prod'] = df['Product_ID'].nunique()
    if {'Quantity', 'Unit_Price'}.issubset(df.columns):
        df['Total_Price'] = df['Quantity'] * df['Unit_Price']
        info['total_sales'] = df['Total_Price'].sum()
    info['missing'] = df.isnull().sum()
    info['duplicates'] = df.duplicated().sum()
    return df, info


def preprocess_data(df):
    """Preprocessing data (TANPA apriori — apriori dijalankan di tab sendiri)."""
    log = []
    df_clean = df.copy()
    n0 = len(df_clean)

    kritikal = [c for c in ['Transaction_ID', 'Customer_ID',
                            'Product_ID', 'Quantity', 'Unit_Price']
                if c in df_clean.columns]
    df_clean = df_clean.dropna(subset=kritikal)
    log.append(("Drop missing (kolom kritis)", n0 - len(df_clean)))

    n0 = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    log.append(("Drop duplikasi", n0 - len(df_clean)))

    if 'Transaction_Date' in df_clean.columns:
        df_clean['Transaction_Date'] = pd.to_datetime(
            df_clean['Transaction_Date'], errors='coerce')
        n0 = len(df_clean)
        df_clean = df_clean.dropna(subset=['Transaction_Date'])
        log.append(("Drop tanggal invalid", n0 - len(df_clean)))

    for col in ['Region', 'Customer_Type', 'Category',
                'Payment_Method', 'Product_Name']:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip().str.title()

    if 'Quantity' in df_clean.columns:
        n0 = len(df_clean)
        df_clean = df_clean[df_clean['Quantity'] > 0]
        log.append(("Drop Quantity <= 0", n0 - len(df_clean)))
    if 'Unit_Price' in df_clean.columns:
        n0 = len(df_clean)
        df_clean = df_clean[df_clean['Unit_Price'] > 0]
        log.append(("Drop Unit_Price <= 0", n0 - len(df_clean)))

    if {'Quantity', 'Unit_Price'}.issubset(df_clean.columns):
        if 'Discount' not in df_clean.columns:
            df_clean['Discount'] = 0
        df_clean['Discount'] = df_clean['Discount'].fillna(0)
        if df_clean['Discount'].max() > 1:
            df_clean['Discount'] /= 100.0
        df_clean['Total_Price'] = (df_clean['Quantity']
                                   * df_clean['Unit_Price']
                                   * (1 - df_clean['Discount']))

    if {'Transaction_ID', 'Product_ID'}.issubset(df_clean.columns):
        bs = (df_clean.groupby('Transaction_ID')['Product_ID']
              .nunique().rename('basket_size').reset_index())
        df_clean = df_clean.merge(bs, on='Transaction_ID', how='left')

    basket = None
    if {'Transaction_ID', 'Product_Name'}.issubset(df_clean.columns):
        basket = (df_clean.groupby(['Transaction_ID', 'Product_Name'])
                  ['Quantity'].sum().unstack().fillna(0))
        # Menggunakan logika boolean, bukan applymap (agar tidak error di pandas versi baru)
        basket = (basket > 0).astype(int)

    rfm = None
    if {'Customer_ID', 'Transaction_Date',
        'Transaction_ID', 'Total_Price'}.issubset(df_clean.columns):
        ref = df_clean['Transaction_Date'].max() + timedelta(days=1)
        rfm = (df_clean.groupby('Customer_ID')
               .agg(Recency=('Transaction_Date', lambda x: (ref - x.max()).days),
                    Frequency=('Transaction_ID', 'nunique'),
                    Monetary=('Total_Price', 'sum'))
               .reset_index())

    return df_clean, log, basket, rfm


def run_apriori(basket, min_support, min_confidence):
    if basket is None or basket.empty:
        return None
    freq = apriori(basket, min_support=min_support, use_colnames=True)
    if freq.empty:
        return None
    rules = association_rules(freq, metric='confidence',
                              min_threshold=min_confidence)
    return rules.sort_values('lift', ascending=False).reset_index(drop=True)


def run_kmeans(rfm, k_min, k_max):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    K_range = list(range(k_min, k_max + 1))
    inertias, silhouettes = [], []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, km.labels_))
    best_k = K_range[int(np.argmax(silhouettes))]
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    rfm_out = rfm.copy()
    rfm_out['Cluster'] = labels

    r_med, f_med, m_med = (rfm_out['Recency'].median(),
                           rfm_out['Frequency'].median(),
                           rfm_out['Monetary'].median())

    def label(c):
        sub = rfm_out[rfm_out['Cluster'] == c]
        r, f, m = (sub['Recency'].mean(),
                   sub['Frequency'].mean(),
                   sub['Monetary'].mean())
        if r < r_med and f > f_med and m > m_med:
            return 'Loyal'
        if r > r_med and f < f_med:
            return 'At Risk'
        if f < f_med and m < m_med:
            return 'New / Potential'
        return 'Regular'

    lmap = {c: label(c) for c in sorted(rfm_out['Cluster'].unique())}
    rfm_out['Segment'] = rfm_out['Cluster'].map(lmap)

    return rfm_out, K_range, inertias, silhouettes, best_k


# ============================================================
# TABS NAVIGASI
# ============================================================
tabs = st.tabs([
    "🏠 Beranda",
    "📊 EDA",
    "🧹 Preprocessing",
    "🛒 Apriori",
    "👥 K-Means",
    "📈 Dashboard BI",
    "💡 Rekomendasi",
])


# ============================================================
# VARIABEL GLOBAL
# ============================================================
df_raw = None
df_clean = None
prep_log = []
basket = None
rfm = None
df_eda = None
eda_info = {}
file_name = None


# ============================================================
# TAB 0 — BERANDA
# ============================================================
with tabs[0]:
    st.markdown(_sec_title("🏛️", "Profil & Identitas"), unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(_info_card(
            "🏛️", "Institusi & Akademik", [
                "<b>Universitas Timor</b>",
                "Kota Kefamenanu, NTT",
                "<hr style='margin: 10px 0; border: 0; border-top: 1px solid #e2e8f0;'>",
                "<b>Fakultas Pertanian Sains & Kesehatan</b>",
                "Program Studi Teknologi Informasi"
            ], 
            color="blue",
            img_b64=img_logo, 
            img_style="width: 90px; margin-bottom: 15px;"
        ), unsafe_allow_html=True)
        
    with c2:
        st.markdown(_info_card(
            "📚", "Mata Kuliah & Dosen", [
                "Mata Kuliah: <b>Business Intelligence</b>",
                "Kode: <b>TIF46212</b>",
                "Semester: <b>VII B</b>",
                "<hr style='margin: 10px 0; border: 0; border-top: 1px solid #e2e8f0;'>",
                "<b>Dosen Pengampu:</b>",
                "Krisantus Jumarto Tey Seran, S.T., M.T."
            ], 
            color="amber"
        ), unsafe_allow_html=True)
        
    with c3:
        st.markdown(_info_card(
            "🧑‍🎓", "Identitas Mahasiswa", [
                "Nama: <b>Elfira Lidya Taolin</b>",
                "NPM: <b>51230014</b>",
                "Kelas: <b>VII B</b>",
                "Prodi: <b>Teknologi Informasi</b>"
            ], 
            color="purple",
            img_b64=img_profil, 
            img_style="width: 100px; height: 100px; object-fit: cover; border-radius: 50%; border: 3px solid #8b5cf6; margin-bottom: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);"
        ), unsafe_allow_html=True)

    st.markdown(_sec_title("📤", "Upload Dataset"), unsafe_allow_html=True)
    st.markdown(_hl(
        "info", "📁", "Format Didukung",
        "Upload file <b>.xlsx</b>, <b>.xls</b>, atau <b>.csv</b> berisi dataset "
        "transaksi penjualan (lihat contoh: <code>P8. UTS_Business Intelligence.xlsx</code>)."
    ), unsafe_allow_html=True)

    up_col1, up_col2 = st.columns([2, 1])
    with up_col1:
        uploaded = st.file_uploader(
            "Pilih file dataset",
            type=['xlsx', 'xls', 'csv'],
            key="main_uploader",
            label_visibility="collapsed"
        )
    with up_col2:
        sheet_name = st.text_input(
            "Nama sheet (untuk .xlsx)",
            value="Data_Raw",
            key="sheet_input"
        )

    if uploaded is not None:
        try:
            df_raw = load_uploaded(uploaded.getvalue(), uploaded.name, sheet_name)
            file_name = uploaded.name
            st.markdown(_hl(
                "success", "✅", "Dataset Berhasil Dimuat",
                f"File <b>{uploaded.name}</b> — "
                f"<b>{_int_fmt(df_raw.shape[0])}</b> baris × "
                f"<b>{_int_fmt(df_raw.shape[1])}</b> kolom. "
                f"Buka tab lain di atas untuk memulai analisis."
            ), unsafe_allow_html=True)

            with st.expander("👀 Preview 20 baris pertama"):
                st.dataframe(_display_df(df_raw.head(20)),
                             use_container_width=True, height=350)
        except Exception as e:
            st.error(f"❌ Gagal memuat data: {e}")


# ============================================================
# JALANKAN PREPROCESSING
# ============================================================
if df_raw is not None:
    df_eda, eda_info = compute_eda(df_raw.copy())
    df_clean, prep_log, basket, rfm = preprocess_data(df_raw.copy())
    st.session_state['df_clean'] = df_clean


def _need_data():
    st.markdown(_hl(
        "warn", "👋", "Silakan Upload Data Dulu",
        "Buka tab <b>🏠 Beranda</b> dan upload file dataset untuk memulai analisis."
    ), unsafe_allow_html=True)


# ============================================================
# TAB 1 — EDA
# ============================================================
with tabs[1]:
    if df_raw is None:
        _need_data()
    else:
        st.markdown(_sec_title("❓", "Pertanyaan Bisnis"), unsafe_allow_html=True)
        q1, q2, q3 = st.columns(3)
        with q1:
            st.markdown(_hl("info", "1️⃣", "Q1",
                            "Produk apa yang sering dibeli bersama?"), unsafe_allow_html=True)
        with q2:
            st.markdown(_hl("info", "2️⃣", "Q2",
                            "Segmen pelanggan mana yang paling bernilai?"), unsafe_allow_html=True)
        with q3:
            st.markdown(_hl("info", "3️⃣", "Q3",
                            "Kapan penjualan meningkat?"), unsafe_allow_html=True)

        st.markdown(_sec_title("📊", "Ringkasan Dataset"), unsafe_allow_html=True)
        st.markdown(_kpi_row([
            {"icon": "📋", "label": "Baris", "value": _int_fmt(df_raw.shape[0]), "color": "blue"},
            {"icon": "📐", "label": "Kolom", "value": _int_fmt(df_raw.shape[1]), "color": "cyan"},
            {"icon": "🧾", "label": "Transaksi Unik",
             "value": _int_fmt(eda_info.get('n_trx', 0)), "color": "green"},
            {"icon": "👥", "label": "Pelanggan Unik",
             "value": _int_fmt(eda_info.get('n_cust', 0)), "color": "purple"},
        ]), unsafe_allow_html=True)

        st.markdown(_kpi_row([
            {"icon": "📦", "label": "Produk Unik",
             "value": _int_fmt(eda_info.get('n_prod', 0)), "color": "amber"},
            {"icon": "🔁", "label": "Duplikasi",
             "value": _int_fmt(eda_info.get('duplicates', 0)), "color": "red"},
            {"icon": "📅", "label": "Rentang Awal",
             "value": str(eda_info.get('min_date', '-'))[:10], "color": "blue"},
            {"icon": "📅", "label": "Rentang Akhir",
             "value": str(eda_info.get('max_date', '-'))[:10], "color": "green"},
        ]), unsafe_allow_html=True)

        if df_clean is not None:
            total_sales = df_clean['Total_Price'].sum() if 'Total_Price' in df_clean.columns else 0
            total_trx = df_clean['Transaction_ID'].nunique() if 'Transaction_ID' in df_clean.columns else 0
            active_cust = df_clean['Customer_ID'].nunique() if 'Customer_ID' in df_clean.columns else 0
            avg_basket = (df_clean.groupby('Transaction_ID')['Product_ID']
                          .nunique().mean()) if {'Transaction_ID','Product_ID'}.issubset(df_clean.columns) else 0
            st.markdown(_sec_title("💰", "KPI Bisnis"), unsafe_allow_html=True)
            st.markdown(_kpi_row([
                {"icon": "💰", "label": "Total Penjualan",
                 "value": rp(total_sales), "color": "green"},
                {"icon": "🧾", "label": "Jumlah Transaksi",
                 "value": _int_fmt(total_trx), "color": "amber"},
                {"icon": "👥", "label": "Pelanggan Aktif",
                 "value": _int_fmt(active_cust), "color": "blue"},
                {"icon": "🛒", "label": "Rata-rata Basket",
                 "value": f"{avg_basket:.2f}", "color": "purple"},
            ]), unsafe_allow_html=True)

        st.markdown(_sec_title("🔍", "Missing Values"), unsafe_allow_html=True)
        mv = eda_info['missing'].reset_index()
        mv.columns = ['Kolom', 'Jumlah Missing']
        mv = mv[mv['Jumlah Missing'] > 0]
        if mv.empty:
            st.markdown(_hl("success", "✅", "Data Bersih",
                            "Tidak ada missing value."), unsafe_allow_html=True)
        else:
            st.dataframe(_display_df(mv, int_cols=["Jumlah Missing"]),
                         use_container_width=True, hide_index=True)

        st.markdown(_sec_title("📐", "Outlier (Metode IQR)"), unsafe_allow_html=True)
        rows = []
        for col in ['Quantity', 'Unit_Price', 'Total_Price']:
            if col in df_eda.columns:
                Q1, Q3 = df_eda[col].quantile(0.25), df_eda[col].quantile(0.75)
                IQR = Q3 - Q1
                lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                n_out = ((df_eda[col] < lo) | (df_eda[col] > hi)).sum()
                rows.append({'Kolom': col, 'Jumlah Outlier': int(n_out)})
        st.dataframe(_display_df(pd.DataFrame(rows), int_cols=["Jumlah Outlier"]),
                     use_container_width=True, hide_index=True)

        st.markdown(_sec_title("📈", "Visualisasi EDA"), unsafe_allow_html=True)
        vc1, vc2 = st.columns(2)
        with vc1:
            if 'Total_Price' in df_eda.columns:
                fig = px.histogram(df_eda, x='Total_Price', nbins=50,
                                   title='Distribusi Total Price',
                                   color_discrete_sequence=['#2563eb'])
                fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                                  font=dict(family="Inter, sans-serif"))
                st.plotly_chart(fig, use_container_width=True)
        with vc2:
            if {'Product_Name', 'Quantity'}.issubset(df_eda.columns):
                top10 = (df_eda.groupby('Product_Name')['Quantity']
                         .sum().nlargest(10).reset_index())
                fig = px.bar(top10, x='Quantity', y='Product_Name',
                             orientation='h', title='Top 10 Produk',
                             color='Quantity', color_continuous_scale='Blues')
                fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                                  font=dict(family="Inter, sans-serif"),
                                  yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        st.markdown(_sec_title("💡", "3 Insight Awal"), unsafe_allow_html=True)
        st.markdown(_hl("success", "1️⃣", "Insight 1",
                        "Distribusi total price condong ke transaksi bernilai kecil."), unsafe_allow_html=True)
        st.markdown(_hl("success", "2️⃣", "Insight 2",
                        "Top 10 produk didominasi kategori Sembako dan Makanan."), unsafe_allow_html=True)
        st.markdown(_hl("success", "3️⃣", "Insight 3",
                        "Terdapat outlier pada Quantity & Unit_Price, perlu ditangani."), unsafe_allow_html=True)


# ============================================================
# TAB 2 — PREPROCESSING
# ============================================================
with tabs[2]:
    if df_raw is None:
        _need_data()
    else:
        st.markdown(_sec_title("🧹", "Laporan Preprocessing"), unsafe_allow_html=True)

        log_df = pd.DataFrame(prep_log, columns=['Tahap', 'Baris Dihapus'])
        st.dataframe(_display_df(log_df, int_cols=["Baris Dihapus"]),
                     use_container_width=True, hide_index=True)

        st.markdown(_kpi_row([
            {"icon": "📥", "label": "Baris Sebelum",
             "value": _int_fmt(df_raw.shape[0]), "color": "blue"},
            {"icon": "✅", "label": "Baris Sesudah",
             "value": _int_fmt(df_clean.shape[0]), "color": "green"},
            {"icon": "🗑️", "label": "Total Dihapus",
             "value": _int_fmt(df_raw.shape[0] - df_clean.shape[0]), "color": "red"},
            {"icon": "📊", "label": "Persentase Bersih",
             "value": f"{(df_clean.shape[0] / max(df_raw.shape[0], 1) * 100):.1f}%",
             "color": "amber"},
        ]), unsafe_allow_html=True)

        st.markdown(_sec_title("👀", "Preview Data Bersih (Semua Baris)"), unsafe_allow_html=True)
        st.dataframe(_display_df(df_clean), use_container_width=True, height=600)

        st.markdown(_sec_title("⬇️", "Unduh Data Bersih"), unsafe_allow_html=True)
        csv_bytes = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Data_Clean.csv",
                           data=csv_bytes,
                           file_name='Data_Clean.csv',
                           mime='text/csv')


# ============================================================
# TAB 3 — APRIORI (FIXED RENDERING)
# ============================================================
with tabs[3]:
    if basket is None:
        if df_raw is None:
            _need_data()
        else:
            st.markdown(_hl("warn", "⚠️", "Basket Tidak Tersedia",
                            "Kolom <b>Transaction_ID</b> dan <b>Product_Name</b> "
                            "tidak ditemukan pada dataset."), unsafe_allow_html=True)
    else:
        # --- MATRIKS TRANSAKSI (Lebih Rapi) ---
        st.markdown(_sec_title("🛒", "Matriks Transaksi (1/0)"), unsafe_allow_html=True)
        
        col_info, col_metric = st.columns([3, 1])
        with col_info:
            st.markdown(_hl(
                "info", "🧮", "Bagaimana Matriks Dibentuk?",
                "Matriks dibentuk murni dari <code>Transaction_ID</code> unik. Jika satu transaksi membeli beberapa barang, "
                "barang-barang tersebut berada dalam <b>satu baris transaksi yang sama</b> (bukan baris terpisah). "
                "Nilai <b>1</b> menandakan produk dibeli, <b>0</b> menandakan tidak dibeli."
            ), unsafe_allow_html=True)
        with col_metric:
            st.metric("Dimensi Matriks", f"{basket.shape[0]} x {basket.shape[1]}")
            st.caption("Baris: Transaksi | Kolom: Produk")

        with st.expander("👀 Preview Matriks Transaksi-Produk (5 Baris Pertama)", expanded=True):
            st.dataframe(basket.head(), use_container_width=True)

        st.markdown(_sec_title("⚙️", "Parameter Apriori"), unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            min_support = st.slider(
                "Minimum Support", 0.005, 0.20, 0.01, 0.005,
                key="apriori_min_support",
                help="Semakin kecil → semakin banyak aturan yang dihasilkan."
            )
        with p2:
            min_confidence = st.slider(
                "Minimum Confidence", 0.05, 0.90, 0.20, 0.05,
                key="apriori_min_confidence",
                help="Semakin besar → aturan makin kuat, jumlah berkurang."
            )

        with st.spinner("Menjalankan Apriori..."):
            rules = run_apriori(basket, min_support, min_confidence)

        st.markdown(_sec_title("🛒", "Hasil Market Basket Analysis"), unsafe_allow_html=True)

        if rules is None or len(rules) == 0:
            st.markdown(_hl("warn", "⚠️", "Tidak Ada Aturan",
                            "Tidak ada aturan asosiasi yang dihasilkan. "
                            "Coba turunkan minimum support / confidence."),
                        unsafe_allow_html=True)
        else:
            st.markdown(_kpi_row([
                {"icon": "🔗", "label": "Jumlah Rules",
                 "value": _int_fmt(len(rules)), "color": "green"},
                {"icon": "📈", "label": "Max Lift",
                 "value": f"{rules['lift'].max():.2f}", "color": "blue"},
                {"icon": "🎯", "label": "Max Confidence",
                 "value": f"{rules['confidence'].max():.2f}", "color": "amber"},
                {"icon": "🛒", "label": "Jumlah Produk",
                 "value": _int_fmt(basket.shape[1]), "color": "purple"},
            ]), unsafe_allow_html=True)

            top10 = rules.head(10).copy()
            top10['antecedents'] = top10['antecedents'].apply(lambda x: ', '.join(list(x)))
            top10['consequents'] = top10['consequents'].apply(lambda x: ', '.join(list(x)))

            st.markdown(_sec_title("🏆", "Top 10 Aturan Asosiasi"), unsafe_allow_html=True)
            top10_display = top10[['antecedents', 'consequents', 'support',
                                    'confidence', 'lift']].copy()
            top10_display['support'] = top10_display['support'].apply(lambda x: f"{x:.4f}")
            top10_display['confidence'] = top10_display['confidence'].apply(lambda x: f"{x:.4f}")
            top10_display['lift'] = top10_display['lift'].apply(lambda x: f"{x:.4f}")
            st.dataframe(top10_display, use_container_width=True, hide_index=True)

            # --- CONFUSION MATRIX VISUAL (Native Streamlit, Bebas Error) ---
            st.markdown(_sec_title("🎯", "Evaluasi Rule Terbaik (Confusion Matrix)"), unsafe_allow_html=True)
            
            best_rule = rules.iloc[0]
            ant_items = list(best_rule['antecedents'])
            cons_items = list(best_rule['consequents'])
            
            ant_prod = ant_items[0]
            cons_prod = cons_items[0]
            
            # Menghitung TP, FP, FN, TN
            tp = ((basket[ant_prod] == 1) & (basket[cons_prod] == 1)).sum()
            fp = ((basket[ant_prod] == 1) & (basket[cons_prod] == 0)).sum()
            fn = ((basket[ant_prod] == 0) & (basket[cons_prod] == 1)).sum()
            tn = ((basket[ant_prod] == 0) & (basket[cons_prod] == 0)).sum()
            
            st.markdown(f"""
            <div style="background: white; padding: 15px; border-radius: 12px; border: 1px solid #e8ecf0; margin-bottom: 10px;">
                <p style="margin:0; font-size: 14px; color:#334155;">
                    Mengevaluasi aturan: <b>Jika beli {ant_prod}, maka beli {cons_prod}</b>.<br>
                    <span style="font-size: 12px; color: #64748b;">(Prediksi Positif = Membeli {ant_prod} | Aktual Positif = Membeli {cons_prod})</span>
                </p>
            </div>
            """.replace("\n", ""), unsafe_allow_html=True)
            
            # Baris Header
            h1, h2, h3 = st.columns([1, 2, 2])
            with h1:
                pass
            with h2:
                st.markdown("<div style='text-align:center; font-weight:bold; padding-bottom:10px;'>Aktual: Positif (1)</div>", unsafe_allow_html=True)
            with h3:
                st.markdown("<div style='text-align:center; font-weight:bold; padding-bottom:10px;'>Aktual: Negatif (0)</div>", unsafe_allow_html=True)

            # Baris 1: Prediksi Positif
            r1_c1, r1_c2, r1_c3 = st.columns([1, 2, 2])
            with r1_c1:
                st.markdown("<div style='text-align:right; font-weight:bold; padding-top:20px; padding-right:10px;'>Prediksi:<br>Positif (1)</div>", unsafe_allow_html=True)
            with r1_c2:
                st.markdown(f"<div style='background:#eff6ff; padding:15px; border:1px solid #bfdbfe; border-radius:10px;'><b>True Positive (TP)</b><br><span style='font-size:12px; color:#64748b;'>(Model memprediksi Positif, dan kenyataannya Benar Positif)</span><h2 style='color:#2563eb; margin:5px 0;'>{tp}</h2></div>", unsafe_allow_html=True)
            with r1_c3:
                st.markdown(f"<div style='background:#fef2f2; padding:15px; border:1px solid #fecaca; border-radius:10px;'><b>False Positive (FP)</b><br><span style='font-size:12px; color:#64748b;'>(Model memprediksi Positif, padahal kenyataannya Negatif)</span><h2 style='color:#ef4444; margin:5px 0;'>{fp}</h2></div>", unsafe_allow_html=True)

            st.write("") # Spacer

            # Baris 2: Prediksi Negatif
            r2_c1, r2_c2, r2_c3 = st.columns([1, 2, 2])
            with r2_c1:
                st.markdown("<div style='text-align:right; font-weight:bold; padding-top:20px; padding-right:10px;'>Prediksi:<br>Negatif (0)</div>", unsafe_allow_html=True)
            with r2_c2:
                st.markdown(f"<div style='background:#fef2f2; padding:15px; border:1px solid #fecaca; border-radius:10px;'><b>False Negative (FN)</b><br><span style='font-size:12px; color:#64748b;'>(Model memprediksi Negatif, padahal kenyataannya Positif)</span><h2 style='color:#ef4444; margin:5px 0;'>{fn}</h2></div>", unsafe_allow_html=True)
            with r2_c3:
                st.markdown(f"<div style='background:#eff6ff; padding:15px; border:1px solid #bfdbfe; border-radius:10px;'><b>True Negative (TN)</b><br><span style='font-size:12px; color:#64748b;'>(Model memprediksi Negatif, dan kenyataannya Benar Negatif)</span><h2 style='color:#2563eb; margin:5px 0;'>{tn}</h2></div>", unsafe_allow_html=True)

            st.write("") # Spacer

            # Grafik Batang pendamping Confusion Matrix
            fig_cm = go.Figure(data=[
                go.Bar(name=f'Beli {cons_prod}', x=[f'Beli {ant_prod}', f'Tidak Beli {ant_prod}'], y=[tp, fn], marker_color='#3b82f6'),
                go.Bar(name=f'Tidak Beli {cons_prod}', x=[f'Beli {ant_prod}', f'Tidak Beli {ant_prod}'], y=[fp, tn], marker_color='#ef4444')
            ])
            fig_cm.update_layout(
                barmode='group', title='Grafik Evaluasi Rule',
                plot_bgcolor='white', paper_bgcolor='white',
                height=300, margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # --- REKOMENDASI CROSS-SELLING & DENAH RAK (Native Streamlit) ---
            st.markdown(_sec_title("🏪", "Rekomendasi Cross-Selling & Tata Letak Rak"), unsafe_allow_html=True)
            top3 = rules.head(3)
            
            st.markdown("### 📋 3 Rekomendasi Utama")
            for i, (idx, row) in enumerate(top3.iterrows(), 1):
                a_prod = list(row['antecedents'])[0]
                c_prod = list(row['consequents'])[0]
                st.markdown(_hl("info", f"0{i}", f"Bundling {a_prod} + {c_prod}", 
                    f"Pelanggan yang membeli <b>{a_prod}</b> memiliki kecenderungan <b>{row['confidence']*100:.1f}%</b> untuk membeli <b>{c_prod}</b> (Lift: {row['lift']:.2f}). Disarankan membuat promo bundling atau menempatkan produk berdekatan."), unsafe_allow_html=True)

            st.markdown("### 🗺️ Saran Denah / Susunan Rak Toko")
            st.markdown("Berdasarkan aturan asosiasi teratas, berikut adalah usulan tata letak rak fisik (kiri ke kanan):")
            
            for i, (idx, row) in enumerate(top3.iterrows(), 1):
                a_prod = list(row['antecedents'])[0]
                c_prod = list(row['consequents'])[0]
                
                st.markdown(f"**Rak {i}**")
                sh_c1, sh_c2, sh_c3 = st.columns([3, 1, 3])
                with sh_c1:
                    st.markdown(f"<div style='background:#dbeafe; color:#1e3a8a; padding:15px; border-radius:10px; text-align:center; font-weight:bold; border:1px solid #93c5fd;'>{a_prod}</div>", unsafe_allow_html=True)
                with sh_c2:
                    st.markdown("<div style='text-align:center; font-size:24px; padding-top:5px; color:#f59e0b;'>➡️</div>", unsafe_allow_html=True)
                with sh_c3:
                    st.markdown(f"<div style='background:#fef3c7; color:#92400e; padding:15px; border-radius:10px; text-align:center; font-weight:bold; border:1px solid #fcd34d;'>{c_prod}</div>", unsafe_allow_html=True)
                st.write("") # Spacer

            st.caption("💡 **Catatan:** Tempatkan produk 'Antecedent' (biru) tepat di sebelah kiri produk 'Consequent' (kuning). Ini memancing pelanggan yang mengambil produk pertama untuk melihat dan mengambil produk kedua (Cross-selling).")

            st.markdown(_sec_title("📊", "Sebaran Support vs Confidence"), unsafe_allow_html=True)
            fig2 = px.scatter(rules, x='support', y='confidence',
                              size='lift', color='lift',
                              color_continuous_scale='Viridis',
                              title='Sebaran Support vs Confidence (ukuran = lift)')
            fig2.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                               font=dict(family="Inter, sans-serif"))
            st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# TAB 4 — K-MEANS
# ============================================================
with tabs[4]:
    if rfm is None:
        if df_raw is None:
            _need_data()
        else:
            st.markdown(_hl("warn", "⚠️", "RFM Tidak Tersedia",
                            "Periksa kolom <b>Customer_ID</b>, "
                            "<b>Transaction_Date</b>, dan <b>Total_Price</b>."),
                        unsafe_allow_html=True)
    else:
        st.markdown(_sec_title("⚙️", "Parameter K-Means"), unsafe_allow_html=True)
        k_range_sel = st.slider(
            "Rentang nilai k (jumlah cluster yang diuji)", 2, 12, (2, 8),
            key="kmeans_k_range",
            help="Aplikasi akan otomatis memilih k terbaik berdasarkan Silhouette Score."
        )
        k_min, k_max = k_range_sel

        with st.spinner("Menjalankan K-Means..."):
            rfm_seg, K_range, inertias, silhouettes, best_k = run_kmeans(rfm, k_min, k_max)

        st.markdown(_hl("success", "🎯", f"k Optimal = {best_k}",
                        f"Berdasarkan <b>Silhouette Score</b>, jumlah cluster "
                        f"terbaik adalah <b>k = {best_k}</b>."),
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=K_range, y=inertias,
                                     mode='lines+markers',
                                     line=dict(color='#2563eb', width=3),
                                     marker=dict(size=10)))
            fig.update_layout(title='Elbow Method',
                              xaxis_title='k', yaxis_title='Inertia',
                              plot_bgcolor='white', paper_bgcolor='white',
                              font=dict(family="Inter, sans-serif"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=K_range, y=silhouettes,
                                     mode='lines+markers',
                                     line=dict(color='#f59e0b', width=3),
                                     marker=dict(size=10)))
            fig.update_layout(title='Silhouette Score',
                              xaxis_title='k', yaxis_title='Score',
                              plot_bgcolor='white', paper_bgcolor='white',
                              font=dict(family="Inter, sans-serif"))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(_sec_title("📋", "Profil Setiap Klaster"), unsafe_allow_html=True)
        profile = (rfm_seg.groupby('Cluster')
                   .agg(Jumlah_Pelanggan=('Customer_ID', 'count'),
                        Recency_Avg=('Recency', 'mean'),
                        Frequency_Avg=('Frequency', 'mean'),
                        Monetary_Avg=('Monetary', 'mean'))
                   .reset_index())
        profile['Segment'] = profile['Cluster'].map(
            rfm_seg.drop_duplicates('Cluster')
                   .set_index('Cluster')['Segment'])

        profile_display = profile.copy()
        profile_display['Jumlah_Pelanggan'] = profile_display['Jumlah_Pelanggan'].apply(_int_fmt)
        profile_display['Recency_Avg'] = profile_display['Recency_Avg'].apply(lambda x: f"{x:.2f}")
        profile_display['Frequency_Avg'] = profile_display['Frequency_Avg'].apply(lambda x: f"{x:.2f}")
        profile_display['Monetary_Avg'] = profile_display['Monetary_Avg'].apply(rp)
        st.dataframe(profile_display, use_container_width=True, hide_index=True)

        st.markdown(_sec_title("📊", "Scatter Plot Segmentasi"), unsafe_allow_html=True)
        fig = px.scatter(rfm_seg, x='Recency', y='Monetary',
                         color='Segment', size='Frequency',
                         hover_data=['Customer_ID', 'Frequency'],
                         color_discrete_sequence=px.colors.qualitative.Set2,
                         title='Recency vs Monetary per Segmen')
        fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                          font=dict(family="Inter, sans-serif"))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(_sec_title("🥧", "Jumlah Pelanggan per Segmen"), unsafe_allow_html=True)
        seg_counts = (rfm_seg.groupby('Segment')['Customer_ID']
                      .count().reset_index(name='Jumlah'))
        fig = px.pie(seg_counts, names='Segment', values='Jumlah',
                     hole=0.5, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 5 — DASHBOARD BI
# ============================================================
with tabs[5]:
    if df_clean is None:
        _need_data()
    else:
        st.markdown(_sec_title("📊", "Dashboard Business Intelligence"), unsafe_allow_html=True)

        df_clean['Month'] = df_clean['Transaction_Date'].dt.to_period('M').astype(str)
        monthly = df_clean.groupby('Month')['Total_Price'].sum().reset_index()

        d1, d2 = st.columns((3, 2))
        with d1:
            fig = px.line(monthly, x='Month', y='Total_Price', markers=True,
                          title='Tren Penjualan Bulanan',
                          color_discrete_sequence=['#2563eb'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                              font=dict(family="Inter, sans-serif"))
            st.plotly_chart(fig, use_container_width=True)
        with d2:
            top_prod = (df_clean.groupby('Product_Name')['Quantity']
                        .sum().nlargest(10).reset_index())
            fig = px.bar(top_prod, x='Quantity', y='Product_Name',
                         orientation='h', title='Top 10 Produk',
                         color='Quantity', color_continuous_scale='Blues')
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                              font=dict(family="Inter, sans-serif"),
                              yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        d3, d4 = st.columns(2)
        with d3:
            if 'Category' in df_clean.columns:
                cat = df_clean.groupby('Category')['Total_Price'].sum().reset_index()
                fig = px.pie(cat, names='Category', values='Total_Price',
                             title='Proporsi Penjualan per Kategori', hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
        with d4:
            if 'Payment_Method' in df_clean.columns:
                pay = (df_clean.groupby('Payment_Method')['Transaction_ID']
                       .nunique().reset_index(name='Jumlah'))
                fig = px.bar(pay, x='Payment_Method', y='Jumlah',
                             title='Transaksi per Metode Pembayaran',
                             color='Jumlah', color_continuous_scale='Blues')
                fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                                  font=dict(family="Inter, sans-serif"))
                st.plotly_chart(fig, use_container_width=True)

        if 'Region' in df_clean.columns:
            reg = (df_clean.groupby('Region')['Total_Price'].sum()
                   .reset_index().sort_values('Total_Price', ascending=False))
            fig = px.bar(reg, x='Region', y='Total_Price',
                         title='Total Penjualan per Region',
                         color='Total_Price', color_continuous_scale='Blues')
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white',
                              font=dict(family="Inter, sans-serif"))
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 6 — REKOMENDASI
# ============================================================
with tabs[6]:
    st.markdown(_sec_title("💡", "Rekomendasi Keputusan"), unsafe_allow_html=True)

    with st.expander("🎯 Rekomendasi 1 — Penempatan / Bundling Produk", expanded=True):
        st.markdown("""
        - **Bukti:** Aturan asosiasi menunjukkan produk yang sering dibeli bersama.
        - **Tindakan:** Tempatkan produk berdekatan atau buat paket bundling (lihat visualisasi **denah rak** di tab 🛒 Apriori).
        - **KPI:** Kenaikan penjualan bundling sebesar **10%**.
        - **Estimasi Biaya:** Rp 500.000 untuk promosi.
        - **Asumsi:** Pelanggan tertarik pada paket hemat.
        - **Evaluasi:** Bandingkan penjualan sebelum vs sesudah 1 bulan.
        """)

    with st.expander("🔄 Rekomendasi 2 — Retensi Pelanggan"):
        st.markdown("""
        - **Bukti:** Segmen *At Risk* memiliki Recency tinggi & Frequency rendah.
        - **Tindakan:** Kirim voucher diskon untuk pelanggan tidak aktif > 60 hari.
        - **KPI:** Retensi naik **15%**.
        - **Estimasi Biaya:** Rp 1.000.000 untuk voucher.
        - **Asumsi:** Voucher menarik pelanggan kembali.
        - **Evaluasi:** Lacak jumlah pelanggan yang kembali bertransaksi.
        """)

    with st.expander("📣 Rekomendasi 3 — Tindakan Pemasaran"):
        st.markdown("""
        - **Bukti:** Segmen *Loyal* memiliki Monetary tinggi.
        - **Tindakan:** Program loyalitas (poin / cashback).
        - **KPI:** AOV (Average Order Value) naik **5%**.
        - **Estimasi Biaya:** Rp 2.000.000 untuk program loyalitas.
        - **Asumsi:** Pelanggan loyal meningkatkan belanja.
        - **Evaluasi:** Ukur AOV sebelum dan sesudah program.
        """)

    st.markdown(_hl("warn", "⚠️", "Catatan Penting",
                    "Pisahkan temuan faktual, estimasi berbasis asumsi, "
                    "dan dampak aktual yang sudah diuji. Jangan menyatakan "
                    "kenaikan penjualan sebagai fakta sebelum diuji."),
                unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="app-footer">
    📊 <strong>UTS Business Intelligence</strong> &nbsp;·&nbsp;
    Universitas Timor &nbsp;·&nbsp; Fakultas Pertanian Sains & Kesehatan &nbsp;·&nbsp; Prodi Teknologi Informasi
</div>
""", unsafe_allow_html=True)