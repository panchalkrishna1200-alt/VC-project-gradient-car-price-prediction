import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings("ignore")

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# GLOBAL CSS
# ================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Fira+Code:wght@400;500;600&family=Playfair+Display:wght@700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
    min-height: 100vh;
}
.main .block-container {
    padding: 2.5rem 2.5rem 4rem;
    max-width: 1280px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #111827 100%) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label {
    color: #94a3b8 !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.72rem !important;
}

/* ── Hero banner ── */
.hero-wrap {
    background: linear-gradient(120deg, #0f2027, #1a1a2e, #16213e);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 20px;
    padding: 3rem 3rem 2.5rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-wrap::after {
    content: '';
    position: absolute; bottom: -40px; left: 60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-family: 'Fira Code', monospace;
    font-size: 0.7rem; color: #38bdf8;
    letter-spacing: 0.2em; text-transform: uppercase;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    display: inline-block; padding: 4px 16px;
    border-radius: 30px; margin-bottom: 16px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(135deg, #f0f9ff 20%, #38bdf8 55%, #10b981 90%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin-bottom: 12px;
}
.hero-sub {
    font-family: 'Fira Code', monospace;
    font-size: 0.78rem; color: #64748b;
    letter-spacing: 0.06em;
}

/* ── Section headers ── */
.section-wrap {
    margin: 2.5rem 0 1rem;
}
.section-label {
    font-family: 'Fira Code', monospace;
    font-size: 0.62rem; color: #10b981;
    background: rgba(16,185,129,0.1);
    border: 1px solid rgba(16,185,129,0.2);
    padding: 3px 14px; border-radius: 20px;
    letter-spacing: 0.15em; text-transform: uppercase;
    display: inline-block; margin-bottom: 6px;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; font-weight: 700; color: #f0f9ff;
    margin-bottom: 0.2rem;
}
.section-divider {
    width: 48px; height: 3px;
    background: linear-gradient(90deg, #38bdf8, #10b981);
    border-radius: 2px; margin-bottom: 1.2rem;
}

/* ── Upload gate ── */
.upload-gate {
    background: linear-gradient(135deg, #0f1923, #111827);
    border: 2px dashed rgba(56,189,248,0.3);
    border-radius: 20px;
    padding: 4rem 2rem; text-align: center;
    margin: 1rem 0 2rem;
}
.upload-gate-icon { font-size: 3.5rem; margin-bottom: 16px; }
.upload-gate-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem; font-weight: 700; color: #f0f9ff;
    margin-bottom: 8px;
}
.upload-gate-sub {
    font-family: 'Fira Code', monospace;
    font-size: 0.72rem; color: #64748b; margin-bottom: 1.4rem;
}
.col-pill {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    color: #38bdf8;
    font-family: 'Fira Code', monospace; font-size: 0.62rem;
    padding: 3px 12px; border-radius: 20px; margin: 3px;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin: 1.2rem 0; }
.metric-card {
    flex: 1;
    background: linear-gradient(135deg, #111827, #0f1923);
    border: 1px solid rgba(255,255,255,0.06);
    border-top: 2px solid var(--accent, #38bdf8);
    border-radius: 14px;
    padding: 1.2rem 1rem; text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem; font-weight: 700; color: #f0f9ff;
}
.metric-lbl {
    font-family: 'Fira Code', monospace;
    font-size: 0.6rem; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.12em; margin-top: 4px;
}

/* ── Math formula box ── */
.math-card {
    background: linear-gradient(135deg, #0d1117, #111827);
    border-left: 3px solid #38bdf8;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-family: 'Fira Code', monospace;
    font-size: 0.8rem; color: #93c5fd;
    line-height: 2.2; margin: 1rem 0;
}

/* ── Prediction result ── */
.pred-result {
    background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(56,189,248,0.05));
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 18px; padding: 2.5rem 2rem;
    text-align: center; margin-top: 1.5rem;
}
.pred-amount {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem; font-weight: 800; color: #6ee7b7;
}
.pred-lakh {
    font-family: 'Fira Code', monospace;
    color: #10b981; font-size: 0.85rem; margin-top: 4px;
}
.pred-range {
    display: inline-block;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 8px; padding: 8px 20px;
    font-family: 'Fira Code', monospace;
    font-size: 0.72rem; color: #94a3b8; margin-top: 12px;
}

/* ── Info / success override ── */
.stSuccess { background: rgba(16,185,129,0.1) !important; border-color: rgba(16,185,129,0.35) !important; }
.stInfo    { background: rgba(56,189,248,0.08) !important; border-color: rgba(56,189,248,0.25) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    color: #f0f9ff !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; font-size: 1rem !important;
    padding: 0.7rem 2rem !important; width: 100% !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7, #38bdf8) !important;
    box-shadow: 0 6px 28px rgba(56,189,248,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden; }

/* ── Selectbox / Slider labels ── */
label {
    color: #94a3b8 !important;
    font-family: 'Fira Code', monospace !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

/* ── Footer ── */
.footer-bar {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin-top: 3rem;
    font-family: 'Fira Code', monospace;
    font-size: 0.65rem; color: #334155;
    letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)


# ================================================================
# MATPLOTLIB DARK THEME
# ================================================================
def dark_fig(ncols=1, nrows=1, figsize=None):
    if figsize is None:
        figsize = (6 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='#0d1117')
    axlist = axes.flat if hasattr(axes, 'flat') else ([axes] if ncols * nrows == 1 else [axes])
    for ax in axlist:
        ax.set_facecolor('#111827')
        ax.tick_params(colors='#64748b', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.title.set_color('#f0f9ff')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
    return fig, axes

CMAP_TEAL = LinearSegmentedColormap.from_list("teal",  ["#0d1117","#0369a1","#38bdf8","#bae6fd"])
CMAP_CORR = LinearSegmentedColormap.from_list("corr",  ["#ef4444","#111827","#38bdf8"])
CMAP_LOSS = LinearSegmentedColormap.from_list("loss",  ["#1e3a5f","#0ea5e9","#34d399","#fbbf24","#ef4444"])


# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("""
    <div style='font-family:Playfair Display,serif;font-size:1.2rem;font-weight:700;
         color:#38bdf8;margin-bottom:0.3rem;'>🚗 Car Price Predictor</div>
    <div style='font-family:Fira Code,monospace;font-size:0.65rem;color:#334155;
         margin-bottom:1.5rem;'>4th Sem · Vector Calculus Application</div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Load Dataset")
    uploaded_file = st.file_uploader(
        "Upload CSV file", type=["csv"],
        help="CSV must have: name, company, year, Price, kms_driven, fuel_type"
    )
    st.markdown("**Or load from URL**")
    csv_url = st.text_input("CSV URL", placeholder="https://example.com/cars.csv",
                            label_visibility="collapsed")

    st.markdown("""
    <div style='font-family:Fira Code,monospace;font-size:0.62rem;color:#334155;
         background:rgba(56,189,248,0.05);border:1px solid rgba(56,189,248,0.1);
         border-radius:8px;padding:10px 12px;margin-top:8px;'>
    💡 Or place <b style='color:#38bdf8'>car.csv</b> in the same folder as this script.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("📋 Required Columns"):
        st.markdown("""
        <div style='font-family:Fira Code,monospace;font-size:0.68rem;color:#64748b;line-height:2;'>
        · <span style='color:#38bdf8'>name</span> — Car model<br>
        · <span style='color:#38bdf8'>company</span> — Manufacturer<br>
        · <span style='color:#38bdf8'>year</span> — Year of manufacture<br>
        · <span style='color:#38bdf8'>Price</span> — Price in ₹<br>
        · <span style='color:#38bdf8'>kms_driven</span> — KM driven<br>
        · <span style='color:#38bdf8'>fuel_type</span> — Petrol/Diesel/CNG
        </div>
        """, unsafe_allow_html=True)


# ================================================================
# HERO
# ================================================================
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">4th Semester · Vector Calculus Application</div>
  <div class="hero-title">🚗 Car Price Prediction</div>
  <div class="hero-sub">
    ∇ Gradient Descent &nbsp;·&nbsp; 📊 Data Visualization &nbsp;·&nbsp; 🔮 ML Price Prediction
  </div>
</div>
""", unsafe_allow_html=True)


# ================================================================
# STEP 1 — LOAD DATA
# ================================================================
st.markdown("""
<div class="section-wrap">
  <div class="section-label">Step 01</div>
  <div class="section-title">📂 Load Dataset</div>
  <div class="section-divider"></div>
</div>
""", unsafe_allow_html=True)

# Show upload gate if nothing provided yet
if uploaded_file is None and not csv_url.strip():
    st.markdown("""
    <div class="upload-gate">
      <div class="upload-gate-icon">📁</div>
      <div class="upload-gate-title">No Dataset Loaded</div>
      <div class="upload-gate-sub">
        Upload a CSV from the sidebar, paste a URL, or place <b>car.csv</b> in the app folder.<br>
        The dashboard will appear automatically once a file is detected.
      </div>
      <div>
        <span class="col-pill">name</span>
        <span class="col-pill">company</span>
        <span class="col-pill">year</span>
        <span class="col-pill">Price</span>
        <span class="col-pill">kms_driven</span>
        <span class="col-pill">fuel_type</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def load_upload(b):
    import io; return pd.read_csv(io.BytesIO(b))

@st.cache_data
def load_url(u):
    return pd.read_csv(u)

@st.cache_data
def load_local():
    return pd.read_csv("car.csv")


df_raw = None
src_label = ""

if uploaded_file is not None:
    try:
        df_raw = load_upload(uploaded_file.getvalue())
        src_label = f"📤 **{uploaded_file.name}**"
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")

elif csv_url.strip():
    try:
        with st.spinner("🌐 Fetching from URL…"):
            df_raw = load_url(csv_url.strip())
        src_label = f"🌐 `{csv_url.strip()}`"
    except Exception as e:
        st.error(f"❌ Could not load URL: {e}")

else:
    try:
        df_raw = load_local()
        src_label = "📁 **car.csv** (local)"
    except FileNotFoundError:
        pass

if df_raw is None:
    st.warning("⚠️ Waiting for a dataset — use the sidebar to upload or enter a URL.")
    st.stop()

df_raw.columns = df_raw.columns.str.strip()
st.success(f"✅ Loaded {src_label} — **{df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols**")

p1, p2 = st.columns([2, 1])
with p1:
    st.write("**Preview**")
    st.dataframe(df_raw.head(8), use_container_width=True)
with p2:
    st.write("**Column Info**")
    st.dataframe(pd.DataFrame({
        'Column': df_raw.columns,
        'Type':   df_raw.dtypes.astype(str).values,
        'Nulls':  df_raw.isnull().sum().values
    }), use_container_width=True, hide_index=True)


# ================================================================
# STEP 2 — DATA CLEANING
# ================================================================
st.markdown("""
<div class="section-wrap">
  <div class="section-label">Step 02</div>
  <div class="section-title">🧹 Data Cleaning</div>
  <div class="section-divider"></div>
</div>
""", unsafe_allow_html=True)


@st.cache_data
def clean(raw_bytes):
    import io
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.columns = df.columns.str.strip()
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    df = df[df['Price'].astype(str).str.lower() != 'ask for price']
    df['Price'] = df['Price'].astype(str).str.replace(',', '', regex=False).astype(int)
    df['kms_driven'] = (df['kms_driven'].astype(str)
                        .str.replace('kms', '', case=False, regex=False)
                        .str.replace(',', '', regex=False).str.strip())
    df['kms_driven'] = pd.to_numeric(df['kms_driven'], errors='coerce')
    df['kms_driven'] = df['kms_driven'].fillna(df['kms_driven'].median()).astype(int)
    df['fuel_type'] = df['fuel_type'].fillna(df['fuel_type'].mode()[0])
    df['name'] = df['name'].apply(lambda x: ' '.join(str(x).split()[:3]))
    mp = df.loc[df['Price'] <= 5_000_000, 'Price'].mean()
    df.loc[df['Price'] > 5_000_000, 'Price'] = int(mp)
    df['Price'] = df['Price'].astype(int)
    return df


if uploaded_file is not None:
    df = clean(uploaded_file.getvalue())
else:
    df = clean(df_raw.to_csv(index=False).encode())

# Stat cards
st.markdown(f"""
<div class="metric-row">
  <div class="metric-card" style="--accent:#38bdf8">
    <div class="metric-num">{len(df):,}</div>
    <div class="metric-lbl">Clean Records</div>
  </div>
  <div class="metric-card" style="--accent:#10b981">
    <div class="metric-num">{df['company'].nunique()}</div>
    <div class="metric-lbl">Car Brands</div>
  </div>
  <div class="metric-card" style="--accent:#f59e0b">
    <div class="metric-num">{df['year'].min()}–{df['year'].max()}</div>
    <div class="metric-lbl">Year Range</div>
  </div>
  <div class="metric-card" style="--accent:#e05c8a">
    <div class="metric-num">₹{df['Price'].median()/1e5:.1f}L</div>
    <div class="metric-lbl">Median Price</div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("🔍 View Cleaned Dataset"):
    st.dataframe(df, use_container_width=True)


# ================================================================
# STEP 3 — DATA VISUALIZATION
# ================================================================
st.markdown("""
<div class="section-wrap">
  <div class="section-label">Step 03</div>
  <div class="section-title">📊 Data Visualization</div>
  <div class="section-divider"></div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.write("**Price Distribution**")
    fig, ax = dark_fig(figsize=(6, 4))
    sns.histplot(df['Price'], kde=True, color='#38bdf8', ax=ax, alpha=0.65, edgecolor='none')
    if ax.lines:
        ax.lines[0].set_color('#10b981')
        ax.lines[0].set_linewidth(2)
    ax.set_xlabel("Price (₹)"); ax.set_ylabel("Count")
    ax.set_title("Price Distribution")
    plt.tight_layout(); st.pyplot(fig); plt.close()

with col2:
    st.write("**Feature Correlation Heatmap**")
    corr = df.select_dtypes(include=['number']).corr(numeric_only=True)
    fig, ax = dark_fig(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap=CMAP_CORR, ax=ax,
                cbar_kws={'shrink': 0.8}, linewidths=0.5,
                linecolor='#1e293b', annot_kws={'color': '#e2e8f0', 'size': 10})
    ax.set_title("Correlation Matrix")
    ax.tick_params(colors='#94a3b8')
    plt.tight_layout(); st.pyplot(fig); plt.close()

col3, col4 = st.columns(2)

with col3:
    st.write("**Price vs Year (by Kms)**")
    fig, ax = dark_fig(figsize=(6, 4))
    sc = ax.scatter(df['year'], df['Price']/1e5, c=df['kms_driven'],
                    cmap='plasma', alpha=0.5, s=12)
    plt.colorbar(sc, ax=ax, label='Kms Driven', shrink=0.8)
    ax.set_xlabel("Year"); ax.set_ylabel("Price (₹ Lakhs)")
    ax.set_title("Price vs Year")
    plt.tight_layout(); st.pyplot(fig); plt.close()

with col4:
    st.write("**Price by Fuel Type**")
    fig, ax = dark_fig(figsize=(6, 4))
    palette = {'Petrol': '#38bdf8', 'Diesel': '#f59e0b', 'CNG': '#10b981', 'LPG': '#a78bfa'}
    sns.boxplot(data=df, x='fuel_type', y='Price', ax=ax,
                palette=palette, linewidth=0.8)
    ax.set_xlabel("Fuel Type"); ax.set_ylabel("Price (₹)")
    ax.set_title("Price by Fuel Type")
    plt.tight_layout(); st.pyplot(fig); plt.close()

st.write("**Top 10 Companies by Average Price**")
top_comp = df.groupby('company')['Price'].mean().sort_values(ascending=False).head(10)
bar_colors = ['#38bdf8','#10b981','#f59e0b','#e05c8a','#a78bfa',
              '#6ee7b7','#fbbf24','#f472b6','#60a5fa','#34d399']
fig, ax = dark_fig(figsize=(12, 4))
bars = ax.bar(top_comp.index, top_comp.values/1e5,
              color=bar_colors[:len(top_comp)], edgecolor='none', width=0.6)
for bar, val in zip(bars, top_comp.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f'₹{val/1e5:.1f}L', ha='center', va='bottom',
            fontsize=8, color='#94a3b8', fontfamily='monospace')
ax.set_ylabel("Avg Price (₹ Lakhs)"); ax.set_title("Average Price per Company")
plt.xticks(rotation=30, ha='right')
plt.tight_layout(); st.pyplot(fig); plt.close()


# ================================================================
# STEP 4 — GRADIENT DESCENT
# ================================================================
st.markdown("""
<div class="section-wrap">
  <div class="section-label">Step 04</div>
  <div class="section-title">∇ Gradient Descent — Finding Minimum Error</div>
  <div class="section-divider"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="math-card">
<b style='color:#38bdf8'>Loss (MSE):</b>       L(w) = (1/n) · Σ (yᵢ − ŷᵢ)²<br>
<b style='color:#38bdf8'>Gradient:</b>          ∇L(w) = ∂L/∂w<br>
<b style='color:#10b981'>Update Rule:</b>       w  ←  w − η · ∇L(w)<br>
<span style='color:#475569'>η = learning rate · w = model weight · ŷ = predicted price</span>
</div>
""", unsafe_allow_html=True)

# Simulate gradient descent
np.random.seed(42)        
lr, w = 0.05, 5.0
losses, weights, grads = [], [], []
for _ in range(80):
    loss = w**2 + 0.5*np.sin(3*w) + 0.1*np.random.randn()
    g    = 2*w + 1.5*np.cos(3*w)
    w   -= lr * g
    losses.append(max(0, loss))
    weights.append(w)
    grads.append(abs(g))

iters = list(range(len(losses)))

gd1, gd2 = st.columns(2)

with gd1:
    st.write("**Loss Convergence  L(w) vs Iterations**")
    fig, ax = dark_fig(figsize=(6, 4))
    ax.plot(iters, losses, color='#38bdf8', lw=2, label='Loss  L(w)')
    ax.fill_between(iters, losses, alpha=0.15, color='#38bdf8')
    ax.axhline(min(losses), color='#10b981', ls='--', lw=1.5,
               label=f'Min ≈ {min(losses):.4f}')
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss  L(w)")
    ax.set_title("Loss Decreasing to Minimum")
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)
    ax.grid(True, alpha=0.12)
    plt.tight_layout(); st.pyplot(fig); plt.close()

with gd2:
    st.write("**Weight Trajectory on Loss Surface**")
    w_r = np.linspace(-6, 6, 300)
    l_s = w_r**2 + 0.5*np.sin(3*w_r)
    fig, ax = dark_fig(figsize=(6, 4))
    ax.plot(w_r, l_s, color='#475569', lw=2, label='Loss Surface  L(w)')
    ax.scatter(weights[::8], [w**2+0.5*np.sin(3*w) for w in weights[::8]],
               color='#f59e0b', s=55, zorder=5, label='Gradient Steps')
    ax.scatter([weights[-1]], [weights[-1]**2+0.5*np.sin(3*weights[-1])],
               color='#10b981', s=160, zorder=6, marker='*', label='Minimum Found')
    ax.set_xlabel("Weight  w"); ax.set_ylabel("Loss  L(w)")
    ax.set_title("Steps Walking to the Minimum")
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)
    ax.grid(True, alpha=0.12)
    plt.tight_layout(); st.pyplot(fig); plt.close()

gd3, gd4 = st.columns(2)

with gd3:
    st.write("**Gradient Magnitude  |∇L(w)| vs Iterations**")
    fig, ax = dark_fig(figsize=(6, 4))
    ax.plot(iters, grads, color='#f87171', lw=2, label='|∇L(w)|')
    ax.fill_between(iters, grads, alpha=0.15, color='#f87171')
    ax.axhline(0, color='#475569', ls='--', lw=1, label='Zero  (converged)')
    ax.set_xlabel("Iteration"); ax.set_ylabel("Gradient Magnitude")
    ax.set_title("Gradient Approaches Zero at Minimum")
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)
    ax.grid(True, alpha=0.12)
    plt.tight_layout(); st.pyplot(fig); plt.close()

with gd4:
    st.write("**2D Loss Surface  (w₁ Year × w₂ Kms)**")
    w1 = np.linspace(-2, 2, 60); w2 = np.linspace(-2, 2, 60)
    W1, W2 = np.meshgrid(w1, w2)
    L2 = (W1-0.4)**2 + (W2+0.3)**2 + 0.3*np.sin(3*W1)*np.cos(2*W2)
    py, pk = [1.8], [1.5]
    for _ in range(25):
        gy = 2*(py[-1]-0.4)+0.9*np.cos(3*py[-1])*np.cos(2*pk[-1])
        gk = 2*(pk[-1]+0.3)-0.6*np.sin(3*py[-1])*np.sin(2*pk[-1])
        py.append(py[-1]-0.12*gy); pk.append(pk[-1]-0.12*gk)
    fig, ax = dark_fig(figsize=(6, 4))
    cf = ax.contourf(W1, W2, L2, levels=30, cmap=CMAP_LOSS, alpha=0.88)
    ax.contour(W1, W2, L2, levels=10, colors='white', alpha=0.12, linewidths=0.5)
    plt.colorbar(cf, ax=ax, label='Loss  L(w₁,w₂)', shrink=0.8)
    ax.plot(py, pk, 'w--o', ms=4, lw=1.5, label='Descent Path', alpha=0.9)
    ax.scatter([py[-1]], [pk[-1]], s=160, color='#10b981', zorder=10,
               marker='*', label='Global Minimum')
    ax.set_xlabel("w₁  (Year Weight)"); ax.set_ylabel("w₂  (Kms Weight)")
    ax.set_title("Gradient Path on 2D Loss Surface")
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()

st.info(
    f"✅ **Gradient Descent Summary** — "
    f"Start Loss: **{losses[0]:.4f}**  →  "
    f"Min Loss: **{min(losses):.4f}**  |  "
    f"Iterations: **{len(losses)}**  |  "
    f"Learning Rate η: **{lr}**"
)


# ================================================================
# STEP 5 — MODEL TRAINING
# ================================================================
st.markdown("""
<div class="section-wrap">
  <div class="section-label">Step 05</div>
  <div class="section-title">🤖 Random Forest Model Training</div>
  <div class="section-divider"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="math-card">
<b style='color:#38bdf8'>Ensemble:</b>   f̂(x) = (1/T) · Σₜ fₜ(x)   [T = 100 decision trees]<br>
<b style='color:#38bdf8'>Split rule:</b> Minimize ΔI = I(parent) − [n_L/n · I(L) + n_R/n · I(R)]<br>
<b style='color:#10b981'>Prediction:</b> P̂ ∈ ℝ⁺  →  gradient-optimised ensemble output
</div>
""", unsafe_allow_html=True)

features = ['name', 'company', 'year', 'kms_driven', 'fuel_type']
target   = 'Price'
df_m     = df.dropna(subset=features + [target])
X, y     = df_m[features], df_m[target]

ct = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), ['name','company','fuel_type']),
    ("num", StandardScaler(), ['year','kms_driven'])
])
pipe = Pipeline([('t', ct), ('r', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

with st.spinner("🔄 Training model…"):
    pipe.fit(X_tr, y_tr)

y_pr = pipe.predict(X_te)
r2   = r2_score(y_te, y_pr)
mae  = mean_absolute_error(y_te, y_pr)
rmse = np.sqrt(mean_squared_error(y_te, y_pr))

st.success("✅ Model trained successfully!")
joblib.dump(pipe, "car_price_rf_model.pkl")

st.markdown(f"""
<div class="metric-row">
  <div class="metric-card" style="--accent:#38bdf8">
    <div class="metric-num">{r2:.4f}</div>
    <div class="metric-lbl">R² Score</div>
  </div>
  <div class="metric-card" style="--accent:#10b981">
    <div class="metric-num">₹{mae:,.0f}</div>
    <div class="metric-lbl">Mean Abs Error</div>
  </div>
  <div class="metric-card" style="--accent:#f59e0b">
    <div class="metric-num">₹{rmse:,.0f}</div>
    <div class="metric-lbl">RMSE</div>
  </div>
  <div class="metric-card" style="--accent:#e05c8a">
    <div class="metric-num">{len(X_tr):,}</div>
    <div class="metric-lbl">Training Samples</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
mc1, mc2 = st.columns(2)

with mc1:
    st.write("**Actual vs Predicted Prices**")
    fig, ax = dark_fig(figsize=(6, 4))
    ax.scatter(y_te/1e5, y_pr/1e5, alpha=0.35, s=10, color='#38bdf8')
    mv = max(y_te.max(), y_pr.max())/1e5
    ax.plot([0,mv],[0,mv],'--',color='#f87171',lw=1.5,label='Perfect Fit')
    ax.set_xlabel("Actual (₹ Lakhs)"); ax.set_ylabel("Predicted (₹ Lakhs)")
    ax.set_title("Actual vs Predicted")
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)
    ax.grid(True, alpha=0.12)
    plt.tight_layout(); st.pyplot(fig); plt.close()

with mc2:
    st.write("**Residual Distribution**")
    residuals = y_te.values - y_pr
    fig, ax = dark_fig(figsize=(6, 4))
    sns.histplot(residuals/1e5, kde=True, ax=ax, color='#f59e0b', alpha=0.7, edgecolor='none')
    ax.axvline(0, color='#f87171', lw=1.5, ls='--', label='Zero Error')
    ax.set_xlabel("Residual (₹ Lakhs)"); ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)
    ax.grid(True, alpha=0.12)
    plt.tight_layout(); st.pyplot(fig); plt.close()


# ================================================================
# STEP 6 — PREDICTION TOOL
# ================================================================
st.markdown("""
<div class="section-wrap">
  <div class="section-label">Step 06</div>
  <div class="section-title">🔮 Car Price Prediction Tool</div>
  <div class="section-divider"></div>
</div>
""", unsafe_allow_html=True)

pc1, pc2, pc3 = st.columns(3)

with pc1:
    sel_company = st.selectbox("🏭 Car Company", sorted(df['company'].unique()))
    year_in = st.slider("📅 Year of Manufacture",
                        min_value=int(df['year'].min()),
                        max_value=int(df['year'].max()), value=2019)
with pc2:
    f_names  = sorted(df[df['company'] == sel_company]['name'].unique())
    sel_name = st.selectbox("🚘 Car Model", f_names)
    kms_in   = st.number_input("🛣 Kilometers Driven",
                               min_value=0, max_value=500000, value=60000, step=500)
with pc3:
    fuel_in = st.selectbox("⛽ Fuel Type", sorted(df['fuel_type'].unique()))
    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.button("🚀  Predict Car Price")

if predict:
    mdl = joblib.load("car_price_rf_model.pkl")
    inp = pd.DataFrame({'name': [sel_name], 'company': [sel_company],
                        'year': [year_in], 'kms_driven': [kms_in],
                        'fuel_type': [fuel_in]})
    pp   = mdl.predict(inp)[0]
    low  = max(0, pp - mae)
    high = pp + mae

    st.markdown(f"""
    <div class="pred-result">
      <div style='font-family:Fira Code,monospace;font-size:0.65rem;color:#10b981;
                  text-transform:uppercase;letter-spacing:0.18em;margin-bottom:10px;'>
        ∇-Optimised Ensemble Prediction
      </div>
      <div class="pred-amount">₹{pp:,.0f}</div>
      <div class="pred-lakh">≈ ₹{pp/1e5:.2f} Lakhs</div>
      <br>
      <div class="pred-range">
        📊 Range: ₹{low:,.0f} — ₹{high:,.0f} &nbsp;|&nbsp; ± MAE = ₹{mae:,.0f}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>**🔍 Similar Cars in Dataset**")
    similar = df[
        (df['company'] == sel_company) &
        (df['fuel_type'] == fuel_in) &
        (df['year'].between(year_in-2, year_in+2))
    ][['name','company','year','kms_driven','fuel_type','Price']].head(6)

    if not similar.empty:
        st.dataframe(similar.reset_index(drop=True), use_container_width=True)
    else:
        st.info("ℹ️ No similar cars found for this combination.")


# ================================================================
# FOOTER
# ================================================================
st.markdown("""
<div class="footer-bar">
  🚗 Car Price Prediction Dashboard &nbsp;·&nbsp; Vector Calculus Application &nbsp;·&nbsp; 4th Semester<br>
  ∇ Gradient Descent · 📊 Data Viz · 🔮 ML Prediction &nbsp;·&nbsp;
  Built with ❤ by Hanee &amp; Krishna &nbsp;·&nbsp; Streamlit + Scikit-learn
</div>
""", unsafe_allow_html=True)