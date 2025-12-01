# Advanced Sales Forecast & Business Intelligence Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import hashlib
import warnings
from datetime import datetime, timedelta
import json
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
import base64

# optional bcrypt (if you have it)
try:
    import bcrypt
    HAVE_BCRYPT = True
except Exception:
    HAVE_BCRYPT = False

warnings.filterwarnings('ignore')

# --------------------------
# Custom CSS and Styling
# --------------------------
def load_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --warning-color: #d62728;
        --info-color: #17a2b8;
        --light-bg: #f8f9fa;
        --dark-bg: #343a40;
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary-color);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"]:not([aria-selected="true"]) {
    background-color: #f0f2f6;  /* light gray or any color you like */
    color: #333333;              /* visible text color */
    font-weight: 600;
    border-radius: 5px 5px 0 0;
}

/* Selected tab styling */
.stTabs [aria-selected="true"] {
    background-color: #ff7f0e;  /* your new primary color */
    color: white;
    font-weight: 700;
}

/* Tab hover effect */
.stTabs [data-baseweb="tab"]:hover {
    background-color: #d9e2f3;  /* optional hover color */
    color: #000;
}
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Alert boxes */
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .alert-info {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid var(--primary-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Data table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------
# Advanced Analytics Functions
# --------------------------
def calculate_advanced_metrics(df):
    """Calculate advanced business metrics"""
    metrics = {}
    
    if df.empty:
        return metrics
    
    # Basic metrics
    metrics['total_sales'] = df['sales'].sum()
    metrics['avg_order_value'] = df['sales'].mean()
    metrics['total_orders'] = len(df)
    
    # Advanced metrics
    metrics['median_order_value'] = df['sales'].median()
    metrics['std_order_value'] = df['sales'].std()
    metrics['cv_order_value'] = (df['sales'].std() / df['sales'].mean()) * 100 if df['sales'].mean() > 0 else 0
    
    # Growth metrics
    if 'date' in df.columns:
        df_sorted = df.sort_values('date')
        if len(df_sorted) > 1:
            first_half = df_sorted.iloc[:len(df_sorted)//2]['sales'].sum()
            second_half = df_sorted.iloc[len(df_sorted)//2:]['sales'].sum()
            metrics['growth_rate'] = ((second_half - first_half) / first_half * 100) if first_half > 0 else 0
    
    return metrics

def detect_seasonality(df, date_col='date', sales_col='sales'):
    """Detect seasonal patterns in sales data"""
    if df.empty or date_col not in df.columns:
        return None
    
    df_seasonal = df.copy()
    df_seasonal['month'] = df_seasonal[date_col].dt.month
    df_seasonal['quarter'] = df_seasonal[date_col].dt.quarter
    df_seasonal['day_of_week'] = df_seasonal[date_col].dt.dayofweek
    
    # Calculate seasonal indices
    monthly_avg = df_seasonal.groupby('month')[sales_col].mean()
    overall_avg = df_seasonal[sales_col].mean()
    seasonal_indices = (monthly_avg / overall_avg) * 100
    
    return {
        'monthly_indices': seasonal_indices,
        'peak_month': monthly_avg.idxmax(),
        'low_month': monthly_avg.idxmin(),
        'seasonality_strength': monthly_avg.std() / monthly_avg.mean() * 100
    }

def forecast_sales(df, periods=6, method='linear'):
    """Generate sales forecasts using different methods"""
    if df.empty or 'date' not in df.columns:
        return None
    
    # Prepare data
    df_forecast = df.copy()
    df_forecast = df_forecast.set_index('date').resample('M')['sales'].sum().reset_index()
    df_forecast = df_forecast.sort_values('date')
    
    if len(df_forecast) < 3:
        return None
    
    # Create time features
    df_forecast['time_index'] = range(len(df_forecast))
    
    # Generate future dates
    last_date = df_forecast['date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
    
    if method == 'linear':
        model = LinearRegression()
        X = df_forecast[['time_index']]
        y = df_forecast['sales']
        model.fit(X, y)
        
        future_time_indices = range(len(df_forecast), len(df_forecast) + periods)
        predictions = model.predict(np.array(future_time_indices).reshape(-1, 1))
    
    elif method == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X = df_forecast[['time_index']]
        y = df_forecast['sales']
        model.fit(X, y)
        
        future_time_indices = range(len(df_forecast), len(df_forecast) + periods)
        predictions = model.predict(np.array(future_time_indices).reshape(-1, 1))
    
    else:  # Simple moving average
        window = min(3, len(df_forecast))
        predictions = [df_forecast['sales'].tail(window).mean()] * periods
    
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': predictions,
        'method': method
    })
    
    return forecast_df

def detect_anomalies(df, sales_col='sales', threshold=2):
    """Detect sales anomalies using statistical methods"""
    if df.empty or sales_col not in df.columns:
        return pd.DataFrame()
    
    sales_data = df[sales_col].values
    mean_sales = np.mean(sales_data)
    std_sales = np.std(sales_data)
    
    # Z-score method
    z_scores = np.abs((sales_data - mean_sales) / std_sales)
    anomalies = df[z_scores > threshold].copy()
    anomalies['z_score'] = z_scores[z_scores > threshold]
    anomalies['anomaly_type'] = np.where(
        anomalies[sales_col] > mean_sales, 'High Sales', 'Low Sales'
    )
    
    return anomalies

# --------------------------
# Helpers
# --------------------------
USER_CSV = "users.csv"

def ensure_users_file():
    if not os.path.exists(USER_CSV):
        pd.DataFrame(columns=["username","password"]).to_csv(USER_CSV, index=False)

def load_users():
    ensure_users_file()
    users = pd.read_csv(USER_CSV)
    return users

def save_user(username, password):
    users = load_users()
    if username in users["username"].values:
        return False
    if HAVE_BCRYPT:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    else:
        hashed = password  # fallback (insecure) if bcrypt not installed
    users = pd.concat([users, pd.DataFrame([{"username":username, "password":hashed}])], ignore_index=True)
    users.to_csv(USER_CSV, index=False)
    return True

def check_login(username, password):
    users = load_users()
    if username in users["username"].values:
        stored = users.loc[users["username"]==username, "password"].values[0]
        if HAVE_BCRYPT:
            return bcrypt.checkpw(password.encode(), stored.encode())
        else:
            return password == stored
    return False

def find_column(df, candidates):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    # exact match or case-insensitive
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # partial match fallback
    for c in cols:
        for cand in candidates:
            if cand.lower() in c.lower() or c.lower() in cand.lower():
                return c
    return None

def detect_columns(df):
    """
    Smartly detect or derive date, sales, and item columns
    Works for BigMart, Invoice, and E-commerce datasets.
    """

    # 1️⃣ Candidate columns for detection
    date_candidates = [
        "Order Date", "Date", "InvoiceDate", "Invoice_Date", "Created_At", "order_date"
    ]
    sales_candidates = [
        "Sales", "Revenue", "Item_Outlet_Sales", "Amount",
        "Total", "TotalPrice", "price", "UnitPrice", "mrp", "selling_price"
    ]
    item_candidates = [
        "Item", "Product", "Item_Type", "Description",
        "product_name", "Item_Identifier", "StockCode", "Product_Name", "name"
    ]

    # 2️⃣ Try to detect them directly
    date_col = find_column(df, date_candidates)
    sales_col = find_column(df, sales_candidates)
    item_col = find_column(df, item_candidates)

    # 3️⃣ If no sales column → compute it when Quantity & UnitPrice exist
    if sales_col is None and "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["Sales"] = pd.to_numeric(df["Quantity"], errors="coerce") * pd.to_numeric(df["UnitPrice"], errors="coerce")
        sales_col = "Sales"

    # 4️⃣ If still no sales column, try fallback patterns (price * 1)
    if sales_col is None and "price" in df.columns:
        df["Sales"] = pd.to_numeric(df["price"], errors="coerce")
        sales_col = "Sales"

    # 5️⃣ If no date column → generate synthetic sequence
    if date_col is None:
        df["Fake_Date"] = pd.date_range(start="2020-01-01", periods=len(df), freq="D")
        date_col = "Fake_Date"

    # 6️⃣ Normalize date column to datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # 7️⃣ Final sanity cleanup
    if df[date_col].isnull().all():
        df["Fake_Date"] = pd.date_range(start="2020-01-01", periods=len(df), freq="D")
        date_col = "Fake_Date"

    return date_col, sales_col, item_col



@st.cache_data
def read_file(uploaded):
    """Reads CSV or Excel file safely, handles £ and other encodings"""
    try:
        name = uploaded.name.lower()
        if name.endswith(".csv"):
            try:
                # Try UTF-8 first
                return pd.read_csv(uploaded, encoding='utf-8')
            except UnicodeDecodeError:
                # Retry with Western European encoding for £ or € symbols
                uploaded.seek(0)
                return pd.read_csv(uploaded, encoding='ISO-8859-1')
        else:
            return pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Failed to read {getattr(uploaded, 'name', str(uploaded))}: {e}")
        return None

def normalize_dataset(raw_df):
    date_col, sales_col, item_col = detect_columns(raw_df)
    if date_col is None or sales_col is None:
        return None
    df = raw_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce').fillna(0)
    if item_col is not None:
        df[item_col] = df[item_col].astype(str)
    # Normalized view
    norm = pd.DataFrame()
    norm['date'] = df[date_col]
    norm['sales'] = df[sales_col]
    norm['item'] = df[item_col] if item_col is not None else np.nan
    norm = norm.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    return {"raw": raw_df, "norm": norm, "meta": {"date_col":date_col, "sales_col":sales_col, "item_col":item_col}}

def make_key(*parts):
    s = "|".join([str(p) for p in parts])
    return hashlib.md5(s.encode()).hexdigest()

def get_monthly_total(df_norm):
    # returns DataFrame with columns ['date','sales'] where date is month-start timestamp
    if df_norm.empty:
        return pd.DataFrame(columns=['date','sales'])
    monthly = df_norm.set_index('date').resample('M')['sales'].sum().reset_index()
    return monthly

def get_monthly_item_sales(df_norm):
    # df_norm has columns date,sales,item
    if 'item' not in df_norm.columns or df_norm['item'].isnull().all():
        return pd.DataFrame(columns=['date','item','sales'])
    tmp = df_norm.copy()
    tmp['month'] = tmp['date'].dt.to_period('M').dt.to_timestamp()
    monthly_items = tmp.groupby(['month','item'], as_index=False)['sales'].sum().rename(columns={'month':'date'})
    return monthly_items

def continuity_summary_from_monthly_items(monthly_items, months=3):
    # monthly_items: columns ['date','item','sales']; date is month timestamp
    if monthly_items.empty:
        return pd.DataFrame(columns=['item','Hike_Percentage_Next_Month','sales_last_n','previous_sales'])
    latest = monthly_items['date'].max()
    last_n = [latest - pd.DateOffset(months=i) for i in range(0, months)]
    previous_n = [latest - pd.DateOffset(months=i) for i in range(months, months*2)]
    
    # Current period sales (last 3 months)
    mask_current = monthly_items['date'].isin(last_n)
    agg_current = monthly_items[mask_current].groupby('item', as_index=False)['sales'].sum().rename(columns={'sales':'sales_last_n'})
    
    # Previous period sales (3 months before that)
    mask_previous = monthly_items['date'].isin(previous_n)
    agg_previous = monthly_items[mask_previous].groupby('item', as_index=False)['sales'].sum().rename(columns={'sales':'previous_sales'})
    
    all_items = pd.DataFrame({'item': monthly_items['item'].unique()})
    summary = all_items.merge(agg_current, on='item', how='left').fillna(0)
    summary = summary.merge(agg_previous, on='item', how='left').fillna(0)
    
    # Calculate percentage hike
    summary['Hike_Percentage_Next_Month'] = np.where(
        summary['previous_sales'] > 0,
        ((summary['sales_last_n'] - summary['previous_sales']) / summary['previous_sales'] * 100).round(2).astype(str) + '%',
        np.where(summary['sales_last_n'] > 0, 'New Product 🆕', 'No Sales ❌')
    )
    
    summary = summary.sort_values(['sales_last_n','item'], ascending=[False,True]).reset_index(drop=True)
    return summary

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(
    page_title=" Sales Forecast Prediction ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# Main header
st.markdown("""
<div class="main-header">
    <h1>📊 Sales Forecast Prediction </h1>
    <p>Comprehensive Business Analytics, Forecasting & Performance Insights</p>
</div>
""", unsafe_allow_html=True)

# auth session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"

# AUTH UI
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        st.header("🔐 Login")
        username = st.text_input("Username", key="lg_user")
        password = st.text_input("Password", type="password", key="lg_pass")
        if st.button("Login"):
            if check_login(username, password):
                st.success(f"Welcome, {username}")
                st.session_state.logged_in = True
                st.session_state.user = username
                st.session_state.page = "app"
                st.rerun()
            else:
                st.error("Invalid credentials")
        if st.button("Go to Signup"):
            st.session_state.page = "signup"
            st.rerun()
        st.stop()

    else:  # signup
        st.header("🆕 Signup")
        new_user = st.text_input("Choose username", key="sg_user")
        new_pass = st.text_input("Choose password", type="password", key="sg_pass")
        if st.button("Create account"):
            if not new_user or not new_pass:
                st.error("Provide username and password")
            else:
                ok = save_user(new_user, new_pass)
                if ok:
                    st.success("Account created — now login")
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error("Username exists")
        if st.button("Back to login"):
            st.session_state.page = "login"
            st.rerun()
        st.stop()

# MAIN APP
st.sidebar.success(f"Logged in as {st.session_state.user}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.page = "login"
    st.rerun()

# Enhanced Sidebar
st.sidebar.markdown("### 📁 Data Management")
uploaded = st.sidebar.file_uploader("Upload CSV / XLSX files", accept_multiple_files=True, type=['csv','xlsx','xls'])
if not uploaded:
    st.info("📊 Upload one or more datasets. Expected columns: Order Date, Item, Revenue or Order Date, Product, Sales.")
    st.stop()

# Data Quality Check
st.sidebar.markdown("### 🔍 Data Quality")
if uploaded:
    total_files = len(uploaded)
    st.sidebar.success(f"✅ {total_files} file(s) uploaded")
    
    # File size check
    total_size = sum(f.size for f in uploaded)
    size_mb = total_size / (1024 * 1024)
    if size_mb > 100:
        st.sidebar.warning(f"⚠ Large dataset: {size_mb:.1f} MB")
    else:
        st.sidebar.info(f"📊 Dataset size: {size_mb:.1f} MB")

# load and normalize
datasets = {}
for f in uploaded:
    raw = read_file(f)
    if raw is None:
        continue
    info = normalize_dataset(raw)
    if info is None:
        st.warning(f"{f.name}: couldn't detect date/sales columns; skipped.")
        continue
    datasets[f.name] = info

if not datasets:
    st.error("No usable datasets.")
    st.stop()

dataset_names = list(datasets.keys())
selected_dataset = st.sidebar.selectbox("Select dataset to analyze", dataset_names)
info = datasets[selected_dataset]
df_norm = info['norm']  # date,sales,item
meta = info['meta']

# filters
st.sidebar.markdown("### Filters")
min_d = df_norm['date'].min().date()
max_d = df_norm['date'].max().date()
date_range = st.sidebar.date_input("Date range", [min_d, max_d], min_value=min_d, max_value=max_d)
item_exists = meta['item_col'] is not None and not df_norm['item'].isnull().all()
items = sorted([i for i in df_norm['item'].dropna().unique()]) if item_exists else []
selected_items = st.sidebar.multiselect("Filter items (optional)", items[:500], default=items[:10] if items else [])
top_n = st.sidebar.slider("Top N products", 5, 50, 10)

# apply filters
df = df_norm.copy()
start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
df = df[(df['date'] >= start) & (df['date'] <= end)]
if item_exists and selected_items:
    df = df[df['item'].isin(selected_items)]

monthly_total = get_monthly_total(df)  # date,sales
monthly_items = get_monthly_item_sales(df)  # date,item,sales

# tabs
tab_overview, tab_product, tab_analytics, tab_continuity = st.tabs([
    "📊 Overview", "🛍 Products", "🔬 Analytics", "📈 Continuity"
])

# ---------- Overview ----------
with tab_overview:
    st.header("📈 Executive Dashboard")
    
    # Calculate advanced metrics
    metrics = calculate_advanced_metrics(df)
    
    # Key Performance Indicators
    st.subheader("🎯 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Sales",
            value=f"${metrics.get('total_sales', 0):,.0f}",
            delta=f"{metrics.get('growth_rate', 0):.1f}% vs Previous Period"
        )
    
    with col2:
        st.metric(
            label="📊 Avg Order Value",
            value=f"${metrics.get('avg_order_value', 0):,.2f}",
            delta=f"Median: ${metrics.get('median_order_value', 0):,.2f}"
        )
    
    with col3:
        st.metric(
            label="📦 Total Orders",
            value=f"{metrics.get('total_orders', 0):,}",
            delta=f"CV: {metrics.get('cv_order_value', 0):.1f}%"
        )
    
    with col4:
        # Calculate MoM change
        latest = monthly_total['sales'].iloc[-1] if not monthly_total.empty else 0.0
        prev = monthly_total['sales'].iloc[-2] if len(monthly_total)>1 else latest
        mom = (latest - prev) / (prev + 1e-9) * 100
        
        st.metric(
            label="📈 Month-over-Month",
            value=f"{mom:.1f}%",
            delta="vs Last Month"
        )

    # Advanced Analytics Section
    st.subheader("🔍 Advanced Analytics")
    
    # Forecasting
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📊 Sales Trend & Forecasting")
        if not monthly_total.empty:
            # Create forecast
            forecast_df = forecast_sales(df, periods=6, method='linear')
            
            if forecast_df is not None:
                # Combine historical and forecast data
                historical = monthly_total.copy()
                historical['type'] = 'Historical'
                forecast = forecast_df.copy()
                forecast = forecast.rename(columns={'forecast': 'sales'})
                forecast['type'] = 'Forecast'
                
                combined_df = pd.concat([historical, forecast], ignore_index=True)
                
                # Create the plot
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=historical['date'],
                    y=historical['sales'],
                    mode='lines+markers',
                    name='Historical Sales',
                    line=dict(color='#1f77b4', width=3),
                    marker=dict(size=6)
                ))
                
                # Forecast data
                fig.add_trace(go.Scatter(
                    x=forecast['date'],
                    y=forecast['sales'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='#ff7f0e', width=3, dash='dash'),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title="Sales Trend & 6-Month Forecast",
                    xaxis_title="Date",
                    yaxis_title="Sales ($)",
                    hovermode='x unified',
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True, key=make_key(selected_dataset, "forecast"))
            else:
                fig = px.line(monthly_total, x='date', y='sales', title=f"Monthly Sales - {selected_dataset}", markers=True)
                st.plotly_chart(fig, use_container_width=True, key=make_key(selected_dataset, "monthly_trend"))
        else:
            st.info("Not enough monthly data for forecasting")
    
    with col2:
        st.markdown("#### 🎯 Business Insights")
        
        # Seasonality analysis
        seasonality = detect_seasonality(df)
        if seasonality:
            st.markdown(f"*Peak Month:* {seasonality['peak_month']}")
            st.markdown(f"*Low Month:* {seasonality['low_month']}")
            st.markdown(f"*Seasonality Strength:* {seasonality['seasonality_strength']:.1f}%")
        
        # Anomaly detection
        anomalies = detect_anomalies(df)
        if not anomalies.empty:
            st.markdown(f"⚠ Anomalies Detected:** {len(anomalies)}")
            if len(anomalies) > 0:
                st.markdown("*Top Anomaly:*")
                top_anomaly = anomalies.nlargest(1, 'z_score')
                st.markdown(f"- {top_anomaly.iloc[0]['anomaly_type']}: ${top_anomaly.iloc[0]['sales']:,.0f}")
        else:
            st.markdown("✅ No significant anomalies detected**")

    # Yearly
    st.subheader("Yearly Sales Summary")
    if not df.empty:
        df_year = df.copy()
        df_year['year'] = df_year['date'].dt.year
        yearly = df_year.groupby('year', as_index=False)['sales'].sum()
        if not yearly.empty:
            fig = px.bar(yearly, x='year', y='sales', title="Sales by Year", text='sales')
            st.plotly_chart(fig, use_container_width=True, key=make_key(selected_dataset,"yearly"))
        else:
            st.info("No yearly data")
    else:
        st.info("No data after filtering")

    # Seasonal pattern (month number)
    st.subheader("Seasonal Pattern (Month of Year)")
    if not df.empty:
        df_month = df.copy()
        df_month['month'] = df_month['date'].dt.month
        month_pattern = df_month.groupby('month', as_index=False)['sales'].sum()
        fig = px.line(month_pattern, x='month', y='sales', markers=True, title="Sales by Month Number")
        st.plotly_chart(fig, use_container_width=True, key=make_key(selected_dataset,"seasonal"))
    else:
        st.info("No data")

    # Distribution
    st.subheader("Sales Distribution")
    if not df.empty:
        fig = px.histogram(df, x='sales', nbins=30, title="Sales histogram")
        st.plotly_chart(fig, use_container_width=True, key=make_key(selected_dataset,"hist"))
        fig2 = px.box(df, y='sales', title="Sales boxplot")
        st.plotly_chart(fig2, use_container_width=True, key=make_key(selected_dataset,"box"))
    else:
        st.info("No transactions in filtered range")

# ---------- Product ----------
with tab_product:
    st.header("Product-level Analysis")
    if not item_exists:
        st.info("No item/product column detected in this dataset.")
    else:
        prod = df.groupby('item', as_index=False)['sales'].sum().sort_values('sales', ascending=False)
        st.subheader(f"Top {top_n} Products")
        top_products = prod.head(top_n)
        st.plotly_chart(px.bar(top_products, x='item', y='sales', text='sales', title="Top products"),
                        use_container_width=True, key=make_key(selected_dataset,"top_products"))
        
        st.subheader(f"Bottom {top_n} Products")
        bottom_products = prod.tail(top_n).sort_values('sales', ascending=True)
        st.plotly_chart(px.bar(bottom_products, x='item', y='sales', text='sales', title="Bottom products"),
                        use_container_width=True, key=make_key(selected_dataset,"bottom_products"))

        st.subheader("Monthly Sales by Top Items (stacked area)")
        if not monthly_items.empty:
            # choose up to 8 top items for clarity
            top_items = prod.head(8)['item'].tolist()
            stacked = monthly_items[monthly_items['item'].isin(top_items)]
            if not stacked.empty:
                fig = px.area(stacked, x='date', y='sales', color='item', title="Top items contribution")
                st.plotly_chart(fig, use_container_width=True, key=make_key(selected_dataset,"stacked"))
            else:
                st.info("No monthly item data for top items.")
        else:
            st.info("No monthly item-level data")

# ---------- Analytics ----------
with tab_analytics:
    st.header("🔬 Advanced Analytics & Insights")
    
    # Analytics options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Seasonality Analysis", "Anomaly Detection", "Trend Analysis", "Performance Metrics"]
        )
    
    with col2:
        time_period = st.selectbox(
            "Time Period",
            ["Last 3 Months", "Last 6 Months", "Last Year", "All Time"]
        )
    
    with col3:
        if analysis_type == "Anomaly Detection":
            anomaly_threshold = st.slider("Anomaly Threshold (Z-Score)", 1.0, 3.0, 2.0, 0.1)
        else:
            anomaly_threshold = 2.0
    
    st.markdown("---")
    
    if analysis_type == "Seasonality Analysis":
        st.subheader("📅 Seasonal Patterns")
        
        seasonality = detect_seasonality(df)
        if seasonality:
            # Monthly seasonality chart
            monthly_data = df.copy()
            monthly_data['month'] = monthly_data['date'].dt.month
            monthly_sales = monthly_data.groupby('month')['sales'].sum().reset_index()
            
            fig = px.bar(
                monthly_sales, 
                x='month', 
                y='sales',
                title="Sales by Month (Seasonal Pattern)",
                labels={'month': 'Month', 'sales': 'Total Sales ($)'}
            )
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Seasonality insights
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Peak Month", f"Month {seasonality['peak_month']}")
            with col2:
                st.metric("Low Month", f"Month {seasonality['low_month']}")
            with col3:
                st.metric("Seasonality Strength", f"{seasonality['seasonality_strength']:.1f}%")
        else:
            st.info("Insufficient data for seasonality analysis")
    
    elif analysis_type == "Anomaly Detection":
        st.subheader("🚨 Anomaly Detection")
        
        anomalies = detect_anomalies(df, threshold=anomaly_threshold)
        if not anomalies.empty:
            st.markdown(f"*Found {len(anomalies)} anomalies with Z-Score > {anomaly_threshold}*")
            
            # Anomaly chart
            fig = px.scatter(
                df, 
                x='date', 
                y='sales',
                title="Sales Data with Anomalies Highlighted",
                labels={'date': 'Date', 'sales': 'Sales ($)'}
            )
            
            # Highlight anomalies
            anomaly_points = anomalies[['date', 'sales']]
            fig.add_trace(go.Scatter(
                x=anomaly_points['date'],
                y=anomaly_points['sales'],
                mode='markers',
                marker=dict(color='red', size=10, symbol='x'),
                name='Anomalies'
            ))
            
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Anomaly details
            st.subheader("Anomaly Details")
            anomaly_display = anomalies[['date', 'sales', 'z_score', 'anomaly_type']].copy()
            anomaly_display['date'] = anomaly_display['date'].dt.strftime('%Y-%m-%d')
            anomaly_display = anomaly_display.rename(columns={
                'date': 'Date',
                'sales': 'Sales ($)',
                'z_score': 'Z-Score',
                'anomaly_type': 'Type'
            })
            st.dataframe(anomaly_display, use_container_width=True)
        else:
            st.success(f"No anomalies detected with Z-Score > {anomaly_threshold}")
    
    elif analysis_type == "Trend Analysis":
        st.subheader("📈 Trend Analysis")
        
        if not monthly_total.empty:
            # Calculate trend
            monthly_data = monthly_total.copy()
            monthly_data['time_index'] = range(len(monthly_data))
            
            # Linear regression for trend
            from sklearn.linear_model import LinearRegression
            X = monthly_data[['time_index']]
            y = monthly_data['sales']
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate trend metrics
            trend_slope = model.coef_[0]
            trend_direction = "Increasing" if trend_slope > 0 else "Decreasing"
            r_squared = model.score(X, y)
            
            # Create trend line
            monthly_data['trend'] = model.predict(X)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_data['date'],
                y=monthly_data['sales'],
                mode='lines+markers',
                name='Actual Sales',
                line=dict(color='#1f77b4', width=3)
            ))
            fig.add_trace(go.Scatter(
                x=monthly_data['date'],
                y=monthly_data['trend'],
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="Sales Trend Analysis",
                xaxis_title="Date",
                yaxis_title="Sales ($)",
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Trend metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Trend Direction", trend_direction)
            with col2:
                st.metric("Monthly Change", f"${trend_slope:,.0f}")
            with col3:
                st.metric("R² Score", f"{r_squared:.3f}")
        else:
            st.info("Insufficient data for trend analysis")
    
    elif analysis_type == "Performance Metrics":
        st.subheader("📊 Performance Metrics")
        
        # Calculate comprehensive metrics
        metrics = calculate_advanced_metrics(df)
        
        # Display metrics in a grid
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Financial Metrics")
            st.metric("Total Revenue", f"${metrics.get('total_sales', 0):,.0f}")
            st.metric("Average Order Value", f"${metrics.get('avg_order_value', 0):,.2f}")
            st.metric("Median Order Value", f"${metrics.get('median_order_value', 0):,.2f}")
            st.metric("Growth Rate", f"{metrics.get('growth_rate', 0):.1f}%")
        
        with col2:
            st.markdown("#### Statistical Metrics")
            st.metric("Total Orders", f"{metrics.get('total_orders', 0):,}")
            st.metric("Order Value Std Dev", f"${metrics.get('std_order_value', 0):,.2f}")
            st.metric("Coefficient of Variation", f"{metrics.get('cv_order_value', 0):.1f}%")
            
            # Calculate additional metrics
            if not df.empty:
                q1 = df['sales'].quantile(0.25)
                q3 = df['sales'].quantile(0.75)
                iqr = q3 - q1
                st.metric("Interquartile Range", f"${iqr:,.2f}")

# ---------- Continuity ----------
with tab_continuity:
    st.header("Continuity: How much hike will each item get next month?")
    if not item_exists or monthly_items.empty:
        st.info("Continuity analysis requires an item column and monthly item data.")
    else:
        summary = continuity_summary_from_monthly_items(monthly_items, months=3)
        # Present clean table: Item | Hike_Percentage_Next_Month | Sales (last 3 months) | Previous Sales
        out = summary.rename(columns={
            'item':'Item',
            'Hike_Percentage_Next_Month':'Hike % Next Month',
            'sales_last_n':'Sales (last 3 months)',
            'previous_sales':'Previous Sales (3 months)'
        })
        # Reset index to start from 1
        out.index = out.index + 1
        st.dataframe(out)
        # download
        csv = out.to_csv(index=False).encode()
        st.download_button("Download continuity summary CSV", csv, f"continuity_{selected_dataset}.csv", "text/csv")
        # interactive exploration
        st.markdown("### Inspect item monthly series")
        item_choice = st.selectbox("Select item", out['Item'].tolist(), key="continuity_item_selector")
        if item_choice:
            item_series = monthly_items[monthly_items['item']==item_choice].rename(columns={'date':'Month','sales':'Sales'})
            if item_series.empty:
                st.info("No monthly records for this item.")
            else:
                fig = px.bar(item_series, x='Month', y='Sales', title=f"Monthly sales: {item_choice}")
                st.plotly_chart(fig, use_container_width=True, key=make_key(selected_dataset,"item_series",item_choice[:30]))


# --------------------------
# Export & Download Section
# --------------------------
st.markdown("---")
st.markdown("### 📥 Export & Download")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Export Overview Report"):
        # Create comprehensive overview report
        report_data = {
            'Dataset': selected_dataset,
            'Total Sales': f"${metrics.get('total_sales', 0):,.0f}",
            'Average Order Value': f"${metrics.get('avg_order_value', 0):,.2f}",
            'Total Orders': f"{metrics.get('total_orders', 0):,}",
            'Growth Rate': f"{metrics.get('growth_rate', 0):.1f}%",
            'Peak Month': seasonality['peak_month'] if seasonality else 'N/A',
            'Low Month': seasonality['low_month'] if seasonality else 'N/A',
            'Seasonality Strength': f"{seasonality['seasonality_strength']:.1f}%" if seasonality else 'N/A'
        }
        
        report_df = pd.DataFrame(list(report_data.items()), columns=['Metric', 'Value'])
        csv = report_df.to_csv(index=False).encode()
        st.download_button(
            label="Download Overview Report (CSV)",
            data=csv,
            file_name=f"overview_report_{selected_dataset}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📈 Export Sales Data"):
        # Export filtered sales data
        export_df = df.copy()
        export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
        csv = export_df.to_csv(index=False).encode()
        st.download_button(
            label="Download Sales Data (CSV)",
            data=csv,
            file_name=f"sales_data_{selected_dataset}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col3:
    if st.button("🔬 Export Analytics"):
        # Export analytics data
        analytics_data = []
        
        # Seasonality data
        if seasonality:
            monthly_data = df.copy()
            monthly_data['month'] = monthly_data['date'].dt.month
            monthly_sales = monthly_data.groupby('month')['sales'].sum().reset_index()
            monthly_sales['analysis_type'] = 'Seasonality'
            analytics_data.append(monthly_sales)
        
        # Anomaly data
        anomalies = detect_anomalies(df)
        if not anomalies.empty:
            anomaly_export = anomalies[['date', 'sales', 'z_score', 'anomaly_type']].copy()
            anomaly_export['analysis_type'] = 'Anomaly'
            analytics_data.append(anomaly_export)
        
        if analytics_data:
            combined_analytics = pd.concat(analytics_data, ignore_index=True)
            csv = combined_analytics.to_csv(index=False).encode()
            st.download_button(
                label="Download Analytics (CSV)",
                data=csv,
                file_name=f"analytics_{selected_dataset}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No analytics data to export")

with col4:
    if st.button("📋 Export All Data"):
        # Create a comprehensive Excel file with multiple sheets
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Overview metrics
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            metrics_df.to_excel(writer, sheet_name='Overview', index=False)
            
            # Sales data
            df_export = df.copy()
            df_export['date'] = df_export['date'].dt.strftime('%Y-%m-%d')
            df_export.to_excel(writer, sheet_name='Sales Data', index=False)
            
            # Monthly totals
            monthly_export = monthly_total.copy()
            monthly_export['date'] = monthly_export['date'].dt.strftime('%Y-%m-%d')
            monthly_export.to_excel(writer, sheet_name='Monthly Totals', index=False)
            
            # Product analysis
            if item_exists:
                prod_export = df.groupby('item', as_index=False)['sales'].sum().sort_values('sales', ascending=False)
                prod_export.to_excel(writer, sheet_name='Product Analysis', index=False)
        
        output.seek(0)
        st.download_button(
            label="Download Complete Report (Excel)",
            data=output.getvalue(),
            file_name=f"complete_report_{selected_dataset}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )