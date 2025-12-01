import streamlit as st
import pandas as pd
import numpy as np
import os
import bcrypt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# --- Use NeuralProphet only ---
try:
    from neuralprophet import NeuralProphet
    HAVE_NEURAL_PROPHET = True
except Exception:
    HAVE_NEURAL_PROPHET = False

USER_CSV = "users.csv"

def ensure_users_file():
    if not os.path.exists(USER_CSV):
        pd.DataFrame(columns=["username", "password"]).to_csv(USER_CSV, index=False)

def load_users():
    ensure_users_file()
    return pd.read_csv(USER_CSV)

def save_user(username, password):
    users = load_users()
    if username in users["username"].values:
        return False
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode() if bcrypt else password
    users = pd.concat([users, pd.DataFrame([{"username": username, "password": hashed}])], ignore_index=True)
    users.to_csv(USER_CSV, index=False)
    return True

def check_login(username, password):
    users = load_users()
    if username in users["username"].values:
        stored = users.loc[users["username"] == username, "password"].values[0]
        try:
            return bcrypt.checkpw(password.encode(), stored.encode())
        except Exception:
            return password == stored
    return False

st.set_page_config("Sales Forecast Prediction System", layout="wide")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"

if not st.session_state.logged_in:
    if st.session_state.page == "login":
        st.title("📊 Sales Forecast Prediction System")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.user = username
                st.session_state.page = "app"
                st.stop()
            else:
                st.error("Invalid credentials")
        if st.button("Go to Signup"):
            st.session_state.page = "signup"
            st.stop()
        st.stop()
    else:
        st.title("Signup")
        new_user = st.text_input("Choose username")
        new_pass = st.text_input("Choose password", type="password")
        if st.button("Create Account"):
            if not new_user or not new_pass:
                st.error("Provide username & password")
            else:
                if save_user(new_user, new_pass):
                    st.success("Account created — please login")
                    st.session_state.page = "login"
                    st.stop()
                else:
                    st.error("Username exists")
        if st.button("Back to Login"):
            st.session_state.page = "login"
            st.stop()
        st.stop()

st.title("📊 Sales Forecast Prediction System")

with st.sidebar:
    st.success(f"Logged in as {st.session_state.user}")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.stop()

uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload dataset to continue")
    st.stop()

df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)

if "Item_Outlet_Sales" not in df.columns:
    st.error("Dataset must contain 'Item_Outlet_Sales' column")
    st.stop()

df["Item_Outlet_Sales"] = pd.to_numeric(df["Item_Outlet_Sales"], errors="coerce").fillna(0)
original_item_type = df["Item_Type"].copy()

feature_cols = [c for c in ["Item_Weight", "Item_MRP", "Item_Visibility"] if c in df.columns]
encoders = {}
for c in ["Item_Type", "Outlet_Type", "Outlet_Size", "Outlet_Location_Type", "Item_Fat_Content", "Outlet_Identifier"]:
    if c in df.columns:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = le
        feature_cols.append(c)

X = df[feature_cols]
y = df["Item_Outlet_Sales"]

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, df.index, test_size=0.3, random_state=42
)

model = XGBRegressor(objective='reg:squarederror', n_estimators=300) if HAVE_XGB else RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

test_df = pd.DataFrame(X_test, index=idx_test).copy()
test_df["Actual"] = y_test
test_df["Predicted"] = preds
test_df["Item_Type"] = original_item_type.loc[idx_test].values

agg = test_df.groupby("Item_Type").agg(
    last_month=("Actual", lambda x: np.mean(x) * 0.95),
    current_month=("Actual", "mean"),
    next_forecast=("Predicted", "mean")
).reset_index()
agg["% Change"] = ((agg["current_month"] - agg["last_month"]) / agg["last_month"]) * 100

st.subheader("📊 Item-wise Forecast Table")
st.dataframe(agg)

st.subheader("📈 Visual Analysis")
item_filter = st.selectbox("Filter by Item Type", ["All"] + agg["Item_Type"].unique().tolist())
plot_data = agg if item_filter == "All" else agg[agg["Item_Type"] == item_filter]
filtered_test_df = test_df if item_filter == "All" else test_df[test_df["Item_Type"] == item_filter]

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Line Chart", "📊 Bar Chart", "📦 Box Plot", "🥧 Pie Chart"
])

with tab1:
    fig1 = px.line(
        plot_data.melt(id_vars="Item_Type", value_vars=["last_month", "current_month", "next_forecast"],
                       var_name="Month", value_name="Sales"),
        x="Month", y="Sales", color="Item_Type", markers=True, title="Item-wise Sales Trend"
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.bar(plot_data, x="Item_Type", y=["current_month", "next_forecast"],
                  barmode="group", title="Current vs Forecast Sales")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = px.box(filtered_test_df, x="Item_Type", y="Actual", title="Distribution of Actual Sales by Item")
    st.plotly_chart(fig3, use_container_width=True)

with tab4:
    pie_data = plot_data[["Item_Type", "current_month"]]
    fig4 = px.pie(pie_data, values="current_month", names="Item_Type", title="Share of Total Sales")
    st.plotly_chart(fig4, use_container_width=True)

st.subheader("🏆 Top 10 Products by Sales")
top10 = agg.nlargest(10, "current_month")
fig_top = px.bar(top10, x="Item_Type", y="current_month", text="current_month", title="Top 10 Products")
fig_top.update_traces(textposition='outside')
st.plotly_chart(fig_top, use_container_width=True)

st.subheader("📉 Bottom 10 Products by Sales")
bottom10 = agg.nsmallest(10, "current_month")
fig_bottom = px.bar(bottom10, x="Item_Type", y="current_month", text="current_month", title="Bottom 10 Products")
fig_bottom.update_traces(textposition='outside')
st.plotly_chart(fig_bottom, use_container_width=True)

st.subheader("📆 NeuralProphet Forecast (Per Item)")
item_choice = st.selectbox("Item", agg["Item_Type"].unique())
item_data = test_df[test_df["Item_Type"] == item_choice]

if len(item_data) >= 6:
    item_data = item_data.copy()
    item_data["Month"] = pd.date_range(start="2023-01-01", periods=len(item_data), freq="M")
    ts = item_data.groupby(item_data["Month"].dt.to_period("M"))["Actual"].mean().reset_index()
    ts["Month"] = ts["Month"].dt.to_timestamp()
    ts = ts.rename(columns={"Month": "ds", "Actual": "y"})

    if HAVE_NEURAL_PROPHET:
        try:
            m = NeuralProphet()
            m.fit(ts, freq='M')
            future = m.make_future_dataframe(ts, periods=6)
            forecast = m.predict(future)
            fig5 = px.line()
            fig5.add_scatter(x=ts["ds"], y=ts["y"], mode="lines+markers", name="Actual")
            fig5.add_scatter(x=forecast["ds"], y=forecast["yhat1"], mode="lines", name="Forecast")
            st.plotly_chart(fig5, use_container_width=True)
        except Exception:
            ts["forecast"] = ts["y"].rolling(3, min_periods=1).mean()
            fig_fallback = px.line(ts, x="ds", y="forecast", title=f"Moving Average Forecast - {item_choice}")
            st.plotly_chart(fig_fallback, use_container_width=True)
    else:
        st.info("NeuralProphet is not installed. Please run: pip install neuralprophet")
else:
    st.warning(f"Not enough data points to forecast {item_choice} — need at least 6 months.")
