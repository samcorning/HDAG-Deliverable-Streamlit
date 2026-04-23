import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="CT Affordable Housing", page_icon="🏠", layout="wide")

st.title("🏠 Connecticut Affordable Housing Explorer")
st.caption("Data: Affordable Housing by Town, 2011–2022")
st.markdown("---")

@st.cache_data
def load_data():
    df = pd.read_csv("Affordable_Housing_by_Town_2011-2022.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Percent Affordable": "Pct_Affordable",
        "Total Assisted Units": "Total_Assisted",
        "2010 Census Units": "Total_Units",
        "Government Assisted": "Gov_Assisted",
        "Tenant Rental Assistance": "Rental_Assist",
        "Single Family CHFA/ USDA Mortgages": "Mortgages",
        "Deed Restricted Units": "Deed_Restricted",
    })
    return df

df = load_data()

# ── SIDEBAR ──────────────────────────────────────────────────────────────────────
st.sidebar.header("Filters & Options")

# Feature 1: Year selector
all_years = sorted(df["Year"].unique())
selected_year = st.sidebar.selectbox("📅 Select Year", all_years, index=len(all_years)-2)

# Feature 2: Min % affordable slider
min_pct = st.sidebar.slider(
    "🔍 Minimum % Affordable",
    min_value=0.0,
    max_value=float(df["Pct_Affordable"].max()),
    value=0.0,
    step=1.0,
)

# Feature 3: Top N towns
top_n = st.sidebar.slider("📊 Top N Towns (bar chart)", min_value=5, max_value=30, value=15)

# Feature 4: Bar color
color_choice = st.sidebar.radio("🎨 Chart color", ["Blue", "Green", "Orange"], horizontal=True)
bar_color = {"Blue": "steelblue", "Green": "seagreen", "Orange": "darkorange"}[color_choice]

st.sidebar.markdown("---")
st.sidebar.info("Explore affordable housing trends across 169 Connecticut towns from 2011 to 2022.")

# ── Filter ───────────────────────────────────────────────────────────────────────
year_df = df[(df["Year"] == selected_year) & (df["Pct_Affordable"] >= min_pct)].copy()

# ── KPI Row ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Towns shown", len(year_df))
k2.metric("Avg % Affordable", f"{year_df['Pct_Affordable'].mean():.1f}%")
k3.metric("Most affordable", year_df.loc[year_df["Pct_Affordable"].idxmax(), "Town"] if len(year_df) > 0 else "—")
k4.metric("Total Assisted Units", f"{year_df['Total_Assisted'].sum():,}")

st.markdown("---")

col1, col2 = st.columns(2)

# ── Bar chart ────────────────────────────────────────────────────────────────────
with col1:
    st.subheader(f"Top {top_n} Towns by % Affordable ({selected_year})")
    top_towns = year_df.nlargest(top_n, "Pct_Affordable")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top_towns["Town"][::-1], top_towns["Pct_Affordable"][::-1], color=bar_color)
    ax.set_xlabel("% Affordable Housing")
    ax.axvline(x=10, color="red", linestyle="--", linewidth=1, label="10% goal")
    ax.legend(fontsize=8)
    ax.set_title(f"Top {top_n} Towns — {selected_year}")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Linear regression ────────────────────────────────────────────────────────────
with col2:
    st.subheader(f"Town Size vs. % Affordable ({selected_year})")

    plot_df = year_df.dropna(subset=["Total_Units", "Pct_Affordable"])
    plot_df = plot_df[plot_df["Total_Units"] > 0]

    X = plot_df["Total_Units"].values.reshape(-1, 1)
    y = plot_df["Pct_Affordable"].values

    model = LinearRegression()
    model.fit(X, y)
    r2 = model.score(X, y)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(X, y, alpha=0.6, color=bar_color, edgecolors="white", linewidths=0.5)
    x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    ax2.plot(x_line, model.predict(x_line), color="red", linewidth=2, label=f"R² = {r2:.3f}")
    ax2.set_xlabel("Total Housing Units (2010 Census)")
    ax2.set_ylabel("% Affordable Housing")
    ax2.set_title(f"Linear Regression — {selected_year}")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


st.markdown("---")

