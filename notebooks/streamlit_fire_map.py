import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_folium import st_folium
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Background
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #151e28 0%, #263449 100%);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 💬 Card Styles (NEW - Add this here)
st.markdown(
    """
    <style>
    .card-style {
        background: rgba(23, 29, 44, 0.94);
        border-radius: 24px;
        box-shadow: 0 6px 36px 0 rgba(0,0,0,0.45), 0 1.5px 8px 0 rgba(30,150,255,0.06);
        padding: 32px 30px 28px 30px;
        margin-bottom: 36px;
        border: 1px solid rgba(30,150,255,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Neon Title CSS
st.markdown("""
<style>
.neon-title {
    font-family: 'Share Tech Mono', 'Orbitron', 'Consolas', monospace;
    color: #00fff7;
    text-shadow: 0 0 8px #00fff7, 0 0 24px #00fff7, 0 0 4px #1a237e;
    letter-spacing: 2px;
    font-size: 2.6rem;
    animation: flicker 2s infinite alternate;
}
@keyframes flicker {
    0%   { opacity: 0.92; text-shadow: 0 0 12px #00fff7, 0 0 32px #00fff7; }
    100% { opacity: 1; text-shadow: 0 0 28px #00fff7, 0 0 64px #00fff7; }
}
</style>
""", unsafe_allow_html=True)

# Neon Title
st.markdown("<h1 class='neon-title'>🔥 Wildfire Map by Year</h1>", unsafe_allow_html=True)

# --- Sidebar: About / Info ---
with st.sidebar:
    st.markdown("""
    ## 🔥 Wildfire Intelligence Dashboard
    **Built by Avery Miller**
    
    _Canadian Wildfire Data (NFDB)_
    
    - Explore wildfires by year, size, and cause
    - See trends, visualize on interactive map
    - Chat with your AI copilot for instant analysis!
    
    [Source code on GitHub](#)
    """)
    st.markdown("---")
    st.write("**Crafted from data and Streamlit | Inspired by Blade Runner vibes**")


st.markdown(
    """
    <style>
    .map-container {
        background: rgba(22, 34, 51, 0.7);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 4px 32px 0 rgba(0,0,0,0.24);
        margin-bottom: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.caption("A Palantir-style portfolio app for Canadian wildfire analysis.")

st.divider()

# Load data
file_path = "data/raw/NFDB_point_20240613.txt"
df = pd.read_csv(file_path, encoding='latin1')

# Filter years (remove 0s or bad data)
valid_years = sorted(df['YEAR'].dropna().unique())
valid_years = [int(y) for y in valid_years if y > 1950 and y < 2025]

# Dropdown to pick a year
selected_year = st.selectbox("Select a year", valid_years, index=valid_years.index(2023))

# Province filter (based on SRC_AGENCY)
province_options = sorted(df['SRC_AGENCY'].dropna().unique())
selected_province = st.selectbox("Select a province/territory", ["All"] + province_options)

# Fire size filter
min_size, max_size = st.slider(
    "Select fire size range (in hectares)",
    min_value=0.0,
    max_value=float(df['SIZE_HA'].max()),
    value=(0.0, float(df['SIZE_HA'].max())),
    step=10.0
)

# 🔘 Large fires toggle — define it BEFORE filtering
large_fires_only = st.checkbox("Show only large fires (≥1000 ha)")

# Base filters: Year + Size
df_filtered = df[
    (df['YEAR'] == selected_year) &
    (df['SIZE_HA'] >= min_size) &
    (df['SIZE_HA'] <= max_size)
].dropna(subset=['LATITUDE', 'LONGITUDE'])

# Optional: Only show large fires
if large_fires_only:
    df_filtered = df_filtered[df_filtered['SIZE_HA'] >= 1000]

# Optional: Filter by province
if selected_province != "All":
    df_filtered = df_filtered[df_filtered['SRC_AGENCY'] == selected_province]

# Search bar for fire name
search_term = st.text_input("Search fire name (optional)").strip().lower()

# If a search term is provided, filter FIRENAME
if search_term:
    df_filtered = df_filtered[
        df_filtered['FIRENAME'].astype(str).str.lower().str.contains(search_term)
    ]

# Sample for performance
df_sampled = df_filtered.sample(n=min(1000, len(df_filtered)), random_state=42)

# Color logic
def get_color(size):
    if size < 10:
        return 'green'
    elif size < 1000:
        return 'orange'
    else:
        return 'red'
    
    # Mapping for cause codes
cause_map = {
    "H": "Human",
    "L": "Lightning",
    "N": "Natural",
    "U": "Unknown",
    "H-PB": "Prescribed Burn",
    "RE": "Re-ignition",
    "O": "Other"
}
    
# Summary stats for filtered fires
total_fires = len(df_filtered)
avg_size = round(df_filtered['SIZE_HA'].mean(), 2)
max_size = round(df_filtered['SIZE_HA'].max(), 2)
most_common_cause = df_filtered['CAUSE'].mode()[0] if not df_filtered.empty else "N/A"
decoded_common_cause = cause_map.get(most_common_cause, most_common_cause)

# --- Stat Cards with Icons (at the top) ---
st.markdown("""
<div style="display: flex; justify-content: space-between; gap: 18px;">
    <div style="background: #181D27; border-radius: 15px; box-shadow: 0 0 15px #f55, 0 0 1px #fff; padding: 18px 24px; min-width: 140px;">
        <h3 style="margin:0; color:#ff8400;">🔥 {}</h3>
        <p style="margin:0; color:#fff;">Total Fires</p>
    </div>
    <div style="background: #181D27; border-radius: 15px; box-shadow: 0 0 15px #0ff, 0 0 1px #fff; padding: 18px 24px; min-width: 140px;">
        <h3 style="margin:0; color:#00ffd0;">🌡️ {}</h3>
        <p style="margin:0; color:#fff;">Avg Size (ha)</p>
    </div>
    <div style="background: #181D27; border-radius: 15px; box-shadow: 0 0 15px #fd0, 0 0 1px #fff; padding: 18px 24px; min-width: 140px;">
        <h3 style="margin:0; color:#fff700;">🚨 {}</h3>
        <p style="margin:0; color:#fff;">Max Size (ha)</p>
    </div>
    <div style="background: #181D27; border-radius: 15px; box-shadow: 0 0 15px #0f0, 0 0 1px #fff; padding: 18px 24px; min-width: 140px;">
        <h3 style="margin:0; color:#19ff14;">⚡ {}</h3>
        <p style="margin:0; color:#fff;">Top Cause</p>
    </div>
</div>
""".format(total_fires, avg_size, max_size, decoded_common_cause), unsafe_allow_html=True)

st.divider()
# 📊 Fires by Province (Horizontal Bar Chart)
st.markdown("### 🔍 Fires by Province/Territory")

province_counts = df_filtered['SRC_AGENCY'].value_counts().sort_values(ascending=True)

fig1, ax1 = plt.subplots(figsize=(8, 6))
sns.barplot(x=province_counts.values, y=province_counts.index, palette="flare", ax=ax1)
ax1.set_xlabel("Number of Fires")
ax1.set_ylabel("Province / Agency")
ax1.set_title("Wildfires Reported by Province (Filtered Selection)")
st.pyplot(fig1)

st.divider()
# 📈 Wildfire Causes Over Time (Line Chart)
st.markdown("### 📈 Wildfire Causes Over Time")

# Simplify causes using cause_map (already defined earlier)
df_filtered['CAUSE_SIMPLE'] = df_filtered['CAUSE'].map(cause_map).fillna("Other")

# Use full dataset (not df_filtered) to show causes over all years
df_cause_chart = df.copy()
if selected_province != "All":
    df_cause_chart = df_cause_chart[df_cause_chart['SRC_AGENCY'] == selected_province]

df_cause_chart['CAUSE_SIMPLE'] = df_cause_chart['CAUSE'].map(cause_map).fillna("Other")
cause_trend = df_cause_chart.groupby(['YEAR', 'CAUSE_SIMPLE']).size().reset_index(name='count')

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.lineplot(data=cause_trend, x='YEAR', y='count', hue='CAUSE_SIMPLE', marker="o", ax=ax2)
ax2.set_title("Wildfire Causes by Year (Filtered View)")
ax2.set_xlabel("Year")
ax2.set_ylabel("Number of Fires")
ax2.legend(title="Cause")
st.pyplot(fig2)

st.divider()
# 🗺️ Wildfire Map (Streamlit + Folium)
st.markdown("### 🗺️ Wildfire Map")

m = folium.Map(location=[56.1304, -106.3468], zoom_start=4, tiles="CartoDB dark_matter")
marker_cluster = MarkerCluster().add_to(m)

for _, row in df_sampled.iterrows():
    color = get_color(row['SIZE_HA'])
    cause_code = row['CAUSE']
    decoded_cause = cause_map.get(cause_code, cause_code)
    tooltip_text = f"Size: {row['SIZE_HA']} ha, Cause: {decoded_cause}"

    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=4 if row['SIZE_HA'] < 10 else 6 if row['SIZE_HA'] < 1000 else 8,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        tooltip=tooltip_text,
        popup=folium.Popup(
            f"""
            <b>Fire Name:</b> {row['FIRENAME']}<br>
            <b>Province:</b> {row['SRC_AGENCY']}<br>
            <b>Size:</b> {row['SIZE_HA']} ha<br>
            <b>Cause:</b> {decoded_cause}
            """,
            max_width=300
        )
    ).add_to(marker_cluster)

st.markdown('<div class="map-container">', unsafe_allow_html=True)
st_data = st_folium(m, width=1000, height=600)
st.markdown('</div>', unsafe_allow_html=True)

st.divider()
# 📁 Export CSV Button
st.markdown("### 📁 Export Filtered Data")

if not df_filtered.empty:
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"wildfires_filtered_{selected_year}.csv",
        mime='text/csv',
    )
else:
    st.warning("No data to export with current filters.")

# 💡 Sample Questions Expander (add this FIRST)
with st.expander("💡 Sample Questions You Can Ask the Copilot", expanded=False):
    st.markdown("""
    - How many fires were there in Ontario in 1999?
    - What was the largest fire in Alberta in 2023?
    - Show average fire size in British Columbia for 2015.
    - List top 5 years by number of fires in Quebec.
    - Which province had the most natural-caused fires in 2022?
    """)

st.divider()
# 🔍 Chat with Wildfire Copilot
st.markdown("### 🤖 Ask the Wildfire Copilot")

# Improved, explicit system instruction
system_prompt = (
    "You are an expert Canadian wildfire data analyst. The user may ask for any statistics, counts, or trends "
    "in the wildfire dataset provided as a pandas DataFrame. The main columns are: "
    "NFDBFIREID (fire ID), SRC_AGENCY (province code), YEAR, MONTH, DAY, LATITUDE, LONGITUDE, "
    "FIRENAME, SIZE_HA (hectares), CAUSE, and more. "
    "Always use the actual DataFrame to answer questions—never guess. "
    "If the user asks for fires in a certain province, check 'SRC_AGENCY'. For year, check 'YEAR'."
)

@st.cache_resource
def create_agent():
    return create_pandas_dataframe_agent(
        ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=st.secrets["gemini_api_key"]
        ),
        df,  # <-- Use the whole dataset for best results
        verbose=False,
        allow_dangerous_code=True,
        system_prompt=system_prompt
    )

agent = create_agent()

user_query = st.text_input("Ask a question about the wildfire dataset")

if user_query:
    with st.spinner("Thinking..."):
        try:
            response = agent.run(user_query)
            st.success(response)
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.markdown("---")
st.markdown(
    "<center><sub>© 2025 Avery Miller • Powered by Streamlit, Gemini AI, and Open Wildfire Data</sub></center>",
    unsafe_allow_html=True
)
# --- Footer ---
st.markdown("""
---
<center>
<span style='color:#ff1c87; font-family:monospace; font-size:18px;'>
&#x1F50C; Data is electric. Stay curious, stay wild. <br>
<i>"More human than human is our motto."</i>
</span>
</center>
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: rgba(20, 20, 40, 0.90);
        color: #ff9800;
        text-align: center;
        font-size: 16px;
        padding: 0.5rem 0;
        letter-spacing: 1px;
        z-index: 100;
        font-family: 'Share Tech Mono', 'Orbitron', 'Consolas', monospace;
        border-top: 1px solid #333955;
        box-shadow: 0px -2px 16px 0px #1a237e22;
    }
    </style>
    <div class="footer">
        RainCode AI – Canadian Wildfire Data &nbsp; | &nbsp; 2024
    </div>
    """,
    unsafe_allow_html=True
)
