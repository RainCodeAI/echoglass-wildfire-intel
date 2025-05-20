import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_folium import st_folium
import os
import plotly.express as px  # <-- Add this here
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Background
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #151e28 0%, #263449 100%);
    }
    .sticky-controls {
        background: rgba(21,30,40,0.95);
        padding: 20px;
        border-radius: 16px;
        box-shadow: 0 0 16px rgba(0, 255, 255, 0.15);
        position: sticky;
        top: 0;
        z-index: 999;
        margin-bottom: 24px;
    }

    /* 🔮 ADD THIS HERE ↓↓↓ */
    .glow-title {
        color: #ffffff;
        font-family: 'Orbitron', sans-serif;
        font-size: 1.4rem;
        transition: all 0.3s ease-in-out;
    }
    .glow-title:hover {
        color: #00fff7;
        text-shadow: 0 0 8px #00fff7, 0 0 18px #00fff7;
        cursor: default;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Card Styles
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

# Neon Title
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

st.markdown("<h1 class='neon-title'>🔥 Wildfire Map by Year</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
.glow-title {
    color: #ffffff;
    font-family: 'Orbitron', sans-serif;
    font-size: 1.4rem;
    transition: all 0.3s ease-in-out;
}
.glow-title:hover {
    color: #00fff7;
    text-shadow: 0 0 8px #00fff7, 0 0 18px #00fff7;
    cursor: default;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.section-title {
    position: relative;
    display: inline-block;
    font-weight: 700;
    font-family: 'Orbitron', sans-serif;
    color: #ffffff;
    margin-bottom: 12px;
}
.section-title::after {
    content: "";
    position: absolute;
    left: 0;
    bottom: -4px;
    width: 100%;
    height: 3px;
    background: linear-gradient(90deg, #00fff7, #ff00c8, #00fff7);
    background-size: 200% 100%;
    animation: glow-underline 3s linear infinite;
    border-radius: 4px;
}
@keyframes glow-underline {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

/* Neon-style spinner */
div[data-testid="stSpinner"] > div {
    color: #00fff7 !important;
    font-family: 'Orbitron', sans-serif;
    font-size: 1.1rem;
    text-shadow: 0 0 4px #00fff7, 0 0 8px #00fff7;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.sticky-filters {
    position: sticky;
    top: 0;
    background-color: #151e28ee;
    padding: 1rem 1rem 0.5rem 1rem;
    z-index: 99;
    border-bottom: 1px solid #444;
    backdrop-filter: blur(4px);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Existing styles */

.sticky-filters {
    position: sticky;
    top: 72px;
    background-color: #1b263b;
    padding: 16px;
    z-index: 999;
    border-radius: 10px;
    border: 1px solid #30455c;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2);
    animation: fadeSlideIn 0.8s ease forwards;
    opacity: 0;
    transform: translateY(-12px);
}

@keyframes fadeSlideIn {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Dark scrollbars for WebKit browsers */
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-track {
    background: #1c2735;
}
::-webkit-scrollbar-thumb {
    background: #44536b;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #607d8b;
}

/* Styled buttons */
.stButton > button {
    background-color: #1f3b57;
    color: white;
    border: 1px solid #3a506b;
    padding: 0.5rem 1.25rem;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #2e5d89;
    border-color: #4a90e2;
    color: #fff;
    box-shadow: 0 0 10px #4a90e2;
}

/* Inputs glow on hover */
.stSelectbox:hover, .stSlider:hover, .stTextInput:hover {
    box-shadow: 0 0 8px #00fff7;
    border: 1px solid #00fff7;
    transition: all 0.3s ease-in-out;
}

/* ✨ Chart Fade-In Animation */
.chart-section {
    opacity: 0;
    transform: translateY(20px);
    animation: fadeSlideIn 1s ease-out forwards;
    animation-delay: 0.3s;
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar Info ---
with st.sidebar:
    st.markdown("## 📊 About This App")
    st.markdown("""
Welcome to the **RainCode AI Wildfire Dashboard** — a Palantir-inspired data intelligence app that lets you explore, analyze, and interact with real Canadian wildfire data.

**🧠 Built by:** Avery Miller  
**📅 Dataset:** National Fire Database (NFDB)  
**🛠️ Tech:** Streamlit · LangChain · Gemini API · Pandas · Plotly · Folium  
**📍 Scope:** 1950–2023 (all provinces & territories)

---

This app lets you:
- 🔎 Filter fires by year, province, size, and cause  
- 🗺️ Visualize fires on an interactive map  
- 📈 See trends by region and year  
- 🤖 Ask natural language questions using the AI Copilot

Stay curious, stay wild. 🌲🔥
""")

import random

wildfire_facts = [
    "Canada experiences over 8,000 wildfires annually.",
    "Lightning causes ~45% of fires but burns ~85% of the area.",
    "2023 was Canada's worst wildfire season on record.",
    "Prescribed burns reduce the risk of mega-fires.",
    "Wind is one of the most unpredictable wildfire factors."
]

st.markdown("---")
st.info("💡 Did you know? " + random.choice(wildfire_facts))

# Load data
file_path = "../data/raw/NFDB_point_20240613.txt"
df = pd.read_csv(file_path, encoding='latin1')

# --- Filters ---
valid_years = sorted(df['YEAR'].dropna().unique())
valid_years = [int(y) for y in valid_years if y > 1950 and y < 2025]
province_options = sorted(df['SRC_AGENCY'].dropna().unique())

# 🔹 Sticky UI Wrapper for Main Filters
st.markdown("""<div class="sticky-controls">""", unsafe_allow_html=True)
st.markdown('<div class="sticky-filters">', unsafe_allow_html=True)

with st.container():
    year_range = st.slider(
        "Select year range",
        min_value=min(valid_years),
        max_value=max(valid_years),
        value=(2020, 2023),  # or set any default you like
        step=1,
        key="year_slider_range",
        help="Choose a range of years to display wildfires for"
    )
    start_year, end_year = year_range

selected_province = st.selectbox(
    "Select a province/territory",
    ["All"] + province_options,
    key="province_select_sticky",
    help="Filter wildfires by province or territory"
)

min_size, max_size = st.slider(
    "Select fire size range (in hectares)",
    min_value=0.0,
    max_value=float(df['SIZE_HA'].max()),
    value=(0.0, float(df['SIZE_HA'].max())),
    step=10.0,
    key="size_slider_sticky",
    help="Filter fires by their size in hectares"
)

with st.expander("⚙️ Advanced Filter Options"):
    large_fires_only = st.checkbox(
        "Show only large fires (≥1000 ha)",
        key="large_fires_checkbox",
        help="Toggle to show only major wildfires over 1000 hectares"
    )

search_term = st.text_input(
    "Search fire name (optional)",
    key="search_input",
    help="Type part of a fire name to filter results"
)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 🔍 Filter Logic
df_filtered = df[
    (df['YEAR'] >= start_year) & (df['YEAR'] <= end_year) &
    (df['SIZE_HA'] >= min_size) & (df['SIZE_HA'] <= max_size)
].dropna(subset=['LATITUDE', 'LONGITUDE']).copy()

if large_fires_only:
    df_filtered = df_filtered[df_filtered['SIZE_HA'] >= 1000]

if selected_province != "All":
    df_filtered = df_filtered[df_filtered['SRC_AGENCY'] == selected_province]

# 🎯 Sample for performance
df_sampled = df_filtered.sample(n=min(1000, len(df_filtered)), random_state=42)

# Color function
def get_color(size):
    if size < 10:
        return 'green'
    elif size < 1000:
        return 'orange'
    else:
        return 'red'

cause_map = {
    "H": "Human",
    "L": "Lightning",
    "N": "Natural",
    "U": "Unknown",
    "H-PB": "Prescribed Burn",
    "RE": "Re-ignition",
    "O": "Other"
}

# --- Stat Cards ---
total_fires = len(df_filtered)
avg_size = round(df_filtered['SIZE_HA'].mean(), 2)
max_size = round(df_filtered['SIZE_HA'].max(), 2)
most_common_cause = df_filtered['CAUSE'].mode()[0] if not df_filtered.empty else "N/A"
decoded_common_cause = cause_map.get(most_common_cause, most_common_cause)

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

# Year-over-year trend calculation
fires_by_year = df_filtered.groupby('YEAR').size().reset_index(name='count').sort_values('YEAR')

if len(fires_by_year) >= 2:
    last_year_count = fires_by_year.iloc[-2]['count']
    this_year_count = fires_by_year.iloc[-1]['count']
    yoy_change = this_year_count - last_year_count
    yoy_percent = (yoy_change / last_year_count) * 100 if last_year_count != 0 else 0

    trend_arrow = "🔺" if yoy_change > 0 else "🔻" if yoy_change < 0 else "➡️"
    trend_color = "#00e676" if yoy_change < 0 else "#ff1744" if yoy_change > 0 else "#cccccc"

    st.markdown(
        f"<div style='margin-top: 12px; font-size: 1.1rem; color: {trend_color}; font-family: Orbitron;'>"
        f"{trend_arrow} Year-over-year change: <strong>{yoy_change:+,}</strong> fires "
        f"({yoy_percent:+.1f}%)</div>",
        unsafe_allow_html=True
    )

elif len(fires_by_year) == 1:
    st.warning("Only 1 year in selection. Add another year to see trend.")
else:
    st.warning("No data available for selected range.")

# 📈 Year-over-Year Trend Summary
try:
    previous_year = selected_year - 1
    prev_year_count = df[df["YEAR"] == previous_year].shape[0]
    curr_year_count = df[df["YEAR"] == selected_year].shape[0]
    delta = curr_year_count - prev_year_count
    trend_emoji = "📈" if delta >= 0 else "📉"
    trend_color = "#00e676" if delta >= 0 else "#ff5252"
    
    st.markdown(f"""
    <div style="background: #10141c; padding: 16px 20px; border-left: 5px solid {trend_color}; border-radius: 10px; margin-top: 18px;">
        <h4 style="color: {trend_color}; font-family: 'Orbitron', sans-serif;">{trend_emoji} Fire Trend vs {previous_year}</h4>
        <p style="margin: 4px 0; color: white;">{selected_year}: <strong>{curr_year_count:,}</strong> fires</p>
        <p style="margin: 4px 0; color: white;">{previous_year}: <strong>{prev_year_count:,}</strong> fires</p>
        <p style="margin: 4px 0; color: white;">Change: <strong style="color: {trend_color};">{delta:+,}</strong></p>
    </div>
    """, unsafe_allow_html=True)
except Exception as e:
    pass  # or optionally log the error if you'd like

# 🔥 Top 5 Largest Fires (Filtered)
st.markdown("<div class='chart-section'><h3 class='glow-title'>🏆 Top 5 Largest Fires</h3></div>", unsafe_allow_html=True)

if not df_filtered.empty:
    top_fires = df_filtered.sort_values(by='SIZE_HA', ascending=False).head(5)
    top_fires_display = top_fires[['FIRENAME', 'SIZE_HA', 'SRC_AGENCY', 'CAUSE']].copy()
    top_fires_display['CAUSE'] = top_fires_display['CAUSE'].map(cause_map).fillna("Unknown")
    top_fires_display.rename(columns={
        'FIRENAME': 'Fire Name',
        'SIZE_HA': 'Size (ha)',
        'SRC_AGENCY': 'Province',
        'CAUSE': 'Cause'
    }, inplace=True)
    
    st.dataframe(top_fires_display.style
        .highlight_max(axis=0, color='#ff6961')
        .set_properties(**{'background-color': '#10141c', 'color': 'white'})
        .format({'Size (ha)': '{:,.1f}'})
    )
else:
    st.info("No data available for current filters.")

st.divider()

# Province Bar Chart (Plotly)
st.markdown("<div class='chart-section'><h3 class='glow-title'>📊 Fires by Province</h3></div>", unsafe_allow_html=True)

province_counts = df_filtered['SRC_AGENCY'].value_counts().reset_index()
province_counts.columns = ['Province', 'Number of Fires']

with st.spinner("Loading interactive chart..."):
    fig1 = px.bar(
        province_counts,
        x='Number of Fires',
        y='Province',
        orientation='h',
        title="Wildfires Reported by Province (Filtered Selection)",
        color='Number of Fires',
        color_continuous_scale='YlOrRd'
    )
    fig1.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig1, use_container_width=True)

st.divider()

# Cause Trend Chart
st.markdown("<div class='chart-section'><h3 class='glow-title'>📈 Wildfire Causes Over Time</h3></div>", unsafe_allow_html=True)

df_filtered['CAUSE_SIMPLE'] = df_filtered['CAUSE'].map(cause_map).fillna("Other")

df_cause_chart = df.copy()
if selected_province != "All":
    df_cause_chart = df_cause_chart[df_cause_chart['SRC_AGENCY'] == selected_province]

df_cause_chart['CAUSE_SIMPLE'] = df_cause_chart['CAUSE'].map(cause_map).fillna("Other")
cause_trend = df_cause_chart.groupby(['YEAR', 'CAUSE_SIMPLE']).size().reset_index(name='count')

with st.spinner("Loading cause trends..."):
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=cause_trend, x='YEAR', y='count', hue='CAUSE_SIMPLE', marker="o", ax=ax2)
    ax2.set_title("Wildfire Causes by Year")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Number of Fires")
    ax2.legend(title="Cause")
    st.pyplot(fig2)


# 🔥 Fire Cause Breakdown (Pie Chart)
st.markdown("<div class='chart-section'><h3 class='glow-title'>🔥 Fire Causes Breakdown</h3></div>", unsafe_allow_html=True)

cause_counts = df_filtered['CAUSE'].value_counts().reset_index()
cause_counts.columns = ['Cause Code', 'Count']
cause_counts['Cause'] = cause_counts['Cause Code'].map(cause_map).fillna("Other")

fig_cause_pie = px.pie(
    cause_counts,
    names='Cause',
    values='Count',
    title='Proportion of Fire Causes (Filtered Selection)',
    color_discrete_sequence=px.colors.sequential.RdBu
)
fig_cause_pie.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font_color='white'
)
st.plotly_chart(fig_cause_pie, use_container_width=True)

st.divider()

# 🔄 Animated Fire Size Over Time
st.markdown("<div class='chart-section'><h3 class='glow-title'>🔄 Fire Size Distribution by Year</h3></div>", unsafe_allow_html=True)

with st.spinner("Rendering animated chart..."):
    size_anim_df = df.copy()
    size_anim_df = size_anim_df[size_anim_df['SIZE_HA'] > 0]
    size_anim_df = size_anim_df[size_anim_df['YEAR'].between(1950, 2023)].sort_values(by="YEAR")

    size_anim = px.histogram(
        size_anim_df,
        x="SIZE_HA",
        nbins=60,
        animation_frame="YEAR",
        range_x=[0, 5000],
        title="Fire Size Distribution (ha) Over Time",
        color_discrete_sequence=["#ff4d4d"]  # vivid red fire glow
    )

    size_anim.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis_title="Fire Size (ha)",
        yaxis_title="Number of Fires"
    )

    st.plotly_chart(size_anim, use_container_width=True)
st.divider()

# 🗺️ Wildfire Map
st.markdown("<div class='chart-section'><h3 class='glow-title'>🗺️ Wildfire Map</h3></div>", unsafe_allow_html=True)

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
            f"<b>Fire Name:</b> {row['FIRENAME']}<br><b>Province:</b> {row['SRC_AGENCY']}<br><b>Size:</b> {row['SIZE_HA']} ha<br><b>Cause:</b> {decoded_cause}",
            max_width=300
        )
    ).add_to(marker_cluster)

with st.spinner("Rendering wildfire map..."):
    st_data = st_folium(m, width=1000, height=600)

st.divider()

# 📁 Export CSV
st.markdown("<div class='chart-section'><h3 class='glow-title'>📁 Export Filtered Data</h3></div>", unsafe_allow_html=True)

if not df_filtered.empty:
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"wildfires_filtered_{start_year}_{end_year}.csv",
        mime='text/csv',
    )
else:
    st.warning("No data to export with current filters.")

st.divider()

# 💡 Sample Questions Expander
with st.expander("💡 **_Sample Questions You Can Ask the Copilot_**", expanded=False):
    st.markdown("""
    - How many fires were there in Ontario in 1999?
    - What was the largest fire in Alberta in 2023?
    - Show average fire size in British Columbia for 2015.
    - List top 5 years by number of fires in Quebec.
    - Which province had the most natural-caused fires in 2022?
    """)

st.caption("✅ Wildfire dataset successfully loaded and ready for Copilot.")

# 🤖 Copilot (Upgraded to actually answer correctly!)
st.markdown("<div class='chart-section'><h3 class='glow-title'>🤖 Ask the Wildfire Copilot</h3></div>", unsafe_allow_html=True)

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Prompt and model setup
system_prompt = (
    "You are an expert Canadian wildfire analyst using a DataFrame named 'df'. "
    "Always base answers on the data provided below, and never guess. "
    "Respond with clear summaries using counts, averages, or sorted lists. "
    "Here are the column meanings:\n"
    "- YEAR: year of fire\n"
    "- SRC_AGENCY: province/territory\n"
    "- SIZE_HA: size in hectares\n"
    "- CAUSE: fire cause code (H=Human, L=Lightning, etc.)\n\n"
    "ONLY answer from the data provided below. If it's not in the data, say so politely."
)

# Agent setup
def create_agent():
    return create_pandas_dataframe_agent(
        ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=st.secrets["gemini_api_key"]
        ),
        df,
        verbose=False,
        allow_dangerous_code=True,
        system_prompt=system_prompt
    )

agent = create_agent()

user_query = st.text_input("Ask a question about wildfires")

response = None  # Make sure response is initialized

if user_query:
    with st.spinner("Thinking..."):
        try:
            preview_rows = df[['YEAR', 'SRC_AGENCY', 'FIRENAME', 'SIZE_HA', 'CAUSE']].dropna().sample(n=min(30, len(df)), random_state=1)
            preview_text = preview_rows.to_markdown(index=False)

            full_prompt = f"""
Here are sample rows from the dataset:

{preview_text}

Now answer this question based on the full dataset columns (assume full data is consistent with these examples):
{user_query}
"""

            response = agent.run(full_prompt)

            # 🔥 Custom-styled response
            st.markdown(
                f"<div style='background-color:#003c3c;padding:12px 16px;border-radius:10px;margin-top:10px;color:#00ffd0;'>"
                f"{response}"
                f"</div>", unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error: {e}")

st.divider()

# --- Copilot Q&A History ---
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# Save successful response to history
if user_query and response:
    st.session_state.qa_history.append((user_query, response))
    # Limit to last 5
    st.session_state.qa_history = st.session_state.qa_history[-5:]

# Show history
if st.session_state.qa_history:
    st.markdown("### 🧠 Previous Questions")
    for i, (q, a) in enumerate(reversed(st.session_state.qa_history), 1):
        with st.expander(f"Q{i}: {q}"):
            st.markdown(a)

st.divider()

# Footer
st.markdown("---")
st.markdown(
    "<center><sub>© 2025 Avery Miller • Powered by Streamlit, Gemini AI, and Open Wildfire Data</sub></center>",
    unsafe_allow_html=True
)
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
