import pandas as pd
import plotly.express as px
import streamlit as st

# Load the data
file_path = 'data/supplements.csv'  # Ensure the CSV file is in the same directory as this script
df = pd.read_csv(file_path)

# Convert 'popularity' to numeric, forcing errors to NaN and then filling them with a default value (e.g., 1)
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(1)

# Create the treemap
fig = px.treemap(df,
                 path=['condition', 'sub_condition', 'type', 'supplement'],
                 values='popularity',
                 color='evidence',
                 hover_data=['link'],
                 color_continuous_scale='RdBu')

# Add hyperlink functionality on hover
fig.update_traces(
    hovertemplate='<b>%{label}</b><br>Evidence: %{color}<br>Link: <a href="%{customdata[0]}" target="_blank">%{customdata[0]}</a><extra></extra>',
    customdata=df[['link']]
)

# Streamlit app
st.title('Longevity Supplements Explorer')
st.plotly_chart(fig)

# Add an explanation of how to use the chart
st.markdown("""
**Instructions:**
- Hover over a rectangle to see more details.
- Click on the link in the tooltip to open it in a new window.
""")

# Initialize a placeholder for displaying the link
link_placeholder = st.empty()

# Add a callback function to handle clicks
if 'click_data' not in st.session_state:
    st.session_state.click_data = None

def display_click_data(click_data):
    if click_data:
        link = click_data['points'][0]['customdata'][0]
        st.session_state.click_data = link

fig.update_traces(
    customdata=df[['link']],
    selector=dict(type='treemap')
)

if st.session_state.click_data:
    link_placeholder.markdown(f"**Selected Link:** [Open Link]({st.session_state.click_data})")

# Capture click events
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# Display the selected link if available
if st.session_state.click_data:
    st.markdown(f"**Selected Link:** [Open Link]({st.session_state.click_data})")
