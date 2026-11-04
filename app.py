"""
Streamlit entry point.

Run from the repository root:
    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Manufacturing Quality & Predictive Maintenance",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
)

pages = st.navigation([
    st.Page("app_pages/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("app_pages/maintenance.py", title="Predictive Maintenance", icon=":material/build:"),
    st.Page("app_pages/quality.py", title="Part Quality Inspection", icon=":material/straighten:"),
    st.Page("app_pages/analytics.py", title="Analytics & Evaluation", icon=":material/insights:"),
    st.Page("app_pages/about.py", title="About the Project", icon=":material/info:"),
])
pages.run()
