import streamlit as st

st.set_page_config(
    page_title="AIC Capital",
    page_icon="📊",
    layout="wide",
)

st.title("Welcome to the AIC Capital Dashboard")

repo_url = "https://github.com/aachen-investment-club"

st.markdown(
    "Check the other repositories of the AIC members out and leave a star! [[Go to Github]](%s)" % repo_url
)