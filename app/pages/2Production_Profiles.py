import streamlit as st

from st_common import st_page_config, st_side_bar

st_page_config(layout="wide")
scenario = st_side_bar()

st.title("Production profiles")

