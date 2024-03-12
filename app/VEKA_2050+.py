import streamlit as st

from st_common import st_page_config
from st_common import st_side_bar

st_page_config(layout="wide")
st_side_bar()

st.title("VEKA 2050+ results explorers")

st.markdown(
    """
    This notebook allows you to the explore the results of `VEKA 2050+`. Please refer to documentation for details.
    
    The following informations are available:
    - Loads
    - Capacities
    - Production profiles
    - Imports & Exports
    - Balancing
    - Costs
""")
