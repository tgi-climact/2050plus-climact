from pathlib import Path

import streamlit as st
from PIL import Image

base_path = Path(__file__).parent
swoosh = Image.open(Path(base_path, "assets", "img", "swoosh.png"))
network_path = Path(base_path, "assets", "data")
scenario_dict = {
    "Average": {
        "path": "VEKA_av_bio_fix_nuc_bev_ccl",
        "fn": "elec_s181_37m_lv3.0__3H-I-T-H-B-A-CCL_YEAR.nc",
    },
    "Électrification": {
        "path": "VEKA_el_bio_fix_nuc_bev_ccl",
        "fn": "elec_s181_37m_lv3.0__3H-I-T-H-B-A-CCL_YEAR.nc",
    }
}


def st_page_config(layout=None):
    if layout is None:
        layout = "centered"
    st.set_page_config(
        page_title="VEKA 2050+",
        page_icon=swoosh,
        layout=layout,
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "mailto:mty@climact.com",
            "Report a bug": "mailto:mty@climact.com",
            "About": "# Experimental Climact carbon footprint app",
        },
    )


# @st.cache_data
# def load_network(scenario, year):
#     return pypsa.Network(Path(network_path,
#                               scenario_dict[scenario]["path"],
#                               scenario_dict[scenario]["fn"].replace("YEAR", str(year))
#                               )
#                          )


def st_side_bar():
    with st.sidebar:
        scenario = st.selectbox(
            "Select your scenario",
            ["Average", "Électrification"]
        )
    return scenario
