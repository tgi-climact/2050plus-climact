from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from st_common import network_path
from st_common import scenario_dict
from st_common import st_page_config
from st_common import st_side_bar

st_page_config(layout="wide")
scenario = st_side_bar()

st.title("Loads per carrier")

df = (
    pd.read_excel(
        Path(network_path,
             scenario_dict[scenario]["path"],
             "graph_extraction_st.xlsx"),
        sheet_name="supply_energy_df",
        header=0
    )
)

all = "EU27 + TYNDP"
country = st.selectbox('Choose your country:', [all] + list(df["node"].unique()))
if country != all:
    df = df.query("node==@country").drop("node", axis=1)
else:
    df = df.groupby(by=["sector", "carrier"]).sum(numeric_only=True).reset_index()
carrier = st.selectbox('Choose your carrier:', df["carrier"].unique())
df = df.query("carrier==@carrier").drop("carrier", axis=1)

df = df.groupby(by="sector").sum()

st.plotly_chart(
    px.bar(
        df,
        title=f"Load in {country} for {carrier} [TWh]",
        barmode="group",
        text_auto=".2s"
    )
    , use_container_width=True
)

st.table(df.style.format(precision=2))
