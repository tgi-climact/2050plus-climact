from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from st_common import network_path
from st_common import scenario_dict
from st_common import st_page_config
from st_common import st_side_bar
from scripts.graph_extraction_utils import query_imp_exp

st_page_config(layout="wide")
scenario = st_side_bar()

st.title("Imports and exports per carrier")

@st.cache_data
def get_data():
    df = (
        pd.read_excel(
            Path(network_path,
                 scenario_dict[scenario]["path"],
                 "graph_extraction_st.xlsx"),
            sheet_name="imports_exports",
            header=0
        )
    )
    return df

df = get_data()

country = st.selectbox('Choose your country:', df["countries"].unique())
carrier = st.selectbox('Choose your carrier:', df['carriers'].unique())
year = st.selectbox('Choose your year:', df["year"].unique())
imports_exports = st.selectbox('Choose if imports or exports', ['imports', 'exports'])
df_imp_exp = (
    pd.concat([query_imp_exp(df, carrier, [country], year, 'imports'),
               query_imp_exp(df, carrier, [country], year, 'exports')],
              axis=1, keys=['imports', 'exports'])
)
df_imp_exp = (
    df_imp_exp.drop(df_imp_exp.query('imports <= 0 and exports <=0')
                    .index)
    .style
    .format(precision=2, thousands=",", decimal='.')
)

df = query_imp_exp(df, carrier, [country], year, imports_exports)
st.plotly_chart(
    px.bar(
        df,
        title=f"{imports_exports.capitalize()} for {country} for {carrier} [TWh]",
        barmode="group",
        text_auto=".2s"
    )
    , use_container_width=True
)

st.table(df_imp_exp)
