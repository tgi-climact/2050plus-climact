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


def query_imp_exp(df, carriers, country, year, imports_exports):
    df_imp_exp = (
        df.query(""
                 "carriers == @carriers & "
                 "year == @year & "
                 "imports_exports == @imports_exports"
                 )
        .drop(["carriers", "year", "imports_exports"], axis=1)
        .set_index('countries')
        [country]
    )
    return df_imp_exp


country = st.selectbox('Choose your country:', df["countries"].unique())
carrier = st.selectbox('Choose your carrier:', df['carriers'].unique())
year = st.selectbox('Choose your year:', df["year"].unique())
df_imp_exp = (
    pd.concat([query_imp_exp(df, carrier, country, year, 'imports'),
               -1 * query_imp_exp(df, carrier, country, year, 'exports')],
              axis=1, keys=['imports', 'exports'])
)

st.plotly_chart(
    px.bar(
        df_imp_exp,
        title=f"Imports / Exports for {country} for {carrier} [TWh]",
        text_auto=".2s"
    )
    , use_container_width=True
)

df_imp_exp_ = df_imp_exp.drop(df_imp_exp.query('imports == 0 and exports ==0').index)

st.table(df_imp_exp_.style.format(precision=2, thousands=",", decimal='.'))
