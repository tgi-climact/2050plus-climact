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

st.title("Production profiles per carrier")

@st.cache_data
def get_df():
    return (
        pd.read_excel(
            Path(network_path,
                 scenario_dict[scenario]["path"],
                 "graph_extraction_st.xlsx"),
            sheet_name="res_temporal",
            header=0,
        )
    )

# %%Cell Name 
# should be able to 
# - Display per carrier
# - 3h load profile
# - eventually per country
years = ['2030','2040','2050']
data = get_df()
df = data.copy()
country = st.selectbox('Choose your country:', ['all'] + list(df["country"].unique()))
if country != 'all':
    df = df.query("country in @country")
df = df.groupby(['carrier']).sum(numeric_only=True).T
carrier = st.selectbox('Choose your carrier:', list(df.index.unique()))
if carrier != 'all':
    df = df.query("carrier in @carrier")
df = df.sum(axis=0).to_frame()
# df = df.groupby(['country','carrier']).sum().T


year = st.selectbox('Choose the year:',years )
df = df.query("index.str.contains(@year)")


fig = px.area(
    df,
    title=f"{carrier.capitalize()} production profile for {country}  [GW]",
)
fig.update_traces(line=dict(width=0.1))
fig.update_layout(legend_traceorder="reversed")
fig.update_yaxes(title_text='Production [GW]')
fig.update_xaxes(title_text='Timesteps')
fig.update_layout(legend_title_text = 'Technologies')

st.plotly_chart(
    fig
    , use_container_width=True
)

df_table = (
    (df.sum()/1e3 #TWh
     *8760/len(df.axes[0]))
    .rename('Annual production [TWh]')
    .to_frame()
    .style
    .format(precision=2, thousands = ",", decimal = '.')
    )

st.table(df_table)
