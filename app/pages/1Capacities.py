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

st.title("Power production installed capacities")


@st.cache_data(show_spinner="Retrieving data ...")
def get_df(scenario):
    return (
        pd.read_excel(
            Path(network_path,
                 scenario_dict[scenario]["path"],
                 "graph_extraction_st.xlsx"),
            sheet_name="power_capacities",
            header=0,
        )
    )


# %%
data = get_df(scenario)
df = data.copy()

st.markdown("NB : Power production units are units able to supply electricity ")

st.header("Installed capacities per country")

all = ['EU27 + TYNDP']
country = st.selectbox('Choose your country:', all + list(df.country.unique()))
if not ('EU27 + TYNDP' in country):
    df = df.query("country in @country")

df = (
    df.drop(columns=['country'])
    .groupby(['sector'])
    .sum(numeric_only=True)
    .rename(index={"sector": "Technologies"})
)

fig = px.bar(
    df,
    title="Power production installed capacities [GW]",
    barmode="group",
    text_auto=".2s"
)

fig.update_yaxes(title_text='Installed capacities [GW]')
fig.update_xaxes(title_text='Countries')
fig.update_traces(hovertemplate="%{y:,.0f}")
fig.update_layout(hovermode="x unified",
                  legend_title_text='Technologies')

st.plotly_chart(
    fig
    , use_container_width=True
)

st.divider()

st.header("Split of capacities per country")

df_bar = data.copy()
technology = st.selectbox('Choose your technology:', list(df_bar.sector.unique()))

df_bar = (df_bar
          .query("sector == @technology")
          .drop(columns=['sector'])
          .set_index('country')
          # .T
          .rename_axis("Investment year")
          )

fig_bar = px.bar(
    df_bar,
    title=f"{technology.capitalize()} installed capacities [GW]",
    barmode="group",

)

fig_bar.update_yaxes(title_text='Installed capacities [GW]')
fig_bar.update_xaxes(title_text='Countries')
fig_bar.update_traces(hovertemplate="%{y:,.1f}",
                      )
fig_bar.update_layout(hovermode="x unified",
                      legend_title_text='Technologies')
# df_tab = (df_tab
#           .style
#           .format(precision=2, thousands=",", decimal='.')
#           )

st.plotly_chart(fig_bar
                , use_container_width=True)
