from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from st_common import get_buses
from st_common import network_path
from st_common import scenario_dict
from st_common import st_page_config
from st_common import st_side_bar

st_page_config(layout="wide")
scenario = st_side_bar()

st.title("Loads per carrier and sector")
st.markdown("The total yearly load per energy carrier, year, country and subsector.")


@st.cache_data(show_spinner="Retrieving data ...")
def get_data(scenario):
    df = (
        pd.read_csv(
            Path(network_path, scenario_dict[scenario]["path"], "supply_energy_df.csv"),
            header=0
        )
    )
    return df


df_raw = get_data(scenario)
all = "ENTSO-E area"
country = st.selectbox('Choose your country:', [all] + list(df_raw["node"].unique()))

# %%
st.header("Loads per carrier")

if country != all:
    df_ca = df_raw.query("node==@country").drop("node", axis=1).copy()
else:
    df_ca = (
        df_raw
        .query("node != 'FL'")
        .groupby(by=["sector", "carrier"]).sum(numeric_only=True)
        .reset_index().copy()
    )

carrier = st.selectbox('Choose your carrier:', df_ca["carrier"].unique(), index=1)
df_ca = df_ca.query("carrier==@carrier").drop("carrier", axis=1)

df_ca = df_ca.groupby(by="sector").sum().sort_values(by="2050", ascending=False)

df_ca_tot = pd.DataFrame(df_ca.sum().rename("Total")).T
df_ca = pd.concat([df_ca, df_ca_tot])
df_ca.index.name = "Annual load [TWh]"

df_map = (
    df_raw
    .query("carrier==@carrier")
    .groupby("node").sum(numeric_only=True)
    .join(get_buses())
    .reset_index()
    .rename(columns={"node": "country"})
    .melt(id_vars=["country", "lat", "lon"], value_name="Load [TWh]", var_name="year")
)
fig_map = px.scatter_mapbox(
    df_map,
    lat="lat",
    lon="lon",
    size="Load [TWh]",
    mapbox_style="carto-positron",
    zoom=2.6,
    height=700,
    hover_name="country",
    animation_frame="year",
    title=f"Cumulated load for {carrier} [TWh]",
)
fig_map.update_layout(sliders=[{"currentvalue": {"prefix": "Year: "}, "len": 0.8, "y": 0.07}])
fig_map.update_layout(updatemenus=[{"y": 0.07}])
st.plotly_chart(fig_map, use_container_width=True)

fig = px.bar(
    df_ca,
    title=f"Load in {country} for {carrier} [TWh]",
    barmode="group",
    text_auto=".2s"
)

fig.update_traces(hovertemplate="%{y:,.0f}")
fig.update_layout(hovermode="x unified")
fig.update_yaxes(title_text='Consumption [TWh]')
fig.update_xaxes(title_text='Sectors')
fig.update_layout(legend_title_text='Years')

st.plotly_chart(
    fig
    , use_container_width=True
)

st.dataframe(df_ca
             .style
             .format(precision=2, thousands=",", decimal='.'),
             use_container_width=True
             )

st.divider()

st.header("Load per sector")

if country != all:
    df_se = df_raw.query("node==@country").drop("node", axis=1).copy()
else:
    df_se = df_raw.query("node!='FL'").groupby(by=["sector", "carrier"]).sum(numeric_only=True).reset_index().copy()

# Add Industry (with and without CC) and Heat production (central and decentral)
df_ind = (
    df_se
    .query("sector.str.contains('Industry') and not carrier.str.contains('CO2')")
    .assign(sector="Industry (with and without CC)")
    .groupby(by=["sector", "carrier"]).sum()
    .reset_index()
)
df_heat = (
    df_se
    .query("sector.str.contains('heat production') and not carrier.str.contains('CO2')")
    .assign(sector="Heat production (central and decentral)")
    .groupby(by=["sector", "carrier"]).sum()
    .reset_index()
)
df_se = pd.concat([df_se, df_ind, df_heat])

sector = st.selectbox('Choose your sector:', df_se["sector"].unique(), index=8)
df_se = df_se.query("sector==@sector").drop("sector", axis=1)

df_se = df_se.groupby(by="carrier").sum().sort_values(by="2050", ascending=False)

df_se_tot = pd.DataFrame(df_se.sum().rename("Total")).T
df_se = pd.concat([df_se, df_se_tot])
df_se.index.name = "Annual load [TWh]"

fig = px.bar(
    df_se,
    title=f"Load in {country} for {sector} [TWh]",
    barmode="group",
    text_auto=".2s"
)

fig.update_traces(hovertemplate="%{y:,.0f}")
fig.update_layout(hovermode="x unified")
fig.update_yaxes(title_text='Consumption [TWh]')
fig.update_xaxes(title_text='Carriers')
fig.update_layout(legend_title_text='Years')

st.plotly_chart(
    fig
    , use_container_width=True
)

st.dataframe(df_se
             .style
             .format(precision=2, thousands=",", decimal='.'),
             use_container_width=True
             )
