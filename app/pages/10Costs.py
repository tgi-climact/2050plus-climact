from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from st_common import COSTS_AREA
from st_common import get_buses
from st_common import network_path
from st_common import scenario_dict
from st_common import st_page_config
from st_common import st_side_bar

st_page_config(layout="wide")
scenario, compare = st_side_bar()

FORMATTER = {"total": "Total", "capital": "CAPEX", "marginal": "OPEX", "elec": "Electricity",
             "gas": "Methane", "H2": "Hydrogen"}

st.title("Costs")


@st.cache_data(show_spinner="Retrieving data ...")
def get_data(scenario, path):
    df = (
        pd.read_csv(
            Path(network_path, scenario_dict[scenario]["path"], path),
            header=0
        )
        .replace(FORMATTER)
        .rename(columns=FORMATTER)
    )
    return df


# %% Cost segment
st.header("Cost by unit segment")
st.markdown("* ENTSO-E area includes all modeled countries, so imports and exports = 0.\n"
            "* A negative value means that the area is exporting and thus making a profit.")

df_cost_segments = get_data(scenario, "costs_segments.csv")
if compare != '-':
    df_compare = get_data(compare, "costs_segments.csv")
    idx = ["config", "cost_segment", "cost/carrier"]
    df_cost_segments = (
        (df_cost_segments.set_index(idx) - df_compare.set_index(idx))
        .reset_index()
    )
df_cost_segments = df_cost_segments.set_index("config")
col1, col2 = st.columns([4, 4])
with col1:
    selected_cost_segment = st.selectbox("Choose your segment :",
                                         ["Total"] + list(df_cost_segments.cost_segment.unique()))
    if selected_cost_segment != "Total":
        df_cost_segments = df_cost_segments.query("cost_segment in @selected_cost_segment")

with col2:
    selected_area = st.selectbox("Choose area :", COSTS_AREA)
    df_cost_segments = df_cost_segments[df_cost_segments.index.str.endswith(COSTS_AREA[selected_area])]

df_cost_segments = df_cost_segments.query("`cost/carrier` != 'Total'")
if selected_cost_segment != "Net_Imports":
    df_cost_segments.loc[df_cost_segments["cost_segment"] == "Net_Imports", "cost/carrier"] = "Imports"
df_cost_segments = df_cost_segments.groupby(by="cost/carrier").sum().drop(columns=["cost_segment"])
df_cost_segments.loc["Total"] = df_cost_segments.sum()
df_cost_segments = df_cost_segments.replace(0, None).dropna(how="all").fillna(0)

df_cost_segments = df_cost_segments.div(1e9)

fig = px.bar(
    df_cost_segments,
    width=1000,
    height=400,
    title=f"{selected_cost_segment} costs for {selected_area} [Billion € / y]",
    barmode="group",
    text_auto=".3s"
)

fig.update_yaxes(title_text="Cost [Billion € / y]")
fig.update_xaxes(title_text="Cost type")
fig.update_traces(hovertemplate="%{y:,.3s}")
fig.update_layout(hovermode="x unified",
                  legend_title_text="Year")

st.plotly_chart(
    fig,
    use_container_width=True
)
df_cost_segments.index.set_names("Costs per type/carrier [B€/year]", inplace=True)
st.dataframe(df_cost_segments.style.format(precision=2, thousands=",", decimal='.'), use_container_width=True)

# %% Cost year
st.header("Cost by year")
df_cost_years = get_data(scenario, "costs_years.csv")
if compare != '-':
    df_compare = get_data(compare, "costs_years.csv")
    idx = ["config", "cost_segment"]
    df_cost_years = (
        (df_cost_years.set_index(idx) - df_compare.set_index(idx))
        .reset_index()
    )

df_cost_years[["year", "area"]] = df_cost_years["config"].str.split('_', expand=True)
df_cost_years = df_cost_years.drop(columns=["config"])
df_cost_years = df_cost_years.set_index("cost_segment")

col1, col2 = st.columns([4, 4])
with col1:
    selected_year = st.selectbox("Choose year :", list(df_cost_years.year.unique()), index=len(list(df_cost_years.year.unique())) - 1)
    df_cost_years = df_cost_years.query("year in @selected_year")
with col2:
    selected_area = st.selectbox("Choose area  :", list(COSTS_AREA))
    selected_area_ = COSTS_AREA[selected_area]
    df_cost_years = df_cost_years.query("area in @selected_area_")

df_cost_years = df_cost_years.drop(columns=["year", "area"])

df_cost_years = df_cost_years.div(1e9)

fig = px.bar(
    df_cost_years,
    width=1000,
    height=400,
    title=f"{selected_year} costs for {selected_area} [Billion € / y]",
    text_auto=".3s"
)

fig.update_yaxes(title_text="Cost [Billion € / y]")
fig.update_xaxes(title_text="Segment")
fig.update_traces(hovertemplate="%{y:,.3s}")
fig.update_layout(hovermode="x unified",
                  legend_title_text="Cost type")

st.plotly_chart(
    fig,
    use_container_width=True
)

df_cost_years.index.set_names("Costs per unit segment/type [B€/year]", inplace=True)
st.dataframe(
    df_cost_years.assign(Total=df_cost_years.sum(axis=1)).style.format(precision=2, thousands=",", decimal='.'),
    use_container_width=True)

# %%
st.header("Marginal price of methane, electricity and hydrogen")
st.markdown(
    "The marginal price represents the cost of an additional unit of an energy carrier at one country at a given time. Please note that the shown value does not take into account taxes nor trading dynamic.")

st.subheader("Compare two areas over years")

df = get_data(scenario, "marginal_prices.csv")
if compare != '-':
    df_compare = get_data(compare, "marginal_prices.csv")
    idx = ["countries", "carrier"]
    df = (
        (df.set_index(idx) - df_compare.set_index(idx))
        .reset_index()
    )

col1, col2 = st.columns([4, 4])
with col1:
    country = st.selectbox("Choose your area:", list(df.countries.unique()))

    if not ("ENTSO-E area" in country):
        df1 = df.query("countries in @country").drop(columns="countries").set_index("carrier")
    else:
        raise Exception("Fix me to remove Flanders")

    df1 = df1.rename(columns=lambda x: x + " Annual average " if not (x.endswith("_std")) else x.replace("_std",
                                                                                                         " Standard Deviation")).T
    df1 = df1.rename(columns=lambda x: x + " [€/MWh]")

    st.dataframe(df1
                 .style
                 .format(precision=2, thousands=",", decimal='.'),
                 use_container_width=True)
with col2:
    country2 = st.selectbox("Choose your country:", list(df.countries.unique()) + [''])

    if not ("ENTSO-E area" in country2):
        df2 = df.query("countries in @country2").drop(columns="countries").set_index("carrier")
    else:
        raise Exception("Fix me to remove Flanders")

    df2 = df2.rename(columns=lambda x: x + " Annual average " if not (x.endswith("_std")) else x.replace("_std",
                                                                                                         " Standard Deviation")).T
    df2 = df2.rename(columns=lambda x: x + " [€/MWh]")

    st.dataframe(df2
                 .style
                 .format(precision=2, thousands=",", decimal='.'),
                 use_container_width=True)

# %% Box plot for costs

st.subheader("Compare variability through Europe")

df_t_ = get_data(scenario, "marginal_prices_t.csv")
if compare != '-':
    df_compare = get_data(compare, "marginal_prices_t.csv")
    idx = ["countries", "carrier", "year"]
    df_t_ = (
        (df_t_.set_index(idx) - df_compare.set_index(idx))
        .reset_index()
    )

col1, col2, col3 = st.columns([0.2, 0.2, .5])
with col1:
    carrier = st.selectbox("Choose areas to compare:", list(df_t_.carrier.unique()), index=1)
with col2:
    year = st.selectbox("Choose year to select:", list(df_t_.year.unique()), index=len(list(df_t_.year.unique())) - 1)
with col3:
    countries_to_display = st.multiselect("Choose your carrier:", list(df.countries.unique()),
                                          default=["GB", "FL", "FR", "LU", "DE", "NL"])

df_t_ = (
    df_t_.query("carrier == @carrier and year==@year")
    .drop(columns=["carrier", "year"])
    .set_index(["countries"])
    .rename_axis(columns=["Local marginal price [€/MWh]"])
    .T
)
df_t_.index = pd.DatetimeIndex(df_t_.index)
df_t = df_t_[countries_to_display]

fig = px.box(df_t)
fig.update_traces(hovertemplate="%{y:,.0f}",
                  line=dict(width=1))
fig.update_yaxes(title_text=f"Local marginal price for {carrier} [€/MWh]")
fig.update_xaxes(title_text="Areas")

st.plotly_chart(fig)

df_map = (
    df_t_
    .resample("D").mean()
    .rename_axis(index="snapshot")
    .melt(value_name="LMP [€/MWh]", var_name="country", ignore_index=False)
    .merge(get_buses(), left_on="country", right_index=True)
    .assign(absolute_size=lambda x: x["LMP [€/MWh]"].abs())
)
df_map.index = pd.DatetimeIndex(df_map.index).strftime(f"{year}-%m-%d")
fig_map = px.scatter_mapbox(
    df_map.reset_index(),
    lat="lat",
    lon="lon",
    size="absolute_size",
    color="LMP [€/MWh]",
    color_continuous_scale="bluered",
    range_color=[0, 100],
    mapbox_style="carto-positron",
    zoom=2.6,
    height=700,
    hover_name="country",
    animation_frame="snapshot",
    title=f"Daily average local marginal price for {carrier} in {year} [€/MWh]",
    hover_data={"LMP [€/MWh]": ":.2f"}
)
fig_map.update_layout(sliders=[{"currentvalue": {"prefix": "Year: "}, "len": 0.8, "y": 0.07,}])
fig_map.update_layout(updatemenus=[{
    "y": 0.07,
    "buttons": [{'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'mode': 'immediate', 'fromcurrent': True,
                                 'transition': {'duration': 100, 'easing': 'linear'}}], 'label': '&#9654;',
                 'method': 'animate'},
                {'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'fromcurrent': True,
                                   'transition': {'duration': 0, 'easing': 'linear'}}], 'label': '&#9724;',
                 'method': 'animate'}],
}])
st.plotly_chart(fig_map, use_container_width=True)
