from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from st_common import PROFILES_AREA
from st_common import get_years
from st_common import network_path
from st_common import scenario_dict
from st_common import st_page_config
from st_common import st_side_bar

st_page_config(layout="wide")
scenario, compare = st_side_bar()

YEARS = get_years(scenario)

st.title("Production per carrier")
st.markdown("The total energy production per year, country and subsector. This data is currently shown at system level (ENTSO-E area), Belgium (BE) and Flanders (FL) due to the very large quantity of data that needs to be handled for every country in the system.")


@st.cache_data(show_spinner="Retrieving data ...")
def get_data(scenario, year, selected_area):
    area = selected_area if selected_area != "ENTSO-E area" else ''
    return (
        pd.read_csv(
            Path(network_path, scenario_dict[scenario]["path"],
                 f"supply_temporal_{area.lower()}_{year}.csv".replace('__', '_')),
            header=[1, 2]
        )
        .set_index(("carrier", "sector"))
    )


col1, col2 = st.columns(2)
with col1:
    selected_area = st.selectbox('Choose area :', PROFILES_AREA)

dfx = []
for y in YEARS:
    dfi = get_data(scenario, y, selected_area)
    if compare != '-':
        df_compare = get_data(compare, y, selected_area)
        dfi = (dfi - df_compare)
    dfi = ((dfi.sum() / 1e3
            * 8760 / len(dfi.axes[0]))
           .to_frame(name=y))
    dfx.append(dfi)
dfx = pd.concat(dfx, axis=1).fillna(0).sort_values(by=y, ascending=False)
dfx.index.name = 'Annual production [TWh]'

with col2:
    carrier = st.selectbox('Choose your carrier:', dfx.index.unique(0).sort_values(),
                           index=dfx.index.unique(0).sort_values().get_loc("Electricity"))
unit_twh = "TWh" if "co2" not in carrier.lower() else "Mt"
unit_gw = "GW" if "co2" not in carrier.lower() else "Mt"
dfx = dfx.loc[carrier]


st.subheader(f"Annual {carrier} production per sector")

total = (dfx.sum())
dfx.loc['Total'] = total
dfx.index.name = f"Annual production [{unit_twh}]"

fig = px.bar(
    dfx,
    title=f"Production in {selected_area} for {carrier} [{unit_twh}]",
    barmode="group",
    text_auto=".2s"
)

fig.update_traces(hovertemplate="%{y:,.0f}")
fig.update_layout(hovermode="x unified")
fig.update_yaxes(title_text=f'Production [{unit_twh}]')
fig.update_xaxes(title_text='Sectors')
fig.update_layout(legend_title_text='Years')

st.plotly_chart(
    fig
    , use_container_width=True
)

st.dataframe(
    dfx
    .style
    .format(precision=2, thousands=",", decimal='.'),
    use_container_width=True
)

st.divider()


st.subheader(f"Production profiles per carrier")

st.markdown(
    "The load 3-hourly profiles for every carrier, year and subsector. You can zoom on these interactive graphs for specific time windows and you can also select/deselect various categories if you want.")

year = st.selectbox('Choose the year:', YEARS, index=len(YEARS) - 1)
data = get_data(scenario, year, selected_area)
if compare != '-':
    df_compare = get_data(compare, year, selected_area)
    data = (data - df_compare)

df = data[carrier]

fig = px.area(
    df,
    title=f"System production profile for {carrier} [{unit_gw}]",
)
fig.update_traces(hovertemplate="%{y:,.0f}",
                  line=dict(width=0.1))
fig.update_layout(legend_traceorder="reversed",
                  hovermode="x unified",
                  legend_title_text='Technologies')
fig.update_yaxes(title_text=f'Production [{unit_gw}]')
fig.update_xaxes(title_text='Timesteps')

st.plotly_chart(
    fig
    , use_container_width=True
)
