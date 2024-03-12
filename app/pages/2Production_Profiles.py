from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from st_common import CLIP_VALUE_TWH
from st_common import network_path
from st_common import scenario_dict
from st_common import st_page_config
from st_common import st_side_bar

st_page_config(layout="wide")
scenario = st_side_bar()

st.title("Production profiles per carrier")
st.text("Data displayed are for EU27 + TYNDP.")


@st.cache_data(show_spinner="Retrieving data ...")
def get_data(scenario):
    return (
        pd.read_excel(
            Path(network_path,
                 scenario_dict[scenario]["path"],
                 "graph_extraction_st.xlsx"),
            sheet_name="supply_temporal",
            header=0
        )
    )


# %%Cell Name
# should be able to 
# - Display per carrier
# - 3h load profile
# - eventually per country
# - eventuelly per subtype of supply
years = ['2030', '2040', '2050']
data = get_data(scenario)
col1, col2 = st.columns(2)
with col1:
    carrier = st.selectbox('Choose your carrier:', data["carrier"].unique(), index=5)
df = data.query("carrier==@carrier").drop("carrier", axis=1)

with col2:
    year = st.selectbox('Choose the year:', years)
df['snapshot'] = pd.to_datetime(pd.DatetimeIndex(df['snapshot'].values, name='snapshots').strftime(f'{year}-%m-%d-%H'))
df = df.pivot(index='snapshot', columns=['sector'], values=year).rename_axis('sector', axis=1)
df = df[(df.std() / df.mean()).sort_values().index]
df = df.loc[:, df.sum() / 1e3 > CLIP_VALUE_TWH]

fig = px.area(
    df,
    title=f"System production profile for {carrier} [GW]",
)
fig.update_traces(hovertemplate="%{y:,.0f}",
                  line=dict(width=0.1))
fig.update_layout(legend_traceorder="reversed",
                  hovermode="x unified",
                  legend_title_text='Technologies')
fig.update_yaxes(title_text='Production [GW]')
fig.update_xaxes(title_text='Timesteps')

st.plotly_chart(
    fig
    , use_container_width=True
)

st.subheader(f"Annual {carrier} production per technology for {year} ")

df_table = (
    (df.sum() / 1e3
     * 8760 / len(df.axes[0]))
    .rename('Annual production [TWh]')
    .to_frame()
    .style
    .format(precision=2, thousands=",", decimal='.')
)

st.table(df_table)
