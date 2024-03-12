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

st.title("Consumption profiles per carrier")


@st.cache_data
def get_data():
    return (
        pd.read_excel(
            Path(network_path,
                 scenario_dict[scenario]["path"],
                 "graph_extraction_st.xlsx"),
            sheet_name="load_temporal",
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
data = get_data()

carrier = st.selectbox('Choose your carrier:', data["carrier"].unique(), index=3)
df = data.query("carrier==@carrier").drop("carrier", axis=1)

year = st.selectbox('Choose the year:', years)
df['snapshot'] = pd.to_datetime(pd.DatetimeIndex(df['snapshot'].values,name='snapshots').strftime(f'{year}-%m-%d-%H'))
df = df.pivot(index='snapshot', columns=['sector'], values=year).rename_axis('sector', axis=1)
df = df[(df.std() / df.mean()).sort_values().index]
df = df.loc[:, df.sum() / 1e3 > CLIP_VALUE_TWH]

fig = px.area(
    df,
    title=f"System consumption profile for {carrier} [GW]",
)
fig.update_traces(hovertemplate="%{y:,.0f}",
                  line=dict(width=0.1))
fig.update_layout(legend_traceorder="reversed",
                  hovermode="x unified",
                  legend_title_text='Technologies')
fig.update_yaxes(title_text='Consumption [GW]')
fig.update_xaxes(title_text='Timesteps')

st.plotly_chart(
    fig
    , use_container_width=True
)

st.subheader(f"Annual {carrier} consumption per sector for {year} ")

st.table(
    (df.sum() / 1e3
     * 8760 / len(df.axes[0]))
    .rename('Annual consumption [TWh]')
    .to_frame()
    .style
    .format(precision=2, thousands=",", decimal='.')
)
