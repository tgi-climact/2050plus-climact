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


@st.cache_data
def get_profiles():
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
data = get_profiles()
carrier = st.selectbox('Choose your carrier:', data["carrier"].unique(), index=4)
df = data.query("carrier==@carrier").drop("carrier", axis=1)

year = st.selectbox('Choose the year:', years)
df = df.pivot(index='snapshot', columns=['sector'], values=year).rename_axis('sector', axis=1)
df = df[(df.std() / df.mean()).sort_values().index]
df = df.loc[:, df.sum() / 1e3 > CLIP_VALUE_TWH]

fig = px.area(
    df,
    title=f"System production profile for {carrier} [GW]",
)
fig.update_traces(line=dict(width=0.1))
fig.update_layout(legend_traceorder="reversed")
fig.update_yaxes(title_text='Production [GW]')
fig.update_xaxes(title_text='Timesteps')
fig.update_layout(legend_title_text='Technologies')

st.plotly_chart(
    fig
    , use_container_width=True
)

df_table = (
    (df.sum() / 1e3
     * 8760 / len(df.axes[0]))
    .rename('Annual production [TWh]')
    .to_frame()
    .style
    .format(precision=2, thousands=",", decimal='.')
)

st.table(df_table)
