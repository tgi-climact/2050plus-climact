# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2024 Climact for The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Create data ready to present (load to streamlit)
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
from scripts.graph_extraction_utils import _load_nodal_oil
from scripts.graph_extraction_utils import HEAT_RENAMER
from scripts.graph_extraction_utils import _load_supply_energy

logger = logging.getLogger(__name__)


def load_supply_energy_df(config, load=True):
    """
    Allow to split _load_supply_energy into its carriers for it
    to be written tab by tab

    """
    dfl = []
    supply_energy_carrier = (
        _load_supply_energy(config, load=load)
        .carrier
        .replace({"AC": "electricity", "low voltage": "electricity"})
        .unique()
    )

    for ca in supply_energy_carrier:
        ca_name = HEAT_RENAMER[ca] if ca in HEAT_RENAMER else ca
        # Todo : should we move the HV/LV/imports/exports to the calling function to keep this function read only (no modifications) ?
        if ca == "electricity":
            df_ac = _load_supply_energy(config, load=load, carriers="AC", aggregate=False)
            df_low = _load_supply_energy(config, load=load, carriers="low voltage", aggregate=False)
            df = pd.concat([df_ac, df_low])
            del df["carrier"]
            df = df.groupby(by=["sector", "node"]).sum().reset_index()

            # if not (load) and countries:
            #     df_imp = _load_imp_exp(config, export=False, countries=countries, carriers='elec',
            #                            years=config["scenario"]["planning_horizons"])
            #     df_exp = _load_imp_exp(config, export=True, countries=countries, carriers='elec',
            #                            years=config["scenario"]["planning_horizons"])
            #     df_net_imp = (df_imp[config["scenario"]["planning_horizons"]] - df_exp[
            #         config["scenario"]["planning_horizons"]]).sum()
            #     df = pd.concat([df, pd.DataFrame(['Net Imports'] + df_net_imp.values.tolist(), index=df.columns).T])
            df.drop(df.query('sector in ["V2G", "Battery charging", "Hydroelectricity"]').index, inplace=True)
            df["carrier"] = ca
            dfl.append(df)
        elif ca == "oil":
            # if load and countries is not None:  # if load and countries exist
            df_eu_load = _load_nodal_oil(config, aggregate=False)
            df_c_load = _load_supply_energy(config, load=load, carriers=ca, aggregate=False)
            dfl.append(pd.concat([df_c_load, df_eu_load]))
            # else:
            #     dfl.append(_load_supply_energy(config, load=load, carriers=ca, aggregate=False))
        else:
            dfl.append(_load_supply_energy(config, load=load, carriers=ca, aggregate=False))

    df = pd.concat(dfl)
    return df

def load_imports_exports(config):
    """
    This function loads the imports and exports for all countries, carriers and years
    considered during the runs. The table loaded is imports_exports, as it is only filtering that is done
    at streamlit level.

    """
    
    return pd.read_csv(Path(config["csvs"], "imports_exports.csv"))

def load_load_temporal(config):
    load = _load_supply_energy(config, load=True, aggregate = True, temporal= True)
    return load

def load_supply_temporal(config):
    supply = _load_supply_energy(config, load=False, aggregate = True, temporal= True)
    return  supply

def load_generation_profiles(config):
    
    return pd.read_csv(Path(config["csvs"], "generation_profiles.csv")) 

#%% Load main
def load_data_st(config):
    logger.info(f"Exporting data to streamlit")

    outputs = [
        "load_temporal",
        "supply_temporal",
        "supply_energy_df",
        "imports_exports",
        "generation_profiles"
    ]

    with pd.ExcelWriter(Path(config["path"]["analysis_path"], "graph_extraction_st.xlsx")) as xl:
        for output in outputs:
            o = globals()["load_" + output](config)
            if isinstance(o, pd.DataFrame):
                o.to_excel(xl, sheet_name=output, index=False)
            elif isinstance(o, dict):
                for k, v in o.items():
                    # Determine sheet name
                    sheet_name = output + "_" + k
                    max_sheet_name_length = 31  # Limit sheet name to 31 char to enhance compatibility support (Excel)
                    overflow_char = ".."
                    sheet_name = (sheet_name[:max_sheet_name_length - len(overflow_char)] + overflow_char) \
                        if len(sheet_name) > max_sheet_name_length - len(overflow_char) else sheet_name
                    v.to_excel(xl, sheet_name=sheet_name, index=False)
            else:
                logging.warning(f"Given output for {output} is not mapped out to output file.")
