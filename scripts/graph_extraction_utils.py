# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2024 Climact for The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Create data ready to present (utils)
"""
import logging
from pathlib import Path

import numpy as np
import yaml

CLIP_VALUE_TWH = 1e-1  # TWh
CLIP_VALUE_GW = 1e-3  # GW
RES = ["solar", "solar rooftop", "offwind", "offwind-ac", "offwind-dc", "onwind"]

logger = logging.getLogger(__name__)

def load_config(config_file, analysis_path, n_path, dir_export):
    logger.info("Loading configuration")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    config["path"] = {
        "analysis_path": analysis_path,
        "n_path": n_path
    }

    years = config["scenario"]["planning_horizons"]
    config["years_str"] = list(map(str, years))
    simpl = config["scenario"]["simpl"][0]
    cluster = config["scenario"]["clusters"][0]
    opts = config["scenario"]["opts"][0]
    sector_opts = config["scenario"]["sector_opts"][0]
    ll = config["scenario"]["ll"][0]
    config["label"] = (cluster, ll, sector_opts, years)

    config["n_name"] = f"elec_s{simpl}_{cluster}_l{ll}_{opts}_{sector_opts}_"
    config["csvs"] = Path(analysis_path, f"{dir_export}_{config['n_name']}")

    # Todo : type cast all references to year into str or int
    config["excel_columns"] = {"all_years": ["carrier", "hist"] + config["years_str"],
                               "all_years_units": ["carrier", "hist"] + config["years_str"] + ["units"],
                               "future_years": ["carrier"] + config["years_str"],
                               "future_years_sector": ["carrier", "sector"] + config["years_str"],
                               "first_year_units": ["carrier"] + [config["years_str"][0], "units"],
                               "last_hist_units": ["carrier", "hist"] + [config["years_str"][-1], "units"],
                               "last_units": ["carrier"] + [config["years_str"][-1], "units"],
                               "first_hist_units": ["carrier", "hist"] + [config["years_str"][0], "units"]}

    return config


def bus_mapper(x, n, column=None):
    if x in n.buses.index:
        return n.buses.loc[x, column]
    else:
        return np.nan

# def extract_production_profiles(n, subset):
#     profiles = []
#     for y, ni in n.items():
#         # Grab data from various sources
#         n_y_t = pd.concat([
#             ni.links_t.p_carrier_nom_opt,
#             ni.generators_t.p,
#             ni.storage_units_t.p
#         ], axis=1)
#         n_y = pd.concat([
#             ni.links,
#             ni.generators,
#             ni.storage_units
#         ])
#         n_y = n_y.rename(index=RENAMER)
#
#         # sorting the carriers
#         n_y_t = n_y_t.loc[:, n_y.index]
#         n_y_t = n_y_t.loc[:, n_y.carrier.isin(subset)]
#         n_y = n_y.loc[n_y.carrier.isin(subset)]
#
#         # mapping the countries
#         buses_links = [c for c in n_y.columns if "bus" in c]
#         country_map = n_y[buses_links].applymap(lambda x: bus_mapper(x, ni, column="country"))
#         n_y_t_co = {}
#         for co in ni.buses.country.unique():
#             if co == 'EU':
#                 continue
#             carrier_mapping = n_y[country_map.apply(lambda L: L.fillna('').str.contains(co)).any(axis=1)] \
#                 .groupby("carrier").apply(lambda x: x)
#             carrier_mapping = dict(zip(carrier_mapping.index.droplevel(0),
#                                        carrier_mapping.index.droplevel(1)))
#             n_y_t_co[co] = (n_y_t.loc[:, n_y_t.columns.isin(list(carrier_mapping.keys()))]
#                             .rename(columns=carrier_mapping)
#                             .groupby(axis=1, level=0)
#                             .sum()).T
#
#         profiles.append(pd.concat({y: pd.concat(n_y_t_co)}, names=["Year", 'Country', "Carrier"]))
#
#     df = pd.concat(profiles)
#     df.insert(0, column="Annual sum [TWh]", value=df.sum(axis=1) / 1e6 * 8760 / len(ni.snapshots))
#     df.loc[(slice(None), slice(None), 'Haber-Bosch'), :] *= 4.415385
#     df.insert(0, column="units", value="MWh_e")
#     df.loc[(slice(None), slice(None), ['Haber-Bosch', 'ammonia cracker']), 'units'] = 'MWh_lhv,nh3'
#     df.loc[(slice(None), slice(None), ['Sabatier']), 'units'] = 'MWh_lhv,h2'
#     return df
