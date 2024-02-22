# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2024 Climact for The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Create data ready to present
"""

import logging
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import yaml
from yaml import safe_load

from make_summary import assign_carriers
from make_summary import assign_locations
from make_summary import calculate_nodal_capacities
from make_summary import calculate_nodal_supply_energy
from plot_network import plot_capacity
from plot_network import plot_series
from prepare_sector_network import get

logger = logging.getLogger(__name__)

# %% Constants
LONG_LIST_LINKS = ["coal/lignite", "oil", "CCGT", "OCGT",
                   "H2 Electrolysis", "H2 Fuel Cell", "battery charger",
                   "home battery charger", "Haber-Bosch", "Sabatier",
                   "ammonia cracker", "helmeth", "SMR", "SMR CC", "hydro"]

LONG_LIST_GENS = ["solar", "solar rooftop", "onwind", "offwind", "offwind-ac", "offwind-dc",
                  "ror", "nuclear", "urban central solid biomass CHP",
                  "home battery", "battery", "H2 Store", "ammonia store"]

RENAMER = {
    # Carriers
    "offwind-dc": "offwind",
    "offwind-ac": "offwind",
    "solar rooftop": "solar",
    "coal": "coal/lignite",
    "lignite": "coal/lignite",
    "battery storage": "EV batteries",

    # Boilers
    "residential rural biomass boiler": "residential / services biomass boiler",
    "residential urban decentral biomass boiler": "residential / services biomass boiler",
    "services rural biomass boiler": "residential / services biomass boiler",
    "services urban decentral biomass boiler": "residential / services biomass boiler",
    "residential rural gas boiler": "residential / services gas boiler",
    "residential urban decentral gas boiler": "residential / services gas boiler",
    "services rural gas boiler": "residential / services gas boiler",
    "services urban decentral gas boiler": "residential / services gas boiler",
    "urban central gas boiler": "residential / services gas boiler",
    "residential rural oil boiler": "residential / services oil boiler",
    "residential urban decentral oil boiler": "residential / services oil boiler",
    "services rural oil boiler": "residential / services oil boiler",
    "services urban decentral oil boiler": "residential / services oil boiler",
    "urban central oil boiler": "residential / services oil boiler",

    # Water tanks
    "residential rural water tanks charger": "residential / services water tanks charger",
    "residential urban decentral water tanks charger": "residential / services water tanks charger",
    "services rural water tanks charger": "residential / services water tanks charger",
    "services urban decentral water tanks charger": "residential / services water tanks charger",
    "urban central water tanks charger": "residential / services water tanks charger",
    "residential rural water tanks discharger": "residential / services water tanks discharger",
    "residential urban decentral water tanks discharger": "residential / services water tanks discharger",
    "services rural water tanks discharger": "residential / services water tanks discharger",
    "services urban decentral water tanks discharger": "residential / services water tanks discharger",
    "urban central water tanks discharger": "residential / services water tanks discharger",

    # Heat pumps
    "residential rural ground heat pump": "residential / services rural ground heat pump",
    "services rural ground heat pump": "residential / services rural ground heat pump",
    "residential urban decentral air heat pump": "residential / services air heat pump",
    "services urban decentral air heat pump": "residential / services air heat pump",
    "urban central air heat pump": "residential / services air heat pump",

    # Resistive heaters
    "residential rural resistive heater": "residential / services resistive heater",
    "residential urban decentral resistive heater": "residential / services resistive heater",
    "services rural resistive heater": "residential / services resistive heater",
    "services urban decentral resistive heater": "residential / services resistive heater",
    "urban central resistive heater": "residential / services resistive heater",

    # Solar thermals
    "residential rural solar thermal": "residential / services solar thermal",
    "residential urban decentral solar thermal": "residential / services solar thermal",
    "services rural solar thermal": "residential / services solar thermal",
    "services urban decentral solar thermal": "residential / services solar thermal",
    "urban central solar thermal": "residential / services solar thermal",
}

HEAT_RENAMER = {"residential rural heat": "dec_heat",
                "services rural heat": "dec_heat",
                "residential urban decentral heat": "dec_heat",
                "services urban decentral heat": "dec_heat",
                "urban central heat": "cent_heat"}

RES = ["solar", "solar rooftop", "offwind", "offwind-ac", "offwind-dc", "onwind"]
LONG_TERM_STORAGE = ["H2 Store", "ammonia store"]
SHORT_TERM_STORAGE = ["battery", "home battery", "EV batteries"]
FF_ELEC = ["OCGT", "CCGT", "coal/lignite"]
FF_HEAT = ["residential / services oil boiler", "residential / services gas boiler"]
PRODUCTION = FF_ELEC + ["PHS", "hydro", "nuclear", "urban central biomass CHP", "solid biomass"] + RES
H2 = ["H2 Electrolysis", "H2 Fuel Cell"]
BALANCE = H2 + ["battery discharger", "home battery discharger", "V2G", "ammonia cracker"]

CLIP_VALUE_TWH = 1e-1  # TWh
CLIP_VALUE_GW = 1e-3  # GW

COST_SEGMENTS = {'prod': 'Production', 'bal': 'Balancing', 'tran': 'Transport', 'net_imp': "Net_Imports"}

EU27_COUNTRIES = ["AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "GR", "ES", "FI", "FR", "HR", "HU", "IE",
                  "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", "SE", "SI", "SK"]


# %% Utils
def assign_countries(n):
    n.buses = (
        n.buses.merge(
            n.buses[(n.buses.location != "EU") & (n.buses.carrier == "AC")]["country"],
            how="left", left_on="location", right_index=True, suffixes=("_old", '')
        )
        .fillna({"country": "EU"})
        .drop(columns="country_old")
    )
    return


def bus_mapper(x, n, column=None):
    if x in n.buses.index:
        return n.buses.loc[x, column]
    else:
        return np.nan


def searcher(x, carrier):
    if carrier in x.to_list():
        return str(x.to_list().index(carrier))
    else:
        return np.nan


def change_p_nom_opt_carrier(n, carriers=['AC'], temporal=True):
    """
    This function expresses for each asset p_nom_opt (whose carrier is not always the same)
    in function of the given carrier, considering the efficiency
    E.g.: having p_nom_opt = 333MW for a nuclear powerplant is equal to say that
            p_carrier_nom_opt = 100MW_e, if the carrier is 'AC', as the efficiency
            from p_nom_opt to AC is 0.333 for this technology

    Currently, carriers should be limited to ['AC'] as no strict testing was made with other carriers.
    In a future development, this function should be able to tackle several carriers

    Parameters
    ----------
    n : pypsa.network

    carriers : List, optional
        List of carriers to use. Must be limited to ['AC'] for the moment, as bugs might arise. The default is ['AC'].

    Returns
    -------
    None.

    """

    # Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    li = n.links
    li_t = n.links_t
    li["efficiency0"] = 1
    li["p_carrier_nom_opt"] = li.p_nom_opt
    li_t["p_carrier_nom_opt"] = li_t.p0
    efficiency_map = li[[c for c in li.columns if "efficiency" in c]].rename(columns={"efficiency": "efficiency1"})
    buses_links = [c for c in li.columns if "bus" in c]
    carrier_map = li[buses_links].applymap(lambda x: bus_mapper(x, n, column="carrier"))

    for carrier in carriers:
        index_map = carrier_map.apply(lambda x: searcher(x, carrier), axis=1).dropna()

        efficiency_map = efficiency_map.loc[index_map.index]
        efficiency_map = efficiency_map.apply(lambda x: x / x[f"efficiency{index_map.loc[x.name]}"], axis=1)
        li.loc[efficiency_map.index, "p_carrier_nom_opt"] = li.loc[efficiency_map.index, "p_nom_opt"] / efficiency_map[
            "efficiency0"]
        if len(li_t.p0.columns):
            li_t.p_carrier_nom_opt.loc[:, efficiency_map.index] = li_t.p0.loc[:, efficiency_map.index] / efficiency_map[
                "efficiency0"]

    return


def reduce_to_countries(df, index):
    buses = [c for c in df.columns if "bus" in c]
    return df.loc[df.loc[:, buses].applymap(lambda x: x in index).values.any(axis=1)]


def get_state_of_charge_t(n, carrier):
    df = n.storage_units_t.state_of_charge.T.reset_index()
    df = df.merge(n.storage_units.reset_index()[["carrier", "StorageUnit"]], on="StorageUnit")
    df = df.groupby(by="carrier").sum()
    df.drop(columns=["StorageUnit"], inplace=True)
    return df.T[[carrier]] / 1e6  # TWh


def get_e_t(n, carrier):
    df = n.stores_t.e.T.reset_index()
    df = df.merge(n.stores.reset_index()[["carrier", "Store"]], on="Store")
    df = df.groupby(by="carrier").sum()
    df.drop(columns=["Store"], inplace=True)
    return df.T[[carrier]] / 1e6  # TWh


def get_p_carrier_nom_t(n, carrier):
    df = n.links_t.p_carrier_nom_opt.T.reset_index()
    df = df.merge(n.links.reset_index()[["carrier", "Link"]], on="Link")
    df = df.groupby(by="carrier").sum()
    df.drop(columns=["Link"], inplace=True)
    return df.T[[carrier]] / 1e3  # GW

def prepare_networks(n_path):
    n = {}
    for y in years:
        run_name = Path(n_path, "postnetworks", n_name + f"{y}.nc")
        n[y] = pypsa.Network(run_name)
        assign_carriers(n[y])
        assign_locations(n[y])
        assign_countries(n[y])
        change_p_nom_opt_carrier(n[y])
    
    # get historical capacities
    n_bf = pypsa.Network(Path(n_path, "prenetworks-brownfield", n_name + f"{2030}.nc"))
    assign_countries(n_bf)
    assign_carriers(n_bf)
    assign_locations(n_bf)

    year_hist = 2026
    n_bf.generators = n_bf.generators.query(f'build_year < {year_hist}')
    n_bf.links = n_bf.links.query(f'build_year < {year_hist}')
    n_bf.storage_units = n_bf.storage_units.query(f'build_year < {year_hist}')
    n_bf.generators.p_nom_opt = n_bf.generators.p_nom
    n_bf.links.p_nom_opt = n_bf.links.p_nom
    n_bf.storage_units.p_nom_opt = n_bf.storage_units.p_nom
    n_bf.lines.carrier = "AC"
    n_bf.lines.s_nom_opt = n_bf.lines.s_nom_min
    n_bf.links.loc[n_bf.links.carrier.isin(["DC"]), 'p_nom_opt'] = n_bf.links.loc[
        n_bf.links.carrier.isin(["DC"]), 'p_nom_min']
    change_p_nom_opt_carrier(n_bf)
    
    
    n_ext = n.copy()
    n_ext['hist'] = n_bf.copy()
    
    return n, n_ext
# %% Extract functions
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


def extract_res_potential(n):
    """
    Extract renewable potentials in GW.
    :param n: Network
    :return: Potentials of renewables in GW
    """
    dfx = []
    rx = re.compile("([A-z]+)[0-9]+\s[0-9]+\s([A-z\-\s]+)-*([0-9]*)")

    for y, ni in n.items():
        df = ni.generators[["p_nom_max", "p_nom_opt"]].reset_index()
        df[["region", "carrier", "build_year"]] = df["Generator"].str.extract(rx)
        df["carrier"] = df["carrier"].str.rstrip("-").replace(RENAMER)
        df["planning horizon"] = y
        df = df[df["carrier"].isin(["onwind", "offwind", "solar"])]
        dfx.append(
            df.groupby(["planning horizon", "carrier", "build_year", "region"]).sum(numeric_only=True) / 1e3
        )  # GW

    dfx = pd.concat(dfx)
    df_potential = pd.concat([
        (
            dfx.loc[
                dfx["p_nom_opt"].index.get_level_values("build_year") <
                dfx["p_nom_opt"].index.get_level_values("planning horizon").astype(str)
                , "p_nom_opt"]
            .groupby(["planning horizon", "carrier", "region"]).sum()
        ),
        (
            dfx.loc[
                dfx["p_nom_max"].index.get_level_values("build_year") ==
                dfx["p_nom_max"].index.get_level_values("planning horizon").astype(str)
                , "p_nom_max"]
            .groupby(["planning horizon", "carrier", "region"]).sum()
        )
    ], axis=1)

    df_potential["potential"] = df_potential["p_nom_max"] + df_potential["p_nom_opt"].fillna(0)
    df_potential = (
        df_potential.reset_index()
        .pivot(index=["carrier", "region"], columns="planning horizon", values="potential")
    )
    df_potential["units"] = "GW_e"
    return df_potential


def extract_marginal_prices(n, carrier_list=['AC']):
    df = []
    for ca in carrier_list:
        prices = pd.DataFrame([]).rename_axis(index='countries')
        for y, ni in n.items():
            if 'hist' != y:
                price_y = (
                    ni.buses_t.marginal_price[ni.buses.query("carrier == @ca ").index]
                    .mean()
                    .groupby(lambda x: x[:2]).mean()
                )
                prices[y] = price_y
        prices["carrier"] = ca.replace('AC', 'elec')
        df.append(prices.reset_index().set_index(['countries', 'carrier']))
    df = pd.concat(df, axis=0)
    return df


def calculate_imp_exp(country_map, transmission_t, y):
    countries = sorted(country_map.stack().unique())

    table_li_co = pd.DataFrame([], index=country_map.index)
    other_bus = pd.DataFrame([], index=transmission_t.columns)
    mat_imp = pd.DataFrame([], columns=countries, index=countries).rename_axis(
        index='countries')
    mat_exp = pd.DataFrame([], columns=countries, index=countries).rename_axis(
        index='countries')
    mat_imp['year'] = y
    mat_exp['year'] = y

    for co in countries:
        if "hist" != y:
            table_li_co[co] = country_map.apply(lambda x: -1 if x.bus0 == co else 0, axis=1)
            table_li_co[co] += country_map.apply(lambda x: 1 if x.bus1 == co else 0, axis=1)

            other_bus[co] = country_map.apply(lambda x: x.bus1 if x.bus0 == co else "", axis=1)
            other_bus[co] += country_map.apply(lambda x: x.bus0 if x.bus1 == co else "", axis=1)

            ie_raw = transmission_t.mul(table_li_co[co]) / 1e6  # TWh
            imp = ie_raw.where(ie_raw > 0, 0).sum(axis=0)
            exp = ie_raw.mask(ie_raw > 0, 0).sum(axis=0)

            mat_imp.loc[other_bus[co].loc[imp[imp > CLIP_VALUE_TWH].index], co] = imp[imp > CLIP_VALUE_TWH].values
            mat_exp.loc[other_bus[co].loc[exp[exp < -CLIP_VALUE_TWH].index], co] = exp[exp < -CLIP_VALUE_TWH].values

    return mat_imp.fillna(0), mat_exp.fillna(0), table_li_co, other_bus


def extract_transmission(n, carriers=["AC", "DC"],
                         units={"AC": "GW_e", "DC": "GW_e",
                                "gas pipeline": "GW_lhv,ch4", "gas pipeline new": "GW_lhv,ch4",
                                "H2 pipeline": "GW_lhv,h2", "H2 pipeline retrofitted": "GW_lhv,h2"}):
    capacities = []
    capacities_countries = []
    imports, exports = [], []
    # Add projected values
    for y, ni in n.items():

        transmission = []
        if "hist" != y:
            transmission_t = []
        for ca in carriers:
            if ca == "AC":
                transmission.append(ni.lines.rename(columns={"s_nom_opt": "p_nom_opt"}))
                if "hist" != y:
                    transmission_t.append(ni.lines_t.p0 * 8760 / len(ni.snapshots))
            else:
                transmission.append(ni.links.query('carrier == @ca'))
                if "hist" != y:
                    transmission_t.append(
                        ni.links_t.p0[ni.links.query('carrier == @ca').index] * 8760 / len(ni.snapshots))

        transmission = pd.concat(transmission)
        if "hist" != y:
            transmission_t = pd.concat(transmission_t, axis=1)

        buses_links = [c for c in transmission.columns if "bus" in c]
        country_map = transmission[buses_links].applymap(lambda x: bus_mapper(x, ni, column="country"))
        transmission_co = {}
        mono_co = {}
        for co in ni.buses.country.unique():
            transmission_co[co] = (transmission
                                   .query("@co == @country_map.bus0 or @co == @country_map.bus1")
                                   .groupby("carrier")
                                   .p_nom_opt.sum()
                                   )

            mono_co[co] = (
                transmission.loc[(transmission.index.str.contains('->')) & (transmission.index.str.contains('<'))]
                .query("@co == @country_map.bus0")
                .groupby("carrier")
                .p_nom_opt.sum()
            )

            if len(mono_co[co]):
                transmission_co[co].loc[mono_co[co].index] -= mono_co[co]

        transmission_co = pd.DataFrame.from_dict(transmission_co, orient='columns').fillna(0) / 1e3
        capacities_countries.append(pd.concat({y: transmission_co}, names=["Year"]))

        transmission_total = pd.DataFrame(transmission.groupby("carrier").p_nom_opt.sum()) / 1e3
        capacities.append(transmission_total.rename(columns={'p_nom_opt': y}))

        # imports/exports
        mat_imp, mat_exp, table_li_co, _ = calculate_imp_exp(country_map, transmission_t, y)
        if "hist" != y:
            imports.append(mat_imp.reset_index().set_index(['countries', 'year']))
            exports.append(mat_exp.reset_index().set_index(['countries', 'year']))

    df = pd.concat(capacities, axis=1)
    df_co = pd.concat(capacities_countries, axis=0)
    df_imp = pd.concat(imports, axis=0).fillna(0)
    df_exp = pd.concat(exports, axis=0).fillna(0)

    df["units"] = df.index.map(units)
    df_co["units"] = df_co.index.get_level_values(level=1).map(units)
    return df, df_co, df_imp


def extract_graphs(n, storage_function, storage_horizon, both=False, units={}, color_shift=None):
    carrier = list(storage_horizon.keys())
    if color_shift:
        pass
    else:
        color_shift = dict(zip(years, ['C' + str(i) for i in range(len(years))]))
    fig = plt.figure(figsize=(14, 8))

    def plotting(ax, title, data, y, unit):
        data.index = pd.to_datetime(pd.DatetimeIndex(data.index.values).strftime('2030-%m-%d-%H'))
        ax.plot(data, label=y, color=color_shift.get(y))
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel(unit, ha='left', y=1.1, rotation=0, labelpad=0.2)
        plt.xticks(rotation=30)
        plt.tight_layout()
        return

    for y, ni in n.items():
        lt = {}
        st = {}
        for car in carrier:
            storage = globals()[storage_function.get(car, "get_e_t")](ni, car)
            if both:
                lt[car] = storage
                st[car] = storage.iloc[:int(8 * 31)]
            elif "L" in storage_horizon.get(car):
                lt[car] = storage
            else:
                st[car] = storage.iloc[:int(8 * 31)]
        for i, (car, s) in enumerate(st.items()):
            ax = plt.subplot(3, 2, 2 * i + 1)
            plotting(ax, car, s, y, unit=r"${}$".format(units.get(car, '[TWh]')))
        for i, (car, l) in enumerate((lt).items()):
            ax = plt.subplot(3, 2, 2 * (i + 1))
            plotting(ax, car, l, y, unit=r"${}$".format(units.get(car, '[TWh]')))

    ax.legend()
    return {"": fig}


def extract_gas_phase_out(n, year):
    dimensions = ["country", "build_year"]
    n_cgt = (
        n[year].links[n[year].links.carrier.str.contains("CGT")]
        .merge(
            n[year].buses["country"].reset_index(),
            left_on="bus1",
            right_on="Bus",
            how="left"
        )
        .groupby(by=dimensions)
        ["p_carrier_nom_opt"]
        .sum(numeric_only=True)
        .reset_index()
    )
    n_cgt.loc[n_cgt["build_year"] < year, "build_year"] = "hist"
    n_cgt = (
                n_cgt
                .groupby(by=dimensions)
                .sum()
                .reset_index()
                .pivot(index="country", columns="build_year", values="p_carrier_nom_opt")
            ) / 1e3  # GW
    n_cgt['units'] = 'GW_e'

    if year in n_cgt.columns:
        sorting = year
    else:
        sorting = 'hist'

    n_cgt = n_cgt.sort_values(by=sorting, ascending=False)
    return n_cgt[n_cgt[sorting] >= 5].fillna(0)


def extract_country_capacities(n):
    # Duplicates scripts.make_summary.calculate_nodal_capacites
    for y, ni in n.items():
        df["nodal_capacities"] = calculate_nodal_capacities(ni, y, df["nodal_capacities"],
                                                            _opt_name={"Store": "e", "Line": "s", "Transformer": "s",
                                                                       "Link": "p_carrier"})

    df_capa = (df["nodal_capacities"]
               .rename(RENAMER)
               .reset_index()
               .rename(columns={"level_0": "unit_type",
                                "level_1": "node",
                                "level_2": "carrier"}))

    df_capa.node = df_capa.node.apply(lambda x: x[:2])

    # add extarction and storage suffixes
    dico = {"generators": "_extraction", "stores": "_stores"}
    for d, suffix in dico.items():
        to_modify = df_capa.query(
            'unit_type in [@d] and carrier in ["gas", "oil", "coal/lignite", "uranium", "solid biomass"]').index
        df_capa.loc[to_modify, ["carrier"]] += suffix

    df_capa = df_capa.groupby(["unit_type", "node", "carrier"]).sum().reset_index(["carrier", "unit_type"])

    df_capa = df_capa.drop(columns="unit_type").groupby(["node", "carrier"]).sum() / 1e3

    df_capa.loc[(slice(None), "Haber-Bosch"), :] *= 4.415385
    df_capa["units"] = "GW_e"
    df_capa.loc[(slice(None), ["Haber-Bosch", "ammonia cracker"]), "units"] = "GW_lhv,nh3"
    df_capa.loc[(slice(None), ["Sabatier"]), "units"] = "GW_lhv,h2"
    df_capa.loc[(slice(None), ["H2 Store"]), "units"] = "GWh_lhv,h2"
    df_capa.loc[(slice(None), ["battery", "home battery"]), "units"] = "GWh_e"
    df_capa.loc[(slice(None), ["gas_extraction", "oil_extraction",
                               "coal/lignite_extraction", "uranium_extraction"]), "units"] = "GW_lhv"
    return df_capa


def extract_nodal_costs():
    # Todo : add handling of multiple runs
    df = (pd.read_csv(Path(path, 'results', 'csvs', 'nodal_costs.csv'),
                      index_col=[0, 1, 2, 3],
                      skiprows=3,
                      header=0)
          .reset_index()
          .rename(columns={"planning_horizon": "type",
                           "level_1": "cost",
                           "level_2": "country",
                           "level_3": "carrier"})
          )
    df['country'] = df['country'].str[:2].fillna('EU')
    fuels = df.query("carrier in ['gas','oil','coal','lignite','uranium'] and cost == 'marginal' and type == 'generators'").index
    biomass = df.query("carrier.str.contains('biomass') and cost == 'marginal' and type == 'stores'").index
    df.loc[fuels.union(biomass),'cost'] = 'fuel'
    df = df.set_index(['type', 'cost', 'country', 'carrier'])
    df = df.fillna(0).groupby(['type', 'cost', 'country', 'carrier']).sum()
    df = df.loc[~df.apply(lambda x: x < 1e3).all(axis=1)]
    df.insert(0, column="units", value="Euro")
    return df

def extract_res_statistics(n):
    
    df = []
    for y, ni in n.items():
        
        res = ni.generators.copy().query("carrier in @RES")
        res_t = ni.generators_t.p[res.index]
        res['year'] = y
        cf = (res_t / (res["p_nom_opt"])).mean()
        p_tot = res_t.sum() * ni.snapshot_weightings.generators.mean()
        opex = p_tot * ni.generators.marginal_cost
        capex = res.capital_cost * res.p_nom_opt
    
        res.loc[cf.index, "cf"] = cf
        res.loc[cf.index, "p_tot"] = p_tot
        res.loc[cf.index, "opex"] = opex
        res.loc[cf.index, "capex"] = capex
        res.loc[cf.index, "totex"] = res.loc[cf.index,
                                             "capex"] + res.loc[cf.index, "opex"]
    

        LCOE = res.groupby(by=["carrier"]).totex.sum() / \
            res.groupby(by=["carrier"]).p_tot.sum()
        LCOE = LCOE.rename("carrier").to_frame().rename(
            columns={"carrier": "LCOE"})
        res = res.reset_index().merge(LCOE, on="carrier").set_index(["Generator","year"])
        res = res.loc[:,
                      ["carrier", 'bus', "capital_cost", "marginal_cost", "p_nom_opt", "build_year", "p_nom", "cf", 'p_tot',
                       "p_nom_max", "LCOE", "opex", "capex"]]
        res = res.sort_values(by="carrier")
        df.append(res)
    
    return pd.concat(df)

rx = re.compile(r"([A-Z]{2})[0-9]\s[0-9]")


def renamer_to_country(x):
    if rx.match(x):
        return rx.match(x).group(1)
    else:
        return x


def extract_nodal_supply_energy(n):
    labels = {y: label[:-1] + (y,) for y in n.keys()}
    columns = pd.MultiIndex.from_tuples(labels.values(), names=["cluster", "ll", "opt", "planning_horizon"])
    df = pd.DataFrame(columns=columns, dtype=float)
    for y, ni in n.items():
        df = calculate_nodal_supply_energy(ni, label=labels[y], nodal_supply_energy=df)
    idx = ["carrier", "component", "node", "item"]
    df.index.names = idx
    df.columns = df.columns.get_level_values(3)

    df = df.reset_index()
    df["node"] = df["node"].map(renamer_to_country)
    df = df.set_index(idx)

    df = df * 1e-6  # TWh
    df["units"] = "TWh"

    sector_mapping = pd.read_csv(
        Path(path.resolve().parents[1], "sector_mapping.csv"), index_col=[0, 1, 2], header=0).dropna()
    df = df.merge(sector_mapping, left_on=["carrier", "component", "item"], right_index=True, how="left")
    return df


def extract_nodal_oil_load(nhours=8760):
    resources = Path(path, "resources")
    nyears = nhours / 8760
    options = config["sector"]

    def read_load(f):
        try:
            df = pd.read_csv(f, index_col=0, header=0)
            return df
        except FileNotFoundError:
            logging.warning(f"Extract nodal oil load miss a file. Mind incoherent results. Missing file : {f.name}.")
            return pd.DataFrame()

    df_shipping = read_load(Path(resources, "shipping_demand_s181_37m.csv"))
    df_transport = read_load(Path(resources, "transport_demand_s181_37m.csv"))
    df_pop_weighted_energy_totals = read_load(Path(resources, "pop_weighted_energy_totals_s181_37m.csv"))
    df_industry = {}
    for y in years:
        df_industry[y] = read_load(Path(resources, f"industrial_energy_demand_elec_s181_37m_{y}.csv"))

    df_oil = []
    for y in years:
        # Land transport
        ice_share = get(options["land_transport_ice_share"], y)
        ice_efficiency = options["transport_internal_combustion_efficiency"]
        transport = (ice_share / ice_efficiency * df_transport.sum() / nhours).rename(index="transport")

        # Shipping
        shipping_oil_share = get(options["shipping_oil_share"], y)
        domestic_navigation = df_pop_weighted_energy_totals["total domestic navigation"]
        international_navigation = (df_shipping * nyears)["0"]
        all_navigation = domestic_navigation + international_navigation
        p_set = all_navigation * 1e6 / nhours
        shipping = (shipping_oil_share * p_set).rename(index="shipping")

        # Industry
        demand_factor = options.get("HVC_demand_factor", 1)
        industry = (demand_factor * df_industry[y]["naphtha"] * 1e6 * nyears / nhours).rename(index="industry")

        # Aviation
        demand_factor = options.get("aviation_demand_factor", 1)
        all_aviation = ["total international aviation", "total domestic aviation"]
        aviation = (demand_factor * df_pop_weighted_energy_totals[all_aviation].sum(axis=1) * 1e6 / nhours) \
            .rename(index="aviation")

        # Agriculture
        oil_share = get(options["agriculture_machinery_oil_share"], y)
        machinery_nodal_energy = df_pop_weighted_energy_totals["total agriculture machinery"]
        agriculture = (oil_share * machinery_nodal_energy * 1e6 / nhours).rename(index="agriculture")

        df = pd.concat([transport, shipping, industry, aviation, agriculture], axis=1)
        df["year"] = y
        df = df.reset_index().rename(columns={"index": "node"})
        df["node"] = df["node"].map(renamer_to_country)
        df = df.groupby(by=["year", "node"]).sum()

        df = df * nhours / 1e6  # from MW to TWh

        df_oil.append(df)

    df_oil = pd.concat(df_oil)

    return df_oil


def extract_loads(n):
    profiles = {}
    for y, ni in n.items():
        loads_t = ni.loads_t.p.T
        loads_t.index.names = ['Load']
        loads_t["country"] = ni.buses.loc[ni.loads.loc[loads_t.index].bus].country.values
        loads_t.reset_index(inplace=True)
        loads_t["Load"].mask(loads_t["Load"].str.contains("NH3"), "NH3 for sectors", inplace=True)
        loads_t["Load"].mask(loads_t["Load"].str.contains("H2"), "H2 for sectors", inplace=True)
        loads_t["Load"].where(loads_t["Load"].str.contains("sectors"), "Electricity demand for sectors", inplace=True)

        loads_t = loads_t.groupby(["country", "Load"]).sum()
        loads_t.insert(0, column="Annual sum [TWh]", value=loads_t.sum(axis=1) / 1e6 * 8760 / len(ni.snapshots))
        profiles[y] = loads_t

    df = pd.concat(profiles, names=["Years"])
    df.insert(0, column="units", value='MW_e')
    df.loc[(slice(None), slice(None), 'H2 for sectors'), 'units'] = 'MW_lhv,h2'
    df.loc[(slice(None), slice(None), 'NH3 for sectors'), 'units'] = 'MW_lhv,nh3'
    return df


def extract_series(n):
    with plt.style.context(["ggplot"]):
        with open(Path(path, 'results/configs/config.snakemake.yaml'), 'r') as f:
            df = safe_load(f)["plotting"]
            plots = {}
            for y, ni in n.items():
                with pd.option_context('mode.chained_assignment', None):
                    plots[y] = plot_series(ni, carrier="electricity", name="electricity", year=str(y),
                                                load_only=True, colors=df["tech_colors"], path=Path(csvs, f"series_AC_{y}.png"), save = False)
    return plots                  

def extract_plot_capacities(n):
    with plt.style.context(["ggplot"]):
        with open(Path(path, 'results/configs/config.snakemake.yaml'), 'r') as f:
            df = safe_load(f)["plotting"]
            plots = {}
            for y, ni in n.items():
                with pd.option_context('mode.chained_assignment', None):
                    plots[y] = plot_capacity(ni, colors=df["tech_colors"], _map_opts=df["map"],
                                                  bus_size_factor=1e5, path=Path(csvs, f"capacities_{y}.png"),
                                                  run_from_rule=False, transmission=True, save = False)
    return plots                

def extract_data(n, n_ext, years, n_path, n_name, color_shift=None):
    df["nodal_capacities"] = pd.DataFrame(columns=years, dtype=float)

    # DataFrames to extract
    capa_country = extract_country_capacities(n_ext)
    n_gas = extract_gas_phase_out(n, years[0])
    n_res_pot = extract_res_potential(n)

    ACDC_grid, ACDC_countries, el_imp = extract_transmission(n_ext)
    H2_grid, H2_countries, H2_imp = extract_transmission(n_ext, carriers=["H2 pipeline", "H2 pipeline retrofitted"])
    gas_grid, gas_countries, gas_imp = extract_transmission(n_ext, carriers=["gas pipeline", "gas pipeline new"])
    el_imp['carriers'] = 'elec'
    el_imp = el_imp.reset_index().set_index(['countries', 'year', 'carriers'])
    H2_imp['carriers'] = 'h2'
    H2_imp = H2_imp.reset_index().set_index(['countries', 'year', 'carriers'])
    gas_imp['carriers'] = 'gas'
    gas_imp = gas_imp.reset_index().set_index(['countries', 'year', 'carriers'])

    imports = pd.concat([el_imp, H2_imp, gas_imp])
    nodal_supply_energy = extract_nodal_supply_energy(n)
    nodal_oil_load = extract_nodal_oil_load(nhours=n_ext['hist'].snapshot_weightings.generators.sum())

    n_costs = extract_nodal_costs()
    marginal_prices = extract_marginal_prices(n, carrier_list=['gas', 'AC'])
    n_loads = extract_loads(n) 
    res_stats = extract_res_statistics(n)

    # n_profile = extract_production_profiles(n, subset=LONG_LIST_LINKS + LONG_LIST_GENS)

    ## Figures to extract
    plt.close('all')
    # Storage
    mpl.rcParams.update(mpl.rcParamsDefault)
    if color_shift:
        pass
    else:
        color_shift = dict(zip(years, ['C0', 'C2', 'C1']))

    storage_function = {"hydro": "get_state_of_charge_t", "PHS": "get_state_of_charge_t"}
    storage_horizon = {"hydro": "LT", "PHS": "ST", "H2 Store": "LT",
                       "battery": "ST", "home battery": "ST",
                       "ammonia store": "LT"}
    n_sto = extract_graphs(n, storage_function, storage_horizon, color_shift=color_shift)
    # h2
    storage_function = {"H2 Fuel Cell": "get_p_carrier_nom_t", "H2 Electrolysis": "get_p_carrier_nom_t"}
    storage_horizon = {"H2 Fuel Cell": "LT", "H2 Electrolysis": "LT", "H2 Store": "LT"}
    n_h2 = extract_graphs(n, storage_function, storage_horizon, color_shift=color_shift,
                          both=True, units={"H2 Fuel Cell": "[GW_e]", "H2 Electrolysis": "[GW_e]",
                                            "H2 Store": "[TWh_{lhv,h2}]"})

    # Figures
    series_consumption = extract_series(n)
    map_capacities = extract_plot_capacities(n)

    outputs = {
        # assets
        'units_capacities_countries' : capa_country,
        'gas_phase_out' : n_gas,
        'res_potentials' : n_res_pot ,
        
        # networks
        'grid_capacities_countries' : ACDC_countries,
        'H2_network_capacities_countries' : H2_countries,
        'gas_network_capacities_countries' : gas_countries,
        'grid_capacities' : ACDC_grid,
        'H2_network_capacities' : H2_grid,
        'gas_network_capacities' : gas_grid,
        
        #energy balance
        'imports_exports' : imports,
        'supply_energy_sectors' : nodal_supply_energy,
        'nodal_oil_load' : nodal_oil_load,
        
        # insights
        'costs_countries' : n_costs,
        'marginal_prices_countries' : marginal_prices,
        'res_statistics' : res_stats,
        'loads_profiles' : n_loads,
        }
    
    figures = {
        'storage_unit' : n_sto,
        'h2_production' : n_h2,
        'series_consumption' : series_consumption,
        'map_capacities' : map_capacities
        }
    
    return outputs, figures

def export_csvs_figures(csvs, outputs, figures):
    csvs.mkdir(parents=True, exist_ok=True)

    for f_name, f in figures.items():
        for y, plot in f.items():
            plot.savefig(Path(csvs,f"{f_name}_{y}.png"), transparent=True)
        
    for o_name, o in outputs.items():
        o.to_csv(Path(csvs,f"{o_name}.csv"))
    
    logger.info(f"Exported files and figures to folder : {csvs}")

    return
    # n_profile.to_csv(Path(csvs, "generation_profiles.csv"))

    return


# %% Unit countries capacities load
def _load_capacities(techs, historical="Historical (installed capacity by 2025)", countries=None):
    """
    
    Parameters
    ----------
    techs : list
        List of techs to filter on.
    historical : str, optional
        String to replace hist with. The default is "Historical (installed capacity by 2025)".
    countries : list, optional
        List of countries to filter on. The default is None.

    Returns
    -------
    DataFrame
        Generic load function, for which are specified the technologies to filter from. If a different
        name from historical is to be set, or if a subset of countries is needed, it can be provided.
.

    """
    df = (
        pd.read_csv(Path(csvs, "units_capacities_countries.csv"), header=0)
        .drop(columns=["units"])
        .query("carrier in @techs")
    )
    if countries:
        df = df.query("node in @countries")

    idx = ["carrier"]
    df = (
        df.groupby(by=idx).sum().reset_index()
        .reindex(columns=excel_columns["all_years"])
        .rename(columns={"hist": historical})
    )

    df = df.set_index(idx)
    df = df[df.sum(axis=1) >= CLIP_VALUE_GW * (len(years) + 1)]
    df = df.reset_index()

    return df


def _load_supply_energy(load=True, carriers=None, countries=None):
    """
    Load nodal supply energy data and aggregate on carrier and sector, given some conditions.
    :param load: If True, keep only load data (negatives values)
    :param carriers: If specified, keep only a given carrier
    :param countries: If specified, keep a specific list of countries
    :return:
    """
    df = (
        pd.read_csv(Path(csvs, "supply_energy_sectors.csv"), header=0)
    )

    def get_load_supply(x):
        if load:
            return x.where(x <= 0, np.nan) * -1
        else:
            return x.where(x > 0, np.nan)

    df[years_str] = df[years_str].apply(get_load_supply)
    df = df.dropna(subset=years_str, how="all")

    if carriers:
        df = df.query("carrier in @carriers")
    if countries:
        df = df.query("node in @countries")

    idx = ["carrier", "sector"]
    df = (
        df.groupby(by=idx).sum().reset_index()
        .reindex(columns=excel_columns["future_years_sector"])
    )

    df = df.set_index(idx)
    df = df[df.sum(axis=1) >= CLIP_VALUE_TWH * len(years)]
    df = df.reset_index()

    return df


def _load_nodal_oil(countries):
    df = (
        pd.read_csv(Path(csvs, "nodal_oil_load.csv"), header=0)
    )

    if countries:
        df = df.query("node in @countries")

    df = (
        df.groupby(by="year").sum(numeric_only=True)
        .T
        .reset_index()
        .rename(columns={"index": "sector"})
    )
    df.columns = [str(c) for c in df.columns]
    df["carrier"] = "oil"
    return df


def _load_supply_energy_dico(load=True, countries=None):
    """
    Allow to split _load_supply_energy into its carriers for it 
    to be written tab by tab

    """
    dico = {}
    supply_energy_carrier = (_load_supply_energy(load=load, countries=countries)
                             .carrier
                             .replace({"AC": "electricity", "low voltage": "electricity"})
                             .unique()
                             )

    for ca in supply_energy_carrier:
        ca_name = HEAT_RENAMER[ca] if ca in HEAT_RENAMER else ca
        # Todo : should we move the HV/LV/imports/exports to the calling function to keep this function read only (no modifications) ? 
        if ca == "electricity":
            df_ac = _load_supply_energy(load=load, countries=countries, carriers="AC")
            df_low = _load_supply_energy(load=load, countries=countries, carriers="low voltage")
            df = pd.concat([df_ac, df_low])
            del df["carrier"]
            df = df.groupby(by="sector").sum().reset_index()

            if not (load) and countries:
                df_imp = _load_imp_exp(export=False, countries=countries, carriers='elec', years=years)
                df_exp = _load_imp_exp(export=True, countries=countries, carriers='elec', years=years)
                df_net_imp = (df_imp[years] - df_exp[years]).sum()
                df = pd.concat([df, pd.DataFrame(['Net Imports'] + df_net_imp.values.tolist(), index=df.columns).T])
            df = pd.concat([pd.DataFrame(df.sum(axis=0).values.tolist(), index=df.columns).T, df])
            df.iloc[0, 0] = 'Total'
            df.drop(df.query('sector == "V2G"').index, inplace=True)
            dico[ca_name] = df
        elif ca == "oil":
            if load and countries is not None:  # if load and countries exist
                df_eu_load = _load_nodal_oil(countries)
                df_c_load = _load_supply_energy(load=load, countries=countries, carriers=ca)
                dico[ca_name] = pd.concat([df_c_load, df_eu_load])
            else:
                dico[ca_name] = _load_supply_energy(load=load, countries=countries, carriers=ca)
        else:
            dico[ca_name] = _load_supply_energy(load=load, countries=countries, carriers=ca)

    return dico


def load_capacities(tech_list, historical):
    """
    Generic function loading for each countries subset the type given in input and the name associated,
    as well as the historical memo. 

    """
    dico = {}
    for co_name, subset in countries.items():
        dico[f"{co_name}"] = _load_capacities(tech_list, countries=subset, historical=historical)
    return dico


def load_res_capacities():
    return load_capacities(RES, historical="Historical (planned by 2022)")


def load_production():
    return load_capacities(PRODUCTION, historical="Historical (installed capacity by 2025)")


def load_balance():
    return load_capacities(BALANCE, historical="Historical")


def load_long_term_storage():
    return load_capacities(LONG_TERM_STORAGE, historical="Historical (installed capacity by 2025)")


def load_short_term_storage():
    return load_capacities(SHORT_TERM_STORAGE, historical="Historical (installed capacity by 2025)")


def load_fossil_fuels():
    return load_capacities(FF_ELEC + FF_HEAT, historical="Historical (installed capacity by 2025)")


def load_h2_capacities():
    return load_capacities(H2, historical="Historical (installed capacity by 2025)")


def load_load_sectors():
    dico = {}
    for co_name, subset in countries.items():
        to_rename = _load_supply_energy_dico(load=True, countries=subset)
        for k, v in to_rename.items():
            dico[f"{co_name}_{k}"] = v
    return dico


def load_supply_sectors():
    dico = {}
    for co_name, subset in countries.items():
        to_rename = _load_supply_energy_dico(load=False, countries=subset)
        for k, v in to_rename.items():
            dico[f"{co_name}_{k}"] = v
    return dico


def load_supply_heat_be():
    dico = _load_supply_energy_dico(load=False, countries=["BE"])
    data = pd.DataFrame()

    for k, v in dico.items():
        to_add = v.copy()
        to_add['carrier'] = HEAT_RENAMER.get(k)
        data = pd.concat([data, to_add], ignore_index=True)

    for heat in ["dec_heat", "cent_heat"]:
        df[heat] = (
            data.loc[data.carrier.isin([heat])]
            .drop(columns="carrier")
            .groupby('sector').sum()
            .reset_index()
        )
    return df


# %% Costs load

# generic function for calling costs
def _load_costs_year_segment(year=None, _countries=None, cost_segment=None):
    """
    Return the costs per segment for a given year or per year for a given segment,
    considering a subset of countries to consider
    Parameters
    ----------
    year : TYPE, optional
        DESCRIPTION. The default is None.
    _countries : TYPE, optional
        DESCRIPTION. The default is None.
    cost_segment : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    """
    df = pd.read_csv(Path(csvs, "costs_countries.csv"), header=0)
    prices = pd.read_csv(Path(csvs, 'marginal_prices_countries.csv'), header=0)

    if _countries:
        df = df.query("country in @_countries")
        countries = list(set(_countries).intersection(set(df.country.unique())))
        
    else:
        countries = prices.countries.unique()

    cost_mapping = pd.read_csv(
        Path(path.resolve().parents[1], "cost_mapping.csv"), index_col=[0, 1], header=0).dropna()
    df = (
        df.merge(cost_mapping, left_on=["carrier", "type"], right_index=True, how="left")
        .groupby(["cost_segment", "cost"]).sum(numeric_only=True)
        .reset_index()
    )

    if cost_segment:
        net_cost = pd.DataFrame([], columns=imp_exp_carriers, index=years)
        if cost_segment != "Net_Imports":
            df = df.query('cost_segment in @cost_segment')

        if cost_segment == "Production" or cost_segment == "Net_Imports":
            for y in years:
                for ca in imp_exp_carriers:
                    imp = _load_imp_exp(export=False, countries=countries, carriers=ca, years=[y]).set_index(
                        'countries') * 1e6  # MWh
                    exp = _load_imp_exp(export=True, countries=countries, carriers=ca, years=[y]).set_index(
                        'countries') * 1e6  # MWh
                    price_ca = prices.query("carrier == @ca").set_index('countries').loc[:, str(y)]  # €/MWh
                    net_cost.loc[y, ca] = 0
                    if len(imp) > 0:
                        net_cost.loc[y, ca] += price_ca.loc[imp.index].dot(imp).sum()
                    if len(exp) > 0:
                        net_cost.loc[y, ca] -= (price_ca.loc[countries].mean() * exp).values.sum()
            df.loc[df.query("'fuel' in cost").index, years_str] += net_cost.sum(axis=1).values
        if cost_segment == "Net_Imports":
            df = net_cost.reset_index()

    else:
        df = (
            df.pivot(columns="cost", values=year, index="cost_segment")
            .fillna(0)
            .reset_index()
        )
    return df


def _load_costs(per_segment=False, per_year=False):
    dico = {}
    for co_name, subset in countries.items():
        if per_segment:
            for seg_name, seg in COST_SEGMENTS.items():
                dico[f"{seg_name}_{co_name}"] = _load_costs_year_segment(_countries=subset, cost_segment=seg)
                # print(seg_name,seg)
        elif per_year:
            for y in years_str:
                dico[f"{y}_{co_name}"] = _load_costs_year_segment(_countries=subset, year=y)
        else:
            logging.warning("Unkown configuration to load costs.")
    return dico


def load_costs_years():
    return _load_costs(per_year=True)


def load_costs_segments():
    return _load_costs(per_segment=True)


# def load_costs_2030_be():
#     return (
#         _load_costs("2030", countries=["BE"])
#     )


# def load_costs_2030_eu27():
#     return (
#         _load_costs("2030", countries=eu27_countries)
#     )


# def load_costs_2040_be():
#     return (
#         _load_costs("2040", countries=["BE"])
#     )


# def load_costs_2040_eu27():
#     return (
#         _load_costs("2030", countries=eu27_countries)
#     )


# def load_costs_2050_be():
#     return (
#         _load_costs("2050", countries=["BE"])
#     )


# def load_costs_2050_eu27():
#     return (
#         _load_costs("2030", countries=eu27_countries)
#     )


# def load_costs_prod_be():
#     return (
#         _load_costs(cost_segment='Production', countries=["BE"])
#     )


# def load_costs_tran_be():
#     return (
#         _load_costs(cost_segment='Transport', countries=["BE"])
#     )


# def load_costs_bal_be():
#     return (
#         _load_costs(cost_segment='Balancing', countries=["BE"])
#     )


def load_costs_total():
    return (
        pd.read_csv(Path(csvs, "costs_countries.csv"), header=0)
    )


# def load_costs_res():
#     # ToDo Add segments and subsegments
#     return pd.DataFrame()


# def load_costs_flex():
#     # ToDo Add segments and subsegments
#     return pd.DataFrame()


# def load_costs_segments():
#     # ToDo Add segments and subsegments
#     return pd.DataFrame()


# def load_costs_thermal():
#     # ToDo Add segments and subsegments
#     return pd.DataFrame()


# def load_costs_type():
#     # ToDo Add segments and subsegments
#     return pd.DataFrame()


# %% Non standard loads
def _load_imp_exp(export=True, countries=None, carriers=None, years=None):
    """
    Return the imports or export of a subset of countries per country external to the subset
    for a given carrier. Since the network imports/exports are zero-sum, the exports can be obtained 
    from the imports matrix
    
    """
    imp_exp = []
    for y in years:
        df_carrier = (
            pd.read_csv(Path(csvs, "imports_exports.csv"), header=0)
            .query('carriers == @carriers')
            .query('year == @y')
            .drop(columns=['carriers', 'year'])
            .set_index('countries')
        )

        if export:
            df_carrier = (df_carrier.T)
        try:
            if countries and len(df_carrier):
                df_carrier = df_carrier.loc[df_carrier.columns.difference(countries),
                df_carrier.columns.intersection(countries)]
        except:
            pass
        imp_exp.append(df_carrier
                       .sum(axis=1)
                       .rename(y)
                       .to_frame())
    imp_exp = pd.concat(imp_exp, axis=1).rename_axis(index="countries")
    return (
        imp_exp.loc[~(imp_exp == 0).all(axis=1)].reset_index()
    )


def load_imports_exports():
    dico = {}

    for imp_exp, exp_value in {'imports': False, 'exports': True}.items():
        for ca in imp_exp_carriers:
            for name, subset in countries.items():
                dico[f"{imp_exp}_{ca}_{name}"] = _load_imp_exp(export=exp_value,
                                                               countries=subset, carriers=ca, years=years)
    return dico


# def load_imports_be():
#     carriers = ['elec']  # ['elec','gas','h2']
#     dico = {}
#     for ca in carriers:
#         dico[ca] = _load_imp_exp(export=False, countries=['BE'], carriers=ca, years=years)
#     return dico


# def load_exports_be():
#     carriers = ['elec']  # ['elec','gas','h2']
#     dico = {}
#     for ca in carriers:
#         dico[ca] = _load_imp_exp(export=True, countries=['BE'], carriers=ca, years=years)
#     return dico


def load_gas_phase_out():
    return (
        pd.read_csv(Path(csvs, "gas_phase_out.csv"), header=0)
        .reindex(columns=excel_columns["first_hist_units"])
        .rename(columns={"hist": "Historical (planned by 2025)"})
    )


def load_grid_capacities():
    return (
        pd.read_csv(Path(csvs, "grid_capacities.csv"), header=0)
        .reindex(columns=excel_columns["all_years_units"])
        .rename(columns={"hist": "Historical (planned by 2025)"})
    )


def load_grid_capacities_countries():
    return (
        pd.read_csv(Path(csvs, "grid_capacities_countries.csv"), header=0)
        .groupby('Year').sum(numeric_only=True)
        .loc[:, ['LU', 'GB', 'NL', 'DE', 'FR']]
        .reset_index()
        .rename(columns={"hist": "Historical (planned by 2025)"})
    )


def load_h2_network_capacities():
    return (
        pd.read_csv(Path(csvs, "H2_network_capacities.csv"), header=0)
        .reindex(columns=excel_columns["all_years_units"])
        .rename(columns={"hist": "Historical (planned by 2025)"})
    )


def load_h2_network_capacities_countries():
    return (
        pd.read_csv(Path(csvs, "H2_network_capacities_countries.csv"), header=0)
        .groupby('Year').sum(numeric_only=True)
        .loc[:, ['LU', 'GB', 'NL', 'DE', 'FR']]
        # .rename(columns={"hist": "Historical (planned by 2025)"})
        .reset_index()
    )


def load_res_potentials():
    return (
        pd.read_csv(Path(csvs, "res_potentials.csv"), header=0)
        .drop(columns=years_str[:-1])
        .groupby(by="carrier").agg({years_str[-1]: "sum", "units": "first"}).reset_index()
        .reindex(columns=excel_columns["last_units"])
    )


def load_res_potentials_be():
    return (
        pd.read_csv(Path(csvs, "res_potentials.csv"), header=0)
        .query("region == 'BE'")
        .drop(columns=years_str[:-1])
        .reindex(columns=excel_columns["last_units"])
    )


# def load_h2_production():
#     return (
#         pd.read_csv(Path(csvs, "generation_profiles.csv"), header=0)
#         .rename(columns={"Carrier": "carrier"})
#         .query("carrier in ['H2 Electrolysis', 'H2 Fuel Cell']")
#         .groupby(by=["Year", "carrier"]).agg({"Annual sum [TWh]": "sum"})
#         .pivot_table(index="carrier", columns="Year", values="Annual sum [TWh]")
#         .reset_index()
#     )


def load_industrial_demand():
    return (
        pd.read_csv(Path(csvs, "loads_profiles.csv"), header=0)
        .query("Load != 'Electricity demand for sectors'")
        .groupby(by=["Load", "Years"]).agg({"Annual sum [TWh]": "sum"}).reset_index()
    )


# def load_production_profile():
#     return (
#         pd.read_csv(Path(csvs, "generation_profiles.csv"), header=0)
#         [["Year", "Country", "Carrier", "Annual sum [TWh]"]]
#         .query("Carrier in ['ammonia cracker', 'battery charger', 'H2 Electrolysis', 'Haber-Bosch', 'helmet', "
#                "'home battery charger', 'Sabatier']")
#         .groupby(by=["Year", "Carrier"]).sum(numeric_only=True).reset_index()
#         .rename(columns={"Carrier": "carrier"})
#         .replace(RENAMER)
#         .pivot_table(index="carrier", columns="Year", values="Annual sum [TWh]")
#         .reset_index()
#     )


# %%
def export_data():
    outputs = [
        # Standard load
        "res_capacities",
        "production",
        "balance",
        "long_term_storage",
        "short_term_storage",
        "fossil_fuels",
        "h2_capacities",
        "load_sectors",
        "supply_sectors",
        "supply_heat_be",

        # Costs
        "costs_years",
        "costs_segments",

        # Non standard
        "gas_phase_out",
        "grid_capacities",
        "grid_capacities_countries",
        "h2_network_capacities",
        "h2_network_capacities_countries",
        "res_potentials",
        "industrial_demand",
        "imports_exports",
        # "production_profile",
    ]

    with pd.ExcelWriter(Path(path, "graph_extraction_raw.xlsx")) as xl:
        for output in outputs:
            o = globals()["load_" + output]()
            if isinstance(o, pd.DataFrame):
                o.to_excel(xl, sheet_name=output, index=False)
            elif isinstance(o, dict):
                for k, v in o.items():
                    v.to_excel(xl, sheet_name=output + "_" + k, index=False)
            else:
                logging.warning(f"Given output for {output} is not mapped out to output file.")
    return


# %% Main
if __name__ == "__main__":
    # for testing
    config_file = "config.VEKA_Average.runner.yaml"
    path = Path("analysis", "VEKA_av_bio_fix_nuc_bev_ccl")
    dir_export = "graph_data"
    n_path = Path(path, "results")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    years = config["scenario"]["planning_horizons"]
    years_str = list(map(str, years))
    simpl = 181
    cluster = "37m"
    opts = ""
    sector_opts = "3H-I-T-H-B-A-CCL"
    ll = "v3.0"
    label = (cluster, ll, sector_opts, years)

    networks_dict = {
        planning_horizon: "results/"
                          + f"/postnetworks/elec_s{si}_{clu}_l{l}_{opt}_{sector_opt}_{planning_horizon}.nc"
        for si in [simpl]
        for clu in [cluster]
        for opt in [opts]
        for sector_opt in [sector_opts]
        for l in [ll]
        for planning_horizon in years
    }

    n_name = f"elec_s{simpl}_{cluster}_l{ll}_{opts}_{sector_opts}_"
    csvs = Path(path, f"{dir_export}_{n_name}")

    df = {}



    # Todo : type cast all references to year into str or int
    excel_columns = {"all_years": ["carrier", "hist"] + years_str,
                     "all_years_units": ["carrier", "hist"] + years_str + ["units"],
                     "future_years": ["carrier"] + years_str,
                     "future_years_sector": ["carrier", "sector"] + years_str,
                     "first_year_units": ["carrier"] + [years_str[0], "units"],
                     "last_hist_units": ["carrier", "hist"] + [years_str[-1], "units"],
                     "last_units": ["carrier"] + [years_str[-1], "units"],
                     "first_hist_units": ["carrier", "hist"] + [years_str[0], "units"]}

    export = 'y'

    # global variables for which to do work
    countries = {"tot": None, "be" : ["BE"], "eu27": EU27_COUNTRIES}
    # countries = {"be": ['BE']}
    imp_exp_carriers = ['elec', 'gas', 'H2']

    logger.info(f"Extracting from {path}")
    n, n_ext = prepare_networks(n_path)
    outputs, figures = extract_data(n, n_ext, years, n_path, n_name)
    export_csvs_figures(csvs, outputs, figures)

    export_data()
