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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

from make_summary import assign_carriers
from make_summary import assign_locations
from make_summary import calculate_nodal_capacities

logger = logging.getLogger(__name__)

LONG_LIST_LINKS = ["coal/lignite", "oil", "CCGT", "OCGT",
                   "H2 Electrolysis", "H2 Fuel Cell", "battery charger",
                   "home battery charger", "Haber-Bosch", "Sabatier",
                   "ammonia cracker", "helmeth", "SMR", "SMR CC", "hydro"]

LONG_LIST_GENS = ["solar", "solar rooftop", "onwind", "offwind", "offwind-ac", "offwind-dc",
                  "ror", "nuclear", "urban central solid biomass CHP",
                  "home battery", "battery", "H2 Store", "ammonia store"]


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


def mapper(x, n, to_apply=None):
    if x in n.buses.index:
        return n.buses.loc[x, to_apply]
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
    carrier_map = li[buses_links].applymap(lambda x: mapper(x, n, to_apply="carrier"))

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


def select_countries(n, countries):
    index = n.buses.loc[n.buses.country.isin(countries)].index
    n.generators = reduce_to_countries(n.generators, index)
    n.lines = reduce_to_countries(n.lines, index)
    n.links = reduce_to_countries(n.links, index)
    n.stores = reduce_to_countries(n.stores, index)
    n.storage_units = reduce_to_countries(n.storage_units, index)
    n.loads_t.p = n.loads_t.p.loc[:, n.loads_t.p.columns.str[:2].isin(countries)]
    n.loads_t.p_set = n.loads_t.p_set.loc[:, n.loads_t.p_set.columns.str[:2].isin(countries)]
    n.buses = n.buses.loc[index]
    return n


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


def extract_production_profiles(n, subset):
    renamer = {"offwind-dc": "offwind", "offwind-ac": "offwind", "coal": "coal/lignite",
               "lignite": "coal/lignite", "ror": "hydro", 'urban central solid biomass CHP': 'biomass CHP'}

    profiles = []
    for y, ni in n.items():
        # Grab data from various sources
        n_y_t = pd.concat([
            ni.links_t.p_carrier_nom_opt,
            ni.generators_t.p,
            ni.storage_units_t.p
        ], axis=1)
        n_y = pd.concat([
            ni.links,
            ni.generators,
            ni.storage_units
        ])
        n_y = n_y.rename(index=renamer)

        # sorting the carriers
        n_y_t = n_y_t.loc[:, n_y.index]
        n_y_t = n_y_t.loc[:, n_y.carrier.isin(subset)]
        n_y = n_y.loc[n_y.carrier.isin(subset)]

        # mapping the countries
        buses_links = [c for c in n_y.columns if "bus" in c]
        country_map = n_y[buses_links].applymap(lambda x: mapper(x, ni, to_apply="country"))
        n_y_t_co = {}
        for co in ni.buses.country.unique():
            if co == 'EU':
                continue
            carrier_mapping = n_y[country_map.apply(lambda L: L.fillna('').str.contains(co)).any(axis=1)] \
                .groupby("carrier").apply(lambda x: x)
            carrier_mapping = dict(zip(carrier_mapping.index.droplevel(0),
                                       carrier_mapping.index.droplevel(1)))
            n_y_t_co[co] = (n_y_t.loc[:, n_y_t.columns.isin(list(carrier_mapping.keys()))]
                            .rename(columns=carrier_mapping)
                            .groupby(axis=1, level=0)
                            .sum()).T

        profiles.append(pd.concat({y: pd.concat(n_y_t_co)}, names=["Year", 'Country', "Carrier"]))

    df = pd.concat(profiles)
    df.insert(0, column="Annual sum [TWh]", value=df.sum(axis=1) / 1e6 * 8760 / len(ni.snapshots))
    df.loc[(slice(None), slice(None), 'Haber-Bosch'), :] *= 4.415385
    df.insert(0, column="units", value="MWh_e")
    df.loc[(slice(None), slice(None), ['Haber-Bosch', 'ammonia cracker']), 'units'] = 'MWh_lhv,nh3'
    df.loc[(slice(None), slice(None), ['Sabatier']), 'units'] = 'MWh_lhv,h2'
    return df


def extract_production_units(n, subset_gen=None, subset_links=None):
    renamer = {"offwind-dc": "offwind", "offwind-ac": "offwind", "coal": "coal/lignite",
               "lignite": "coal/lignite", "ror": "hydro", 'urban central biomass CHP': 'biomass CHP'}
    dischargers = ["battery discharger", "home battery discharger"]
    balance_exclude = ["H2 Electrolysis", "H2 Fuel Cell", "battery charger",
                       "home battery charger", "Haber-Bosch", "Sabatier",
                       "ammonia cracker", "helmeth", "SMR", "SMR CC"]
    carriers_links = ["coal", "lignite", "oil"]  # same carrier name than link
    carriers = carriers_links + ["gas", "uranium", "biomass"]  # different carrier name than link
    transmissions = ["DC", "gas pipeline", "gas pipeline new", "CO2 pipeline",
                     "H2 pipeline", "H2 pipeline retrofitted", "electricity distribution grid"]
    balance_carriers_transmission_exclude = balance_exclude + carriers + transmissions + dischargers

    n_prod = {}
    for y, ni in n.items():
        # Grab data from various sources
        n_y = pd.concat([
            ni.links.groupby(by="carrier").sum().p_carrier_nom_opt,
            ni.generators.groupby(by="carrier").sum().p_nom_opt,
            ni.storage_units.groupby(by="carrier").sum().p_nom_opt,
            ni.stores.groupby(by="carrier").sum().e_nom_opt
        ])
        n_y = n_y.rename(index=renamer)

        if subset_gen:
            n_y = n_y[n_y.index.isin(subset_gen)]
        else:
            n_y = n_y[~n_y.index.isin(balance_carriers_transmission_exclude)]

        # Grab exceptions for carriers/links   
        n_y_except = ni.links.groupby(by="carrier").sum().p_carrier_nom_opt
        n_y_except = n_y_except.rename(index=renamer)
        if subset_links:
            n_y_except = n_y_except[n_y_except.index.isin(subset_links)]
        else:
            n_y_except = n_y_except[n_y_except.index.isin(carriers_links)]
        n_prod[y] = pd.concat([n_y, n_y_except])

    df = pd.concat({k: ni.groupby(by="carrier").sum() / 1e3 for k, ni in n_prod.items()}, axis=1).fillna(0)

    if 'Haber-Bosch' in df.index:
        df.loc['Haber-Bosch', :] *= 4.415385
    df["units"] = "GW_e"

    unit_change = {'Haber-Bosch': 'GW_lhv,nh3', 'ammonia cracker': 'GW_lhv,nh3', 'Sabatier': 'GW_lhv,h2',
                   'H2 Store': "GWh_lhv,h2", "home battery": "GWh_e", "battery": "GWh_e"}
    for i, j in unit_change.items():
        if i in df.index:
            df.loc[i, 'units'] = j
    return df


def extract_res_potential(n):
    dfx = []
    dimensions = ["region", "carrier", "build_year"]
    rx = re.compile("([A-z]+)[0-9]+\s[0-9]+\s([A-z\-\s]+)-*([0-9]*)")
    renamer = {"offwind-dc": "offwind", "offwind-ac": "offwind",
               "solar rooftop": "solar", "coal": "coal/lignite",
               "lignite": "coal/lignite", "ror": "hydro",
               'urban central biomass CHP': 'biomass CHP'}

    for y, ni in n.items():
        df_max = pd.DataFrame(ni.generators.p_nom_max)
        df_opt = pd.DataFrame(ni.generators.p_nom_opt)
        df = df_max.join(df_opt).reset_index()
        df[dimensions] = df["Generator"].str.extract(rx)
        df["carrier"] = df["carrier"].str.rstrip("-").replace(renamer)
        df["planning horizon"] = y
        df = df[df["carrier"].isin(["onwind", "offwind", "solar"])]
        dfx.append(df.groupby(["planning horizon", "carrier", "build_year"]).sum(numeric_only=True) / 1e3)  # GW

    dfx = pd.concat(dfx)
    df_potential = pd.concat([
        dfx.loc[dfx["p_nom_opt"].index.get_level_values("build_year") != dfx["p_nom_opt"].index.get_level_values(
            "planning horizon").astype(str), "p_nom_opt"].groupby(["planning horizon", "carrier"]).sum(),
        dfx.loc[dfx["p_nom_max"].index.get_level_values("build_year") == dfx["p_nom_max"].index.get_level_values(
            "planning horizon").astype(str), "p_nom_max"].groupby(["planning horizon", "carrier"]).sum()
    ], axis=1)
    df_potential["potential"] = df_potential["p_nom_max"] + df_potential["p_nom_opt"]
    df_potential = df_potential.reset_index().pivot(index="carrier", columns="planning horizon", values="potential")
    df_potential["units"] = "GW_e"
    return df_potential


def extract_transmission(n, carriers=["AC","DC"],
                         units = {"AC" : "GW_e", "DC": "GW_e",
                                  "gas pipeline" : "GW_lhv,ch4", "gas pipeline new" : "GW_lhv,ch4",
                                  "H2 pipeline" : "GW_lhv,h2", "H2 pipeline retrofitted" : "GW_lhv,h2"}):
    capacity = []
    capacity_countries = []

    # Add projected values
    for y, ni in n.items():
        
        transmission = []
        for ca in carriers:
            if ca == "AC":
                transmission.append(ni.lines.rename(columns={"s_nom_opt": "p_nom_opt"}))
            else:
                transmission.append(ni.links.query('carrier == @ca'))
                
        transmission = pd.concat(transmission)

        buses_links = [c for c in transmission.columns if "bus" in c]
        country_map = transmission[buses_links].applymap(lambda x: mapper(x, ni, to_apply="country")).fillna('')
        transmission_co = {}
        mono_co = {}
        for co in ni.buses.country.unique():
            transmission_co[co] =(transmission
                    .query("@co == @country_map.bus0 or @co == @country_map.bus1")
                    .groupby("carrier") 
                    .p_nom_opt.sum()
                )
            
            mono_co[co] = (transmission
                        .query("(Link.str.contains('->')) and not(Link.str.contains('<'))")
                        .query("@co == @country_map.bus0")
                        .groupby("carrier") 
                        .p_nom_opt.sum()
                    )
            
            if len(mono_co[co]):
                transmission_co[co].loc[mono_co[co].index] -= mono_co[co]

        transmission_co = pd.DataFrame.from_dict(transmission_co, orient='columns').fillna(0) / 1e3
        capacity_countries.append(pd.concat({y: transmission_co}, names=["Year"]))

        transmission_total = pd.DataFrame(transmission.groupby("carrier").p_nom_opt.sum()) / 1e3
        capacity.append(transmission_total.rename(columns={'p_nom_opt': y}))

    df = pd.concat(capacity, axis=1)
    df_co = pd.concat(capacity_countries, axis=0)
    df["units"] = df.index.map(units)
    df_co["units"] = df_co.index.get_level_values(level=1).map(units)
    return df, df_co


def extract_storage_units(n, color_shift, storage_function, storage_horizon, both=False, unit={}):
    carrier = list(storage_horizon.keys())

    fig = plt.figure(figsize=(14, 8))

    def plotting(ax, title, data, y, unit):
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
            plotting(ax, car, s, y, unit=unit.get(car, '[TWh]'))
        for i, (car, l) in enumerate((lt).items()):
            ax = plt.subplot(3, 2, 2 * (i + 1))
            plotting(ax, car, l, y, unit=unit.get(car, '[TWh]'))

    ax.legend()
    return fig


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
    n_cgt.loc[n_cgt["build_year"] != year, "build_year"] = "historical"
    n_cgt = (
                n_cgt
                .groupby(by=dimensions)
                .sum()
                .reset_index()
                .pivot(index="country", columns="build_year", values="p_carrier_nom_opt")
                .sort_values(by=year, ascending=False)
            ) / 1e3  # GW
    n_cgt['units'] = 'GW_e'
    return n_cgt[n_cgt[year] >= 1]


def extract_country_capacities(n):
    # Duplicates scripts.make_summary.calculate_nodal_capacites

    renamer = {"offwind-dc": "offwind", "offwind-ac": "offwind", "coal": "coal/lignite",
               "lignite": "coal/lignite", "ror": "hydro", 'urban central biomass CHP': 'biomass CHP'}

    for y, ni in n.items():
        df["nodal_capacities"] = calculate_nodal_capacities(ni, y, df["nodal_capacities"],
                                                            _opt_name={"Store": "e", "Line": "s", "Transformer": "s",
                                                                       "Link": "p_carrier"})

    df_capa = (df["nodal_capacities"]
               .rename(renamer)
               .reset_index()
               .rename(columns={"level_0": "unit_type",
                                "level_1": "node",
                                "level_2": "carrier"}))

    df_capa.node = df_capa.node.apply(lambda x: x[:2])
    df_capa = df_capa.groupby(["unit_type", "node", "carrier"]).sum().reset_index(["carrier", "unit_type"])
    df_capa = df_capa.query('unit_type in ["generators","links","storage_units"] or carrier in @LONG_LIST_GENS')
    df_capa = df_capa.drop(columns='unit_type').groupby(['node', "carrier"]).sum() / 1e3

    df_capa.loc[(slice(None), 'Haber-Bosch'), :] *= 4.415385
    df_capa['units'] = 'GW_e'
    df_capa.loc[(slice(None), ['Haber-Bosch', 'ammonia cracker']), 'units'] = 'GW_lhv,nh3'
    df_capa.loc[(slice(None), ['Sabatier']), 'units'] = 'GW_lhv,h2'
    df_capa.loc[(slice(None), ['H2 Store']), 'units'] = 'GWh_lhv,h2'
    df_capa.loc[(slice(None), ['battery', 'home battery']), 'units'] = 'GWh_e'
    df_capa.loc[(slice(None), ['gas']), 'units'] = 'GW_lhv,ch4'
    df_capa.loc[(slice(None), ['oil', 'coal/lignite', 'uranium']), 'units'] = 'GW_lhv'
    return df_capa


def extract_nodal_costs():
    # Todo : add handling of multiple runs
    df = (pd.read_csv(Path(path, 'results', 'csvs', 'nodal_costs.csv'),
                      index_col=[0, 1, 2, 3],
                      skiprows=3,
                      header=0)
          .reset_index()
          .rename(columns={"planning_horizon": "Type",
                           "level_1": "Cost",
                           "level_2": "Country",
                           "level_3": "Tech"})
          )
    df['Country'] = df['Country'].str[:2].fillna('')
    df.loc[df.Tech.isin(["gas", "biomass"]) & df.Cost.str.contains('marginal') & df.Type.str.contains(
        'generators'), 'Cost'] = 'fuel'
    df = df.set_index(['Type', 'Cost', 'Country', 'Tech'])
    df = df.fillna(0).groupby(['Type', 'Cost', 'Country', 'Tech']).sum()
    df = df.loc[~df.apply(lambda x: x < 1e3).all(axis=1)]
    df.insert(0, column="units", value="Euro")
    return df


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


def extract_graphs(years, n_path, n_name, countries=None, color_shift={2030: "C0", 2035: "C1", 2040: "C2"}):
    n = {}
    df["nodal_capacities"] = pd.DataFrame(columns=years, dtype=float)

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

    plt.close('all')
    # Extract full country list before selection of countries
    storage_function = {"hydro": "get_state_of_charge_t", "PHS": "get_state_of_charge_t"}
    storage_horizon = {"hydro": "LT", "PHS": "ST", "H2 Store": "LT",
                       "battery": "ST", "home battery": "ST",
                       "ammonia store": "LT"}
    n_sto = extract_storage_units(n, color_shift, storage_function, storage_horizon)
    # h2
    storage_function = {"H2 Fuel Cell": "get_p_carrier_nom_t", "H2 Electrolysis": "get_p_carrier_nom_t"}
    storage_horizon = {"H2 Fuel Cell": "LT", "H2 Electrolysis": "LT", "H2 Store": "LT"}
    n_h2 = extract_storage_units(n, color_shift, storage_function, storage_horizon,
                                 both=True, unit={"H2 Fuel Cell": "[GW_e]", "H2 Electrolysis": "[GW_e]",
                                                  "H2 Store": "[TWh_lhv,h2]"})
    ACDC_grid, ACDC_countries = extract_transmission(n_ext)
    H2_grid, H2_countries = extract_transmission(n_ext,carriers=["H2 pipeline","H2 pipeline retrofitted"],)
    n_costs = extract_nodal_costs()

    n_profile = extract_production_profiles(n,
                                            subset=LONG_LIST_LINKS + LONG_LIST_GENS)
    n_gas = extract_gas_phase_out(n, 2030)

    capa_country = extract_country_capacities(n_ext)

    for v in n.values():
        if countries:
            select_countries(v, countries)

    for v in n_ext.values():
        if countries:
            select_countries(v, countries)

    n_res_pot = extract_res_potential(n)

    # country specific extracts
    n_prod = extract_production_units(n_ext)
    n_res = extract_production_units(n_ext, subset_gen=["solar", "onwind", "offwind", "ror"],
                                     subset_links=[""])
    n_bal = extract_production_units(n_ext, subset_gen=[""],
                                     subset_links=["H2 Electrolysis", "H2 Fuel Cell", "battery charger",
                                                   "home battery charger", "Haber-Bosch", "Sabatier",
                                                   "ammonia cracker", "helmeth", "SMR", "SMR CC"])
    n_capa = extract_production_units(n_ext, subset_gen=LONG_LIST_GENS,
                                      subset_links=LONG_LIST_LINKS)

    n_loads = extract_loads(n)

    # Todo : put in a separate function
    # extract
    csvs.mkdir(parents=True, exist_ok=True)
    for csv in [csvs, Path(path, dir_export)]:
        n_capa.to_csv(Path(csv, "unit_capacities.csv"))
        n_sto.savefig(Path(csv, "storage_unit.png"))
        n_h2.savefig(Path(csv, "h2_production.png"))
        n_prod.to_csv(Path(csv, "power_production_capacities.csv"))
        n_res_pot.to_csv(Path(csv, "res_potentials.csv"))
        n_res.to_csv(Path(csv, "res_capacities.csv"))
        ACDC_grid.to_csv(Path(csv, "grid_capacities.csv"))
        H2_grid.to_csv(Path(csv, "H2_network_capacities.csv"))
        n_bal.to_csv(Path(csv, "power_balance_capacities.csv"))
        n_gas.to_csv(Path(csv, "gas_phase_out.csv"))

        # extract profiles
        n_loads.to_csv(Path(csv, "loads_profiles.csv"))
        n_profile.to_csv(Path(csv, "generation_profiles.csv"))
        n_costs.to_csv(Path(csv, 'costs_countries.csv'))

        # extract country specific
        ACDC_countries.to_csv(Path(csv, "grid_capacity_countries.csv"))
        H2_countries.to_csv(Path(csv, "H2_network_capacity_countries.csv"))
        capa_country.to_csv(Path(csv, "units_capacity_countries.csv"))
        logger.info(f"Exported files to folder : {csvs}")
    return


def load_gas_phase_out():
    return (
        pd.read_csv(Path(path, dir_export, "gas_phase_out.csv"), header=0)
        .reindex(columns=["country", "historical", "2030", "units"])
        .rename(columns={"historical": "Historical"})
    )


def load_res_capacities():
    return (
        pd.read_csv(Path(path, dir_export, "res_capacities.csv"), header=0)
    )


def load_production_eu27():
    return (
        pd.read_csv(Path(path, dir_export, "power_production_capacities.csv"), header=0)
        .reindex(columns=["carrier", "hist", "2030", "2035", "2040"])
        .replace({"CCGT": "gas", "OCGT": "gas"})
        .groupby(by="carrier").sum().reset_index()
        .query("carrier not in ['home battery', 'ammonia store', 'battery', 'co2 stored', 'H2 Store']")
    )


def load_production_total():
    return (
        pd.read_csv(Path(path, dir_export, "units_capacity_countries.csv"), header=0)
        .replace({"CCGT": "gas", "OCGT": "gas"})
        .query("carrier in ['gas', 'hydro', 'nuclear', 'offwind', 'carrier', 'PHS', "
               "'solar', 'solar rooftop', 'urban central biomass CHP']")
        .groupby(by="carrier").sum().reset_index()
        .reindex(columns=["carrier", "hist", "2030", "2035", "2040"])
    )


def load_balance_total():
    return (
        pd.read_csv(Path(path, dir_export, "units_capacity_countries.csv"), header=0)
        .query("carrier in ['ammonia cracker', 'battery charger', 'H2 Electrolysis', 'H2 Fuel Cell', "
               "'Haber-Bosch', 'home battery charger']")
        .groupby(by="carrier").agg({"hist": "sum", "2030": "sum", "2035": "sum", "2040": "sum", "units": "first"})
        .reset_index()
    )


def load_balance_eu27():
    return (
        pd.read_csv(Path(path, dir_export, "unit_capacities.csv"), header=0)
        .query("carrier in ['ammonia cracker', 'battery charger', 'H2 Electrolysis', 'H2 Fuel Cell', "
               "'Haber-Bosch', 'home battery charger']")
        .reindex(columns=["carrier", "hist", "2030", "2035", "2040", "units"])
    )


def load_grid_capacity():
    return (
        pd.read_csv(Path(path, dir_export, "grid_capacities.csv"), header=0)
        .reindex(columns=["carrier", "Historical", "2030", "2035", "2040"])
    )


def load_res_potentials():
    return (
        pd.read_csv(Path(path, dir_export, "grid_capacities.csv"), header=0)
        .drop(columns=["2030", "2035"])
        .reindex(columns=["carrier", "Historical", "2040", "units"])
    )


def load_h2_network_capacity():
    return (
        pd.read_csv(Path(path, dir_export, "H2_network_capacities.csv"), header=0)
        .reindex(columns=["carrier", "hist", "2030", "2035", "2040", "units"])
    )


def load_fossil_fuels():
    return (
        pd.read_csv(Path(path, dir_export, "fossil_fuels.csv"), header=0)
        .reindex(columns=["carrier", "hist", "2030", "2035", "2040", "units"])
    )


def load_costs_total():
    return (
        pd.read_csv(Path(path, dir_export, "costs_countries.csv"), header=0)
    )


def load_costs_res():
    # ToDo Add segments and subsegments
    return pd.DataFrame()
    # return (
    #     pd.read_csv(Path(path, dir_export, "costs_countries.csv"), header=0)
    #     .query("segment = 'RES'")
    #     .drop(columns=["Type", "Cost", "Country", "Tech", "segment"])
    #     .groupby(by="Subsegment").sum().reset_index()
    # )


def load_costs_flex():
    # ToDo Add segments and subsegments
    return pd.DataFrame()
    # return (
    #     pd.read_csv(Path(path, dir_export, "costs_countries.csv"), header=0)
    #     .query("segment = 'Flex'")
    #     .drop(columns=["Type", "Cost", "Country", "Tech", "segment"])
    #     .groupby(by="Subsegment").sum().reset_index()
    # )


def load_costs_segments():
    # ToDo Add segments and subsegments
    return pd.DataFrame()
    # return (
    #     pd.read_csv(Path(path, dir_export, "costs_countries.csv"), header=0)
    #     .drop(columns=["Type", "Cost", "Country", "Tech"])
    #     .groupby(by="Subsegment").sum().reset_index()
    # )


def load_costs_thermal():
    # ToDo Add segments and subsegments
    return pd.DataFrame()
    # return (
    #     pd.read_csv(Path(path, dir_export, "costs_countries.csv"), header=0)
    #     .query("segment = 'Thermal'")
    #     .drop(columns=["Type", "Cost", "Country", "Tech", "segment"])
    #     .groupby(by="Subsegment").sum().reset_index()
    # )


def load_costs_type():
    # ToDo Add segments and subsegments
    return pd.DataFrame()
    # return (
    #     pd.read_csv(Path(path, dir_export, "costs_countries.csv"), header=0)
    #     .drop(columns=["Type", "Country", "Tech", "units", "Segment", "Subsegment"])
    #     .groupby(by="Cost").sum().reset_index()
    #     .replace({"marginal": "opex"})
    # )


def load_h2_production():
    return (
        pd.read_csv(Path(path, dir_export, "generation_profiles.csv"), header=0)
        .rename(columns={"Carrier": "carrier"})
        .query("carrier in ['H2 Electrolysis', 'H2 Fuel Cell']")
        .groupby(by=["Year", "carrier"]).agg({"Annual sum [TWh]": "sum"})
        .pivot_table(index="carrier", columns="Year", values="Annual sum [TWh]")
        .reset_index()
    )


def load_h2_production_bis():
    return (
        pd.read_csv(Path(path, dir_export, "loads_profiles.csv"), header=0)
        .query("Load != 'Electricity demand for sectors'")
        .groupby(by=["Load", "Years"]).agg({"Annual sum [TWh]": "sum"}).reset_index()
    )


def load_h2_capacities():
    return (
        pd.read_csv(Path(path, dir_export, "units_capacity_countries.csv"), header=0)
        .query("carrier in ['H2 Electrolysis', 'H2 Fuel Cell']")
        .groupby(by="carrier").agg({"2030": "sum", "2035": "sum", "2040": "sum", "units": "first"})
        .reset_index()
    )


def load_production_profile():
    return (
        pd.read_csv(Path(path, dir_export, "generation_profiles.csv"), header=0)
        [["Year", "Country", "Carrier", "Annual sum [TWh]"]]
        .query("Carrier in ['ammonia cracker', 'battery charger', 'H2 Electrolysis', 'Haber-Bosch', 'helmet', "
               "'home battery charger', 'Sabatier']")
        .groupby(by=["Year", "Carrier"]).sum(numeric_only=True).reset_index()
        .rename(columns={"Carrier": "carrier"})
        .replace({"urban central solid biomass CHP": "biomass CHP"})
        .pivot_table(index="carrier", columns="Year", values="Annual sum [TWh]")
        .reset_index()
    )


def load_long_term_storage():
    return (
        pd.read_csv(Path(path, dir_export, "units_capacity_countries.csv"), header=0)
        .query("carrier in ['ammonia store', 'H2 Store']")
        .groupby(by="carrier").sum(numeric_only=True).reset_index()
        .reindex(columns=["carrier", "hist", "2030", "2035", "2040"])
    )


def load_short_term_storage():
    return (
        pd.read_csv(Path(path, dir_export, "units_capacity_countries.csv"), header=0)
        .query("carrier in ['battery', 'home battery']")
        .groupby(by="carrier").sum(numeric_only=True).reset_index()
        .reindex(columns=["carrier", "hist", "2030", "2035", "2040"])
    )


def load_long_term_storage_eu27():
    return (
        pd.read_csv(Path(path, dir_export, "unit_capacities.csv"), header=0)
        .query("carrier in ['ammonia store', 'H2 Store']")
        .drop(columns="hist")
    )


def load_short_term_storage_eu27():
    return (
        pd.read_csv(Path(path, dir_export, "unit_capacities.csv"), header=0)
        .query("carrier in ['battery', 'home battery']")
        .drop(columns="hist")
    )


def load_balance_eu27_bis():
    return (
        pd.read_csv(Path(path, dir_export, "unit_capacities.csv"), header=0)
        .query("carrier in ['ammonia cracker', 'battery charger', 'H2 Electrolysis', 'H2 Fuel Cell', 'Haber-Bosch', "
               "'home battery charger']")
        .reindex(columns=["carrier", "hist", "2030", "2035", "2040"])
    )


def export_data():
    outputs = [
        "gas_phase_out",
        "res_capacities",
        "production_eu27",
        "production_total",
        "balance_total",
        "balance_eu27",
        "grid_capacity",
        "res_potentials",
        "h2_network_capacity",
        "fossil_fuels",
        "costs_total",
        "costs_res",
        "costs_flex",
        "costs_segments",
        "costs_thermal",
        "costs_type",
        "h2_production",
        "h2_production_bis",
        "h2_capacities",
        "production_profile",
        "long_term_storage",
        "short_term_storage",
        "long_term_storage_eu27",
        "short_term_storage_eu27",
        "balance_eu27_bis",
    ]

    with pd.ExcelWriter(Path(path, "graph_extraction_raw.xlsx")) as xl:
        for output in outputs:
            globals()["load_" + output]().to_excel(xl, sheet_name=output, index=False)
    return


if __name__ == "__main__":
    # for testing
    years = [2030, 2035, 2040]
    path = Path("analysis", "CANEurope_no_SMR_oil_with_and_without_CCL_social17")
    dir_export = "csvs_for_graphs"
    n_path = Path(path, "results")

    simpl = 181
    cluster = 37
    opts = ""
    sector_opts = "3H"
    ll = "v3.0"

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

    df = {}

    eu25_countries = ["AT", "BE", "BG", "HR", "CZ", "DK", "EE", "FI", "FR", "DE", "HU", 'GR', "IE", "IT", "LV", "LT",
                      "LU", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"]
    n_name = f"elec_s{simpl}_{cluster}_l{ll}_{opts}_{sector_opts}_"

    csvs = Path(path, "csvs_for_graphs_" + n_name)

    countries = None  # ["BE","DE","FR","UK"]
    export = 'y'
    if csvs.exists() and csvs.is_dir() and export:
        export = str(input(
            "Folder already existing. Make sure to backup any data needed before extracting new ones. Do you want to continue (y/n)?"))
    if ('y' in export) or ('Y' in export):
        export = True
        countries = None  # ["BE","DE","FR","UK"]
        logger.info(f"Extracting from {path}")
        extract_graphs(years, n_path, n_name, countries=eu25_countries)
        export_data()
