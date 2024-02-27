# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2024 Climact for The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

"""
Create data ready to present (transform)
"""
import logging
import re
from pathlib import Path

import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
from yaml import safe_load

from scripts.graph_extraction_utils import CLIP_VALUE_TWH
from scripts.graph_extraction_utils import TRANSMISSION_RENAMER
from scripts.graph_extraction_utils import RES
from scripts.graph_extraction_utils import bus_mapper
from scripts.make_summary import calculate_nodal_capacities
from scripts.make_summary import calculate_nodal_supply_energy
from scripts.plot_network import plot_capacity
from scripts.plot_network import plot_series
from scripts.prepare_sector_network import get

logger = logging.getLogger(__name__)

RES
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


def extract_nodal_oil_load(config, nhours=8760):
    resources = Path(config["path"]["analysis_path"], "resources")
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
    for y in config["scenario"]["planning_horizons"]:
        df_industry[y] = read_load(Path(resources, f"industrial_energy_demand_elec_s181_37m_{y}.csv"))

    df_oil = []
    for y in config["scenario"]["planning_horizons"]:
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
        res = res.reset_index().merge(LCOE, on="carrier").set_index(["Generator", "year"])
        res = res.loc[:,
              ["carrier", 'bus', "capital_cost", "marginal_cost", "p_nom_opt", "build_year", "p_nom", "cf", 'p_tot',
               "p_nom_max", "LCOE", "opex", "capex", "totex"]]
        res = res.sort_values(by="carrier")
        df.append(res)

    return pd.concat(df)

def extract_production_profiles(config, n, regionalized = False):
    production_profiles = []
    for carrier in config['carriers_to_plot']:
        df = []
        for y, ni in n.items():            
            df.append(plot_series(ni, carrier=carrier, name=carrier, year=str(y),
                                   return_data = True, load_only = True, regionalized=regionalized))
            print(f'Extracted {carrier} for year {y}')
        df = pd.concat(df)
        df['carrier'] = carrier
        production_profiles.append(df)
    production_profiles = (pd.concat(production_profiles)
                           .reset_index()
                           .set_index(["carrier","snapshots","country"] if regionalized else ["carrier","snapshots"])
                           )
    return production_profiles


def extract_country_capacities(config, n):
    df = {}
    df["nodal_capacities"] = pd.DataFrame(columns=config["scenario"]["planning_horizons"], dtype=float)

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

    # add extraction and storage suffixes
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
            mat_exp.loc[other_bus[co].loc[exp[exp < -CLIP_VALUE_TWH].index], co] = -exp[exp < -CLIP_VALUE_TWH].values

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
    df_imp['imports_exports'] = 'imports'
    df_exp = pd.concat(exports, axis=0).fillna(0)
    df_exp['imports_exports'] = 'exports'


    df["units"] = df.index.map(units)
    df_co["units"] = df_co.index.get_level_values(level=1).map(units)
    df_imp_exp = pd.concat([df_imp,df_exp])
    df_imp_exp["carriers"] = TRANSMISSION_RENAMER.get(carriers[0])

    return df, df_co, df_imp_exp


def extract_nodal_costs(config):
    # Todo : add handling of multiple runs
    df = (pd.read_csv(Path(config["path"]["analysis_path"], 'results', 'csvs', 'nodal_costs.csv'),
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
    fuels = df.query(
        "carrier in ['gas','oil','coal','lignite','uranium'] and cost == 'marginal' and type == 'generators'").index
    biomass = df.query("carrier.str.contains('biomass') and cost == 'marginal' and type == 'stores'").index
    df.loc[fuels.union(biomass), 'cost'] = 'fuel'
    df = df.set_index(['type', 'cost', 'country', 'carrier'])
    df = df.fillna(0).groupby(['type', 'cost', 'country', 'carrier']).sum()
    df = df.loc[~df.apply(lambda x: x < 1e3).all(axis=1)]
    df.insert(0, column="units", value="Euro")
    return df


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


rx = re.compile(r"([A-Z]{2})[0-9]\s[0-9]")


def renamer_to_country(x):
    if rx.match(x):
        return rx.match(x).group(1)
    else:
        return x


def extract_nodal_supply_energy(config, n):
    labels = {y: config["label"][:-1] + (y,) for y in n.keys()}
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
        Path(config["path"]["analysis_path"].resolve().parents[1], "sector_mapping.csv"), index_col=[0, 1, 2],
        header=0).dropna()
    df = df.merge(sector_mapping, left_on=["carrier", "component", "item"], right_index=True, how="left")
    return df


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


def _extract_graphs(config, n, storage_function, storage_horizon, both=False, units={}, color_shift=None):
    carrier = list(storage_horizon.keys())
    if color_shift:
        pass
    else:
        color_shift = dict(zip(config["scenario"]["planning_horizons"],
                               ['C' + str(i) for i in range(len(config["scenario"]["planning_horizons"]))]))
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


def extract_graphs(config, n, color_shift=None):
    ## Figures to extract
    plt.close('all')
    # Storage
    mpl.rcParams.update(mpl.rcParamsDefault)
    if color_shift:
        pass
    else:
        color_shift = dict(zip(config["scenario"]["planning_horizons"], ['C0', 'C2', 'C1']))

    storage_function = {"hydro": "get_state_of_charge_t", "PHS": "get_state_of_charge_t"}
    storage_horizon = {"hydro": "LT", "PHS": "ST", "H2 Store": "LT",
                       "battery": "ST", "home battery": "ST",
                       "ammonia store": "LT"}
    n_sto = _extract_graphs(config, n, storage_function, storage_horizon, color_shift=color_shift)
    # h2
    storage_function = {"H2 Fuel Cell": "get_p_carrier_nom_t", "H2 Electrolysis": "get_p_carrier_nom_t"}
    storage_horizon = {"H2 Fuel Cell": "LT", "H2 Electrolysis": "LT", "H2 Store": "LT"}
    n_h2 = _extract_graphs(config, n, storage_function, storage_horizon, color_shift=color_shift,
                           both=True, units={"H2 Fuel Cell": "[GW_e]", "H2 Electrolysis": "[GW_e]",
                                             "H2 Store": "[TWh_{lhv,h2}]"})
    return n_sto, n_h2


def extract_series(config, n):
    with plt.style.context(["ggplot"]):
        with open(Path(config["path"]["analysis_path"], 'results/configs/config.snakemake.yaml'), 'r') as f:
            df = safe_load(f)["plotting"]
            plots = {}
            for y, ni in n.items():
                with pd.option_context('mode.chained_assignment', None):
                    plots[y] = plot_series(ni, carrier="electricity", name="electricity", year=str(y),
                                           load_only=True, colors=df["tech_colors"],
                                           path=Path(config["csvs"], f"series_AC_{y}.png"), save=False)
    return plots


def extract_plot_capacities(config, n):
    with plt.style.context(["ggplot"]):
        with open(Path(config["path"]["analysis_path"], 'results/configs/config.snakemake.yaml'), 'r') as f:
            df = safe_load(f)["plotting"]
            plots = {}
            for y, ni in n.items():
                with pd.option_context('mode.chained_assignment', None):
                    plots[y] = plot_capacity(ni, colors=df["tech_colors"], _map_opts=df["map"],
                                             bus_size_factor=1e5, path=Path(config["csvs"], f"capacities_{y}.png"),
                                             run_from_rule=False, transmission=True, save=False)
    return plots


def export_csvs_figures(csvs, outputs, figures):
    csvs.mkdir(parents=True, exist_ok=True)

    for f_name, f in figures.items():
        for y, plot in f.items():
            plot.savefig(Path(csvs, f"{f_name}_{y}.png"), transparent=True)

    for o_name, o in outputs.items():
        o.to_csv(Path(csvs, f"{o_name}.csv"))

    logger.info(f"Exported files and figures to folder : {csvs}")

    return


def transform_data(config, n, n_ext, color_shift=None):
    logger.info(f"Transforming data")

    # DataFrames to extract
    prod_profiles = extract_production_profiles(config,n)
    n_loads = extract_loads(n)
    nodal_oil_load = extract_nodal_oil_load(config, nhours=n_ext['hist'].snapshot_weightings.generators.sum())
    n_res_pot = extract_res_potential(n)
    res_stats = extract_res_statistics(n)
    capa_country = extract_country_capacities(config, n_ext)
    ACDC_grid, ACDC_countries, el_imp_exp = extract_transmission(n_ext)
    H2_grid, H2_countries, H2_imp_exp = extract_transmission(n_ext, carriers=["H2 pipeline", "H2 pipeline retrofitted"])
    gas_grid, gas_countries, gas_imp_exp = extract_transmission(n_ext, carriers=["gas pipeline", "gas pipeline new"])
    n_costs = extract_nodal_costs(config)
    marginal_prices = extract_marginal_prices(n, carrier_list=['gas', 'AC', 'H2'])
    nodal_supply_energy = extract_nodal_supply_energy(config, n)
    n_gas_out = extract_gas_phase_out(n, config["scenario"]["planning_horizons"][0])
    # n_profile = extract_production_profiles(n, subset=LONG_LIST_LINKS + LONG_LIST_GENS)

    imp_exp = pd.concat([y.reset_index()
                         .set_index(['imports_exports','countries', 'year', 'carriers'])
                         for y in [el_imp_exp, H2_imp_exp, gas_imp_exp]])

    # Figures to extract
    n_sto, n_h2 = extract_graphs(config, n, color_shift)
    series_consumption = extract_series(config, n)
    map_capacities = extract_plot_capacities(config, n)

    # Define outputs and export them
    outputs = {
        # assets
        'units_capacities_countries': capa_country,
        'gas_phase_out': n_gas_out,
        'res_potentials': n_res_pot,

        # networks
        'grid_capacities_countries': ACDC_countries,
        'H2_network_capacities_countries': H2_countries,
        'gas_network_capacities_countries': gas_countries,
        'grid_capacities': ACDC_grid,
        'H2_network_capacities': H2_grid,
        'gas_network_capacities': gas_grid,

        # energy balance
        'imports_exports': imp_exp,
        'supply_energy_sectors': nodal_supply_energy,
        'nodal_oil_load': nodal_oil_load,

        # insights
        'costs_countries': n_costs,
        'marginal_prices_countries': marginal_prices,
        'res_statistics': res_stats,
        'loads_profiles': n_loads,
        "generation_profiles" : prod_profiles
    }

    figures = {
        'storage_unit': n_sto,
        'h2_production': n_h2,
        'series_consumption': series_consumption,
        'map_capacities': map_capacities
    }

    export_csvs_figures(config["csvs"], outputs, figures)

    return outputs, figures
