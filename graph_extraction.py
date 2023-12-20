# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:35:07 2023

@author: VincentLaguna
"""

from pathlib import Path
import pypsa
import pandas as pd
import pylab
import re
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np

from make_summary import (change_p_nom_opt_carrier,
                            make_summaries,
                            mapper,
                            searcher,
                            calculate_nodal_capacities,
                            assign_carriers,
                            assign_locations,
                            assign_countries)

from prepare_sector_network import prepare_costs

import logging

logger = logging.getLogger(__name__)

def reduce_to_countries(df,index):
    buses = [c for c in df.columns if "bus" in c]
    return df.loc[df.loc[:,buses].applymap(lambda x : x in index).values.any(axis=1)]

def select_countries(n,countries):
    index = n.buses.loc[n.buses.country.isin(eu25_countries)].index
    n.generators = reduce_to_countries(n.generators,index)
    n.lines = reduce_to_countries(n.lines,index)
    n.links = reduce_to_countries(n.links,index)
    n.stores = reduce_to_countries(n.stores,index)
    n.storage_units = reduce_to_countries(n.storage_units,index)
    n.loads_t.p = n.loads_t.p.loc[:,n.loads_t.p.columns.str[:2].isin(countries)]
    n.loads_t.p_set = n.loads_t.p_set.loc[:,n.loads_t.p_set.columns.str[:2].isin(countries)]
    n.buses = n.buses.loc[index]
    return n

def get_state_of_charge_t(n, carrier):
    df = n.storage_units_t.state_of_charge.T.reset_index()
    df = df.merge(n.storage_units.reset_index()[["carrier", "StorageUnit"]], on="StorageUnit")
    df = df.groupby(by="carrier").sum()
    df.drop(columns=["StorageUnit"],inplace=True)
    return df.T[[carrier]]

def get_e_t(n, carrier):
    df = n.stores_t.e.T.reset_index()
    df = df.merge(n.stores.reset_index()[["carrier", "Store"]], on="Store")
    df = df.groupby(by="carrier").sum()
    df.drop(columns=["Store"],inplace=True)
    return df.T[[carrier]]

def extract_production_profiles(n, subset):
    renamer = {"offwind-dc": "offwind", "offwind-ac": "offwind",
               "solar rooftop": "solar", "coal": "coal/lignite",
               "lignite": "coal/lignite","ror" : "hydro"} 
    dischargers = ["battery discharger", "home battery discharger"]
    balance_exclude = ["H2 Electrolysis", "H2 Fuel Cell", "battery charger",
                       "home battery charger", "Haber-Bosch", "Sabatier", 
                       "ammonia cracker", "helmeth", "SMR", "SMR CC"]
    carriers_links = ["coal", "lignite", "oil"] # same carrier name than link
    carriers = carriers_links + ["gas", "uranium", "biomass"] # different carrier name than link
    transmissions = ["DC", "gas pipeline", "gas pipeline new", "CO2 pipeline",
                     "H2 pipeline", "H2 pipeline retrofitted", "electricity distribution grid"]
    balance_carriers_transmission_exclude = balance_exclude + carriers + transmissions + dischargers
    
    
    profiles = []
    for y, ni in n.items():
        # Grab data from various sources
        n_y_t = pd.concat([
                    ni.links_t.p_carrier_nom_opt,
                    ni.generators_t.p, 
                    ni.storage_units_t.p
                ],axis=1)
        n_y = pd.concat([
            ni.links,
            ni.generators, 
            ni.storage_units
        ])
        n_y = n_y.rename(index=renamer)
        
        #sorting the carriers
        n_y_t = n_y_t.loc[:,n_y.index]
        n_y_t = n_y_t.loc[:,n_y.carrier.isin(subset)]
        n_y = n_y.loc[n_y.carrier.isin(subset)]
        
        #mapping the countries
        buses_links = [c for c in n_y.columns if "bus" in c]
        country_map = n_y[buses_links].applymap(lambda x : mapper(x,ni,to_apply="country"))
        n_y_t_co = {}
        for co in ni.buses.country.unique():
            if co =='EU':
                continue
            carrier_mapping = n_y[country_map.apply(lambda L : L.fillna('').str.contains(co)).any(axis=1)] \
                          .groupby("carrier").apply(lambda x:x)
            carrier_mapping = dict(zip(carrier_mapping.index.droplevel(0),
                                       carrier_mapping.index.droplevel(1)))
            n_y_t_co[co] = (n_y_t.loc[:,n_y_t.columns.isin(list(carrier_mapping.keys()))]
                            .rename(columns=carrier_mapping)
                            .groupby(axis=1,level=0)
                            .sum()).T
 
        profiles.append(pd.concat({y: pd.concat(n_y_t_co)}, names=['Year']))
        
    df = pd.concat(profiles)
    return df

def extract_production_units(n,subset_gen=None,subset_links=None):

    renamer = {"offwind-dc": "offwind", "offwind-ac": "offwind",
                "solar rooftop": "solar", "coal": "coal/lignite",
                "lignite": "coal/lignite","ror" : "hydro",
                'urban central biomass CHP' : 'biomass CHP'}  
    dischargers = ["battery discharger", "home battery discharger"]
    balance_exclude = ["H2 Electrolysis", "H2 Fuel Cell", "battery charger",
                        "home battery charger", "Haber-Bosch", "Sabatier", 
                        "ammonia cracker", "helmeth", "SMR", "SMR CC"]
    carriers_links = ["coal", "lignite", "oil"] # same carrier name than link
    carriers = carriers_links + ["gas", "uranium", "biomass"] # different carrier name than link
    transmissions = ["DC", "gas pipeline", "gas pipeline new", "CO2 pipeline",
                      "H2 pipeline", "H2 pipeline retrofitted", "electricity distribution grid"]
    balance_carriers_transmission_exclude = balance_exclude + carriers + transmissions + dischargers
        
    n_prod  ={}
    for y, ni in n.items():
        # Grab data from various sources
        n_y = pd.concat([
            ni.links.groupby(by="carrier").sum().p_carrier_nom_opt,
            ni.generators.groupby(by="carrier").sum().p_nom_opt, 
            ni.storage_units.groupby(by="carrier").sum().p_nom_opt
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
        
    return pd.concat({k: ni.groupby(by="carrier").sum()/1e3 for k, ni in n_prod.items()},axis=1).fillna(0)

def extract_res_potential_old(n):
    dfx = []
    dimensions = ["region", "carrier", "build_year"]
    rx = re.compile("([A-z]+)[0-9]+\s[0-9]+\s([A-z\-\s]+)-*([0-9]*)")
    renamer = {"offwind-dc": "offwind", "offwind-ac": "offwind",
               "solar rooftop": "solar", "coal": "coal/lignite",
               "lignite": "coal/lignite","ror" : "hydro",
               'urban central biomass CHP' : 'biomass CHP'} 
    
    for y, ni in n.items():
        df_max = pd.DataFrame(ni.generators.p_nom_max)
        df_opt = pd.DataFrame(ni.generators.p_nom_opt)
        df = df_max.join(df_opt).reset_index()
        df[dimensions] = df["Generator"].str.extract(rx)
        df["carrier"] = df["carrier"].str.rstrip("-").replace(renamer)
        df["planning horizon"] = y
        df = df[df["carrier"].isin(["onwind", "offwind", "solar"])]
        dfx.append(df.groupby(["planning horizon", "carrier", "build_year"]).sum(numeric_only=True)/1e3) #GW
    
    dfx = pd.concat(dfx)
    df_potential = pd.concat([
        dfx.loc[dfx["p_nom_opt"].index.get_level_values("build_year") != dfx["p_nom_opt"].index.get_level_values("planning horizon").astype(str), "p_nom_opt"].groupby(["planning horizon", "carrier"]).sum(), 
        dfx.loc[dfx["p_nom_max"].index.get_level_values("build_year") == dfx["p_nom_max"].index.get_level_values("planning horizon").astype(str), "p_nom_max"].groupby(["planning horizon", "carrier"]).sum()
        ], axis=1)
    df_potential["potential"] = df_potential["p_nom_max"] + df_potential["p_nom_opt"]
    df_potential = df_potential.reset_index().pivot(index="carrier", columns="planning horizon", values="potential")

    return df_potential
  
def extract_res_potential(n):
    dfx = []
    dimensions = ["region", "carrier", "build_year"]
    rx = re.compile("([A-z]+)[0-9]+\s[0-9]+\s([A-z\-\s]+)-*([0-9]*)")
    renamer = {"offwind-dc": "offwind", "offwind-ac": "offwind",
               "solar rooftop": "solar", "coal": "coal/lignite",
               "lignite": "coal/lignite","ror" : "hydro"} 
    
    gens = n.generators
    df_max = pd.DataFrame(gens.p_nom_max)
    df_opt = pd.DataFrame(gens.p_nom_opt)
    df = df_max.join(df_opt).reset_index()
    df[dimensions] = df["Generator"].str.extract(rx)
    df["carrier"] = df["carrier"].str.rstrip("-").replace(renamer)
    df["planning horizon"] = "historical"
    df = df[df["carrier"].isin(["onwind", "offwind", "solar"])]
    dfx.append(df.groupby(["planning horizon", "carrier", "build_year"]).sum(numeric_only=True)/1e3) #GW
    
    dfx = pd.concat(dfx)
    df_potential = pd.DataFrame()
    df_potential["potential"] = dfx["p_nom_max"]
    df_potential = df_potential.reset_index().pivot(index="carrier", columns="planning horizon", values="potential")

    return df_potential
  
def extract_transmission_AC_DC(n, n_path, n_name):
    #Localy extend network collection 
    capacity = []
    capacity_countries = []
    n_copy = n.copy()
    
    # Set historical values
    n_hist = pypsa.Network(Path(n_path, "prenetworks", n_name + f"{2030}.nc"))
    assign_countries(n_hist)
    n_hist.lines.carrier = "AC"
    n_hist.lines.s_nom_opt = n_hist.lines.s_nom_min
    n_hist.links.loc[n_hist.links.carrier.isin(["DC"]),'p_nom_opt'] = n_hist.links.loc[n_hist.links.carrier.isin(["DC"]),'p_nom_min']
    
    n_copy["Historical"] =  n_hist
    
    # Add projected values
    for y, ni in n_copy.items():
        AC = ni.lines.rename(columns={"s_nom_opt":"p_nom_opt"})
        DC = ni.links[ni.links.carrier=="DC"]
        AC_DC = pd.concat([AC,DC])
        
        buses_links = [c for c in AC_DC.columns if "bus" in c]
        country_map = AC_DC[buses_links].applymap(lambda x : mapper(x,ni,to_apply="country"))
        AC_DC_co = {}
        for co in ni.buses.country.unique():
            AC_DC_co[co] = AC_DC[country_map.apply(lambda L : L.fillna('').str.contains(co)).any(axis=1)] \
                          .groupby("carrier").p_nom_opt.sum()

        AC_DC_co =  pd.DataFrame.from_dict(AC_DC_co, orient = 'columns').fillna(0)/1e3
        capacity_countries.append(pd.concat({y: AC_DC_co}, names=['Year']))
        # for co in ni.buses.country.unique():
        #     lines_co[co] = n.lines[country_map_lines.apply(lambda L : L.str.contains(co).fillna(False)).any(axis=1)].s_nom_opt.sum()
            
        AC_DC_total = pd.DataFrame(AC_DC.groupby("carrier").p_nom_opt.sum())/1e3
        capacity.append(AC_DC_total.rename(columns={'p_nom_opt':y}))
        
    df = pd.concat(capacity,axis=1)
    df_co = pd.concat(capacity_countries,axis=0)
    return df,df_co,n_hist

def extract_transmission_H2(n):
    # Add projected values
    capacity = []
    capacity_countries = []
    for y, ni in n.items():
        
        H2_pipelines = pd.concat([ni.links[ni.links.carrier.isin(["H2 pipeline"])],
                                  ni.links[ni.links.carrier.isin(["H2 pipeline retrofitted"])]
                                  ])
        
        buses_links = [c for c in H2_pipelines.columns if "bus" in c]
        country_map = H2_pipelines[buses_links].applymap(lambda x : mapper(x,ni,to_apply="country")) 
        
        H2_per_country = {}
        for co in ni.buses.country.unique():
            H2_per_country[co] = H2_pipelines[country_map.apply(lambda L : L.fillna('').str.contains(co)).any(axis=1)] \
                                .groupby("carrier").p_nom_opt.sum()
        H2_per_country = pd.DataFrame.from_dict(H2_per_country, orient = 'columns').fillna(0)/1e3
        capacity_countries.append(pd.concat({y: H2_per_country}, names=['Year']))
        
        H2_total = pd.DataFrame(H2_pipelines.groupby("carrier").p_nom_opt.sum())/1e3
        capacity.append(H2_total.rename(columns={'p_nom_opt':y}))

    df = pd.concat(capacity,axis=1)
    df_co = pd.concat(capacity_countries,axis=0)
    return df,df_co

def extract_storage_units(n, color_shift):
    n_store = {}
    storage_function = {"hydro" : "get_state_of_charge_t", "PHS" : "get_state_of_charge_t"}
    storage_horizon = {"hydro" : "LT", "PHS" : "ST", "H2 Store" : "LT",
            "battery" : "ST", "home battery" : "ST",
            "ammonia store" : "LT"}
    
    carrier = list(storage_horizon.keys())
    
    plt.close('all')
    fig = plt.figure(figsize=(14,8))
    def plotting(ax,title,data,y):
        ax.plot(data,label = y,color=color_shift.get(y))
        ax.set_title(title)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel("[TWh]", ha='left', y=1.1, rotation=0, labelpad=0.2)
        plt.xticks(rotation=30)
        plt.tight_layout()
        return 
    
    for y, ni in n.items():
        lt = {}
        st = {}
        for car in carrier:
            storage = globals()[storage_function.get(car,"get_e_t")](ni, car)/1e6
            if "L" in storage_horizon.get(car):
                lt[car] = storage
            else:
                st[car] = storage.iloc[:int(8*31)]    
        for i,(car,s) in enumerate(st.items()) :
            ax = plt.subplot(3,2,2*i+1)
            plotting(ax,car,s,y)
        for i,(car,l) in enumerate((lt).items()) :
            ax = plt.subplot(3,2,2*(i+1))
            plotting(ax,car,l,y)

    ax.legend()
    return fig

def extract_gas_phase_out(n, year, subset=None):
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
    n_cgt.loc[n_cgt["build_year"]!=year, "build_year"] = "historical"
    n_cgt = (
        n_cgt
        .groupby(by=dimensions)
        .sum()
        .reset_index()
        .pivot(index="country", columns="build_year", values="p_carrier_nom_opt")
        .sort_values(by=year, ascending=False)
    ) / 1e3 # GW
    
    return n_cgt[n_cgt[year] >= 1]

def extract_nodal_capacities(n):
    renamer = {"offwind-dc": "offwind", "offwind-ac": "offwind",
               "solar rooftop": "solar", "coal": "coal/lignite",
               "lignite": "coal/lignite","ror" : "hydro",
               'urban central biomass CHP' : 'biomass CHP'} 
    dischargers = ["battery discharger", "home battery discharger"]
    balance_exclude = ["H2 Electrolysis", "H2 Fuel Cell", "battery charger",
                       "home battery charger", "Haber-Bosch", "Sabatier", 
                       "ammonia cracker", "helmeth", "SMR", "SMR CC"]
    carriers_links = ["coal", "lignite", "oil"] # same carrier name than link
    carriers = carriers_links + ["gas", "uranium", "biomass"] # different carrier name than link
    transmissions = ["DC", "gas pipeline", "gas pipeline new", "CO2 pipeline",
                     "H2 pipeline", "H2 pipeline retrofitted", "electricity distribution grid"]
    balance_carriers_transmission_exclude = balance_exclude + carriers + transmissions + dischargers
    
    for y in years:
        df["nodal_capacities"] = calculate_nodal_capacities(n[y],y,df["nodal_capacities"])
        
    df_capa = (      df["nodal_capacities"]
                       .rename(renamer)         
                       .reset_index()
                        .rename(columns={"level_0":"unit_type",
                                 "level_1":"node",
                                 "level_2":"carrier"}))
    
    df_capa.node = df_capa.node.apply(lambda x: x[:2])
    df_capa = df_capa.groupby(["unit_type","node","carrier"]).sum().reset_index(["carrier","unit_type"])
    df_capa = df_capa.loc[df_capa.unit_type.isin(["generators","links","storage_units"])]
    df_capa = df_capa.drop(columns='unit_type').groupby(['node','carrier']).sum()
    return df_capa

def extract_nodal_costs(n):
    #Todo : add handling of multiple runs 
    df = (pd.read_csv(Path(path,'results','csvs','nodal_costs.csv'),
                     index_col = [0,1,2,3],
                     skiprows=3,
                     header=0,
                     names=['Type','Cost','Country','Tech','2030','2035','2040'])
                    .reset_index())
    df['Country'] = df['Country'].str[:2].fillna('')
    df = df.set_index(['Type','Cost','Country','Tech'])
    df = df.fillna(0).groupby(['Type','Cost','Country','Tech']).sum()
    df = df.loc[~df.apply(lambda x : x<1e3).all(axis=1)]
    return df
    
def extract_graphs(years, n_path, n_name, countries=None, subset_production=None,
                   subset_balancing=None, color_shift = {2030:"C0",2035:"C2",2040:"C1"},export=False):
    
    n = {}
    for y in years:
        run_name = Path(n_path, "postnetworks", n_name + f"{y}.nc")
        n[y] = pypsa.Network(run_name)
        assign_carriers(n[y])
        assign_locations(n[y])
        assign_countries(n[y])
        change_p_nom_opt_carrier(n[y])

    #Extract full country list before selection of countries
    capa_country = extract_nodal_capacities(n)
    n_sto = extract_storage_units(n,color_shift)
    ACDC_grid,ACDC_countries,n_hist = extract_transmission_AC_DC(n,n_path,n_name)
    H2_grid,H2_countries = extract_transmission_H2(n)
    n_costs = extract_nodal_costs(n,countries)
    
    for y in years:
        if countries:
            select_countries(n[y], countries)
            
    n_gas = extract_gas_phase_out(n,2030)
    n_res_pot = extract_res_potential(n_hist)

        
    long_list_links = ["coal/lignite", "oil","CCGT","OCGT",
                    "H2 Electrolysis", "H2 Fuel Cell", "battery charger",
                       "home battery charger", "Haber-Bosch", "Sabatier", 
                       "ammonia cracker", "helmeth", "SMR", "SMR CC"]
    
    long_list_gens = ["solar", "onwind", "offwind", "ror", "nuclear", "biomass CHP"]
    
    #country specific extracts   
   
    n_profile  = extract_production_profiles(n, 
                                     subset = long_list_links + long_list_gens)
    n_prod = extract_production_units(n)
    n_res = extract_production_units(n,subset_gen = ["solar","onwind","offwind","ror"],
                                     subset_links = [""])
    n_bal = extract_production_units(n,subset_gen = [""], 
                                     subset_links = ["H2 Electrolysis", "H2 Fuel Cell", "battery charger",
                                                        "home battery charger", "Haber-Bosch", "Sabatier", 
                                                        "ammonia cracker", "helmeth", "SMR", "SMR CC"])
    n_capa = extract_production_units(n,subset_gen = long_list_gens ,
                                             subset_links = long_list_links)
    n_ff  = extract_production_units(n,subset_gen = [""], 
                                     subset_links = ["coal/lignite", "oil", "CCGT","OCGT"] )

    n_loads = extract_loads(n)

    #Todo : put in a separate function
    if export :
        #extract
        csvs.mkdir(parents=True, exist_ok=True)
        for csv in [csvs,Path(path,'csvs_for_graphs')]:
            n_capa.to_csv(Path(csv,"capacities.csv"))
            capa_country.to_csv(Path(csv,"capacities_countries.csv"))
            n_sto.savefig(Path(csv,"storage_unit.png"))
            n_prod.to_csv(Path(csv,"power_production_capacities.csv"))
            n_res_pot.to_csv(Path(csv,"res_potentials.csv"))
            n_res.to_csv(Path(csv,"res_capacities.csv"))
            ACDC_grid.to_csv(Path(csv,"grid_capacity.csv"))
            H2_grid.to_csv(Path(csv,"H2_network_capacity.csv"))
            n_bal.to_csv(Path(csv,"power_balance_capacities.csv"))
            n_gas.to_csv(Path(csv,"gas_phase_out.csv"))
            n_ff.to_csv(Path(csv,"fossil_fuels.csv"))
            
            
            #extract profiles
            n_loads.to_csv(Path(csv,"loads_profiles.csv"))
            n_profile.to_csv(Path(csv,"generation_profiles.csv"))
            n_costs.to_csv(Path(csv,'costs_countries.csv'))
            
            #extract country specific
            ACDC_countries.to_csv(Path(csv,"grid_capacity_countries.csv"))
            H2_countries.to_csv(Path(csv,"H2_network_capacity_countries.csv"))
            capa_country.to_csv(Path(csv,"units_capacity_countries.csv"))
            logger.info(f"Exported files to folder : {csvs}")
    return 

def extract_loads(n):
    profiles =  {}
    for y, ni in n.items():
        loads_t = ni.loads_t.p.T
        loads_t.index.names= ['Load']
        loads_t["country"] = ni.buses.loc[ni.loads.loc[loads_t.index].bus].country.values
        loads_t.reset_index(inplace=True)
        loads_t["Load"].mask(loads_t["Load"].str.contains("NH3"),"NH3 for industry",inplace=True)
        loads_t["Load"].mask(loads_t["Load"].str.contains("H2"),"H2 for industry",inplace=True)
        loads_t["Load"].mask(loads_t["Load"].str.contains("industry electricity"),"Industry",inplace=True)
        loads_t["Load"].where(loads_t["Load"].str.contains("industry"),"Load",inplace=True)
        
        loads_t = loads_t.groupby(["country","Load"]).sum()
        loads_t.insert(0,column="Annual sum [TWh]",value= loads_t.sum(axis=1)/1e6*8760/len(ni.snapshots))
        profiles[y] = loads_t
    return pd.concat(profiles,names=["Years"])
    
if __name__ == "__main__":
    
    import os 
    os.chdir('C:/Users/VincentLaguna/pypsa-eur-climact')
    
    
    #for testing
    years = [2030, 2035]
    path = Path("analysis", "CANEurope_industry_no_SMR_oil_3H")
    
    simpl = 181
    cluster = '37m'
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
    df["nodal_capacities"] = pd.DataFrame(columns=years, dtype=float)

    
    eu25_countries = ["AT", "BE", "BG", "HR", "CZ", "DK", "EE", "FI", "FR", "DE", "HU", 'GR', "IE", "IT", "LV", "LT",
            "LU", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"]
    n_path = Path(path,"results")
    n_name = f"elec_s{simpl}_{cluster}_l{ll}_{opts}_{sector_opts}_"
    
    csvs = Path(path,"csvs_for_graphs_"+n_name)
    
    countries= None #["BE","DE","FR","UK"]
    export = True
    if csvs.exists() and csvs.is_dir() and export:
        logger.info("Folder already existing. Make sure to backup any data needed before extracting new ones")
    else :
        countries= None #["BE","DE","FR","UK"]
        logger.info(f"Extracting from {path}")
        extract_graphs(years,n_path,n_name,countries=eu25_countries,export=export)
    
