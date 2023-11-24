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

from make_sumamry import change_p_nom_opt_carrier


def select_countries(ni,countries):
    return ni[ni.index.str[:2].isin(countries)]
    
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

def extract_production_units(n,subset_gen=None,subset_links=None):
    var = "p_nom_opt","p_nom"
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
    
    
    n_prod = {}
    for y, ni in n.items():
        # Grab data from various sources
        n_y = pd.concat([
            ni.links.groupby(by="carrier").sum(),
            ni.generators.groupby(by="carrier").sum(), 
            ni.storage_units.groupby(by="carrier").sum()
        ])
        n_y = n_y.rename(index=renamer)
        
        if subset_gen:
            n_y = n_y[n_y.index.isin(subset_gen)]
        else:
            n_y = n_y[~n_y.index.isin(balance_carriers_transmission_exclude)]
            
        # Grab exceptions for carriers/links   
        n_y_except = ni.links.groupby(by="carrier").sum()
        n_y_except = n_y_except.rename(index=renamer)
        if subset_links:
            n_y_except = n_y_except[n_y_except.index.isin(subset_links)]
        else:
            n_y_except = n_y_except[n_y_except.index.isin(carriers_links)]
        n_prod[y] = pd.concat([n_y, n_y_except])
        
    return {v: pd.concat({k: ni.groupby(by="carrier").sum()[v]/1e3 for k, ni in n_prod.items()},axis=1).fillna(0) for v in var}

def extract_res_potential(n):
    dfx = []
    dimensions = ["region", "carrier", "build_year"]
    rx = re.compile("([A-z]+)[0-9]+\s[0-9]+\s([A-z\-\s]+)-*([0-9]*)")
    renamer = {"offwind-dc": "offwind", "offwind-ac": "offwind",
               "solar rooftop": "solar", "coal": "coal/lignite",
               "lignite": "coal/lignite","ror" : "hydro"} 
    
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
    
def extract_transmission_AC_DC(n,n_path,n_name):
    #Localy extend network collection 
    capacity = []
    n_copy = n.copy()
    
    # Set historical values
    n_hist = pypsa.Network(Path(n_path, "prenetworks", n_name + f"{2030}.nc"))
    assign_countries(n_hist)
    n_hist.lines.carrier = "AC"
    n_hist.lines.s_nom_opt = n_hist.lines.s_nom_min
    n_hist.links.loc[n_hist.links.carrier.isin(["DC"]),'p_nom_opt'] = n_hist.links.p_nom_min
    
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

        pd.DataFrame.from_dict(AC_DC_co, orient = 'columns').fillna(0)/1e3

        # for co in ni.buses.country.unique():
        #     lines_co[co] = n.lines[country_map_lines.apply(lambda L : L.str.contains(co).fillna(False)).any(axis=1)].s_nom_opt.sum()
            
        AC_DC_total = pd.DataFrame(AC_DC.groupby("carrier").p_nom_opt.sum())/1e3
        capacity.append(AC_DC_total.rename(columns={'p_nom_opt':y}))
        
    df = pd.concat(capacity,axis=1)
    
    return df

def extract_transmission_H2(n):
    # Add projected values
    capacity = []
    for y, ni in n.items():
        
        H2_pipelines = pd.concat([ni.links[ni.links.carrier.isin(["H2 pipeline"])],
                                  ni.links[ni.links.carrier.isin(["H2 pipeline retrofitted"])]
                                  ])
        
        buses_links = [c for c in H2_pipelines.columns if "bus" in c]
        country_map = H2_pipelines[buses_links].applymap(lambda x : mapper(x,ni,to_apply="country")) 
        
        H2_per_country = {}
        for co in ni.buses.country.unique():
            H2_per_country[co] = H2_pipelines[country_map.apply(lambda L : L.fillna('').str.contains(co)).any(axis=1)] \
                                .groupby("carrier").p_nom_opt.sum().fillna(0)/1e3
        pd.DataFrame.from_dict(H2_per_country, orient = 'columns')
        
        H2_total = pd.DataFrame(H2_pipelines.groupby("carrier").p_nom_opt.sum())/1e3
        capacity.append(H2_total.rename(columns={'p_nom_opt':y}))

    df = pd.concat(capacity,axis=1)
    
    return df

def extract_storage_units(n,color_shift):
    n_store = {}
    storage_function = {"hydro" : "get_state_of_charge_t", "PHS" : "get_state_of_charge_t"}
    storage_horizon = {"hydro" : "LT", "PHS" : "ST", "H2 Store" : "LT",
            "battery" : "ST", "home battery" : "ST",
            "ammonia store" : "LT"}
    
    carrier = list(storage_horizon.keys())
    
    plt.close('all')
    plt.figure(figsize=(14,8))
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

    plt.legend()
    plt.savefig(Path(path,"storage_unit.png"))
    return n_store

def extract_gas_phase_out(n,year,subset=None):
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
        ["p_nom_opt"]
        .sum(numeric_only=True)
        .reset_index()
    )
    n_cgt.loc[n_cgt["build_year"]!=year, "build_year"] = "historical"
    n_cgt = (
        n_cgt
        .groupby(by=dimensions)
        .sum()
        .reset_index()
        .pivot(index="country", columns="build_year", values="p_nom_opt")
        .sort_values(by=year, ascending=False)
    ) / 1e3 # GW
    
    return n_cgt[n_cgt[year] >= 1]

def extract_graphs(years,n_path,n_name,countries=None,subset_production=None,subset_balancing=None, color_shift = {2030:"C0",2035:"C2",2040:"C1"}):
    
    n = {}
    for y in years:
        run_name = Path(n_path, "postnetworks", n_name + f"{y}.nc")
        n[y] = pypsa.Network(run_name)
        assign_countries(n[y])
        change_p_nom_opt_carrier(n[y],carrier='AC')
        
    #non-country specific extracts   
    ACDC_grid = extract_transmission_AC_DC(n,n_path,n_name)
    H2_grid = extract_transmission_H2(n)
    n_gas = extract_gas_phase_out(n,2030)
    
    for y in years:
        if countries:
            n[y].generators = select_countries(n[y].generators,countries)
            n[y].links = select_countries(n[y].links,countries)
            n[y].storage_units = select_countries(n[y].storage_units,countries)
            n[y].stores = select_countries(n[y].stores,countries)
        
    #country specific extracts   
    n_sto = extract_storage_units(n,color_shift)
    n_prod = extract_production_units(n).get("p_nom_opt")
    n_res_pot = extract_res_potential(n)
    n_res = extract_production_units(n,subset_gen = ["solar","onwind","offwind","ror"],
                                     subset_links = [""]).get("p_nom_opt")
    n_bal = extract_production_units(n,subset_gen = [""], 
                                     subset_links = ["H2 Electrolysis", "H2 Fuel Cell", "battery charger",
                                                        "home battery charger", "Haber-Bosch", "Sabatier", 
                                                        "ammonia cracker", "helmeth", "SMR", "SMR CC"]).get("p_nom_opt")
    n_ff  = extract_production_units(n,subset_gen = [""], 
                                     subset_links = ["coal", "lignite", "oil","CCGT","OCGT"] ).get("p_nom_opt")

    #extract
    n_prod.to_csv(Path(csvs,"power_production_capacities.csv"))
    n_res_pot.to_csv(Path(csvs,"res_potentials.csv"))
    n_res.to_csv(Path(csvs,"res_capacities.csv"))
    ACDC_grid.to_csv(Path(csvs,"grid_capacity.csv"))
    H2_grid.to_csv(Path(csvs,"H2_network_capacity.csv"))
    n_bal.to_csv(Path(csvs,"power_balance_capacities.csv"))
    n_gas.to_csv(Path(csvs,"gas_phase_out.csv"))
    n_ff.to_csv(Path(csvs,"fossil_fuels.csv"))
    return 


# #%%
def assign_countries(n):
    n.buses.loc[n.buses.location!="EU","country"] = n.buses.loc[n.buses.loc[n.buses.location!="EU","location"].values,"country"].values
    n.buses.loc[n.buses.location=="EU","country"] = 'EU'
    return

def mapper(x,n,to_apply=None):
        if x in n.buses.index:
            return n.buses.loc[x,to_apply]
        else:
            return np.nan    
        
def searcher(x,carrier):
        if carrier in x.to_list():
            return str(x.to_list().index(carrier))
        else:
            return np.nan
        
def extract_loads(n):
    profiles =  {}
    for y, ni in n.items():
        loads_t = ni.loads_t.p.T
        loads_t = loads_t.loc[~(loads_t.index.str.contains('H2')|loads_t.index.str.contains("NH3"))]
        loads_t["country"] = ni.buses.loc[ni.loads.loc[loads_t.index].bus].country.values
        loads_t.reset_index(inplace=True)
        loads_t["Load"].where(loads_t["Load"].str.contains("industry"),"Load",inplace=True)
        loads_t["Load"].mask(loads_t["Load"].str.contains("industry"),"Industry",inplace=True)
        loads_t = loads_t.groupby(["country","Load"]).sum()
        profiles[y] = loads_t
    return pd.concat(profiles,names=["Years"])
    
if __name__ == "__main__":
    
    #for testing
    years = [2030, 2035, 2040]
    path = Path("analysis", "Graph_Extraction_template")
    
    n_path = Path(path,"results")
    n_name = "elec_s181_37m_lv3.0__3H_"
    csvs = Path(path,"csvs_for_graphs")
    
    countries= None #["BE","DE","FR","UK"]
    
    extract_graphs(years,n_path,n_name,countries=countries)
    
