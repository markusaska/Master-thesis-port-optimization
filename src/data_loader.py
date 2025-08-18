from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

import itertools as it

import pandas as pd


def _lower(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def load_data(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    xls = pd.ExcelFile(path)

    data: Dict[str, Any] = {}

    ports_df = xls.parse("ports")
    data["ports"] = _lower(ports_df["port_id"]).tolist()

    fuels_df = xls.parse("fuels")
    all_fuels = _lower(fuels_df["fuel_id"]).tolist()
    data["fuels"] = [f for f in all_fuels if f != "el"]

    periods_df = xls.parse("periods")
    data["periods"] = periods_df["t"].astype(int).tolist()

    ops_df = xls.parse("op_steps")
    ops_df["t"]   = ops_df["t"].astype(int)
    ops_df["tau"] = ops_df["tau"].astype(int)
    ops_df["is_refill"] = ops_df["is_refill"].astype(int)
    ops_df["t"] = ops_df["t"].astype(int)

    data["op_steps"] = {t: df["tau"].astype(int).tolist()
                        for t, df in ops_df.groupby("t")}

    data["refill_steps"] = {t: set(df.loc[df["is_refill"] == 1, "tau"].astype(int))
                            for t, df in ops_df.groupby("t")}

    if "is_home" in ops_df.columns:
        data["home_steps"] = {
            t: set(df.loc[df["is_home"] == 1, "tau"].astype(int))
            for t, df in ops_df.groupby("t")
        }
    else:
        data["home_steps"] = {t: set() for t in data["op_steps"]}


    store_df = xls.parse("storage_options")
    store_df["port_id"]   = _lower(store_df["port_id"])
    store_df["fuel_id"]   = _lower(store_df["fuel_id"])
    store_df["option_id"] = _lower(store_df["option_id"])
    store_df["capacity"] = pd.to_numeric(store_df["capacity"])
    store_df["capex"]    = pd.to_numeric(store_df["capex"])

    store_df.set_index(["port_id", "fuel_id", "option_id"], inplace=True)
    data["storage_options"] = store_df


    demand_frames: list[pd.DataFrame] = []
    sheet_names = [s.strip().lower() for s in xls.sheet_names]

    for sh in sheet_names:
        if sh.startswith("demand_"):
            fuel_name = sh[len("demand_"):].strip().lower()
            df = xls.parse(sh)
            df["fuel_id"] = fuel_name
            demand_frames.append(df)

    if "demand" in sheet_names:
        demand_frames.append(xls.parse("demand"))

    if not demand_frames:
        raise ValueError(
        )

    demand_df = pd.concat(demand_frames, ignore_index=True)

    demand_df["port_id"] = _lower(demand_df["port_id"])
    demand_df["fuel_id"] = _lower(demand_df["fuel_id"])
    demand_df["t"]   = demand_df["t"].astype(int)
    demand_df["tau"] = demand_df["tau"].astype(int)
    demand_df["demand"] = pd.to_numeric(demand_df["demand"])

    el_mask = demand_df["fuel_id"] == "el"
    if el_mask.any():
        el_df = demand_df.loc[el_mask].copy()
        el_df = el_df.drop(columns=["fuel_id"])
        el_df.set_index(["port_id", "t", "tau"], inplace=True)
        data["demand_el"] = el_df["demand"]
        demand_df = demand_df.loc[~el_mask]

    demand_df.set_index(["port_id", "fuel_id", "t", "tau"], inplace=True)
    data["demand"] = demand_df["demand"]

    if "vessels" not in sheet_names:
        raise ValueError("Sheet 'vessels' is required for Phase‑3 bunkering vessels.")

    ves_df = xls.parse("vessels")
    ves_df["v_id"]      = _lower(ves_df["v_id"])
    ves_df["fuel_id"]   = _lower(ves_df["fuel_id"])
    ves_df["home_port"] = _lower(ves_df["home_port"])
    ves_df["capacity"]  = pd.to_numeric(ves_df["capacity"])
    ves_df["k_tr"]      = pd.to_numeric(ves_df["k_tr"])
    ves_df.set_index("v_id", inplace=True)

    data["vessels"] = ves_df
    data["K_TR"]    = ves_df["k_tr"]

    data["vessel_ids"]       = ves_df.index.tolist()
    data["vessel_fuel"]     = ves_df["fuel_id"]
    data["vessel_home"]     = ves_df["home_port"]
    data["vessel_capacity"] = ves_df["capacity"]


    if "pipeline_pairs" in sheet_names:
        pl_df = xls.parse("pipeline_pairs")
        pl_df["i_from"]  = _lower(pl_df["i_from"])
        pl_df["j_to"]    = _lower(pl_df["j_to"])
        pl_df["fuel_id"] = _lower(pl_df["fuel_id"])
        pl_df["capex"]   = pd.to_numeric(pl_df["capex"])
        pl_df.set_index(["i_from", "j_to", "fuel_id"], inplace=True)
        data["K_PL"] = pl_df["capex"]
    else:
        data["K_PL"] = pd.Series(dtype=float)

    if "k_t" in sheet_names:
        kt_df = xls.parse("k_t")
        kt_df["i_from"]  = _lower(kt_df["i_from"])
        kt_df["j_to"]    = _lower(kt_df["j_to"])
        kt_df["fuel_id"] = _lower(kt_df["fuel_id"])
        kt_df["value"]   = pd.to_numeric(kt_df["value"])
        kt_df.set_index(["i_from", "j_to", "fuel_id"], inplace=True)
        data["K_T"] = kt_df["value"]
    else:
        data["K_T"] = pd.Series(dtype=float)

    if "lambda" in sheet_names:
        data["Lambda"] = float(xls.parse("lambda").iloc[0, 0])

    globals_raw = xls.parse("globals")
    globals_raw["key"] = globals_raw["key"].astype(str).str.strip().str.lower()
    globals_df = globals_raw.set_index("key")["value"]

    required_global_keys: set[str] = {
        "r", "m", "l_s", "n_years_per_period"
    }
    missing_keys = required_global_keys - set(globals_df.index)
    if missing_keys:
        raise ValueError(f"Globals sheet is missing required keys: {missing_keys}")

    data["globals"] = {
        "R": float(globals_df["r"]),
        "M": float(globals_df["m"]),
        "L_S": int(globals_df["l_s"]),
        "L_D": int(globals_df.get("l_d", 0)),
        "n_years_per_period": float(globals_df["n_years_per_period"]),
        "Lambda": float(globals_df.get("lambda", 1e4)),
    }

    if "Lambda" not in data:
        data["Lambda"] = float(data["globals"]["Lambda"])

    # ---------------- cost parameters from sheets ----------------
    def _fuel_series(sheet_name: str) -> pd.Series:
        df = xls.parse(sheet_name)
        df.columns = df.columns.str.strip().str.lower()
        if 'fuel' in df.columns and 'fuel_id' not in df.columns:
            df.rename(columns={'fuel': 'fuel_id'}, inplace=True)
        if 'cost' in df.columns and 'value' not in df.columns:
            df.rename(columns={'cost': 'value'}, inplace=True)
        df["fuel_id"] = _lower(df["fuel_id"])
        df["value"]   = pd.to_numeric(df["value"])
        return df.set_index("fuel_id")["value"]

    data["K_DE"] = _fuel_series("k_de")   
    data["Pi"]   = _fuel_series("pi")     

    data["K_HS"] = _fuel_series("k_hs")   
    data["K_VD"] = _fuel_series("k_vd")   

    if "k_lp" in sheet_names:
        klp_df = xls.parse("k_lp")
        klp_df.columns = klp_df.columns.str.strip().str.lower()
        if "value" in klp_df.columns:
            klp_val = float(klp_df.loc[0, "value"]) 
        else:
            klp_val = float(klp_df.iloc[0, 0])
        data["K_LP_EL"] = klp_val
    else:
        data["K_LP_EL"] = 0.0

    data["K_LP"] = pd.Series(dtype=float)

    if "k_hv" in sheet_names:
        hv_df = xls.parse("k_hv")
        hv_df.columns = hv_df.columns.str.strip().str.lower()
        if "v" in hv_df.columns and "v_id" not in hv_df.columns:
            hv_df.rename(columns={"v": "v_id"}, inplace=True)
        hv_df["v_id"] = _lower(hv_df["v_id"])
        hv_df["value"] = pd.to_numeric(hv_df["value"])
        hv_df.set_index("v_id", inplace=True)
        data["K_HV"] = hv_df["value"]    
    else:
        data["K_HV"] = pd.Series(dtype=float) 

    sheet_names = [s.strip().lower() for s in xls.sheet_names]  

    if "production_caps" in sheet_names:
        prodcap_df = xls.parse("production_caps")
        prodcap_df.columns = prodcap_df.columns.str.strip().str.lower()
        if "port" in prodcap_df.columns and "port_id" not in prodcap_df.columns:
            prodcap_df.rename(columns={"port":"port_id"}, inplace=True)
        if "fuel" in prodcap_df.columns and "fuel_id" not in prodcap_df.columns:
            prodcap_df.rename(columns={"fuel":"fuel_id"}, inplace=True)
        for c in ("capacity","cap"):
            if c in prodcap_df.columns and "value" not in prodcap_df.columns:
                prodcap_df.rename(columns={c:"value"}, inplace=True)
        prodcap_df["port_id"] = _lower(prodcap_df["port_id"])
        prodcap_df["fuel_id"] = _lower(prodcap_df["fuel_id"])
        prodcap_df["value"]   = pd.to_numeric(prodcap_df["value"])
        prodcap_df.set_index(["port_id","fuel_id"], inplace=True)
        data["C_PR"] = prodcap_df["value"]
    else:
        data["C_PR"] = pd.Series(dtype=float)

    if "production_caps_el" in sheet_names:
        elcap_df = xls.parse("production_caps_el")
        elcap_df.columns = elcap_df.columns.str.strip().str.lower()
        if "port" in elcap_df.columns and "port_id" not in elcap_df.columns:
            elcap_df.rename(columns={"port":"port_id"}, inplace=True)
        for c in ("value","capacity","cap"):
            if c in elcap_df.columns and "value" not in elcap_df.columns:
                elcap_df.rename(columns={c:"value"}, inplace=True)
        elcap_df["port_id"] = _lower(elcap_df["port_id"])
        elcap_df["value"]   = pd.to_numeric(elcap_df["value"])
        elcap_df.set_index("port_id", inplace=True)
        data["C_PR_EL"] = elcap_df["value"]
    else:
        data["C_PR_EL"] = pd.Series(dtype=float)

    if "grid_caps" in sheet_names:
        gridcap_df = xls.parse("grid_caps")
        gridcap_df.columns = gridcap_df.columns.str.strip().str.lower()
        if "port" in gridcap_df.columns and "port_id" not in gridcap_df.columns:
            gridcap_df.rename(columns={"port": "port_id"}, inplace=True)
        if "capacity" in gridcap_df.columns and "value" not in gridcap_df.columns:
            gridcap_df.rename(columns={"capacity": "value"}, inplace=True)
        if "cap" in gridcap_df.columns and "value" not in gridcap_df.columns:
            gridcap_df.rename(columns={"cap": "value"}, inplace=True)
        gridcap_df["port_id"] = _lower(gridcap_df["port_id"])
        gridcap_df["value"]   = pd.to_numeric(gridcap_df["value"])
        gridcap_df.set_index("port_id", inplace=True)
        data["C_GR"] = gridcap_df["value"]           
        data["grid_caps"] = gridcap_df["value"]     
    else:
        data["C_GR"] = pd.Series(dtype=float)
        data["grid_caps"] = pd.Series(dtype=float)

    if "prod_costs" in sheet_names:
        pr = _fuel_series("prod_costs")   
        if "el" in pr.index:
            data["K_PR_EL"] = pr.loc["el"]
            pr = pr.drop("el")
        else:
            data["K_PR_EL"] = 0.0
        data["K_PR"] = pr
    else:
        data["K_PR"]    = pd.Series(dtype=float)
        data["K_PR_EL"] = 0.0

    if "grid_costs" in sheet_names:
        grd_df = xls.parse("grid_costs")
        grd_df.columns = grd_df.columns.str.strip().str.lower()

        
        if {"port_id", "t", "tau", "value"}.issubset(grd_df.columns):
            grd_df["port_id"] = _lower(grd_df["port_id"])
            grd_df["t"]   = grd_df["t"].astype(int)
            grd_df["tau"] = grd_df["tau"].astype(int)
            grd_df["value"] = pd.to_numeric(grd_df["value"])
            grd_df.set_index(["port_id", "t", "tau"], inplace=True)
            data["K_GR_time"] = grd_df["value"]   
            data["K_GR"] = 0.0                   

       
        elif "port_id" in grd_df.columns:
            grd_df["port_id"] = _lower(grd_df["port_id"])
            grd_df["value"]   = pd.to_numeric(grd_df["value"])
            grd_df.set_index("port_id", inplace=True)
            data["K_GR"] = grd_df["value"]       

        else:
            data["K_GR"] = float(grd_df.iloc[0, 0]) 

    else:
        data["K_GR"] = 0.0

    battery_sheet_name = next(
        (s for s in xls.sheet_names if s.strip().lower() in {"battery_options", "battery options"}),
        None,
    )

    if battery_sheet_name is not None:
        st_df = xls.parse(battery_sheet_name)
        st_df.columns = st_df.columns.str.strip().str.lower()
        for c in ("option", "battery_id", "id"):
            if c in st_df.columns and "option_id" not in st_df.columns:
                st_df.rename(columns={c: "option_id"}, inplace=True)
        for c in ("capacity", "cap", "value"):
            if c in st_df.columns and "capacity" not in st_df.columns:
                st_df.rename(columns={c: "capacity"}, inplace=True)
        if "cost" in st_df.columns and "capex" not in st_df.columns:
            st_df.rename(columns={"cost": "capex"}, inplace=True)

        # --- mandatory columns check ------------------------------------------
        mandatory_cols = {"port_id", "option_id", "capacity", "capex"}
        missing = mandatory_cols - set(st_df.columns)
        if missing:
            raise ValueError(
                f"Battery sheet '{battery_sheet_name}' is missing required columns: {missing}"
            )

        st_df["port_id"]   = _lower(st_df["port_id"])
        st_df["option_id"] = _lower(st_df["option_id"])

        num_cols = st_df.columns.difference(["port_id", "option_id"])
        for col in num_cols:
            st_df[col] = pd.to_numeric(st_df[col], errors="coerce")

        st_df.set_index(["port_id", "option_id"], inplace=True)

        data["battery_options"] = st_df

        data["storage_opts_el"] = st_df[["capacity", "capex"]]

        data["B"]    = st_df.index.get_level_values("option_id").unique().tolist()
        data["S_ST"] = st_df["capacity"]     
        data["K_ST"] = st_df["capex"]        
    else:
        data["battery_options"] = pd.DataFrame()
        data["storage_opts_el"] = pd.DataFrame()
        data["B"]    = []
        data["S_ST"] = pd.Series(dtype=float)
        data["K_ST"] = pd.Series(dtype=float)

    if "charge_efficiency" in sheet_names:
        eff_df = xls.parse("charge_efficiency")
        data["eta_CH"] = float(eff_df.iloc[0, 0])
    else:
        data["eta_CH"] = 1.0

    return data

