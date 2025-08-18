from __future__ import annotations
import datetime as _dt
import pathlib   as _pl
import re
import pandas as _pd
from gurobipy import Model as _Model

class _ZeroVar:
    X = 0.0

_zero_var = _ZeroVar()   

def _vars_like(m: _Model, pattern: str):
    rx = re.compile(pattern)
    return [v for v in m.getVars() if rx.match(v.VarName)]

def _to_df(varlist, colnames):
    rows = []
    for v in varlist:
        name = v.VarName
        if "_" in name:
            parts = name.split("_")
        elif "[" in name and name.endswith("]"):
            inner = name[name.find("[") + 1 : -1]
            parts = re.split(r"[,\s]+", inner)
            parts.insert(0, name[: name.find("[")])  
        else:         
            continue

        if len(parts) < len(colnames):
            continue
        rows.append({c: p for c, p in zip(colnames, parts)} | {"value": v.X})
    return _pd.DataFrame(rows)

def _tupledict_to_df(tdict, col_names):
    rows = []
    for k, var in tdict.items():
        if isinstance(k, tuple):
            rows.append({c: str(v) for c, v in zip(col_names, k)} | {"value": var.X})
        else:          
            rows.append({col_names[0]: str(k), "value": var.X})
    return _pd.DataFrame(rows)


def dump_results(model: _Model, *, outdir: str | _pl.Path = ".", tag: str | None = None):

    outdir = _pl.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if tag is None:
        tag = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = outdir / f"results_{tag}.xlsx"

    # ---- collectors ------------------------------------------------------- #
    sheets: dict[str, _pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Capture objective value (may be unavailable for infeasible/no‑solution)
    # ------------------------------------------------------------------
    try:
        total_obj = model.ObjVal if model.SolCount > 0 else None
    except Exception:
        total_obj = None

    # ------------------------------------------------------------------
    # Simple cost summary (CAPEX vs OPEX w/out penalties)
    # ------------------------------------------------------------------
    capex_prefix = ("zS", "zST", "zLP", "bPL")
    pen_prefix   = ("u",)      # unmet‑demand penalties

    capex_total  = 0.0
    opex_total   = 0.0
    pen_total    = 0.0
    if total_obj is not None:
        for v in model.getVars():
            if v.Obj == 0:
                continue
            if v.VarName.startswith(capex_prefix):
                capex_total += v.X * v.Obj
            elif v.VarName.startswith(pen_prefix):
                pen_total   += v.X * v.Obj

        opex_total = total_obj - capex_total - pen_total

        cost_df = _pd.DataFrame(
            {
                "category": ["CAPEX", "OPEX", "Penalties", "Objective"],
                "value":    [capex_total, opex_total, pen_total, total_obj],
            }
        )
        sheets["cost_summary"] = cost_df

    explicit = getattr(model, "_vars", {})

    # 1) Storage investments  z^S_ifkt
    if "zS" in explicit:
        df = _tupledict_to_df(explicit["zS"], ["port", "fuel", "size", "t"])
    else:
        zs = _vars_like(model, r"^zS")
        df = _to_df(zs, ["zS", "port", "fuel", "size", "t"])
    if not df.empty:
        sheets["storage_invest"] = df

    # 1b) Local production capacity investments  zLP_i_f_t
    if "zLP" in explicit:
        df = _tupledict_to_df(explicit["zLP"], ["port", "fuel", "t"])
    else:
        zlp = _vars_like(model, r"^zLP")
        df = _to_df(zlp, ["zLP", "port", "fuel", "t"])
    if not df.empty:
        sheets["localprod_invest"] = df

    # 1c) Battery‑storage investments  zST_i_opt_t
    if "zST" in explicit:
        df = _tupledict_to_df(explicit["zST"], ["port", "option", "t"])
    else:
        zst = _vars_like(model, r"^zST")
        df = _to_df(zst, ["zST", "port", "option", "t"])
    if not df.empty:
        sheets["battery_invest"] = df

    # 2) Storage level  s_ifτ
    ss = _vars_like(model, r"^s")
    if ss:
        df = _to_df(ss, ["s", "port", "fuel", "t", "tau"])
        if not df.empty:
            sheets["storage_level"] = df

    # 3) Vessel inventory   a_v_t_tau
    if "a" in explicit:
        df = _tupledict_to_df(explicit["a"], ["vessel", "t", "tau"])
    else:
        aa = _vars_like(model, r"^a")
        df = _to_df(aa, ["a", "vessel", "t", "tau"])
    if not df.empty:
        sheets["vessel_level"] = df

    # 4) Battery energy  e_i_t_tau
    ee = _vars_like(model, r"^e")
    if ee:
        df = _to_df(ee, ["e", "port", "t", "tau"])
        if not df.empty:
            sheets["battery_level"] = df

    # 5) Generic delivered quantity   q_ifτ
    qq = _vars_like(model, r"^q")
    if qq:
        df = _to_df(qq, ["q", "port", "fuel", "t", "tau"])
        if not df.empty:
            sheets["flows_raw"] = df

    # 5b) Electricity mass‑balance snapshot  (qEL, qPR_EL, qGR, qDC, qCH)
    # ------------------------------------------------------------------
    if all(k in explicit for k in ("qEL", "qPR_EL", "qGR", "qDC", "qCH")):
        qel = explicit["qEL"]
        qpr = explicit["qPR_EL"]
        qgr = explicit["qGR"]
        qdc = explicit["qDC"]
        qch = explicit["qCH"]

        def _v(tdict, key):
            """Safe value lookup for tupledicts (0.0 if key absent)."""
            var = tdict.get(key)
            return var.X if var is not None else 0.0

        rows = []
        for (i, t, tau), var in qel.items():
            rows.append(
                {
                    "port":  i,
                    "t":     t,
                    "tau":   tau,
                    "demand":       var.X,                       # qEL
                    "local_prod":   _v(qpr, (i, t, tau)),        # qPR_EL
                    "grid":         _v(qgr, (i, t, tau)),        # qGR
                    "discharge":    _v(qdc, (i, t, tau)),        # qDC
                    "charge":       _v(qch, (i, t, tau)),        # qCH
                }
            )

        df = _pd.DataFrame(rows)
        if not df.empty:
            sheets["el_balance"] = df

# ------------------------------------------------------------------
# Grid‑imports sheet  qGR_i_t_tau  → sheets["grid_imports"]
# ------------------------------------------------------------------
    if "qGR" in explicit:
        qgr_td = explicit["qGR"]           
        gi_rows = [
            {"port": i, "t": t, "tau": tau, "qGR": var.X}
            for (i, t, tau), var in qgr_td.items()
            if var.X != 0
        ]
        gi_df = _pd.DataFrame(gi_rows)
        if not gi_df.empty:
            sheets["grid_imports"] = gi_df

    # ------------------------------------------------------------------
    # 5d) Unmet‑demand sheet  u_i_f_t_tau
    # ------------------------------------------------------------------
    if "u" in explicit:
        u_td = explicit["u"]                    
        u_rows = [
            {"port": i, "fuel": f, "t": t, "tau": tau, "unmet": var.X}
            for (i, f, t, tau), var in u_td.items() if var.X > 0
        ]
        u_df = _pd.DataFrame(u_rows)
        if not u_df.empty:
            sheets["unmet_demand"] = u_df

    # ------------------------------------------------------------------
    # 5e) Storage utilisation – inventory / capacity (%)
    # ------------------------------------------------------------------
    if all(k in explicit for k in ("s", "cS")):
        s_td  = explicit["s"]     
        c_td  = explicit["cS"]     
        util_rows = []
        for (i, f, t, tau), s_var in s_td.items():
            cap = c_td.get((i, f, t))
            cap_val = cap.X if cap is not None else 0.0
            if cap_val > 0:
                util_rows.append({
                    "port": i, "fuel": f, "t": t, "tau": tau,
                    "inventory": s_var.X,
                    "capacity": cap_val,
                    "util_pct": s_var.X / cap_val
                })
        util_df = _pd.DataFrame(util_rows)
        if not util_df.empty:
            sheets["storage_util"] = util_df

    # ------------------------------------------------------------------
    # 5f) Vessel movements – rows where bTR == 1
    # ------------------------------------------------------------------
    if "bTR" in explicit:
        btr_td = explicit["bTR"]   
        move_rows = [
            {"vessel": v, "from": i, "to": j, "t": t, "tau": tau}
            for (v, i, j, t, tau), var in btr_td.items() if var.X > 0.5
        ]
        mv_df = _pd.DataFrame(move_rows)
        if not mv_df.empty:
            sheets["vessel_moves"] = mv_df


# ------------------------------------------------------------------
# Capacity built per period – storage tanks & batteries
# ------------------------------------------------------------------
    if all(k in explicit for k in ("cS", "cST")):
        cS_td  = explicit["cS"]     
        cST_td = explicit["cST"]    
        cap_rows = []

        for (i, f, t), cap_var in cS_td.items():
            if t == 0:
                continue
            prev = cS_td.get((i, f, t-1), _zero_var).X
            delta = cap_var.X - prev
            if delta > 1e-6:
                cap_rows.append({"port": i, "t": t, "fuel": f, "capacity": delta})

        for (i, t), bat_var in cST_td.items():
            if t == 0:
                continue
            prev = cST_td.get((i, t-1), _zero_var).X
            delta = bat_var.X - prev
            if delta > 1e-6:
                cap_rows.append({"port": i, "t": t, "fuel": "el", "capacity": delta})

        cap_df = _pd.DataFrame(cap_rows)
        if not cap_df.empty:
            sheets["capacity_built"] = cap_df

    # ------------------------------------------------------------------
    # NPV summary – CAPEX, OPEX, Objective w/o penalties, Levelised cost
    # ------------------------------------------------------------------
    if total_obj is not None and capex_total is not None and opex_total is not None and \
       "q" in explicit and "qEL" in explicit:
        delivered_vol = (
            sum(v.X for v in explicit["q"].values()) +
            sum(v.X for v in explicit["qEL"].values())
        )
        obj_wo_pen = capex_total + opex_total           
        levelised  = obj_wo_pen / delivered_vol if delivered_vol > 0 else None

        npv_df = _pd.DataFrame(
            {
                "metric": ["CAPEX NPV", "OPEX NPV",
                           "Objective w/o penalties", "Levelised cost"],
                "value":  [capex_total, opex_total, obj_wo_pen, levelised],
            }
        )
        sheets["npv_summary"] = npv_df

    # 5c) Fuel‑mass balances (non‑electric)  -------------------------------
    if all(k in explicit for k in ("q", "qDE", "qPR")):
        q_tot = explicit["q"]      
        q_de  = explicit["qDE"]
        q_pr  = explicit["qPR"]
        u_md  = explicit.get("u", {})        
        qT_td = explicit.get("qT", {})         
        d_td  = explicit.get("d", {})          
      
        vessel_fuel_guess = {}
        F_list = model._sets.get("F", [])
        for (v_id, *_rest) in d_td.keys():
            if v_id in vessel_fuel_guess:
                continue
            v_low = str(v_id).lower()
            vessel_fuel_guess[v_id] = next(
                (f for f in F_list if f in v_low),
                None  
            )

        def _v(tdict, key):
            var = tdict.get(key)
            return var.X if var is not None else 0.0

        P   = model._sets["P"]
        T   = model._sets["T"]
        Tau = model._sets["Tau"]

        for f in F_list:
            rows = []
            for i in P:
                for t in T:
                    for tau in Tau[t]:
                        key = (i, f, t, tau)
                        demand   = _v(q_tot, key) + _v(u_md, key)   
                        from_sto = _v(q_de, key)
                        local_pr = _v(q_pr, key)
                      
                        piped_in = sum(_v(qT_td, (j, i, f, t, tau)) for j in P)
                        vessel_in = sum(
                            _v(d_td, (v, i, t, tau))
                            for v in vessel_fuel_guess
                            if vessel_fuel_guess[v] == f
                        )
                        unmet = _v(u_md, key)
                        rows.append(
                            dict(
                                port=i, t=t, tau=tau, demand=demand,
                                from_storage=from_sto, local_prod=local_pr,
                                pipeline=piped_in, vessel=vessel_in, unmet=unmet
                            )
                        )
            df = _pd.DataFrame(rows)
            if not df.empty:
                sheet_name = f"{f}_balance"
                sheet_name = sheet_name[:31]
                sheets[sheet_name] = df

    # ------------------------------------------------------------------
    # Additional KPI sheets (per‑port KPIs, infrastructure footprint,
    # and pivot‑friendly aggregated flows)                         
    # ------------------------------------------------------------------
    P   = model._sets.get("P", [])
    F   = model._sets.get("F", [])
    T   = model._sets.get("T", [])
    Tau = model._sets.get("Tau", {})

    # ---- 1) Comparative KPIs by port ----------------------------------
    if all(k in explicit for k in ("q", "u", "qEL")):
        delivered_td   = explicit["q"]     
        unmet_td       = explicit["u"]     
        el_delivered   = explicit["qEL"]  

        port_rows = {i: {"delivered_nonEL": 0.0,
                         "unmet_nonEL":    0.0,
                         "delivered_EL":   0.0}
                     for i in P}

        
        for (i, f, t, tau), var in delivered_td.items():
            port_rows[i]["delivered_nonEL"] += var.X
        for (i, f, t, tau), var in unmet_td.items():
            port_rows[i]["unmet_nonEL"] += var.X
        
        for (i, t, tau), var in el_delivered.items():
            port_rows[i]["delivered_EL"] += var.X

       
        if "storage_util" in sheets:
            util_df = sheets["storage_util"]
            util_avg = util_df.groupby("port")["util_pct"].mean().to_dict()
        else:
            util_avg = {}

        kpi_rows = []
        for i, rec in port_rows.items():
            demand_nonEL = rec["delivered_nonEL"] + rec["unmet_nonEL"]
            kpi_rows.append(
                dict(
                    port=i,
                    demand_nonEL=demand_nonEL,
                    delivered_nonEL=rec["delivered_nonEL"],
                    unmet_nonEL=rec["unmet_nonEL"],
                    delivered_pct_nonEL=rec["delivered_nonEL"]/demand_nonEL
                    if demand_nonEL > 0 else None,
                    delivered_EL=rec["delivered_EL"],
                    avg_storage_util=util_avg.get(i, None),
                )
            )
        kpi_df = _pd.DataFrame(kpi_rows)
        if not kpi_df.empty:
            sheets["port_kpi"] = kpi_df

    # ---- 3) Infrastructure footprint by port -------------------------
    if all(k in explicit for k in ("cS", "cST", "qGR")):
        cS_td  = explicit["cS"]    
        cST_td = explicit["cST"]   
        qGR_td = explicit["qGR"]   
        bAT_td = explicit.get("bAT", {})  

        rows = []
        for i in P:
            storage_cap_tot = 0.0
            for f in F:
                var = cS_td.get((i, f, max(T)))
                storage_cap_tot += var.X if var is not None else 0.0
            battery_cap = cST_td.get((i, max(T)), _zero_var).X
            peak_grid   = max(
                (qGR_td.get((i, t, tau), _zero_var).X
                 for t in T for tau in Tau[t]),
                default=0.0
            )
            vessels_here = {
                v for (v, ip, t, tau), var in bAT_td.items()
                if ip == i and var.X > 0.5
            }
            rows.append(
                dict(port=i,
                     storage_capacity=storage_cap_tot,
                     battery_capacity=battery_cap,
                     peak_grid_draw=peak_grid,
                     vessels_observed=len(vessels_here))
            )
        infra_df = _pd.DataFrame(rows)
        if not infra_df.empty:
            sheets["port_infra"] = infra_df

    # ---- 6) Pivot‑friendly aggregated delivered flows ----------------
    if "flows_raw" in sheets:
        fr = sheets["flows_raw"]
        agg = (
            fr.groupby(["port", "fuel", "t"], as_index=False)
              .agg(total_delivered=("value", "sum"))
        )
        if not agg.empty:
            sheets["flows_agg_t"] = agg



    if not sheets or all(df.empty for df in sheets.values()):
        sheets["empty"] = _pd.DataFrame({"msg": ["No recognised variables found."]})

    with _pd.ExcelWriter(outfile, engine="xlsxwriter") as xw:
        for name, frame in sheets.items():
            frame = frame.copy()
            frame.reset_index(drop=True, inplace=True)
            xw.book.use_zip64()  # safety for big cases
            frame.to_excel(xw, sheet_name=name, index=False)

    # ------------------------------------------------------------------
    #  Auto‑generate quick‑look PNG figures (best‑effort; non‑fatal)
    # ------------------------------------------------------------------
    try:
        from plots import generate_plots
        figs_dir = outdir / "figures"
        generate_plots(sheets, figs_dir, tag)
    except Exception as _e:
        print(f"Plot generation skipped: {_e}")

    print(f"✔ Results written to {outfile}")
    return outfile