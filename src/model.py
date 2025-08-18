"""
This module:
  • loads cleaned input data (expects a dict from the loader)
  • builds a Gurobi model
  • returns the ready-to-solve model so that higher-level scripts
    can optimise and post-process results.
"""
from __future__ import annotations
import itertools as it
from pathlib import Path
from typing import Dict, Any
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from collections import defaultdict

class _ZeroVar:
    X = 0.0

_zero_var = _ZeroVar()           

def _wrap_var_dict(td):
    return defaultdict(lambda: _zero_var, td)


def build_phase4_model(data: Dict[str, Any], *, name: str | None = None, log: bool = True) -> gp.Model:
    m = gp.Model(name or "phase4_ports")
    if not log:
        m.Params.LogToConsole = 0

    # ------------------------------------------------------------------
    # Unpack sets and global scalars
    # ------------------------------------------------------------------
    P  = data["ports"]               
    F  = data["fuels"]               
    T  = data["periods"]            
    Tau = data["op_steps"]         
    TRF = data["refill_steps"]       

    globals_ = data["globals"]
    R   = globals_["R"]            
    M   = globals_["M"]              
    L_S = globals_["L_S"]            

    n_years = globals_["n_years_per_period"]  
   
    deltaP = {t: (1 + R) ** (-t * n_years) for t in T}              # CAPEX
    deltaO = {t: (1 + R) ** (-t * n_years / 2.0) for t in T}        # OPEX

    # ------------------------------------------------------------------
    # Time‑expansion weights  W_O[t]
    # ------------------------------------------------------------------

    weeks_simulated = 2       

    if "W_O" in data:
        W_O = data["W_O"]                
    else:
        W_O = {t: (52 / weeks_simulated) * n_years for t in T}

    # ------------------------------------------------------------------
    # Electricity‑specific data 
    # ------------------------------------------------------------------
    EL = "EL"                                  
    grid_cap        = data["grid_caps"]
    prod_cap_el     = data["C_PR_EL"]            
    prod_cap        = data["C_PR"]                
    K_PR            = data["K_PR"]               
    K_PR_EL         = data["K_PR_EL"]             
    K_GR            = data["K_GR"]               
    K_GR_time      = data.get("K_GR_time", None)
    storage_opts_el = data["storage_opts_el"]
    eta_CH          = data["eta_CH"]              
    demand_el       = data["demand_el"]
    K_ST            = data["K_ST"]

    S_opts = data["storage_options"]   
    S_opts = S_opts.sort_index()  

    # --- Pipeline data -------------------------------------------------
    pairs  = data["K_PL"].index            
    Lambda = data["Lambda"]
    K_PL   = data["K_PL"]                  
    K_T    = data["K_T"]                   

    # --- Bunkering‑vessel data ---------------------------------------
    vessels = data["vessels"]                      
    V       = vessels.index.tolist()               
    F_v     = vessels["fuel_id"].to_dict()        
    H_v     = vessels["home_port"].to_dict()      
    C_B     = vessels["capacity"].to_dict()        
    K_TR_v  = vessels["k_tr"].to_dict()          
    home_steps = data["home_steps"]              

    D = data["demand"]  

    K_DE = data["K_DE"]      
    Pi   = data["Pi"]       
    K_HS            = data["K_HS"]                  
    K_VD            = data["K_VD"]                                   
    K_HV_v          = data["K_HV"]                  

    K_LP_EL = data.get("K_LP_EL", 0.0) 

    K_LP_dict = {f: 0.0 for f in F}
    K_LP_dict[EL] = K_LP_EL  

    # ------------------------------------------------------------------
    # Decision variables
    # ------------------------------------------------------------------

    zS = m.addVars(
        ((i, f, tank_id, t) for (i, f, tank_id) in S_opts.index for t in T),
        vtype=GRB.INTEGER,
        name="zS",
    )

    yS = m.addVars(P, F, T, vtype=GRB.BINARY, name="yS")

    cS = m.addVars(P, F, T, lb=0, name="cS")

    # Operational flows & states (continuous, ≥0)
    qDE = m.addVars(P, F, T, max(len(v) for v in Tau.values()), lb=0, name="qDE")  
    q   = m.addVars(P, F, T, max(len(v) for v in Tau.values()), lb=0, name="qTOT") 
    qRF = m.addVars(P, F, T, max(len(v) for v in Tau.values()), lb=0, name="qRF") 
    s   = m.addVars(P, F, T, max(len(v) for v in Tau.values()), lb=0, name="s")   
    u   = m.addVars(P, F, T, max(len(v) for v in Tau.values()), lb=0, name="u")  

    # --- Bunkering‑vessel decision variables ---------------------------------
    a   = m.addVars(V, T, max(len(v) for v in Tau.values()), lb=0, name="a")           
    bAT = m.addVars(V, P, T, max(len(v) for v in Tau.values()), vtype=GRB.BINARY, name="bAT")
    bDE = m.addVars(V, P, T, max(len(v) for v in Tau.values()), vtype=GRB.BINARY, name="bDE")
    bRE = m.addVars(V, P, T, max(len(v) for v in Tau.values()), vtype=GRB.BINARY, name="bRE")
    bTR = m.addVars(V, P, P, T, max(len(v) for v in Tau.values()), vtype=GRB.BINARY, name="bTR")
    d   = m.addVars(V, P, T, max(len(v) for v in Tau.values()), lb=0, name="d")        
    r   = m.addVars(V, P, T, max(len(v) for v in Tau.values()), lb=0, name="r")       

    # --- Non-electric local production variables  -------
    qPR = m.addVars(
        ((i, f, t, tau) for i in P for f in F for t in T for tau in Tau[t]),
        lb=0, name="qPR"
    )

    # ----- Electricity variables ---------------------------------
    max_tau = max(len(v) for v in Tau.values())

    qPR_EL = m.addVars(P, T, max_tau, lb=0, name="qPR_EL")  
    qGR    = m.addVars(P, T, max_tau, lb=0, name="qGR")     
    qCH    = m.addVars(P, T, max_tau, lb=0, name="qCH")   
    qDC    = m.addVars(P, T, max_tau, lb=0, name="qDC")  
    eBat   = m.addVars(P, T, max_tau, lb=0, name="e")      
    cST    = m.addVars(P, T,           lb=0, name="cST")    

    # --- Local‑production capacity investments ------------------------------
    pVar = m.addVars(P, F + [EL], T, lb=0, name="p")          
    zLP  = m.addVars(P, F + [EL], T, lb=0, name="zLP")        

    # storage investment decisions
    zST = m.addVars(
        ((i, opt, t) for (i, opt) in storage_opts_el.index for t in T),
        vtype=GRB.INTEGER,
        name="zST",
    )

    # electricity delivered to meet demand
    qEL   = m.addVars(P, T, max_tau, lb=0, name="qEL")

    # --- Pipeline decision variables ------------------------------------------
    if len(pairs) > 0:
        bPL = m.addVars(pairs, T, vtype=GRB.BINARY,   name="bPL")  
        bEX = m.addVars(pairs, T, vtype=GRB.BINARY,   name="bEX")  
        qT  = m.addVars(((i, j, f, t, tau)
                         for (i, j, f) in pairs
                         for t in T
                         for tau in Tau[t]),
                        lb=0, vtype=GRB.CONTINUOUS, name="qT")
    else:
        bPL = bEX = qT = {}

    def var(vdict: gp.tupledict, i: str, f: str, t: int, tau: int):
        return vdict[i, f, t, tau]

    # ------------------------------------------------------------------
    # Generic Constraints
    # ------------------------------------------------------------------
    for (i, f, t, tau), demand_val in D.items():
        m.addConstr(var(q, i, f, t, tau) + var(u, i, f, t, tau) == demand_val,
                    name=f"demand_{i}_{f}_{t}_{tau}")


    for i, f in it.product(P, F):
        if (i, f) not in S_opts.index:
            continue
        m.addConstr(cS[i, f, 0] == 0, name=f"cS_init_{i}_{f}")
        tank_ids = S_opts.loc[(i, f)].index.get_level_values("option_id").unique()
        for t in T[1:]:
            rhs = cS[i, f, t - 1] + gp.quicksum(
                S_opts.loc[(i, f, tank_id), "capacity"] * zS[i, f, tank_id, t - 1]
                for tank_id in tank_ids
            )
            m.addConstr(cS[i, f, t] == rhs, name=f"cS_accum_{i}_{f}_{t}")

    for i, f, t in it.product(P, F, T):
        if (i, f) not in S_opts.index:
            continue
        m.addConstr(gp.quicksum(zS[i, f, opt, t] for opt in S_opts.loc[(i, f)].index.get_level_values('option_id').unique())
                    <= M * yS[i, f, t], name=f"linkS_{i}_{f}_{t}")

    for i, t in it.product(P, T):
        m.addConstr(gp.quicksum(yS[i, f, t] for f in F) <= L_S, name=f"fuelLimS_{i}_{t}")

    for i, f, t in it.product(P, F, T):
        m.addConstr(s[i, f, t, 0] == 0, name=f"s_init_{i}_{f}_{t}")
        for tau in Tau[t][1:]:
            rhs = (
                s[i, f, t, tau - 1]
                + qRF[i, f, t, tau - 1]
                - qDE[i, f, t, tau - 1]
                - gp.quicksum(qT[i, j, f, t, tau - 1] for j in P if (i, j, f) in pairs)
            )
            m.addConstr(
                s[i, f, t, tau] == rhs,
                name=f"s_bal_{i}_{f}_{t}_{tau}")

    # ---------- Pipeline constraints ----------------------------------
    if len(pairs) > 0:
        for (i, j, f) in pairs:
            for t in T:
                for tau in Tau[t]:
                    m.addConstr(qT[i, j, f, t, tau] <= Lambda * bEX[i, j, f, t],
                                name=f"PipeFlowCap_{i}_{j}_{f}_{t}_{tau}")

        for (i, j, f) in pairs:
            for t in T:
                m.addConstr(bEX[i, j, f, t] == gp.quicksum(bPL[i, j, f, k] for k in T if k < t),
                            name=f"PipeExistCum_{i}_{j}_{f}_{t}")

        for (i, j, f) in pairs:
            m.addConstr(gp.quicksum(bPL[i, j, f, t] for t in T) <= 1,
                        name=f"PipeSingleBuild_{i}_{j}_{f}")

    for i, f, t in it.product(P, F, T):
        allowed = TRF[t]
        for tau in Tau[t]:
            if tau not in allowed:
                m.addConstr(qRF[i, f, t, tau] == 0, name=f"noRF_{i}_{f}_{t}_{tau}")

    for i, f, t in it.product(P, F, T):
        for tau in TRF[t]:
            m.addConstr(qRF[i, f, t, tau] <= cS[i, f, t] - s[i, f, t, tau],
                        name=f"RFspace_{i}_{f}_{t}_{tau}")
            
    for i, f, t in it.product(P, F, T):
        for tau in Tau[t]:
            m.addConstr(s[i, f, t, tau] <= cS[i, f, t], name=f"sCap_{i}_{f}_{t}_{tau}")

    for i, f, t in it.product(P, F, T):
        for tau in Tau[t]:
            m.addConstr(qDE[i, f, t, tau] <= s[i, f, t, tau], name=f"invBound_{i}_{f}_{t}_{tau}")

    # ------- Local production capacity dynamics -------------------
    for i in P:
        for f in F + [EL]:
            m.addConstr(pVar[i, f, 0] == 0, name=f"pInit_{i}_{f}")
            for t in T[1:]:
                m.addConstr(
                    pVar[i, f, t] == pVar[i, f, t-1] + zLP[i, f, t-1],
                    name=f"pAccum_{i}_{f}_{t}"
                )
                lim = prod_cap_el[i] if f == EL else prod_cap[i, f]
                m.addConstr(pVar[i, f, t] <= lim,
                            name=f"pCap_{i}_{f}_{t}")

    for i, f, t in it.product(P, F, T):
        for tau in Tau[t]:
            m.addConstr(
                qPR[i, f, t, tau] <= pVar[i, f, t],
                name=f"ProdCap_{i}_{f}_{t}_{tau}"
            )

    for i, f, t in it.product(P, F, T):
        for tau in Tau[t]:
            m.addConstr(
                q[i, f, t, tau] ==
                qDE[i, f, t, tau]
                + qPR[i, f, t, tau]
                + gp.quicksum(qT[j, i, f, t, tau] for j in P if (j, i, f) in pairs)
                + gp.quicksum(d[v, i, t, tau] for v in V if F_v[v] == f),
                name=f"massBal_{i}_{f}_{t}_{tau}"
            )
    # ---------- Bunkering‑vessel constraints -------------------------
    for v in V:
        f_v = F_v[v]
        h   = H_v[v]
        for t in T:
            m.addConstr(a[v, t, 0] == 0, name=f"aInit_{v}_{t}")

            for tau in Tau[t]:
                m.addConstr(gp.quicksum(bAT[v, i, t, tau] for i in P) <= 1,
                            name=f"locUnique_{v}_{t}_{tau}")

                for i in P:
                    m.addConstr(r[v, i, t, tau] <= C_B[v] * bRE[v, i, t, tau],
                                name=f"rBind_{v}_{i}_{t}_{tau}")
                    m.addConstr(d[v, i, t, tau] <= C_B[v] * bDE[v, i, t, tau],
                                name=f"dBind_{v}_{i}_{t}_{tau}")
                m.addConstr(
                    gp.quicksum(d[v, i, t, tau] for i in P) <= a[v, t, tau],
                    name=f"delivInv_{v}_{t}_{tau}"
                )
                for i in P:
                    m.addConstr(bRE[v, i, t, tau] + bDE[v, i, t, tau] <= bAT[v, i, t, tau],
                                name=f"atCheck_{v}_{i}_{t}_{tau}")

                m.addConstr(a[v, t, tau] + gp.quicksum(r[v, i, t, tau] for i in P) <= C_B[v],
                            name=f"cap_{v}_{t}_{tau}")

                m.addConstr(gp.quicksum(bTR[v, i, j, t, tau] for i in P for j in P if j != i) <= 1,
                            name=f"moveLim_{v}_{t}_{tau}")

                m.addConstr(
                    gp.quicksum(bRE[v, i, t, tau] for i in P) +
                    gp.quicksum(bDE[v, i, t, tau] for i in P) +
                    gp.quicksum(bTR[v, i, j, t, tau] for i in P for j in P if j != i) <= 1,
                    name=f"stateLim_{v}_{t}_{tau}"
                )

                m.addConstr(
                    gp.quicksum(bAT[v, i, t, tau] for i in P) +
                    gp.quicksum(bTR[v, i, j, t, tau] for i in P for j in P if j != i) == 1,
                    name=f"locComplete_{v}_{t}_{tau}"
                )

            for tau in Tau[t][1:]:
                m.addConstr(
                    a[v, t, tau] ==
                    a[v, t, tau - 1] +
                    gp.quicksum(r[v, i, t, tau - 1] for i in P) -
                    gp.quicksum(d[v, i, t, tau - 1] for i in P),
                    name=f"aBal_{v}_{t}_{tau}"
                )

            for tau in home_steps.get(t, set()):
                m.addConstr(bAT[v, h, t, tau] == 1,
                            name=f"home_{v}_{t}_{tau}")

            for i in P:
                if i != h:
                    for tau in Tau[t]:
                        m.addConstr(bRE[v, i, t, tau] == 0,
                                    name=f"noRefuelAway_{v}_{i}_{t}_{tau}")

            for tau in Tau[t][1:]:
                for i in P:
                    m.addConstr(
                        gp.quicksum(bTR[v, i, j, t, tau] for j in P if j != i) <= bAT[v, i, t, tau - 1],
                        name=f"depart_{v}_{i}_{t}_{tau}"
                    )
                for i in P:
                    for j in P:
                        if i != j and (tau + 1) in Tau[t]:
                            m.addConstr(bTR[v, i, j, t, tau] <= bAT[v, j, t, tau + 1],
                                        name=f"arrive_{v}_{i}_{j}_{t}_{tau}"
                            )
            for tau in Tau[t][1:]:
                for i in P:
                    for j in P:
                        if i != j:
                            m.addConstr(
                                bAT[v, i, t, tau - 1] + bAT[v, j, t, tau] - bTR[v, i, j, t, tau - 1] <= 1,
                                name=f"moveConsistency_{v}_{i}_{j}_{t}_{tau}"
                            )

    # ------------------------------------------------------------------
    # Electricity constraints 
    # ------------------------------------------------------------------
    for i, t in it.product(P, T):
        for tau in Tau[t]:
            m.addConstr(qPR_EL[i, t, tau] <= pVar[i, EL, t],
                        name=f"ProdCapEL_{i}_{t}_{tau}")

            m.addConstr(
                qPR_EL[i, t, tau] + qGR[i, t, tau] + qDC[i, t, tau]
                == qEL[i, t, tau] + qCH[i, t, tau],
                name=f"ELbal_{i}_{t}_{tau}"
            )

            m.addConstr(eBat[i, t, tau] <= cST[i, t],
                        name=f"eCap_{i}_{t}_{tau}")

            m.addConstr(qGR[i, t, tau] <= grid_cap[i],
                        name=f"GridCap_{i}_{t}_{tau}")

            if tau == 0:
                m.addConstr(qDC[i, t, 0] <= eBat[i, t, 0],
                            name=f"dcLim0_{i}_{t}")
                m.addConstr(qCH[i, t, 0] <= cST[i, t] - eBat[i, t, 0],
                            name=f"chLim0_{i}_{t}")
            else:
                m.addConstr(qDC[i, t, tau] <= eBat[i, t, tau-1],
                            name=f"dcLim_{i}_{t}_{tau}")
                m.addConstr(qCH[i, t, tau] <= cST[i, t] - eBat[i, t, tau-1],
                            name=f"chLim_{i}_{t}_{tau}")

    for i, t in it.product(P, T):
        m.addConstr(eBat[i, t, 0] == 0, name=f"eInit_{i}_{t}")
        for tau in Tau[t][1:]:
            m.addConstr(
                eBat[i, t, tau] ==
                eBat[i, t, tau-1] + eta_CH * qCH[i, t, tau-1] - qDC[i, t, tau-1],
                name=f"eBal_{i}_{t}_{tau}"
            )

    for i in P:
        m.addConstr(cST[i, 0] == 0, name=f"cSTInit_{i}")
        try:
            opts_i = storage_opts_el.loc[i].index.tolist()
        except KeyError:
            opts_i = []
        for t in T[1:]:
            rhs = cST[i, t-1] + gp.quicksum(
                storage_opts_el.loc[(i, opt), "capacity"] * zST[i, opt, t-1]
                for opt in opts_i
            )
            m.addConstr(cST[i, t] == rhs, name=f"cSTaccum_{i}_{t}")

    for (i, t, tau), dval in demand_el.items():
        m.addConstr(qEL[i, t, tau] == dval, name=f"ELdemand_{i}_{t}_{tau}")

    # ------------------------------------------------------------------
    # Objective – Cost components
    # ------------------------------------------------------------------

    capex_localProd = gp.quicksum(
        deltaP[t] * K_LP_dict.get(f, 0.0) * zLP[i, f, t]
        for i in P for f in F + [EL] for t in T
    )

    cost_hold_shore = gp.quicksum(
        deltaO[t] * W_O[t] * K_HS[f] * s[i, f, t, tau]
        for i, f, t in it.product(P, F, T)
        for tau in Tau[t]
    )

    cost_hold_vessel = gp.quicksum(
        deltaO[t] * W_O[t] * K_HV_v[v] * a[v, t, tau]
        for v in V for t in T for tau in Tau[t]
    )

    cost_deliv_vessel = gp.quicksum(
        deltaO[t] * W_O[t] * K_VD[F_v[v]] * d[v, i, t, tau]
        for v in V for i in P for t in T for tau in Tau[t]
    )

    capex_storage = gp.quicksum(
        deltaP[idx[3]] * S_opts.loc[idx[:3], "capex"] * zS[idx]
        for idx in zS.keys()
    )

    if len(pairs) > 0:
        capex_pipeline = gp.quicksum(
            deltaP[t] * K_PL[i, j, f] * bPL[i, j, f, t]
            for (i, j, f) in pairs for t in T)

        opex_transfer = gp.quicksum(
            deltaO[t] * W_O[t] * K_T[i, j, f] * qT[i, j, f, t, tau]
            for (i, j, f) in pairs for t in T for tau in Tau[t])
    else:
        capex_pipeline = 0
        opex_transfer  = 0

    opex_delivery = gp.quicksum(
        deltaO[t] * W_O[t] * K_DE[f] * qDE[i, f, t, tau]     
        for i, f, t in it.product(P, F, T)
        for tau in Tau[t]
    )

    cost_unmet = gp.quicksum(
        deltaO[t] * W_O[t] * Pi[f] * u[i, f, t, tau]
        for i, f, t in it.product(P, F, T)
        for tau in Tau[t]
    )


    cost_move   = gp.quicksum(deltaO[t] * W_O[t] * K_TR_v[v] * bTR[v, i, j, t, tau]
                              for v in V for t in T for tau in Tau[t]
                              for i in P for j in P if i != j)

    opex_prod_nonEL = gp.quicksum(
        deltaO[t] * W_O[t] * K_PR[f] * qPR[i, f, t, tau]
        for i, f, t in it.product(P, F, T)
        for tau in Tau[t]
    )

    capex_eStorage = gp.quicksum(
        deltaP[t] * storage_opts_el.loc[(i, opt), "capex"] * zST[i, opt, t]
        for (i, opt, t) in zST.keys()
    )

    opex_prod_EL = gp.quicksum(
        deltaO[t] * W_O[t] * K_PR_EL * qPR_EL[i, t, tau]
        for i in P for t in T for tau in Tau[t]
    )

    if K_GR_time is not None:
        opex_grid = gp.quicksum(
            deltaO[t] * W_O[t] * K_GR_time[i, t, tau] * qGR[i, t, tau]
            for i in P for t in T for tau in Tau[t]
        )
    else:
        opex_grid = gp.quicksum(
            deltaO[t] * W_O[t] * (K_GR[i] if isinstance(K_GR, pd.Series) else K_GR) * qGR[i, t, tau]
            for i in P for t in T for tau in Tau[t]
        )

    m.setObjective(
        capex_storage + capex_pipeline +
        opex_delivery + opex_transfer + cost_move + cost_unmet
        + capex_eStorage + opex_prod_EL + opex_grid + opex_prod_nonEL
        + capex_localProd
        + cost_hold_shore + cost_hold_vessel + cost_deliv_vessel,
        GRB.MINIMIZE)

    m._vars = dict(
        zS=zS, cS=cS, qDE=qDE, q=q, qRF=qRF, s=s, u=u,
        qPR=qPR 
    )
    m._vars.update(dict(a=a, bAT=bAT, bDE=bDE, bRE=bRE, bTR=bTR, d=d, r=r))
    m._vars.update(dict(
        qPR_EL=qPR_EL, qGR=qGR, qCH=qCH, qDC=qDC,
        e=eBat, cST=cST, zST=zST, qEL=qEL
    ))
    m._vars.update(dict(p=pVar, zLP=zLP))
    m._sets = dict(P=P, F=F, T=T, Tau=Tau)
    m._deltaP = deltaP
    m._deltaO = deltaO

    if len(pairs) > 0:
        m._vars["bPL"] = bPL
        m._vars["bEX"] = bEX
        m._vars["qT"]  = qT

    for _k in ("d", "r", "a", "qPR"):
        m._vars[_k] = _wrap_var_dict(m._vars[_k])

    return m


# -----------------------------------------------------------------------------
# Helper to run standalone (quick sanity test with a minimal JSON)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import json, sys
    from model import build_phase4_model

    if len(sys.argv) != 2:
        print("Usage: python phase4_model.py data.json")
        sys.exit(1)

    data_path = Path(sys.argv[1])
    from data_loader import load_data  # noqa: local import
    data = load_data(data_path)
    model = build_phase4_model(data)
    model.write("phase4.lp")
    model.optimize()
    print("Objective:", model.objVal)
