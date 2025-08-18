from __future__ import annotations
import pathlib as _pl
import matplotlib.pyplot as _plt
import pandas as _pd


def _save(fig, path: _pl.Path):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=150)
    _plt.close(fig)


def _capex_opex(cs, out_dir, tag):
    if cs is not None and not cs.empty:
        fig, ax = _plt.subplots()
        ax.bar(cs["category"], cs["value"])
        ax.set_ylabel("€ (NPV)")
        ax.set_title("Total CAPEX / OPEX / Penalties")
        _save(fig, out_dir / f"capex_opex_{tag}.png")


def _cost_components(cs, ns, out_dir, tag):
    if (
        cs is not None
        and ns is not None
        and (not cs.empty)
        and (not ns.empty)
    ):
        capex_val = cs.loc[cs["category"].str.lower() == "capex", "value"].iloc[0]
        opex_val = cs.loc[cs["category"].str.lower() == "opex", "value"].iloc[0]
        obj_no_pen = ns.loc[
            ns["metric"].str.lower().str.contains("w/o penalties"), "value"
        ].iloc[0]
        fig, ax = _plt.subplots()
        ax.bar(
            ["CAPEX", "OPEX", "CAPEX+OPEX"],
            [capex_val, opex_val, capex_val + opex_val],
        )
        ax.set_ylabel("€ (NPV)")
        ax.set_title("CAPEX / OPEX / CAPEX+OPEX")
        _save(fig, out_dir / f"cost_components_{tag}.png")


def _delivery_mix(sheets, out_dir, tag):
    for name, df in sheets.items():
        if name.endswith("_balance") and not name.startswith("el_") and not df.empty:
            p0 = df["port"].iloc[0]
            fuel = name.replace("_balance", "")
            d = df[df["port"] == p0].sort_values(["t", "tau"])
            if d.empty:
                continue
            x = range(len(d))
            fig, ax = _plt.subplots()
            ax.stackplot(
                x,
                d["vessel"],
                d["from_storage"],
                d["local_prod"],
                d["pipeline"],
                labels=["vessel-delivery", "storage", "local prod", "pipeline"],
            )
            ax.set_title(f"Delivery mix – {fuel} @ {p0}")
            ax.set_xlabel("(t, τ) sequence")
            ax.set_ylabel("volume")
            ax.legend()
            _save(fig, out_dir / f"delivery_mix_{fuel}_{p0}_{tag}.png")


def _storage_util(su, out_dir, tag):
    if su is not None and not su.empty:
        for p in su["port"].unique():
            d = su[su["port"] == p]
            if d.empty:
                continue
            piv = d.pivot_table(index="t", columns="tau", values="util_pct")
            fig, ax = _plt.subplots()
            im = ax.imshow(piv.values, aspect="auto")
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels(piv.index)
            xticks_all = list(piv.columns)
            step = max(1, len(xticks_all) // 6)
            xtick_sel = xticks_all[::step]
            ax.set_xticks([piv.columns.get_loc(c) for c in xtick_sel])
            ax.set_xticklabels(xtick_sel)
            ax.set_title(f"Storage utilisation – {p}")
            ax.set_xlabel("τ")
            ax.set_ylabel("t")
            fig.colorbar(im, ax=ax, label="util %")
            _save(fig, out_dir / f"storage_util_{p}_{tag}.png")


def _battery_soc(bl, out_dir, tag):
    if bl is not None and not bl.empty:
        p0 = bl["port"].iloc[0]
        d = bl[bl["port"] == p0].sort_values(["t", "tau"])
        fig, ax = _plt.subplots()
        ax.plot(d["value"].reset_index(drop=True))
        ax.set_title(f"Battery SoC – {p0}")
        ax.set_ylabel("MWh")
        _save(fig, out_dir / f"battery_soc_{p0}_{tag}.png")


def _unmet_demand(ud, out_dir, tag):
    if ud is not None and not ud.empty:
        agg = ud.groupby("fuel")["unmet"].sum()
        fig, ax = _plt.subplots()
        ax.bar(agg.index, agg.values)
        ax.set_title("Unmet demand by fuel")
        ax.set_ylabel("volume")
        _save(fig, out_dir / f"unmet_by_fuel_{tag}.png")


def _vessel_moves(mv, out_dir, tag):
    required_cols = {"vessel", "i_from", "j_to", "t", "tau"}
    if mv is None or mv.empty:
        return
    if not required_cols.issubset(mv.columns):
        missing = required_cols - set(mv.columns)
        raise KeyError(f"Missing columns in vessel_moves sheet: {missing}")
    ports_sorted = sorted({*mv["i_from"].unique(), *mv["j_to"].unique()})
    port_map = {p: idx for idx, p in enumerate(ports_sorted)}

    for v_id, v_df in mv.groupby("vessel"):
        v_df = v_df.sort_values(["t", "tau"])
        x = range(len(v_df))
        y = [port_map[p] for p in v_df["j_to"]]
        fig, ax = _plt.subplots()
        ax.plot(x, y, marker="o")
        ax.set_title(f"Vessel trajectory – {v_id}")
        ax.set_xlabel("(t, τ) sequence")
        ax.set_ylabel("port")
        ax.set_yticks(list(port_map.values()))
        ax.set_yticklabels(list(port_map.keys()))
        _save(fig, out_dir / f"vessel_route_{v_id}_{tag}.png")


def _levelised_cost(ns, out_dir, tag):
    if ns is not None and not ns.empty:
        lc = ns.loc[ns["metric"] == "Levelised cost", "value"].iloc[0]
        fig, ax = _plt.subplots()
        ax.bar(["co-operative run"], [lc])
        ax.set_title("Levelised cost of fuel delivered")
        ax.set_ylabel("€/unit")
        _save(fig, out_dir / f"levelised_cost_{tag}.png")


def _grid_imports(gi, out_dir, tag):
    if gi is not None and not gi.empty:
        gdf = gi.copy()
        gdf.columns = gdf.columns.str.lower()
        gdf.rename(columns={"qgr": "q_gr"}, inplace=True)
        for p, sub in gdf.groupby("port"):
            sub = sub.sort_values(["t", "tau"])
            y = sub["q_gr"].reset_index(drop=True)
            fig, ax = _plt.subplots()
            ax.plot(range(len(sub)), y)
            ax.set_title(f"Grid imports – {p}")
            ax.set_ylabel("Grid import (MWh)")
            ax.set_xlabel("(t, τ) sequence")
            _save(fig, out_dir / f"grid_imports_{p}_{tag}.png")


def _capacity_built(cap_built, out_dir, tag):
    """
    Line plot of cumulative storage capacity per fuel at the end of each
    planning period t, one figure per port.
    """
    if cap_built is None or cap_built.empty:
        return

    df = cap_built.copy()
    df.columns = df.columns.str.lower()
    if "value" in df.columns and "capacity" not in df.columns:
        df = df.rename(columns={"value": "capacity"})

    required_cols = {"port", "fuel", "t", "capacity"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise KeyError(f"capacity_built sheet missing required cols: {missing}")

    df["t"] = df["t"].astype(int)

    for p, pdf in df.groupby("port"):
        piv = pdf.pivot_table(index="t", columns="fuel", values="capacity", aggfunc="last")

        piv = piv.sort_index()

        fig, ax = _plt.subplots()
        for fuel in piv.columns:
            ax.plot(piv.index, piv[fuel], marker="o", label=fuel)

        ax.set_title(f"Cumulative storage capacity – {p}")
        ax.set_xlabel("planning period t")
        ax.set_ylabel("capacity (units)")
        ax.set_xticks(piv.index)  
        ax.legend(title="fuel", loc="best")
        _save(fig, out_dir / f"capacity_added_{p}_{tag}.png")


def generate_plots(
    sheets: dict[str, _pd.DataFrame], out_dir: str | _pl.Path, tag: str = "run"
) -> None:
    out_dir = _pl.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _safe(plot_name, fn):
        try:
            fn()
        except Exception as exc:
            print(f"[plots] {plot_name} skipped: {exc}")

    cs = sheets.get("cost_summary")
    ns = sheets.get("npv_summary")
    su = sheets.get("storage_util")
    bl = sheets.get("battery_level")
    ud = sheets.get("unmet_demand")
    mv = sheets.get("vessel_moves")
    gi = sheets.get("grid_imports")
    cap_built = sheets.get("capacity_built")

    _safe("capex/opex summary", lambda: _capex_opex(cs, out_dir, tag))
    _safe("cost components", lambda: _cost_components(cs, ns, out_dir, tag))
    _safe("delivery mix", lambda: _delivery_mix(sheets, out_dir, tag))
    _safe("storage utilisation", lambda: _storage_util(su, out_dir, tag))
    _safe("battery soc", lambda: _battery_soc(bl, out_dir, tag))
    _safe("unmet demand", lambda: _unmet_demand(ud, out_dir, tag))
    _safe("vessel moves", lambda: _vessel_moves(mv, out_dir, tag))
    _safe("levelised cost", lambda: _levelised_cost(ns, out_dir, tag))
    _safe("grid imports", lambda: _grid_imports(gi, out_dir, tag))
    _safe("capacity built", lambda: _capacity_built(cap_built, out_dir, tag))
