import pandas as pd


def load_price_vector(path: str) -> pd.Series:
    """Read a CSV, keep only numeric cells, return a 56‑element vector."""
    raw = pd.read_csv(path, header=None).stack()         
    vec = pd.to_numeric(raw, errors="coerce").dropna()   
    vec = vec.reset_index(drop=True)
    if len(vec) != 56:
        raise ValueError(
            f"{path} must contain exactly 56 numeric entries, found {len(vec)}"
        )
    return vec

typical = load_price_vector("grid_price_typical_week.csv")
high    = load_price_vector("grid_price_high_week.csv")

week_vector = pd.concat([typical, high], ignore_index=True)   

ESCALATORS = {
    0: 0.00,   # 2025‑30  (all prices set to zero)
    1: 1.00,   # 2030‑35
    2: 1.10,   # 2035‑40
    3: 1.20,   # 2040‑45
    4: 1.30,   # 2045‑50
}

records = []
for t, factor in ESCALATORS.items():
    df_t = pd.DataFrame({
        "planning_period": t,
        "tau": range(112),
        "price_EUR_per_MWh": week_vector * factor,
    })
    records.append(df_t)

all_prices = pd.concat(records, ignore_index=True)

all_prices.to_csv("grid_price_4periods.csv", index=False)
all_prices.to_excel("grid_price_4periods.xlsx", index=False)

print("Wrote grid_price_4periods.csv and grid_price_4periods.xlsx")
