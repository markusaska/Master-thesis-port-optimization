
import pandas as pd
TIMESTAMP_COL = "Datetime (Local)"    
PRICE_COL     = "Price (EUR/MWhe)"

df = (
    pd.read_csv("Norway.csv")               
      .rename(columns=lambda c: c.strip())  
)
df[TIMESTAMP_COL] = (
    pd.to_datetime(df[TIMESTAMP_COL], utc=True)
      .dt.tz_convert("Europe/Oslo")
)
df = df.set_index(TIMESTAMP_COL).sort_index()

price = df[PRICE_COL].astype(float)

price_2h = price.resample("2H").mean()   

price_2h = price_2h.to_frame("price")
price_2h["week_id"] = price_2h.index.to_period("W")  

weekly_avg = price_2h.groupby("week_id")["price"].mean()

mean_price = weekly_avg.mean()                       
typical_week = (weekly_avg - mean_price).abs().idxmin()

highpct = 0.90                                      
high_price_week = weekly_avg[weekly_avg >= weekly_avg.quantile(highpct)].idxmin()

print(f"Typical week  : {typical_week}  (avg ≈ {weekly_avg[typical_week]:.1f} €/MWh)")
print(f"High-price week: {high_price_week} (avg ≈ {weekly_avg[high_price_week]:.1f} €/MWh)")

def week_vector(week_id):
    week_df = price_2h.loc[price_2h["week_id"] == week_id, "price"].copy()
    week_df = week_df.reset_index(drop=True)         
    return week_df.iloc[:56]

typ_vec = week_vector(typical_week)
hi_vec  = week_vector(high_price_week)

assert len(typ_vec) == len(hi_vec) == 56

typ_vec.to_csv("grid_price_typical_week.csv", index=False, header=["€/MWh"])
hi_vec.to_csv("grid_price_high_week.csv",    index=False, header=["€/MWh"])

print("Files written: grid_price_typical_week.csv  |  grid_price_high_week.csv")