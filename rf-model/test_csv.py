# weekly_preparation.py
import pandas as pd

# -------------------------------
# Step 1. Load daily dataset
# -------------------------------
df = pd.read_csv("weather_mock.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)

print("✅ Daily dataset loaded")
print(df.head())

# -------------------------------
# Step 2. Convert daily → weekly
# -------------------------------
weekly = df.resample("W").agg({
    "Rainfall": "sum",        # total rainfall in a week
    "TempMin": "mean",
    "TempMax": "mean",
    "TempAvg": "mean",
    "Humidity": "mean",
    "WindSpeed": "mean",
    "WindDir": "mean",
    "Cases": "sum"            # total dengue cases per week
}).reset_index()

print("\n✅ Weekly dataset created")
print(weekly.head())

# -------------------------------
# Step 3. Define outbreak rule
# -------------------------------
# Example: outbreak if Cases >= 100 in that week
weekly["Outbreak"] = (weekly["Cases"] >= 80).astype(int)

print("\n✅ Outbreak column added")
print(weekly[["Date", "Cases", "Outbreak"]].head(15))

# -------------------------------
# Step 4. Save to CSV
# -------------------------------
weekly.to_csv("weekly_prepared.csv", index=False)
print("\n💾 Weekly dataset saved as weekly_prepared.csv")
