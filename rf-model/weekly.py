import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1. Load weekly dataset
# -------------------------------
df = pd.read_csv("weekly_prepared.csv", parse_dates=["Date"])

# -------------------------------
# Step 2. Weekly Dengue Cases + Outbreak
# -------------------------------
plt.figure(figsize=(14,5))
plt.plot(df["Date"], df["Cases"], marker='o', label="Weekly Dengue Cases")
plt.fill_between(df["Date"], 0, df["Cases"], where=df["Outbreak"]==1, color='red', alpha=0.2, label="Outbreak Weeks")
plt.xlabel("Week")
plt.ylabel("Dengue Cases")
plt.title("Weekly Dengue Cases & Outbreaks")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# Step 3. Weather vs Outbreak
# -------------------------------
plt.figure(figsize=(14,5))
plt.plot(df["Date"], df["Rainfall"], label="Rainfall (mm)")
plt.plot(df["Date"], df["TempAvg"], label="Average Temp (°C)")
plt.plot(df["Date"], df["Humidity"], label="Humidity (%)")
plt.scatter(df["Date"][df["Outbreak"]==1], df["Rainfall"][df["Outbreak"]==1], color='red', label="Outbreak Week Rainfall")
plt.xlabel("Week")
plt.ylabel("Weather Values")
plt.title("Weekly Weather & Outbreaks")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -------------------------------
# Step 4. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(10,6))
corr = df.drop(columns=["Date"]).corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Weekly Features")
plt.tight_layout()
plt.show()
