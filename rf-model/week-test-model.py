import pandas as pd
import joblib  # for loading your saved RF model


daily_df = pd.read_csv("weekly_test_weather.csv", parse_dates=["Date"])
daily_df.set_index("Date", inplace=True)


weekly_test = daily_df.resample("W").agg({
    "Rainfall": "sum",
    "TempMin": "mean",
    "TempMax": "mean",
    "TempAvg": "mean",
    "Humidity": "mean",
    "WindSpeed": "mean",
    "WindDir": "mean"
}).reset_index()

print("✅ Weekly test row prepared:")
print(weekly_test)


rf_model = joblib.load("rf_dengue_model.pkl")


trained_features = rf_model.feature_names_in_

# Add missing columns with 0 (for features like 'Cases' if present)
for col in trained_features:
    if col not in weekly_test.columns:
        weekly_test[col] = 0

# Ensure same column order as training
X_test = weekly_test[trained_features]


y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:,1]  

weekly_test["Predicted_Outbreak"] = y_pred
weekly_test["Probability"] = y_prob

print("\n✅ Prediction result:")
print(weekly_test)
