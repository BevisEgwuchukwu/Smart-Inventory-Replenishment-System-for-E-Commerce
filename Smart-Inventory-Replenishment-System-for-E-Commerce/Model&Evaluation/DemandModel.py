import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv(
    "/Users/macintosh/Desktop/Dissertation/Smart-Inventory-Replenishment-System-for-E-Commerce/EDA/Model_data"
)

# drop missing values
df.dropna(
    subset=["Daily Demand", "Warehouse Inventory", "Lead Time", "Safety Stock"],
    inplace=True,
)

# Convert dates to datetime
df["date"] = pd.to_datetime(df["date"])
df["Order Day"] = pd.to_datetime(df["Order Day"])
df["Shipment Day"] = pd.to_datetime(df["Shipment Day"])
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# Forecast daily demand
features_demand = [
    col
    for col in df.columns
    if col
    not in [
        "Daily Demand",
        "Stockout",
        "Restock Now",
        "Order ID",
        "Order Item ID",
        "Product Department",
        "Product Category",
        "Customer ID",
        "Order Time",
        "Order Day",
        "Shipment Day",
        "date",
        "Order Quantity",
        "Gross Sales",
        "Profit",
        "Profit Margin",
        "Inventory Cost",
        "Inventory Efficiency",
        "Estimated Ordering Cost",
        "EOQ",
        "Reorder Point",
        "Inventory Value",
        "Ordering Cost",
        "Holding Cost",
        "Holding Rate",
        "days of supply",
    ]
]
X_demand = df[features_demand]
y_demand = df["Daily Demand"]

X_train, X_test, y_train, y_test = train_test_split(
    X_demand, y_demand, test_size=0.2, random_state=42
)
model_demand = RandomForestRegressor(n_estimators=100, random_state=42)
model_demand.fit(X_train, y_train)
df["Predicted_Daily_Demand"] = model_demand.predict(X_demand)

# Evaluate the demand forecast model
plt.figure(figsize=(12, 6))
plt.plot(df["date"], df["Daily Demand"], label="Actual Daily Demand", color="blue")
plt.plot(
    df["date"],
    df["Predicted_Daily_Demand"],
    label="Predicted Daily Demand",
    color="orange",
)
plt.title("Daily Demand Forecast")
plt.xlabel("Date")
plt.ylabel("Daily Demand")
plt.legend()
plt.show()
mae_demand = mean_absolute_error(y_test, model_demand.predict(X_test))
rmse_demand = np.sqrt(mean_squared_error(y_test, model_demand.predict(X_test)))
print(f"Demand Forecast MAE: {mae_demand:.2f}, RMSE: {rmse_demand:.2f}")

# Predict stockout
features_stockout = features_demand + [
    "Predicted_Daily_Demand",
    "Warehouse Inventory",
    "Lead Time",
    "Safety Stock",
]
X_stockout = df[features_stockout]
y_stockout = df["Stockout"]

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_stockout, y_stockout, test_size=0.2, random_state=42
)
model_stockout = RandomForestClassifier(n_estimators=100, random_state=42)
model_stockout.fit(X_train_s, y_train_s)
df["Stockout_Prob"] = model_stockout.predict_proba(X_stockout)[:, 1]

# Evaluate the stockout prediction model
plt.figure(figsize=(12, 6))
sns.countplot(x="Stockout", data=df, palette="Set1")
plt.title("Stockout Distribution")
plt.xlabel("Stockout")
plt.ylabel("Count")
plt.show()
mae_stockout = mean_absolute_error(y_test, model_demand.predict(X_test))
rmse_stockout = np.sqrt(mean_squared_error(y_test, model_demand.predict(X_test)))
print(f"Demand Forecast MAE: {mae_stockout:.2f}, RMSE: {rmse_stockout:.2f}")

# Replenishment Rule
df["Calculated_Reorder_Point"] = (
    df["Predicted_Daily_Demand"] * df["Lead Time"] + df["Safety Stock"]
)
df["Restock_Recommendation"] = (
    df["Warehouse Inventory"] <= df["Calculated_Reorder_Point"]
)
df["Recommended_Order_Quantity"] = np.where(df["Restock_Recommendation"], df["EOQ"], 0)

# Visualize restock recommendations
sns.histplot(df["Recommended_Order_Quantity"], bins=50)
plt.title("Recommended Order Quantities Distribution")
plt.xlabel("Order Quantity")
plt.ylabel("Frequency")
plt.show()

# Export final dataset with recommendations
df.to_csv("inventory_replenishment_output.csv", index=False)
print("Output saved to inventory_replenishment_output.csv")
