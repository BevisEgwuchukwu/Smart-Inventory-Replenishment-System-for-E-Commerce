import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

df_os = pd.read_csv("/Users/macintosh/Desktop/Dissertation/orders_and_shipments.csv")
df_inv = pd.read_csv("/Users/macintosh/Desktop/Dissertation/inventory.csv")

# merge both files
merged_df = pd.merge(df_os, df_inv, on="Product Name", how="inner")

# save to csv
merged_df.to_csv("merged.csv", index=False)

merged_df.columns = merged_df.columns.str.strip()
merged_df.info()

# Create a date column and drop individual year, month and day columns
merged_df["Order_Date"] = pd.to_datetime(
    merged_df[["Order Year", "Order Month", "Order Day"]].rename(
        columns={"Order Year": "year", "Order Month": "month", "Order Day": "day"}
    )
)

merged_df["Shipment_Date"] = pd.to_datetime(
    merged_df[["Shipment Year", "Shipment Month", "Shipment Day"]].rename(
        columns={
            "Shipment Year": "year",
            "Shipment Month": "month",
            "Shipment Day": "day",
        }
    )
)

merged_df.drop(
    [
        "Order Year",
        "Order Month",
        "Order Day",
        "Shipment Year",
        "Shipment Month",
        "Shipment Day",
    ],
    axis=1,
    inplace=True,
)

# Check for null values
merged_df.isnull().sum()

# Check for duplicates
merged_df.duplicated().any()
