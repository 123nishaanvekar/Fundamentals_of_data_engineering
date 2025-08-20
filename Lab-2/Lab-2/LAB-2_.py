import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# -------------------------------
# Step 1: Load raw dataset
# -------------------------------
raw_path = "./raw_data/customer_purchases.csv"
df = pd.read_csv(raw_path)

# Sample expected columns: customer_id, purchase_amount, purchase_date

# -------------------------------
# Step 2: Feature Engineering
# -------------------------------
# Total purchase amount per customer
total_purchase = df.groupby("customer_id")["purchase_amount"].sum().rename("total_purchase")

# Purchase frequency
purchase_freq = df.groupby("customer_id")["purchase_date"].count().rename("purchase_frequency")

# Average transaction value
avg_transaction = (total_purchase / purchase_freq).rename("avg_transaction_value")

# Merge into a feature dataframe
features_df = pd.concat([total_purchase, purchase_freq, avg_transaction], axis=1)

# -------------------------------
# Step 3: Preprocessing
# -------------------------------
features_df = features_df.fillna(0)  # Handle missing values
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)

# -------------------------------
# Step 4: Clustering (K-Means)
# -------------------------------
kmeans = KMeans(n_clusters=2, random_state=42)  # Two groups: VIP & non-VIP
features_df["cluster"] = kmeans.fit_predict(scaled_features)

# -------------------------------
# Step 5: Assign VIP labels
# -------------------------------
# Assume the cluster with the highest average purchase is VIP
vip_cluster = features_df.groupby("cluster")["total_purchase"].mean().idxmax()
features_df["VIP_status"] = features_df["cluster"].apply(lambda x: "VIP" if x == vip_cluster else "Non-VIP")

# -------------------------------
# Step 6: Reverse ETL - Update original dataset
# -------------------------------
df = pd.merge(df, features_df["VIP_status"], on="customer_id", how="left")

# Save enriched dataset
os.makedirs("reverse_etl_output", exist_ok=True)
output_path = "reverse_etl_output/enriched_customer_data.csv"
df.to_csv(output_path, index=False)

print(f"Reverse ETL completed. Enriched data saved to {output_path}")
