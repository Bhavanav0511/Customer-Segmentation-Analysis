import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load Dataset
df = pd.read_csv("../Dataset/store_customers.csv")

# Display Dataset
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Data Visualization

plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=20)
plt.title("Age Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Annual Income (k$)"], bins=20)
plt.title("Annual Income Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(df["Spending Score (1-100)"], bins=20)
plt.title("Spending Score Distribution")
plt.show()

# Scatter Plot

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)"
)
plt.title("Income vs Spending")
plt.show()

# Elbow Method

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# K-Means

kmeans = KMeans(n_clusters=5, random_state=42)

df["Cluster"] = kmeans.fit_predict(X)

# Final Visualization

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    s=80
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    c="black",
    marker="X",
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means")

plt.legend()

plt.show()

print(df.head())

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Fill missing values
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Annual Income (k$)"] = df["Annual Income (k$)"].fillna(df["Annual Income (k$)"].mean())
df["Spending Score (1-100)"] = df["Spending Score (1-100)"].fillna(df["Spending Score (1-100)"].mean())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Fill missing values
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Annual Income (k$)"] = df["Annual Income (k$)"].fillna(df["Annual Income (k$)"].mean())
df["Spending Score (1-100)"] = df["Spending Score (1-100)"].fillna(df["Spending Score (1-100)"].mean())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Cleaning missing values
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Annual Income (k$)"] = df["Annual Income (k$)"].fillna(df["Annual Income (k$)"].mean())
df["Spending Score (1-100)"] = df["Spending Score (1-100)"].fillna(df["Spending Score (1-100)"].mean())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=20)
plt.title("Age Distribution")
plt.savefig("../Graphs/age_distribution.png")
plt.savefig("../Graphs/income_distribution.png")
plt.show()

# Income Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Annual Income (k$)"], bins=20)
plt.title("Annual Income Distribution")
plt.savefig("../Graphs/income_distribution.png")
plt.show()

# Spending Score Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["Spending Score (1-100)"], bins=20)
plt.title("Spending Score Distribution")
plt.savefig("../Graphs/spending_distribution.png")
plt.show()

# Save the clustered dataset
df.to_csv("../Dataset/customer_segments.csv", index=False)

print("\nClustered dataset saved successfully!")

print("\nCustomers in Each Cluster:")
print(df["Cluster"].value_counts())

print("\nAverage Values for Each Cluster:")
print(df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)", "Age"]].mean())