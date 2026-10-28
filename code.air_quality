import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




# 1. DATA COLLECTION / LOADING


# Upload AirQuality.csv to Google Colab

from google.colab import files

files.upload()

df_raw = pd.read_csv("city_day.csv")

print("Dataset loaded successfully!")

print("\nRaw shape:", df_raw.shape)

print("\nFirst 5 rows:")
print(df_raw.head())


# 2. PREPROCESSING

df = df_raw.copy()


# 2.1 Convert Date into datetime
print("Convert Date into datetime")

if "Date" in df.columns:

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )


# 2.2 Convert pollution columns into numbers

pollution_columns = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3"
]

for column in pollution_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# 2.3 Handle missing values

print("\nMissing values before:")

print(df.isnull().sum())


if "City" in df.columns and "Date" in df.columns:

    df = df.dropna(
        subset=["City", "Date"]
    )


# Fill pollution missing values

for column in pollution_columns:

    if column in df.columns:

        df[column] = df[column].fillna(
            df[column].median()
        )


# Fill AQI missing values

if "AQI" in df.columns:

    df["AQI"] = pd.to_numeric(
        df["AQI"],
        errors="coerce"
    )

    df["AQI"] = df["AQI"].fillna(
        df["AQI"].median()
    )


# 2.4 Remove duplicates

print(
    "\nDuplicates before:",
    df.duplicated().sum()
)

df = df.drop_duplicates()

print(
    "Duplicates after:",
    df.duplicated().sum()
)


# 2.5 Remove impossible negative values

for column in pollution_columns:

    if column in df.columns:

        df = df[df[column] >= 0]


# 2.6 Feature Engineering

# Create Year column

if "Date" in df.columns:

    df["Year"] = df["Date"].dt.year


# Create Month column

if "Date" in df.columns:

    df["Month"] = df["Date"].dt.month


# Create Month Name column

if "Date" in df.columns:

    df["Month_Name"] = df["Date"].dt.month_name()


# Create Day column

if "Date" in df.columns:

    df["Day"] = df["Date"].dt.day


print("\nCleaned shape:", df.shape)

print("\nCleaned data:")

print(df.head())


# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n--- Basic Statistics ---")


available_columns = []

for column in pollution_columns:

    if column in df.columns:

        available_columns.append(column)


if len(available_columns) > 0:

    print(
        df[available_columns].describe()
    )


print("\nAverage Pollution Values:")

if len(available_columns) > 0:

    print(
        df[available_columns].mean().round(2)
    )


print("\nAverage AQI:")

if "AQI" in df.columns:

    print(
        df["AQI"].mean().round(2)
    )


print("\nAverage AQI by Month:")

if "Month" in df.columns and "AQI" in df.columns:

    print(
        df.groupby("Month")["AQI"]
        .mean()
        .round(2)
    )


# 4. VISUALISATIONS

plt.figure(figsize=(14, 10))


# **1. Monthly Average AQI


plt.subplot(2, 2, 1)

if "Month" in df.columns and "AQI" in df.columns:

    monthly_aqi = df.groupby(
        "Month"
    )["AQI"].mean()

    plt.plot(
        monthly_aqi.index,
        monthly_aqi.values,
        marker="o"
    )

plt.title("Monthly Average AQI")

plt.xlabel("Month")

plt.ylabel("Average AQI")

plt.xticks(
    range(1, 13)
)


# **2. Monthly Average PM2.5

plt.subplot(2, 2, 2)

if "Month" in df.columns and "PM2.5" in df.columns:

    monthly_pm25 = df.groupby(
        "Month"
    )["PM2.5"].mean()

    plt.bar(
        monthly_pm25.index,
        monthly_pm25.values
    )

plt.title("Monthly Average PM2.5")

plt.xlabel("Month")

plt.ylabel("Average PM2.5")

plt.xticks(
    range(1, 13)
)


# **3. Monthly Pollution Comparison

plt.subplot(2, 2, 3)

monthly_pollution = []

for column in [
    "PM2.5",
    "PM10",
    "NO2",
    "CO"
]:

    if column in df.columns:

        monthly_pollution.append(
            df.groupby("Month")[column].mean()
        )


if len(monthly_pollution) > 0:

    monthly_data = pd.concat(
        monthly_pollution,
        axis=1
    )

    monthly_data.columns = [
        column
        for column in [
            "PM2.5",
            "PM10",
            "NO2",
            "CO"
        ]
        if column in df.columns
    ]

    monthly_data.plot(
        marker="o",
        ax=plt.gca()
    )

plt.title("Monthly Pollution Trends")

plt.xlabel("Month")

plt.ylabel("Average Value")

plt.xticks(
    range(1, 13)
)

plt.legend()

# **4. Monthly AQI Category

plt.subplot(2, 2, 4)

if "Month" in df.columns and "AQI_Bucket" in df.columns:

    monthly_category = pd.crosstab(
        df["Month"],
        df["AQI_Bucket"]
    )

    monthly_category.plot(
        kind="bar",
        stacked=True,
        ax=plt.gca()
    )

plt.title("Monthly AQI Category Distribution")

plt.xlabel("Month")

plt.ylabel("Number of Records")

plt.xticks(
    rotation=0
)

plt.legend(
    title="AQI Category"
)


plt.tight_layout()

plt.show()


# 5. KEY INSIGHTS

print("\n===== KEY INSIGHTS =====")


print(
    f"Total records after cleaning : {len(df)}"
)


if "City" in df.columns:

    print(
        f"Number of cities             : "
        f"{df['City'].nunique()}"
    )


if "PM2.5" in df.columns:

    print(
        f"Average PM2.5                : "
        f"{df['PM2.5'].mean():.2f}"
    )


if "PM2.5" in df.columns:

    print(
        f"Highest PM2.5                : "
        f"{df['PM2.5'].max():.2f}"
    )


if "AQI" in df.columns:

    print(
        f"Average AQI                  : "
        f"{df['AQI'].mean():.2f}"
    )


if "AQI" in df.columns:

    print(
        f"Highest AQI                  : "
        f"{df['AQI'].max():.2f}"
    )


if "Month" in df.columns and "AQI" in df.columns:

    highest_aqi_month = (
        df.groupby("Month")["AQI"]
        .mean()
        .idxmax()
    )

    print(
        f"Month with highest average AQI : "
        f"{highest_aqi_month}"
    )


if "Month" in df.columns and "PM2.5" in df.columns:

    highest_pm25_month = (
        df.groupby("Month")["PM2.5"]
        .mean()
        .idxmax()
    )

    print(
        f"Month with highest average PM2.5 : "
        f"{highest_pm25_month}"
    )


# 6. SAVE CLEANED DATA

df.to_csv(
    "cleaned_air_quality_data.csv",
    index=False
)

print(
    "\nCleaned data saved as "
    "cleaned_air_quality_data.csv"
)
