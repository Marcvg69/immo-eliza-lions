# === Imports ===
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load the raw dataset
# This loads the CSV file you previously scraped or downloaded.
input_path = "data/raw/immoweb-dataset.csv"
df = pd.read_csv(input_path)

# DROP LOW-QUALITY COLUMNS
# These columns are removed because they fall into one or more of the following categories:
# - ❌ Too sparse: More than 40% of the values are missing, making them unreliable for modeling.
# - 📉 Low predictive value: The feature is either redundant, uninformative, or unlikely to influence price significantly.
# - 🔁 Duplicated or implied: Some features are already covered by other variables (e.g., 'roomCount' vs. 'bedroomCount').
# - 🧱 Metadata: Technical information or identifiers not useful for analysis or machine learning.

# This code calculates the percentage of missing values for each column.
# Columns with more than 40% or 80% missing values are considered unreliable for modeling.
# Step 1: Calculate the percentage of missing values for each column
missing_percentages = df.isnull().mean() * 100
# Step 2: Identify columns where more than 40% of the data is missing
cols_to_drop = missing_percentages[missing_percentages > 40].index.tolist()
# Step 3: Print a sorted list of columns with their % of missing data (only the ones to drop)
print("\n❌ Columns with more than 40% missing values (recommended to drop):")
print(missing_percentages[cols_to_drop].sort_values(ascending=False))
# Step 4: Drop these columns from the DataFrame
df.drop(columns=cols_to_drop, inplace=True)
# Step 5: Confirm and list what was dropped
print(f"\n🧹 Dropped {len(cols_to_drop)} columns with >40% missing values:")
for col in cols_to_drop:
    print(f" - {col} ({missing_percentages[col]:.1f}% missing)")

# Additional columns to drop
columns_to_drop = [
    "Unnamed: 0", # leftover index column from CSV export
    # 🔗 Metadata or URLs (not used in modeling)
    "url", "id",
    # 📏 Not critical for value estimation ????
    #    "facedeCount", # not sure yet????
    # 🌍 Geographic details – either redundant or too granular for our use case, leave in for now as it is numeric
    # "postCode",  
]
# Drop only if the column exists
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Clean up text fields and duplicates 
# Replace empty strings with NaN (standard missing value marker in pandas)
df.replace('', pd.NA, inplace=True)
# Strip leading/trailing whitespace from all text (object) columns ????
# Determine which columns where affected ????
for col in df.select_dtypes(include='object'):
    df[col] = df[col].astype(str).str.strip()

# Remove exact duplicate rows
df.drop_duplicates(inplace=True)

# Drop rows with missing price ===
# Price is our target variable — we can’t model or analyse without it.
df = df[df['price'].notna()].copy()

# Add REGION column
# Group provinces into their major Belgian regions (Flanders, Wallonia, Brussels)
belgium_regions = {
    'Antwerp': 'Flanders', 'Limburg': 'Flanders', 'East Flanders': 'Flanders',
    'Flemish Brabant': 'Flanders', 'West Flanders': 'Flanders',
    'Hainaut': 'Wallonia', 'Walloon Brabant': 'Wallonia', 'Namur': 'Wallonia',
    'Liège': 'Wallonia', 'Luxembourg': 'Wallonia',
    'Brussels': 'Brussels'
}
df['region'] = df['province'].map(belgium_regions)

# Convert categorical strings to numerical values
# These mappings convert qualitative labels to ordinal numerical values for modeling.

# Clean EPC scores like 'EPC_SCORE_A' into just 'A'
def extract_epc(value):
    return value.split('_')[-1] if isinstance(value, str) and '_' in value else value

# Define conversion tables
epc_mapping = {'A++': 9, 'A+': 8, 'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
condition_mapping = {
    'JUST_RENOVATED': 6, 'AS_NEW': 5, 'GOOD': 4,
    'TO_BE_DONE_UP': 3, 'TO_RENOVATE': 2, 'TO_RESTORE': 1
}
type_mapping = {'APARTMENT': 1, 'HOUSE': 0}

# Apply the conversions
df['epcScore'] = df['epcScore'].apply(extract_epc).map(epc_mapping)
df['buildingCondition'] = df['buildingCondition'].map(condition_mapping)
df['type'] = df['type'].map(type_mapping)

# Add price per m² metric
# Helps standardize property value comparisons across different sizes
df['price_per_m2'] = df['price'] / df['habitableSurface']

# REMOVE OUTLIERS
def remove_outliers_iqr(df, columns):
    df_filtered = df.copy()
    for col in columns:
        Q1 = df_filtered[col].quantile(0.25)
        Q3 = df_filtered[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Print stats
        print(f"\n📊 Outlier analysis for '{col}':")
        print(f"  Q1 (25th percentile): {Q1:,.2f}")
        print(f"  Q3 (75th percentile): {Q3:,.2f}")
        print(f"  IQR: {IQR:,.2f}")
        print(f"  Outlier threshold: < {lower:,.2f} or > {upper:,.2f}")

        original_len = len(df_filtered)
        outlier_count = df_filtered[(df_filtered[col] < lower) | (df_filtered[col] > upper)].shape[0]
        pct_removed = 100 * outlier_count / original_len
        print(f"  Outliers to remove: {outlier_count} ({pct_removed:.2f}%)")

        # PLOT before removal
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df_filtered[col])
        plt.title(f"Before: Outliers in {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

        # Remove outliers
        df_filtered = df_filtered[(df_filtered[col] >= lower) & (df_filtered[col] <= upper)]

        # PLOT after removal
        plt.figure(figsize=(10, 4))
        sns.boxplot(x=df_filtered[col])
        plt.title(f"After: Outliers removed from {col}")
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

    return df_filtered

# Columns to check for outliers
columns_to_filter = ['price', 'price_per_m2', 'habitableSurface', 'bedroomCount', 'bathroomCount']
# Apply outlier removal
df_no_outliers = remove_outliers_iqr(df, columns_to_filter)

# Save result
# Create output folder if not present
os.makedirs("data/cleaned", exist_ok=True)
output_path = "data/cleaned/immoweb-dataset_cleaned_mvg.csv"
df_no_outliers.to_csv(output_path, index=False)
print(f"✅ Cleaned dataset saved to: {output_path}")
print(f"Final shape: {df_no_outliers.shape}")

# Save mapping tables as separate CSVs for transparency and documentation
pd.DataFrame(list(epc_mapping.items()), columns=["EPC Label", "EPC Score"]).to_csv("data/cleaned/mapping_epcScore.csv", index=False)
pd.DataFrame(list(condition_mapping.items()), columns=["Building Condition", "Score"]).to_csv("data/cleaned/mapping_buildingCondition.csv", index=False)
pd.DataFrame(list(type_mapping.items()), columns=["Property Type", "Code"]).to_csv("data/cleaned/mapping_type.csv", index=False)
print("📄 Mapping tables saved: epcScore, buildingCondition, type.")

# End of code ======================================================================================


# Maybe later to prepare for ML
# Convert remaining categorical columns to numeric
# Many ML models only work with numbers, so we convert text labels using one-hot encoding
# This creates new columns for each category (e.g. province_Liège = 1 if applicable)
#cat_cols = df.select_dtypes(include='object').columns.tolist()
#if cat_cols:
#    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Drop any rows that still contain missing values
# This is a simple, safe strategy when building first models ?????????????? maybe later befor ML....
# df = df.dropna()
