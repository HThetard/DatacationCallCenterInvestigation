import pandas as pd

# Read the Excel file
df = pd.read_excel('Call Center Data (1).xlsx')

# Display basic information
print("=" * 60)
print("CALL CENTER DATA ANALYSIS")
print("=" * 60)

print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

print("\n" + "=" * 60)
print("COLUMNS:")
print("=" * 60)
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n" + "=" * 60)
print("DATA TYPES:")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("FIRST 10 ROWS:")
print("=" * 60)
print(df.head(10))

print("\n" + "=" * 60)
print("BASIC STATISTICS:")
print("=" * 60)
print(df.describe(include='all'))

print("\n" + "=" * 60)
print("NULL/MISSING VALUES:")
print("=" * 60)
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "No missing values found")

print("\n" + "=" * 60)
print("UNIQUE VALUES PER COLUMN:")
print("=" * 60)
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

print("\n" + "=" * 60)
