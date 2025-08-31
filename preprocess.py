import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# =====================
# 1. Collecting the Dataset
# =====================
# (Dataset already downloaded from Kaggle and placed in project folder)

# =====================
# 2. Importing the Dataset
# =====================
df = pd.read_csv("water_potability.csv")
print("Initial shape:", df.shape)

# =====================
# 3. Preparing the Dataset
# =====================
# ---- Data Cleaning ----
# Fill missing values with column mean
df.fillna(df.mean(), inplace=True)
print("Missing values after filling:", df.isnull().sum().sum())

# Outlier handling
# Clip pH values to [0, 14]
df['ph'] = df['ph'].clip(0, 14)
# Remove unrealistic solids (> 100000 ppm)
df = df[df['Solids'] < 100000]
print("After outlier removal:", df.shape)


# ---- Feature Engineering ----
# Add categorical feature for pH
# Acidic (<6.5), Neutral (6.5-7.5), Alkaline (>7.5)
df['pH_category'] = pd.cut(
    df['ph'], bins=[0, 6.5, 7.5, 14], labels=['Acidic', 'Neutral', 'Alkaline']
)
# One-hot encode categorical feature
df = pd.get_dummies(df, columns=['pH_category'], drop_first=True)

# =====================
# 4. Splitting into Input (X) and Output (y)
# =====================
X = df.drop("Potability", axis=1)
y = df["Potability"]

# =====================
# 5. Feature Scaling
# =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# =====================
# 6. Handling Class Imbalance with SMOTE
# =====================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("Before SMOTE:", y.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_resampled).value_counts().to_dict())

# =====================
# 7. Splitting into Train and Test
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

# =====================
# 8. Exploratory Data Analysis (EDA)
# =====================
plt.figure(figsize=(12,8))
sns.heatmap(pd.DataFrame(X_resampled, columns=X.columns).corr(), annot=False, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# =====================
# 9. Save Preprocessed Data
# =====================
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
pd.Series(y_train).to_csv("y_train.csv", index=False)
pd.Series(y_test).to_csv("y_test.csv", index=False)

print("Preprocessing completed and files saved!")
