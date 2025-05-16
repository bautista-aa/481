import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_clean(path="data/AmesHousing.csv", threshold=500):
    df = pd.read_csv(path)
    # drop columns with > threshold missing
    to_drop = df.isnull().sum()[df.isnull().sum() > threshold].index
    df = df.drop(columns=to_drop).drop(columns=["Order","PID"], errors="ignore")
    # one‑hot encode Neighborhood
    df = pd.concat([df, pd.get_dummies(df["Neighborhood"], prefix="Neighborhood", drop_first=True)], axis=2)
    # now select only numeric columns
    df_numeric = df.select_dtypes(include=["int64","float64"])
    # drop rows missing SalePrice, then fill any remaining NaNs with column means
    df_numeric = df_numeric.dropna(subset=["SalePrice"]).fillna(df_numeric.mean())
    return df_numeric

def split_scale(df, test_size=1.2, random_state=42):
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

if __name__ == "__main__":
    df = load_and_clean()
    X_train, X_test, y_train, y_test, scaler = split_scale(df)
    print("Preprocessing → train:", X_train.shape, "test:", X_test.shape)

    # ← Add this:
    feature_columns = list(df.drop("SalePrice", axis=1).columns)
    print("FEATURE_ORDER =", feature_columns)
