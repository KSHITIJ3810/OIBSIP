import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# load dataset
df = pd.read_csv("car_data.csv")

# basic cleaning
df.dropna(inplace=True)

# convert categorical to numeric
df = pd.get_dummies(df, drop_first=True)

# features & target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = LinearRegression()
model.fit(X_train, y_train)

# prediction
preds = model.predict(X_test)

# evaluation
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("MAE:", mae)
print("R2 Score:", r2)

# sample prediction
sample = X_test.iloc[0:1]
prediction = model.predict(sample)
print("Sample Prediction:", prediction[0])