print("Starting script")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# load dataset
df = pd.read_csv("Advertising.csv")

# remove unwanted column
df = df.loc[:, ~df.columns.str.contains('Unnamed')]

print(df.head())

# features & target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = LinearRegression()
model.fit(X_train, y_train)

# prediction
preds = model.predict(X_test)

# evaluation
print("MAE:", mean_absolute_error(y_test, preds))
print("R2 Score:", r2_score(y_test, preds))

# sample prediction
sample = [[150, 25, 10]]
print("Predicted Sales:", model.predict(sample)[0])