import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# load data
df = pd.read_csv("Iris.csv")

if "Id" in df.columns:
    df.drop("Id", axis=1, inplace=True)

df = df.loc[:, ~df.columns.str.contains('Unnamed')]
df.columns = df.columns.str.strip()

# features & target
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df['Species']

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=model.classes_,
            yticklabels=model.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# sample prediction
sample = [[5.1, 3.5, 1.4, 0.2]]
print("Prediction:", model.predict(sample)[0])