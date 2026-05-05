import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# clean columns (extra columns hata rahe)
df = df.iloc[:, :2]
df.columns = ['label', 'message']

# convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# features & target
X = df['message']
y = df['label']

# text → numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = MultinomialNB()
model.fit(X_train, y_train)

# prediction
y_pred = model.predict(X_test)

# results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

# sample test
sample = ["Win money now!!!"]
sample_vec = vectorizer.transform(sample)

print("\nSample Prediction:",
      "Spam" if model.predict(sample_vec)[0] == 1 else "Not Spam")