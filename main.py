import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("/content/zomato.csv", encoding="latin-1")

df = df[df['Cuisines'].notnull()]
df = df[df['Aggregate rating'] > 0]

df['Primary Cuisine'] = df['Cuisines'].apply(lambda x: x.split(',')[0].strip())

top_cuisines = df['Primary Cuisine'].value_counts().head(15).index
df = df[df['Primary Cuisine'].isin(top_cuisines)]
df = df.reset_index(drop=True)
le = LabelEncoder()
df['Cuisine_Label'] = le.fit_transform(df['Primary Cuisine'])
tf = TfidfVectorizer(stop_words='english')
X = tf.fit_transform(df['Restaurant Name'])
y = df['Cuisine_Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LogisticRegression(max_iter=300)
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print("Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, pred_lr))
print("Precision:", precision_score(y_test, pred_lr, average='weighted'))
print("Recall:", recall_score(y_test, pred_lr, average='weighted'))
print("F1:", f1_score(y_test, pred_lr, average='weighted'))
print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, pred_rf))
print("Precision:", precision_score(y_test, pred_rf, average='weighted'))
print("Recall:", recall_score(y_test, pred_rf, average='weighted'))
print("F1:", f1_score(y_test, pred_rf, average='weighted'))
labels = np.unique(y_test)
print("\nClassification Report (Random Forest)")
print(classification_report(
    y_test,
    pred_rf,
    labels=labels,
    target_names=le.inverse_transform(labels),
    zero_division=0
))

results = pd.DataFrame({
    "Restaurant Name": df.loc[y_test.index, 'Restaurant Name'].values,
    "Actual Cuisine": le.inverse_transform(y_test),
    "Predicted Cuisine": le.inverse_transform(pred_rf)
})

print("\nSample Predictions:")
print(results.head(15))

cm = confusion_matrix(y_test, pred_rf)