import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# ✅ Fix Unicode issues in Windows terminal (optional but recommended)
try:
    sys.stdout.reconfigure(encoding='utf-8')
except:
    pass

# ✅ Load dataset
df = pd.read_csv("expenses_dataset.csv")

# ✅ Replace unsupported '₹' with 'Rs.' to prevent Unicode errors
df["text"] = df["text"].str.replace("₹", "Rs.", regex=False)

# ✅ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['category'], test_size=0.2, random_state=42, stratify=df['category']
)

# ✅ Text Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ✅ Train Model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test_vec)
print("Classification Report:\n", classification_report(y_test, y_pred))

# ✅ Test on Custom Inputs
test_samples = ["Swiggy 400", "Uber 250", "Amazon 2000", "Netflix 500", "Paid Rs.1500 to Airtel Recharge"]
test_samples = [t.replace("₹", "Rs.") for t in test_samples]  # Avoid Unicode issues
test_vec = vectorizer.transform(test_samples)
predictions = model.predict(test_vec)

print("\n--- Sample Predictions ---")
for text, pred in zip(test_samples, predictions):
    print(f"{text} -> {pred}")
