from main import load_model
import pandas as pd
from sklearn.metrics import classification_report

# Test data (mock)
df = pd.read_csv('data.csv')
X = df.drop('disease', axis=1)
y = df['disease']
model = load_model()
y_pred = model.predict(X)

print(classification_report(y, y_pred))
