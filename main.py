import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    df = pd.read_csv('data.csv')
    X = df.drop('disease', axis=1)
    y = df['disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, 'model.pkl')
    print("Model trained and saved.")

def load_model():
    return joblib.load('model.pkl')

def predict_disease(age, fever, cough, headache, model=None):
    if model is None:
        model = load_model()
    data = pd.DataFrame([[age, fever, cough, headache]],
                        columns=['age', 'fever', 'cough', 'headache'])
    return model.predict(data)[0]

if _name_ == "_main_":
    train_model()
