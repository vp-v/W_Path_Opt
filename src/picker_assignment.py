# Picker assigment module using Random Forest Regressor
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 

# Function to train a Random Forest model for picker assignment
def train_model(training_data):
    X = training_data.drop("completion_time", axis=1)
    y = training_data["completion_time"]

    model=RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X,y)
    return model
# Function to assign the best picker based on predicted completion time
def assign_picker(model, order_features, picker_features):
    scores = {}
    for picker_id, features in picker_features.items():
        input_data = {**order_features, **features}
        df = pd.DataFrame([input_data])
        scores[picker_id] = model.predict(df)[0]   

        return min(scores, key=scores.get)
