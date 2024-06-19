from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import re
import pandas as pd
from datetime import datetime, timedelta

app = FastAPI()

# Load the pre-trained model
with open("C:/Users/menna/app4/model4.pkl", "rb") as pickle_in:
    model = pickle.load(pickle_in)

class ConsumptionRequest(BaseModel):
    datetime: datetime
    weather: str

class PredictionOutput(BaseModel):
    datetime: datetime
    predicted_consumption: float

def extract_temp(x):
    matches = re.findall(r'\d+\.\d+', x)
    return float(matches[0]) if matches else None

def generate_future_dates(start_date, days):
    return [start_date + timedelta(days=i) for i in range(days)]

def prepare_input_data(start_date, weather):
    future_dates = generate_future_dates(start_date, 7)
    future_data = pd.DataFrame({'datetime': future_dates})
    future_data['datetime'] = pd.to_datetime(future_data['datetime'])
    future_data['hour'] = future_data['datetime'].dt.hour
    future_data['dayofweek'] = future_data['datetime'].dt.dayofweek
    future_data['weather'] = weather  # Assuming the weather is constant for simplicity
    future_data['weather'] = future_data['weather'].apply(extract_temp)
    future_data = future_data.dropna()  # Drop rows with missing weather data
    return future_data[['hour', 'dayofweek', 'weather']], future_dates

@app.get('/')
def index():
    return {'message': 'Hello, world'}

@app.get('/{name}')
def get_name(name: str):
    return {'welcome to my model': f'Hello, {name}'}

@app.post("/predict/", response_model=list[PredictionOutput])
async def predict_next_week_consumption(request: ConsumptionRequest):
    # Data preprocessing for prediction
    try:
        start_date = request.datetime
        weather = request.weather
        
        future_data, future_dates = prepare_input_data(start_date, weather)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {e}")

    if future_data.empty:
        raise HTTPException(status_code=400, detail="Missing or invalid weather data")

    # Prediction
    try:
        predicted_consumption = model.predict(future_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    results = [
        PredictionOutput(datetime=date, predicted_consumption=prediction)
        for date, prediction in zip(future_dates, predicted_consumption)
    ]

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
