from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

app = FastAPI(title="Bank Marketing API")

MODEL_PATH = Path("models/final_model.pkl")

model = joblib.load(MODEL_PATH)

FEATURES = [
    'age', 'education', 'campaign', 'pdays', 'previous', 'emp.var.rate',
    'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
    'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management',
    'job_retired', 'job_self-employed', 'job_services', 'job_student',
    'job_technician', 'job_unemployed', 'marital_married', 'marital_single',
    'default_yes', 'housing_yes', 'loan_yes', 'contact_telephone',
    'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar',
    'month_may', 'month_nov', 'month_oct', 'month_sep', 'day_of_week_mon',
    'day_of_week_thu', 'day_of_week_tue', 'day_of_week_wed',
    'poutcome_nonexistent', 'poutcome_success', 'contacted_before',
    'campaign_intensity', 'has_loan_or_housing'
]

class ClientInput(BaseModel):
    age: float
    education: float
    campaign: float
    pdays: float
    previous: float
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float
    job_blue_collar: int
    job_entrepreneur: int
    job_housemaid: int
    job_management: int
    job_retired: int
    job_self_employed: int
    job_services: int
    job_student: int
    job_technician: int
    job_unemployed: int
    marital_married: int
    marital_single: int
    default_yes: int
    housing_yes: int
    loan_yes: int
    contact_telephone: int
    month_aug: int
    month_dec: int
    month_jul: int
    month_jun: int
    month_mar: int
    month_may: int
    month_nov: int
    month_oct: int
    month_sep: int
    day_of_week_mon: int
    day_of_week_thu: int
    day_of_week_tue: int
    day_of_week_wed: int
    poutcome_nonexistent: int
    poutcome_success: int
    contacted_before: int
    campaign_intensity: float
    has_loan_or_housing: int

class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    label: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"model": "BaggingClassifier", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionOutput)
def predict(client: ClientInput):
    try:
        data = pd.DataFrame([{
            'age': client.age,
            'education': client.education,
            'campaign': client.campaign,
            'pdays': client.pdays,
            'previous': client.previous,
            'emp.var.rate': client.emp_var_rate,
            'cons.price.idx': client.cons_price_idx,
            'cons.conf.idx': client.cons_conf_idx,
            'euribor3m': client.euribor3m,
            'nr.employed': client.nr_employed,
            'job_blue-collar': client.job_blue_collar,
            'job_entrepreneur': client.job_entrepreneur,
            'job_housemaid': client.job_housemaid,
            'job_management': client.job_management,
            'job_retired': client.job_retired,
            'job_self-employed': client.job_self_employed,
            'job_services': client.job_services,
            'job_student': client.job_student,
            'job_technician': client.job_technician,
            'job_unemployed': client.job_unemployed,
            'marital_married': client.marital_married,
            'marital_single': client.marital_single,
            'default_yes': client.default_yes,
            'housing_yes': client.housing_yes,
            'loan_yes': client.loan_yes,
            'contact_telephone': client.contact_telephone,
            'month_aug': client.month_aug,
            'month_dec': client.month_dec,
            'month_jul': client.month_jul,
            'month_jun': client.month_jun,
            'month_mar': client.month_mar,
            'month_may': client.month_may,
            'month_nov': client.month_nov,
            'month_oct': client.month_oct,
            'month_sep': client.month_sep,
            'day_of_week_mon': client.day_of_week_mon,
            'day_of_week_thu': client.day_of_week_thu,
            'day_of_week_tue': client.day_of_week_tue,
            'day_of_week_wed': client.day_of_week_wed,
            'poutcome_nonexistent': client.poutcome_nonexistent,
            'poutcome_success': client.poutcome_success,
            'contacted_before': client.contacted_before,
            'campaign_intensity': client.campaign_intensity,
            'has_loan_or_housing': client.has_loan_or_housing,
        }])

        pred = int(model.predict(data)[0])
        prob = float(model.predict_proba(data)[0][1])
        label = "Suscribe" if pred == 1 else "No suscribe"

        return PredictionOutput(prediction=pred, probability=prob, label=label)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))