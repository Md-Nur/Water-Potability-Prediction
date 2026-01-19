import gradio as gr
import pandas as pd
import pickle
from is_the_water_drinkable import capping
import numpy as np

with open("water_model.pkl", "rb") as f:
    model = pickle.load(f)


def predict_gpa(ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity):
    input_df = pd.DataFrame([[
        ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity
    ]],
      columns=[
        'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
    ])
    
    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        return "Drinkable"
    else:
        return "Not Drinkable"

inputs = [
    gr.Slider(0, 14, step=0.1, label="pH"),
    gr.Number(0, 1000, step=1, label="Hardness"),
    gr.Number(0, 100000, step=1, label="Solids"),
    gr.Slider(0, 100, step=0.5, label="Chloramines"),
    gr.Number(0, 1000, step=1, label="Sulfate"),
    gr.Number(0, 1000, step=1, label="Conductivity"),
    gr.Slider(0, 4, step=0.1, label="Organic_carbon"),
    gr.Number(0, 1000, step=1, label="Trihalomethanes"),
    gr.Number(0, 1000, step=1, label="Turbidity"),
]

app = gr.Interface(
    fn=predict_gpa,
    inputs=inputs,
    outputs="text", 
    title="Water Potability Predictor"
    )

app.launch()