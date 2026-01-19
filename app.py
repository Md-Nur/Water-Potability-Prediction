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
    gr.Slider(3.8, 10.3, step=0.1, label="pH"),
    gr.Slider(117, 277, step=1, label="Hardness"),
    gr.Slider(320, 44832, step=1, label="Solids"),
    gr.Slider(3.1, 11.1, step=0.1, label="Chloramines"),
    gr.Slider(267, 401, step=1, label="Sulfate"),
    gr.Slider(191, 656, step=1, label="Conductivity"),
    gr.Slider(5.5, 23.5, step=0.5, label="Organic_carbon"),
    gr.Slider(26.5, 107, step=0.5, label="Trihalomethanes"),
    gr.Slider(1.8, 6.1, step=0.1, label="Turbidity"),
]

app = gr.Interface(
    fn=predict_gpa,
    inputs=inputs,
    outputs="text", 
    title="Water Potability Predictor"
    )

app.launch()