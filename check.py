from AI_Vs_Human_Geoguess.utils.common import actual_coords,model_predictions
import pandas as pd
from keras.models import load_model
model=load_model(r"C:\Users\\bisht\Downloads\\best_model.keras")
print(model.summary())
    