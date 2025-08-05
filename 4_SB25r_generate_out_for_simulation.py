# # Setting up the environment
# Import necessary libraries
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import joblib  # For loading the saved model
import json

# Start tracking time for performance evaluation
start_time = time.time()

# # Load Pre-Trained Model and Selected Features
# The model and selected features are loaded from files in the `Models` folder.
dataset_name = 'SB25r'
input_file = os.path.join('Data_Preparation/InputModel', f'{dataset_name}_n0.2_process_filt.csv')
model_file = os.path.join('Models', f'{dataset_name}_n0.2_process_filt.joblib') 
features_file = os.path.join('Models', f'{dataset_name}_n0.2_process_filt.json')


# Load the input file as a DataFrame
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file {input_file} not found.")
df = pd.read_csv(input_file)

# Create interaction features
df["ndvi_soil_moisture"] = df["veg_mean_ndvi"] * df["era5_swi"]
df["precip_temp_ratio"] = df["era5_totprec"] / (df["era5_temp2m"] + 1)

df["ndvi_temp_interaction"] = df["veg_mean_ndvi"] * df["era5_temp2m"]
df["lai_precip_ratio"] = df["veg_mean_lai"] / (df["era5_totprec"] + 1)
############################################################################################################
# # Model Prediction
############################################################################################################


# edit the df by keeping  only field:  FIELDID,	Year,	week, the fields listed in features_file and    Crop_num

df1 = df[['FIELDID', 'Year', 'week'] + json.load(open(features_file))['features']]
print(df.head())

# add Crop_num to the df1 as the last column
df1['Crop_num'] = df['Crop_num']
print(df1.head())

# Save the df1 to a csv file OutputSimulation as input1.csv. Use dataset_name instead of 'All2' to make the code more general.
df1.to_csv(f'OutputSimulation/{dataset_name}_input.csv', index=False)




