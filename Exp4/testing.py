import pandas as pd
from pycaret.classification import *

# Load test dataset
test_df = pd.read_csv('Testing.csv')

# Drop unwanted column if it exists
if 'Unnamed: 133' in test_df.columns:
    test_df.drop('Unnamed: 133', axis=1, inplace=True)

# Load the trained model
model = load_model('xgboost_model')

# Align columns with training data (intersect only the needed columns)
test_df = test_df.loc[:, test_df.columns.isin(model.feature_names_in_)]

# Predict
predictions = predict_model(model, data=test_df)

# Save predictions to CSV
predictions.to_csv('predictions.csv', index=False)

print("Prediction completed. Results saved to predictions.csv")



# from pycaret.classification import *
# xgboost = load_model("C:/Users/kambl/Desktop/Aniket/Degree/Main/BE/Sem 2/BI/BI/Exp4/")
# dataset = predict_model(xgboost,data=dataset)