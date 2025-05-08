import pandas as pd
from pycaret.classification import *

# Load training dataset
train_df = pd.read_csv('Training.csv')

# Drop unwanted column if it exists
if 'Unnamed: 133' in train_df.columns:
    train_df.drop('Unnamed: 133', axis=1, inplace=True)

# Set up classification environment
clf = setup(train_df, target='prognosis', session_id=123, verbose=False)

# Create and train model (use alternative if xgboost not available)
model = create_model('rf')  # Use 'rf' (Random Forest) as it's always available

# Finalize model (freeze pipeline)
final_model = finalize_model(model)

# Save model
save_model(final_model, 'xgboost_model')  # name doesn't have to match actual algorithm




# 'dataset' holds the input data for this script
# from pycaret.classification import *

# clf1 = setup(dataset, target='prognosis')

# xgboost = create_model('xgboost', verbose=False)
# final_xgboost = finalize_model(xgboost)
# save_model(final_xgboost, "C:/Users/kambl/Desktop/Aniket/Degree/Main/BE/Sem 2/BI/BI/Exp4/")