import pandas as pd
from pycaret.clustering import *

# Load your dataset
df = pd.read_csv('clustering.csv')

# Save original data for merging later
original_df = df.copy()

# Initialize PyCaret (this creates a transformed internal dataset)
setup(data=df, ignore_features=['Country'], session_id=123)

# Create clustering model
model = create_model('kmeans', num_clusters=5)

# Assign cluster labels to the dataset
clustered = assign_model(model)

# Add the cluster column to the original dataframe
original_df['Cluster'] = clustered['Cluster']

# Save to CSV
original_df.to_csv('clustered_output.csv', index=False)
