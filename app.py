import pickle
import numpy as np
# Load the model from the .pkl file
with open('RandomForest.pkl', 'rb') as f:
    model = pickle.load(f)
print(model.predict(np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])))
