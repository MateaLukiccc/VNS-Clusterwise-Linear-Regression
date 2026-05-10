import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from alg import CLR_VND

# --- 1. LOAD LOCAL DATA ---
def load_local_car_data():
    data_path = 'auto-mpg.data'
    data = pd.read_csv(data_path, header=None, sep='\s+', na_values='?')
    data = data.dropna()
    X = pd.get_dummies(data.iloc[:, 1:-1], columns=[7]).values.astype(float)
    y = data[0].values.astype(float)
    
    # Scaling to [-1, 1] range
    X -= np.min(X, axis=0, keepdims=True)
    max_val = np.max(X, axis=0, keepdims=True)
    max_val[max_val == 0] = 1.0 
    X /= max_val / 2.0
    X -= 1.0
    
    return X, y

# --- 2. RUN ALGORITHM ---
X, y = load_local_car_data()

# Using K=2 and l_max=1 for a quick, efficient run
model = CLR_VND(K=2, l_max=1, strategy="first", random_state=42, alpha=0.7)
model.fit(X, y)

# --- 3. CALCULATE METRICS ---
predictions = model.predict(X)
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictions)

# --- 4. DISPLAY RESULTS ---
print("\n" + "="*35)
print("       CAR DATASET RESULTS")
print("="*35)
print(f"Total Samples:   {X.shape[0]}")
print(f"MSE:             {mse:.4f}")
print(f"R-Squared (R2):  {r2:.4f}")
print(f"Cluster Sizes:   {model.cluster_sizes()}")
print(f"Objective Error: {model.err_:.4f}")
print("="*35)

# K=2 l_max=1 alpha=1 strategy=first
# ===================================
# Total Samples:   392
# MSE:             9.0275
# R-Squared (R2):  0.8514
# Cluster Sizes:   [204 188]
# Objective Error: 3.8965
# ===================================


# K=2 l_max=1 alpha=0.7 strategy=first
# ===================================
#        CAR DATASET RESULTS
# ===================================
# Total Samples:   392
# MSE:             10.1370
# R-Squared (R2):  0.8332
# Cluster Sizes:   [176 216]
# Objective Error: 2.6037
# ===================================