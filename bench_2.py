import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from alg import CLR_VND
from sklearn.linear_model import LinearRegression

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

X, y = load_local_car_data()
lin_reg = LinearRegression().fit(X, y)
y_pred_lin = lin_reg.predict(X)

rmse_1 = np.sqrt(mean_squared_error(y, y_pred_lin))
r2_1 = r2_score(y, y_pred_lin)

results = []
results.append(['K=1 (MLR)', rmse_1, r2_1])

for k in [2, 3, 4]:
    print(f"Treniram model za K={k}...")
    model = CLR_VND(K=k, l_max=1, strategy="first", random_state=42)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    rmse_k = np.sqrt(mean_squared_error(y, y_pred))
    r2_k = r2_score(y, y_pred)
    
    results.append([f'K={k} (CLR)', rmse_k, r2_k])

print("\n{:<15} | {:<10} | {:<10}".format("Model", "RMSE", "R2 Score"))
print("-" * 40)
for res in results:
    print("{:<15} | {:<10.4f} | {:<10.4f}".format(res[0], res[1], res[2]))


# Model           | RMSE       | R2 Score  
# ----------------------------------------
# K=1 (MLR)       | 3.2684     | 0.8242    
# K=2 (CLR)       | 3.0046     | 0.8514    
# K=3 (CLR)       | 2.8439     | 0.8669    
# K=4 (CLR)       | 2.6659     | 0.8830   