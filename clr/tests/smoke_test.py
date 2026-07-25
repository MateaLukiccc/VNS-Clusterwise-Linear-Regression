import os
import sys
import numpy as np

# Go up 3 levels: tests/ -> clr/ -> VNS-Clusterwise-Linear-Regression/
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from clr.models import CLR_VND_Logistic, CLR_VND_Tree, CLR_VND_TreeIterative, CLR_Kipok
from clr.metrics import rmse

def generate_data():
    """Generates a very small, simple 2-cluster dataset."""
    rng = np.random.default_rng(42)
    X1 = rng.uniform(0, 5, (30, 2))
    y1 = 2 * X1[:, 0] - 1 * X1[:, 1] + rng.normal(0, 0.1, 30)
    X2 = rng.uniform(5, 10, (30, 2))
    y2 = -1 * X2[:, 0] + 3 * X2[:, 1] + rng.normal(0, 0.1, 30)
    X = np.vstack([X1, X2])
    y = np.concatenate([y1, y2])
    return X, y

def main():
    print("=== Running Smoke Test ===\n")
    X, y = generate_data()
    
    models_to_test = [
        ("CLR_VND_Logistic", CLR_VND_Logistic(K=2, l_max=1, strategy="first")),
        ("CLR_VND_Tree", CLR_VND_Tree(K=2, l_max=1, strategy="first")),
        ("CLR_VND_TreeIterative", CLR_VND_TreeIterative(K=2, l_max=1, strategy="first")),
        ("CLR_Kipok", CLR_Kipok(K=2, num_tries=2, max_iter=3)),
    ]
    
    all_passed = True
    
    for name, model in models_to_test:
        try:
            print(f"Testing {name}...")
            model.fit(X, y)
            preds = model.predict(X)
            
            # Basic sanity checks
            assert preds.shape == (60,), f"Predictions shape mismatch for {name}"
            err = rmse(y, preds)
            assert err < 5.0, f"RMSE too high for {name}: {err}"
            assert len(model.cluster_sizes()) == 2, f"Cluster sizes wrong for {name}"
            
            print(f"  -> Success! RMSE: {err:.4f}, Cluster Sizes: {model.cluster_sizes()}\n")
            
        except Exception as e:
            print(f"  -> FAILED! Error: {e}\n")
            all_passed = False
            
    if all_passed:
        print("=== SMOKE TEST PASSED === All modules are wired up correctly.")
    else:
        print("=== SMOKE TEST FAILED === Check errors above.")

if __name__ == "__main__":
    main()