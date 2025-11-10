import numpy as np
import pandas as pd
from scipy.optimize import minimize
import sys

# --- 1. Load and Inspect Data ---
try:
    # Load the dataset
    data = pd.read_csv('xy_data.csv')
    
    # Extract x and y data columns
    x_data = data['x'].values
    y_data = data['y'].values

    # Generate the corresponding 't' values, assuming uniform sampling
    # as per the problem description (6 < t < 60)
    t_values = np.linspace(6, 60, len(x_data))
    
    print(f"Successfully loaded {len(x_data)} data points.")

except FileNotFoundError:
    print("Error: 'xy_data.csv' not found.")
    sys.exit()
except KeyError:
    print("Error: CSV file does not contain 'x' and 'y' columns.")
    sys.exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    sys.exit()

# --- 2. Define Parametric Functions ---

def x_equation(t, theta_deg, M, X):
    """Calculates the x-component of the parametric equation."""
    # Convert degrees to radians for numpy's trig functions
    theta_rad = np.radians(theta_deg) 
    term1 = t * np.cos(theta_rad)
    term2 = np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta_rad)
    return term1 - term2 + X

def y_equation(t, theta_deg, M):
    """Calculates the y-component of the parametric equation."""
    # Convert degrees to radians for numpy's trig functions
    theta_rad = np.radians(theta_deg) 
    term1 = 42 + t * np.sin(theta_rad)
    term2 = np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta_rad)
    return term1 + term2

# --- 3. Define the L1 Error (Loss) Function ---

def calculate_l1_error(params):
    """
    Calculates the total L1 distance (Manhattan distance)
    between the data points and the predicted points.
    
    'params' is a list: [theta, M, X]
    """
    theta, M, X = params
    
    # 1. Calculate all predicted x and y values
    x_pred = x_equation(t_values, theta, M, X)
    y_pred = y_equation(t_values, theta, M)
    
    # 2. Calculate the L1 error (sum of absolute differences)
    error_x = np.sum(np.abs(x_pred - x_data))
    error_y = np.sum(np.abs(y_pred - y_data))
    
    # 3. Return the total error
    total_error = error_x + error_y
    
    return total_error

# --- 4. Run the Optimization ---

# Initial guess: [theta, M, X] (midpoints of the allowed ranges)
initial_guess = [25.0, 0.0, 50.0] 

# Bounds: [(min_theta, max_theta), (min_M, max_M), (min_X, max_X)]
# We use a small epsilon (1e-6) to be strictly *within* the bounds.
bounds = [(1e-6, 50.0 - 1e-6), (-0.05 + 1e-6, 0.05 - 1e-6), (1e-6, 100.0 - 1e-6)]

print("\nStarting optimization...")

# Run the minimizer using 'L-BFGS-B' which handles bounds.
result = minimize(
    calculate_l1_error,
    initial_guess,
    method='L-BFGS-B',
    bounds=bounds,
    options={'disp': False, 'ftol': 1e-10, 'gtol': 1e-7} # Set display=False for cleaner output
)

# --- 5. Print the Solution ---
print("\n--- Optimization Results ---")
if result.success:
    optimal_params = result.x
    final_theta = optimal_params[0]
    final_M = optimal_params[1]
    final_X = optimal_params[2]
    
    print(f"Optimal Theta (θ): {final_theta:.8f} degrees")
    print(f"Optimal M:         {final_M:.8f}")
    print(f"Optimal X:         {final_X:.8f}")
    print(f"Final L1 Error:    {result.fun:.8f}")
    
    # Also print the required radian value for theta for the submission
    final_theta_rad = np.radians(final_theta)
    print(f"Optimal Theta (rad): {final_theta_rad:.8f} radians")
    
else:
    print("\n--- Optimization Failed ---")
    print(f"Message: {result.message}")

print("----------------------------")