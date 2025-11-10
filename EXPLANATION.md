This problem was solved as a nonlinear curve-fitting task. The objective was to find the parameters $(\theta, M, X)$ that make the given parametric equations match the xy_data.csv data as closely as possible. The "closeness" was measured using the L1 distance (or Manhattan distance), which is the sum of the absolute differences between the predicted and actual data points.

Here are the steps followed:

Load Data: 
The xy_data.csv file was loaded using the pandas library. This gave us two arrays: x_data and y_data, each containing 1500 points.

Generate 't' Values: 
The problem states the data lies on the curve for $6 < t < 60$. We created a t_values array using numpy.linspace(6, 60, 1500) to map each $(x, y)$ data point to a corresponding value of the parameter $t$.

Define Parametric Functions: 
Two Python functions, x_equation(t, theta, M, X) and y_equation(t, theta, M), were created to represent the mathematical formulas given in the problem. A key step inside these functions was to convert $\theta$ from degrees (which we are optimizing) to radians (which Python's np.cos and np.sin require) using np.radians().

Define the L1 Error Function: 
A single "loss" function (calculate_l1_error) was created. This function takes a list of parameters [theta, M, X], calculates the predicted $x$ and $y$ values for all $t$ values, and returns the total L1 error: $\sum |x_{\text{pred}} - x_{\text{data}}| + \sum |y_{\text{pred}} - y_{\text{data}}|$.

Run Optimization: 
The scipy.optimize.minimize function was used to find the parameters $(\theta, M, X)$ that produce the smallest possible L1 error.

Method: We used 'L-BFGS-B', a powerful method that can handle parameter bounds.

Bounds: The ranges specified in the problem ($0 < \theta < 50$, $-0.05 < M < 0.05$, $0 < X < 100$) were given to the optimizer.

Initial Guess: We started the optimizer with a guess at the midpoint of the bounds: $[\theta=25, M=0, X=50]$.

Extract Results: 
The optimizer successfully converged and returned the optimal values for $\theta$, $M$, and $X$ listed above, which minimize the L1 erro
