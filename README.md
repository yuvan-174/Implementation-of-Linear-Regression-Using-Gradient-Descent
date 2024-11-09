# EX-03-Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize parameters theta_0 and theta_1 to zero, and set learning rate alpha and number of iterations.
2. Normalize feature X by centering and scaling.
3. Add a column of ones to X  for the intercept term.
4. Perform gradient descent for a specified number of iterations to update theta.
5. Use theta to plot the regression line along with the data points.

## Program:
```py
/*
Program to implement the linear regression using gradient descent.
Developed by: YUVAN SUNDAR S
RegisterNumber:  212223040250
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = '50_Startups.csv'
data = pd.read_csv(file_path)

X = data['R&D Spend'].values
y = data['Profit'].values

X_normalized = (X - X.mean()) / X.std()

X_normalized = np.c_[np.ones(X_normalized.shape[0]), X_normalized]

theta = np.zeros(X_normalized.shape[1])

alpha = 0.01 
iterations = 1000 

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (alpha / m) * X.T.dot(errors)
    return theta

theta = gradient_descent(X_normalized, y, theta, alpha, iterations)
print(f"Intercept (θ₀): {theta[0]}")
print(f"Coefficient for R&D Spend (θ₁): {theta[1]}")
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, X_normalized.dot(theta), color='red', label='Regression line')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.title('R&D Spend vs Profit')
plt.legend()
plt.show()
```


## Output:
![Screenshot 2024-08-31 163646](https://github.com/user-attachments/assets/d980d367-53a4-406d-a872-16c9c0438bd6)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
