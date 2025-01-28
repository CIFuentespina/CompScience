import numpy as np
import matplotlib.pyplot as plt


X = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([2, 3, 5, 6, 8, 10])


mean_x = sum(X) / len(X)
mean_y = sum(Y) / len(Y)

numerator = sum((X - mean_x) * (Y - mean_y))  
denominator = sum((X - mean_x) ** 2)        

m = numerator / denominator  
b = mean_y - m * mean_x      

print(f"Equation of Regression Line: Y = {m:.2f}X + {b:.2f}")
Y_pred = m * X + b
mse = sum((Y - Y_pred) ** 2) / len(Y)
print(f"Mean Squared Error (MSE): {mse:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(X, Y, color="red", label="Actual Data")  
plt.plot(X, Y_pred, color="blue", linestyle="--", label="Regression Line")  
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Manual Linear Regression")
plt.legend()
plt.grid(True)
plt.show()