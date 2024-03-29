import matplotlib.pyplot as plt
import numpy as np

size = 9  # Define the size of the square side
x_values = np.linspace(-1, 1, size)
y_values = np.linspace(-1, 1, size)

X, Y = np.meshgrid(x_values, y_values)

# Plotting
plt.figure(figsize=(8, 8))
plt.scatter(X, Y, color='blue')  # Just plotting the points in blue
plt.title('Matrix of Points in the Shape of a Square')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid(True)
plt.show()

