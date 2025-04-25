from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

N_points = 40
N_neurons = 10
t = np.linspace(0, np.pi * 2, N_points)
x = t
y = np.sin(t)

som = MiniSom(1, N_neurons, 2, sigma=3, learning_rate=0.4, neighborhood_function='gaussian', random_seed=0)
points = np.array([x, y]).T
som.random_weights_init(points)

plt.figure(figsize=(10, 9))
total_iter = 0
for i, iterations in enumerate(range(5, 116, 10)): # note: increasing training periods
	som.train(points, iterations, verbose=False, random_order=False) # continued training
	total_iter += iterations
	plt.subplot(3, 4, i + 1)
	plt.scatter(x, y, color='red', s=10)
	# print(som.get_weights())
	plt.plot(som.get_weights()[0][:, 0], som.get_weights()[0][:, 1], 'green', marker='o')
	plt.title("Iterations: %d\nError: %.3f" % (total_iter, som.quantization_error(points)))
	plt.xticks([])
	plt.yticks([])
plt.tight_layout()
plt.show()