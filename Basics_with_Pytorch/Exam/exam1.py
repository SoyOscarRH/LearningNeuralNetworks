import numpy as np

roundList = lambda l: [round(elem, 3) for elem in l]

x1 = np.array([-1, -2, 1, 1.5])
x2 = np.array([203, 405, 126, 228])
x3 = np.array([0.01, 0.22, 1.23, -0.2])
x4 = np.array([-400, 328, -52, -25])

for i, xi in enumerate([x1, x2, x3, x4]):
  mean = np.mean(xi)
  std = np.std(xi)

  print(f"i={i+1} \t mean={round(mean, 3)} \t std={round(std, 3)}")
  print(xi)
  print(roundList((xi - mean) / std), end="\n\n")