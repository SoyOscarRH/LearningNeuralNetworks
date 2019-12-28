import numpy as np


# if only 3 digits
h1s = np.array([0.848, 0.033,
                0.032, 0.0])
h2s = np.array([0.005, 0.132,
                0.134, 0.818])

# if all digits
h1s = np.array([0.8481288363433407, 0.032926394798136256,
                0.032295464698450495, 0.00020342697805520653])
h2s = np.array([0.005220125693558397, 0.13238887354206538,
                0.13354172253321245, 0.8175744761936437])

for i, hi in enumerate([h1s, h2s]):
  mean = np.mean(hi)
  std = np.std(hi)

  print(f"i={i+1} \t mean={round(mean, 3)} \t std={round(std, 3)}")
  print(np.around((hi - mean) / std, 3), end="\n\n")

  

