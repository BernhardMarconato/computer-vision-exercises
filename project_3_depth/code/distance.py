import pickle
import numpy as np

# VIDEO 1
# frame 665
l = np.array([668, 499, 1])
r = np.array([912, 499, 1])
filename = "params3"
# 263.59

# VIDEO 2
# frame 1029
#l = np.array([691, 439, 1])
#r = np.array([886, 420, 1])
#filename = "params2"
# 296.47

# VIDEO 3
# frame 867
#l = np.array([700, 410, 1])
#r = np.array([833, 410, 1])
#filename = "params3"
# 491.02

with open(filename, "rb") as file:
    ret, mtx, dist, _, _ = pickle.load(file)

print("Mtx inv")
Ki = np.linalg.inv(mtx)
print(Ki)

print("Kl")
Kl = np.matmul(Ki, l)
print(Kl)

print("Kr")
Kr = np.matmul(Ki, r)
print(Kr)

print("Angle")
alpha = np.arccos(np.dot(Kl, Kr) / (np.linalg.norm(Kl) * np.linalg.norm(Kr)))
print(alpha)

print("Distance")
dp_cm2 = 50 / (2 * np.tan(alpha / 2))
print(dp_cm2)
# 263.59

print("Error")
err = 250 / dp_cm2
print(1 - err)