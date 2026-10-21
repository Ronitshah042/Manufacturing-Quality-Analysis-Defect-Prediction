import numpy as np

force_readings = np.array([12.5, 14.2, 13.8, 18.9, 12.1, 19.5])

print (force_readings.shape)
print(force_readings.dtype)

newtons = force_readings * 1000
print (newtons, "milliNewtond (mN)")

print("Mean force:", np.mean(force_readings))
print("Max force:", np.max(force_readings))
print("Standard Deviation:", np.std(force_readings))

over_threshold =force_readings[force_readings > 14.0]
print (force_readings, "over_threshold")
print("Count over threshold:", over_threshold.size)
