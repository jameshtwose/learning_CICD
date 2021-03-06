"""
=============================
Plotting Jms Transformer
=============================

An example plot of :class:`jms_estimator.jms_estimator.JmsTransformer`
"""
import numpy as np
from matplotlib import pyplot as plt
from jms_estimator import JmsTransformer

X = np.arange(50, dtype=float).reshape(-1, 1)
X /= 50
estimator = JmsTransformer()
X_transformed = estimator.fit_transform(X)

plt.plot(X.flatten(), label='Original Data')
plt.plot(X_transformed.flatten(), label='Transformed Data')
plt.title('Plots of original and transformed data')

plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Index')
plt.ylabel('Value of Data')

plt.show()
