"""
===========================
Plotting Jms Estimator
===========================

An example plot of :class:`jms_estimator.jms_estimator.JmsEstimator`
"""
import numpy as np
from matplotlib import pyplot as plt
from jms_estimator import JmsEstimator

X = np.arange(100).reshape(100, 1)
y = np.zeros((100, ))
estimator = JmsEstimator()
estimator.fit(X, y)
plt.plot(estimator.predict(X))
plt.show()
