import pytest

from sklearn.utils.estimator_checks import check_estimator

from jms_estimator import JmsEstimator
from jms_estimator import JmsClassifier
from jms_estimator import JmsTransformer


@pytest.mark.parametrize(
    "estimator",
    [JmsEstimator(), JmsTransformer(), JmsClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
