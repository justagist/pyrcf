"""Defines state estimators that can be used in the control loop.
Currently, because we operate only in simulation, we directly use
data from the simulator and do not use any custom state estimation
algorithms."""

from .state_estimator_base import StateEstimatorBase, DummyStateEstimator
