"""Failures of the environment, not of the agent. They abort the instance and leave no scored result (see runner.Run.generate)."""


class InfraError(RuntimeError):
    """The environment broke (image pull, container start, disk, provider outage, failed reset verification).
    Scoring infrastructure failures as agent failures would corrupt the experiment."""
