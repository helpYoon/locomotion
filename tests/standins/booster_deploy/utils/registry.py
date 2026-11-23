"""Stand-in for booster_deploy.utils.registry."""

_TASK_REGISTRY = {}


def register_task(name, cfg) -> None:
    if name in _TASK_REGISTRY:
        raise KeyError(f"Task '{name}' already registered")
    _TASK_REGISTRY[name] = cfg


def get_task(name):
    return _TASK_REGISTRY[name]


def list_tasks():
    return dict(_TASK_REGISTRY)
