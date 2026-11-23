"""Stand-in for booster_deploy.utils.isaaclab.configclass: a dataclass whose values are deep-copied at construction
(the vendor's _custom_post_init does that to every non-callable member) and that has replace()."""

import copy
import dataclasses


def configclass(cls):
    def __post_init__(self):
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if not callable(value):
                setattr(self, f.name, copy.deepcopy(value))

    cls.__post_init__ = __post_init__
    cls.replace = lambda self, **kw: dataclasses.replace(self, **kw)
    return dataclasses.dataclass(cls, kw_only=True)  # the vendor allows MISSING fields in any order
