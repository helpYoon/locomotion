"""robot.spec.get_robot: the T1 and the T2 resolve; every other name fails at once (Review Focus 2)."""

import pytest
from robot.spec import get_robot

@pytest.mark.parametrize("name", ["t1", "t2"])
def test_robots_resolve(name):
    assert get_robot(name).name == name

@pytest.mark.parametrize("name", ["mjlab_g1", "g1", "spec", "kinematics"])
def test_other_robots_fail_fast(name):
    with pytest.raises(ValueError, match="unknown robot"):
        get_robot(name)
