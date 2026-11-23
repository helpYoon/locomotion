"""A plain `bash utils/t1_run.sh <run>` (STRETCH unset, so 2) trains src/train.py's default schedule, the deployed
controller's recipe; STRETCH=1 is BlinkMimic 2's own schedule.

Read from the sources without running either: train.py starts training on import, the driver starts it when run."""

import ast
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KEYS = ("--iterations", "--ckpt_every", "--phase")


def train_flags(keys: tuple[str, ...]) -> dict:
    """{flag: {"default": ..., "action": ...}} of the flags ``keys`` in train.py, the keywords they give, a module-level
    name resolved."""
    tree = ast.parse((REPO / "src" / "train.py").read_text())
    names = {t.id: n.value for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value in keys):
            out[node.args[0].value] = {k.arg: ast.literal_eval(names[k.value.id] if isinstance(k.value, ast.Name)
                                                               else k.value)
                                       for k in node.keywords if k.arg in ("default", "action")}  # fmt: skip
    return out


def train_defaults() -> dict:
    """The default= of --iterations, --ckpt_every and --phase in train.py, a module-level name resolved."""
    return {k: str(v["default"]) for k, v in train_flags(KEYS).items()}


def driver_schedule(stretch: int | None) -> dict:
    """The driver's SCHED flags (STRETCH None: unset, the driver's default); only its X, x() and SCHED lines run."""
    lines = [l for l in (REPO / "utils" / "t1_run.sh").read_text().splitlines() if l.startswith(("X=", "x()", "SCHED="))]
    assert len(lines) == 3, lines
    env = {k: v for k, v in os.environ.items() if k != "STRETCH"} | ({} if stretch is None else {"STRETCH": str(stretch)})
    script = "\n".join(lines) + '\nprintf "%s\\n" "${SCHED[@]}"'
    sched = subprocess.run(["bash", "-c", script], env=env, check=True, capture_output=True, text=True).stdout.splitlines()
    return dict(zip(sched[::2], sched[1::2]))


def test_plain_driver_is_train_defaults():
    assert train_defaults() == driver_schedule(None) == driver_schedule(2)


def test_landing_command_is_off_by_default():
    """The deployed recipe trains no landing command: --land_frac 0, no --init_from, --freeze_base off."""
    flags = train_flags(("--land_frac", "--init_from", "--freeze_base"))
    assert flags == {"--land_frac": {"default": 0.0}, "--init_from": {"default": ""},
                     "--freeze_base": {"action": "store_true"}}  # fmt: skip


def test_stretch_1_is_blinkmimic2():
    assert driver_schedule(1) == {
        "--iterations": "6500",
        "--ckpt_every": "500",
        "--phase": "0:exam_frac=0,front_share=0,fail_penalty=0,reset_aprev_ref=0,lr_decay_iters=250000,lr_end=0.03,wstream_frac=0;"
                   "2000:exam_frac=0.5,front_share=1,fail_penalty=3,reset_aprev_ref=1,lr_decay_iters=5757,lr_end=0.3;"
                   "3000:exam_frac=0.5,lr_decay_iters=6200,wstream_frac=0.25,critic_epochs=1;"
                   "4500:exam_frac=0.9",
    }
