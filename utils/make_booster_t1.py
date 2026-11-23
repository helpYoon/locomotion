"""Build <soma-retargeter>/local-robots/booster_t1: soma's bundled Booster T1 with each joint range tightened to its
intersection with bfm_wm's MJCF (the model the policy is trained on), so retargeted motions stay inside both.
Kinematics, configs and meshes are soma's own (symlinked). Registered via "extra_robot_paths", it overrides the bundled
booster_t1. utils/retarget_bones_seed_t1.sh runs this first.

usage: python3 utils/make_booster_t1.py <soma-retargeter root>  (standard library only)
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.abspath(sys.argv[1])
SRC = os.path.join(ROOT, "soma_retargeter", "assets", "robotics", "booster", "t1")
DST = os.path.join(ROOT, "local-robots", "booster_t1")
BFM_MJCF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "t1", "t1_description",
                        "mjcf", "t1.xml")

mjcf_range = {
    j.get("name"): [float(x) for x in j.get("range").split()]
    for j in ET.parse(BFM_MJCF).getroot().iter("joint")
    if j.get("range")
}

os.makedirs(os.path.join(DST, "desc"), exist_ok=True)
for link, target in (("configs", os.path.join(SRC, "configs")), (os.path.join("desc", "meshes"), os.path.join(SRC, "desc", "meshes"))):
    path = os.path.join(DST, link)
    if not os.path.islink(path):
        os.symlink(os.path.relpath(target, os.path.dirname(path)), path)
with open(os.path.join(SRC, "manifest.json")) as f:
    manifest = json.load(f)
with open(os.path.join(DST, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=4)

tree = ET.parse(os.path.join(SRC, "desc", "T1_29dof.urdf"))
for joint in tree.getroot().iter("joint"):
    limit, name = joint.find("limit"), joint.get("name")
    if limit is None or name not in mjcf_range:
        continue
    lo, hi = float(limit.get("lower")), float(limit.get("upper"))
    new_lo, new_hi = max(lo, mjcf_range[name][0]), min(hi, mjcf_range[name][1])
    if (new_lo, new_hi) != (lo, hi):
        print(f"{name:20s} [{lo:+.4f}, {hi:+.4f}] -> [{new_lo:+.4f}, {new_hi:+.4f}]")
        limit.set("lower", f"{new_lo:.6g}")
        limit.set("upper", f"{new_hi:.6g}")
tree.write(os.path.join(DST, "desc", "T1_29dof.urdf"), xml_declaration=True, encoding="utf-8")
print("wrote", DST)
