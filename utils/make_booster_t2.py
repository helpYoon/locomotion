"""Build <soma-retargeter>/local-robots/booster_t2: Booster's T2_31dof_Revo2Hand URDF as a soma target.

- the `world` link and floating `world_joint` are dropped (the Revo2 file has them commented out; without that, Newton
  imports a second free joint), so the floating base is `trunk`, the free-joint body of our MJCF;
- the 22 finger joints become fixed at 0 (fingers locked open; finger links kept), leaving the 31 body joints;
- ankle pitch gets the base T2's range on both sides (the user's decision; the Revo2 file says [-0.838, +0.576]).
The meshes are the vendor's (symlinked). The manifest declares one target per pilot scale, booster_t2_075 and
booster_t2_080, which differ only in their retargeter config; utils/soma_t2_configs.py writes configs/ (run it next).
Registered via "extra_robot_paths" like utils/make_booster_t1.py's booster_t1.

usage: python3 utils/make_booster_t2.py <soma-retargeter root>  (standard library only)
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.abspath(sys.argv[1])
VENDOR = os.path.expanduser("~/booster_assets/robots/T2/T2_31dof_Revo2Hand")  # booster_assets 12a9751
DST = os.path.join(ROOT, "local-robots", "booster_t2")
ANKLE_PITCH = (-0.645771823, 0.331612558)  # base T2_31dof.urdf
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
TARGETS = ("booster_t2_075", "booster_t2_080")  # human_scale_ratio 0.75 / 0.80

os.makedirs(os.path.join(DST, "desc"), exist_ok=True)
meshes = os.path.join(DST, "desc", "meshes")
if not os.path.islink(meshes):
    os.symlink(os.path.join(VENDOR, "meshes"), meshes)
manifest = {"targets": [{"name": t, "desc": {"urdf_path": "desc/T2_31dof.urdf"},
                         "retarget_configs": {"soma": f"configs/soma_to_{t}_retargeter_config.json"}} for t in TARGETS]}
with open(os.path.join(DST, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=4)

tree = ET.parse(os.path.join(VENDOR, "T2_31dof_Revo2Hand.urdf"))  # drops comments
robot = tree.getroot()
for link in robot.findall("link"):
    if link.get("name") == "world":
        robot.remove(link)
n_fixed = 0
for joint in robot.findall("joint"):
    name = joint.get("name")
    if joint.get("type") == "floating":
        robot.remove(joint)
    elif joint.get("type") == "revolute" and any(k in name for k in FINGERS):
        joint.set("type", "fixed")  # its origin is the pose at q = 0
        for tag in ("axis", "limit", "mimic", "dynamics"):
            for e in joint.findall(tag):
                joint.remove(e)
        n_fixed += 1
    elif name in ("left_ankle_pitch_joint", "right_ankle_pitch_joint"):
        limit = joint.find("limit")
        print(f"{name:24s} [{float(limit.get('lower')):+.4f}, {float(limit.get('upper')):+.4f}] -> "
              f"[{ANKLE_PITCH[0]:+.4f}, {ANKLE_PITCH[1]:+.4f}]")
        limit.set("lower", repr(ANKLE_PITCH[0]))
        limit.set("upper", repr(ANKLE_PITCH[1]))
revolute = [j.get("name") for j in robot.findall("joint") if j.get("type") == "revolute"]
assert n_fixed == 22 and len(revolute) == 31, (n_fixed, len(revolute))
tree.write(os.path.join(DST, "desc", "T2_31dof.urdf"), xml_declaration=True, encoding="utf-8")
print(f"wrote {DST} ({n_fixed} finger joints fixed, {len(revolute)} revolute)")
