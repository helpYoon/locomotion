"""Build assets/t2/t2_description/mjcf/t2.xml from Booster's T2_31dof_Revo2Hand.xml, the hand's fingers locked open.

Source: booster_assets robots/T2/T2_31dof_Revo2Hand/ at 12a9751 (BSD-3; assets/t2/SOURCE.txt). The vendor XML is
edited as text (MjSpec.to_xml would round every number to 6 digits), so bodies, inertia, the 31 body joints (axes,
ranges, armature, effort), colliders, motors, floor and IMU stay Booster's exact values, except:
- the 22 finger joints and their motors are removed. The finger bodies stay, welded at angle 0 (open: each finger
  joint's lower limit), with their mass and inertia;
- each hand's 16 finger collision meshes and its palm mesh collider (wrist_roll_link_collision, a 2,200-vertex hull)
  become one box: the bounding box of their vertices in the wrist_roll_link frame at that pose;
- both ankle pitch ranges are the base T2_31dof.xml's (lines 190, 225), not the Revo2 file's [-0.837758, 0.575959].
Only the meshes the output references are copied, to assets/t2/t2_description/meshes/.

usage (from the repo root): python -m utils.build_t2_mjcf [booster_assets root, default ~/booster_assets]
"""

import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESC = os.path.join(ROOT, "assets", "t2", "t2_description")
OUT = os.path.join(DESC, "mjcf", "t2.xml")
SRC = os.path.join("robots", "T2", "T2_31dof_Revo2Hand", "T2_31dof_Revo2Hand.xml")  # in booster_assets
COMMIT = "12a9751"
FINGER = re.compile(r"(left|right)_(thumb|index|middle|ring|pinky)_\w+")  # finger joints, links and their geoms
ANKLE_PITCH_RANGE = "-0.645771823 0.331612558"  # base T2_31dof.xml; the user's choice for training and retargeting
MASS = 42.24  # kg: the base T2's 41.99 with the Revo2 wrist and hands
SIDES = ("left", "right")


def _colliders(m: mujoco.MjModel) -> list[int]:
    return [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]


def hand_boxes(m: mujoco.MjModel) -> dict[str, tuple[list[str], np.ndarray, np.ndarray]]:
    """side -> (its palm and finger colliders' names, box centre, box half sizes) in the wrist_roll_link frame, at qpos0 =
    fingers open. MuJoCo's mesh_vert is in the geom's compiled frame. ponytail: the box is axis-aligned in the wrist
    frame, not a fitted oriented box."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    out = {}
    for side in SIDES:
        w = m.body(f"{side}_wrist_roll_link").id
        geoms = [g for g in _colliders(m) if FINGER.fullmatch(name := m.body(m.geom_bodyid[g]).name) and name.startswith(side)]
        geoms.append(m.geom(f"{side}_wrist_roll_link_collision").id)  # the palm
        assert len(geoms) == 17 and all(m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH for g in geoms), side
        pts = []
        for g in geoms:
            k = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[k] : m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            pts.append((d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T - d.xpos[w]) @ d.xmat[w].reshape(3, 3))
        lo, hi = np.concatenate(pts).min(0), np.concatenate(pts).max(0)
        out[side] = [m.geom(g).name for g in geoms], (hi + lo) / 2, (hi - lo) / 2
    return out


def build(src_xml: str) -> ET.ElementTree:
    tree = ET.parse(src_xml)
    root = tree.getroot()
    root.set("model", "t2")
    root.find("compiler").set("meshdir", "../meshes/")
    parent = {c: p for p in root.iter() for c in p}
    boxes = hand_boxes(mujoco.MjModel.from_xml_path(src_xml))
    doomed = {n for names, _, _ in boxes.values() for n in names}
    drop = [j for j in root.iter("joint") if FINGER.fullmatch(j.get("name"))]
    drop += [a for a in root.find("actuator") if FINGER.fullmatch(a.get("joint"))]
    for g in root.iter("geom"):
        if g.get("name", "").endswith("_wrist_roll_link_collision"):  # the palm: the box takes its place
            side = g.get("name").split("_")[0]
            _, c, h = boxes[side]
            box = ET.Element("geom", {"name": f"{side}_hand_collision", "class": "collision", "type": "box",
                                      "size": " ".join(f"{x:.6f}" for x in h), "pos": " ".join(f"{x:.6f}" for x in c)})  # fmt: skip
            body = parent[g]
            body.insert(list(body).index(g) + 1, box)
    drop += [g for g in root.iter("geom") if g.get("name") in doomed]
    assert len(drop) == 22 + 22 + 34
    for e in drop:  # a finger body without its joint keeps its frame, i.e. joint angle 0
        parent[e].remove(e)
    for j in root.iter("joint"):
        if j.get("name").endswith("_ankle_pitch_joint"):
            j.set("range", ANKLE_PITCH_RANGE)
    used = {g.get("mesh") for g in root.iter("geom")}
    asset = root.find("asset")
    for mesh in [mesh for mesh in asset.findall("mesh") if mesh.get("name") not in used]:
        asset.remove(mesh)
    ET.indent(tree, "  ")
    return tree


def check(m: mujoco.MjModel, vendor: mujoco.MjModel) -> None:
    """The output against the vendor model: the same bodies and visuals, the 31 body joints and motors, the finger
    colliders replaced by the two hand boxes, the base ankle pitch range."""
    hinges = [m.joint(j).name for j in range(1, m.njnt)]
    assert m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and (m.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE).all()
    assert hinges == [n for n in (vendor.joint(j).name for j in range(1, vendor.njnt)) if not FINGER.fullmatch(n)]
    assert len(hinges) == 31 and [m.actuator(i).name for i in range(m.nu)] == hinges  # Booster's body motors, in order
    assert m.nbody == vendor.nbody and abs(m.body_mass.sum() - MASS) < 0.01, m.body_mass.sum()
    for f in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat"):
        np.testing.assert_array_equal(getattr(m, f), getattr(vendor, f), err_msg=f)
    visual = lambda x: sorted((x.body(x.geom_bodyid[g]).name, x.mesh(x.geom_dataid[g]).name, *x.geom_pos[g], *x.geom_quat[g])
                              for g in sorted(set(range(x.ngeom)) - set(_colliders(x))) if x.geom_dataid[g] >= 0)  # noqa: E731
    assert visual(m) == visual(vendor) and m.ngeom - len(_colliders(m)) == vendor.ngeom - len(_colliders(vendor))
    names = lambda x: [x.geom(g).name for g in _colliders(x)]  # noqa: E731
    assert sorted(names(m)) == sorted([n for n in names(vendor) if not FINGER.fullmatch(n.removesuffix("_1").removesuffix("_collision"))
                                       and not n.endswith("_wrist_roll_link_collision")]
                                      + ["left_hand_collision", "right_hand_collision"])  # fmt: skip
    for side in SIDES:
        g = m.geom(f"{side}_hand_collision")
        assert m.body(g.bodyid[0]).name == f"{side}_wrist_roll_link" and g.type[0] == mujoco.mjtGeom.mjGEOM_BOX
        np.testing.assert_array_equal(m.joint(f"{side}_ankle_pitch_joint").range, [float(x) for x in ANKLE_PITCH_RANGE.split()])
    for j in range(1, m.njnt):
        v = vendor.joint(m.joint(j).name).id
        np.testing.assert_array_equal([m.dof_armature[m.jnt_dofadr[j]], *m.jnt_axis[j], *m.jnt_actfrcrange[j], *m.jnt_pos[j]],
                                      [vendor.dof_armature[vendor.jnt_dofadr[v]], *vendor.jnt_axis[v], *vendor.jnt_actfrcrange[v],
                                       *vendor.jnt_pos[v]])  # fmt: skip
        if "ankle_pitch" not in m.joint(j).name:
            np.testing.assert_array_equal(m.jnt_range[j], vendor.jnt_range[v])


if __name__ == "__main__":
    assets = os.path.abspath(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/booster_assets"))
    src = os.path.join(assets, SRC)
    tree = build(src)
    os.makedirs(os.path.join(DESC, "meshes"), exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    meshes = [mesh.get("file") for mesh in tree.getroot().find("asset").findall("mesh")]
    for f in meshes:
        shutil.copy2(os.path.join(os.path.dirname(src), "meshes", f), os.path.join(DESC, "meshes", f))
    with open(OUT, "w") as f:
        f.write(f"<!-- Generated by utils/build_t2_mjcf.py from booster_assets {COMMIT} {SRC}; edit that script, not this. -->\n")
        f.write(ET.tostring(tree.getroot(), encoding="unicode") + "\n")
    with open(os.path.join(ROOT, "assets", "t2", "SOURCE.txt"), "w") as f:
        f.write(f"booster_assets (BSD-3), commit {COMMIT}: {SRC} and the meshes it references in "
                f"{os.path.dirname(SRC)}/meshes/, built into t2_description/ by utils/build_t2_mjcf.py "
                "(fingers welded open, one box per hand over palm and fingers, the base T2_31dof's ankle pitch range).\n")
    check(mujoco.MjModel.from_xml_path(OUT), mujoco.MjModel.from_xml_path(src))
    print(f"wrote {OUT} and {len(meshes)} meshes")
