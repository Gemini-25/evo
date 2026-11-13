"""Writes tests/data/audit/*.evoenv: the six audit states of design D40/T31 (one problem each).

Run from the repo root: .venv/bin/python tests/data/audit/make_audit.py tests/data/audit
"""
import copy, json, sys
from pathlib import Path
from evolab.spec.io import parse_env, dump_env, compile_env
from evolab.spec.compiled import check_env
from evolab.app.editor.document import editor_checks

BASE = {
    "version": 1, "name": "검사 견본",
    "world": {"gravity": [0.0, -9.81], "max_time": 10.0, "bounds": [-20.0, -5.0, 20.0, 20.0]},
    "agent": {"spawn": {"x": 0.0, "y": 1.2}},
    "bodies": [
        {"id": "ground", "label": "땅", "owner": "world", "type": "static", "pos": [0.0, 0.0],
         "shapes": [{"kind": "box", "w": 30.0, "h": 0.4}], "tags": ["ground"]},
        {"id": "torso", "label": "몸통", "pos": [0.0, 0.0], "shapes": [{"kind": "box", "w": 0.4, "h": 0.6}]},
        {"id": "leg", "label": "다리", "place_by": "hip", "shapes": [{"kind": "box", "w": 0.15, "h": 0.5}]},
    ],
    "joints": [{"id": "hip", "label": "엉덩이", "kind": "pivot", "a": "torso", "b": "leg",
                "anchor_a": [0.0, -0.3], "anchor_b": [0.0, 0.25]}],
    "actuators": [{"kind": "motor", "id": "m_hip", "label": "엉덩이 모터", "joint": "hip", "max_torque": 20.0,
                   "max_speed": 6.0, "keys_pos": ["Q"], "keys_neg": ["A"]}],
    "sensors": [{"kind": "body_state", "id": "st", "label": "몸통 상태", "body": "torso", "fields": ["vx", "vy", "angle"]}],
    "termination": [{"event": "timeout", "outcome": "neutral", "label": "시간 끝"}],
    "scoring": {"blocks": [{"kind": "progress", "id": "s_dist", "label": "간 거리", "body": "torso", "weight": 1.0}]},
}

def variant(code):
    d = copy.deepcopy(BASE)
    if code == "ground_is_body":
        d["bodies"][0].update(owner="agent", type="dynamic", pos=[0.0, -1.2], tags=["ground"])
        del d["bodies"][0]["type"]
        d["bodies"][0]["shapes"] = [{"kind": "box", "w": 30.0, "h": 0.4}]
        d["bodies"].append({"id": "floor", "label": "바닥", "owner": "world", "type": "static", "pos": [0.0, -3.0],
                            "shapes": [{"kind": "box", "w": 40.0, "h": 0.4}], "tags": ["floor"]})
    elif code == "score_far_is_good":
        d["regions"] = [{"id": "goal", "label": "골", "shape": {"kind": "box", "w": 1.0, "h": 2.0}, "pos": [8.0, 1.0]}]
        d["scoring"]["blocks"].append({"kind": "distance_to", "id": "s_near", "label": "골까지", "body": "torso",
                                       "target": "goal", "weight": 1.0, "agg": "min"})
    elif code == "no_keys":
        d["bodies"].append({"id": "leg2", "label": "다리2", "place_by": "hip2", "shapes": [{"kind": "box", "w": 0.15, "h": 0.5}]})
        d["bodies"].append({"id": "arm", "label": "팔", "place_by": "sh", "shapes": [{"kind": "box", "w": 0.1, "h": 0.4}]})
        d["joints"] += [{"id": "hip2", "label": "엉덩이2", "kind": "pivot", "a": "torso", "b": "leg2",
                         "anchor_a": [0.1, -0.3], "anchor_b": [0.0, 0.25]},
                        {"id": "sh", "label": "어깨", "kind": "pivot", "a": "torso", "b": "arm",
                         "anchor_a": [0.0, 0.3], "anchor_b": [0.0, 0.2]}]
        d["actuators"] = [{"kind": "motor", "id": m, "label": m, "joint": j, "max_torque": 20.0, "max_speed": 6.0}
                          for m, j in (("m_hip", "hip"), ("m_hip2", "hip2"), ("m_sh", "sh"))]
    elif code == "motor_twice":
        d["actuators"].append({"kind": "motor", "id": "m_hip2", "label": "둘째 모터", "joint": "hip", "max_torque": 20.0,
                               "max_speed": 6.0, "keys_pos": ["W"], "keys_neg": ["S"]})
    elif code == "joint_apart":
        leg = d["bodies"][2]
        del leg["place_by"]
        leg["pos"] = [0.25, -0.55]
    elif code == "never_stops":
        d["world"]["gravity"] = [0.0, 0.0]
        d["bodies"].append({"id": "ball", "label": "공", "owner": "world", "pos": [3.0, 1.0],
                            "shapes": [{"kind": "circle", "r": 0.3}]})
    return d

out = Path(sys.argv[1])
for code in ("ground_is_body", "score_far_is_good", "no_keys", "motor_twice", "joint_apart", "never_stops"):
    d = variant(code)
    env = parse_env(d)
    text = dump_env(env)
    spec = compile_env(json.loads(text))
    eng = check_env(env)
    ed = editor_checks(parse_env(text), spec)
    print(code, [(i.severity, i.code, i.message) for i in eng], [(i.severity, i.code, i.where, i.message) for i in ed])
    (out / f"{code}.evoenv").write_text(text, encoding="utf-8", newline="\n")
