"""Shared by the annotation server, the plan maker and the export script.

An annotation is a list of CHANGE POINTS, not a label per frame: "from frame f on, the state is L".
The state then holds until the next change point or the end of the clip. That is how a person
watches football -- a possession starts, a possession ends -- and it is roughly a hundred times
less clicking than labelling frames.

    {"clip": "c03", "annotator": "fosco", "start": 30000, "end": 32999,
     "changes": [{"frame": 30000, "label": "H", "doubt": false, "note": ""},
                 {"frame": 30412, "label": "A", "doubt": true,  "note": "deviazione?"}],
     "done": false, "updated": "2026-10-28T10:00:00", "format": 1}

Everything under annotation/labels/ is hand-made labels for the match: gitignored, like the
submissions in evaluation/results/. See README.md in this folder before sharing it anywhere.
"""

import datetime
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LABEL_DIR = os.path.join(HERE, "labels")
PLAN_PATH = os.path.join(LABEL_DIR, "plan.json")
OUT_DIR = os.path.join(LABEL_DIR, "out")

# Params
LABELS = ("H", "A", "D")
NAME_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
NOTE_MAX = 200
FORMAT = 1


def load_plan():
    if not os.path.isfile(PLAN_PATH):
        raise IOError("no annotation plan yet -- run:  python annotation/make_plan.py")
    with open(PLAN_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def clip_by_id(plan, clip_id):
    for clip in plan["clips"]:
        if clip["id"] == clip_id:
            return clip
    raise KeyError("no clip {!r} in the plan".format(clip_id))


def check_name(name, what):
    if not NAME_RE.match(name or ""):
        raise ValueError("{} {!r}: use lowercase letters, digits, _ or -, at most 32".format(what, name))
    return name


def annotation_path(annotator, clip_id):
    return os.path.join(LABEL_DIR, check_name(annotator, "annotator"),
                        check_name(clip_id, "clip") + ".json")


def annotators():
    if not os.path.isdir(LABEL_DIR):
        return []
    return sorted(name for name in os.listdir(LABEL_DIR)
                  if NAME_RE.match(name) and os.path.isdir(os.path.join(LABEL_DIR, name)))


def empty_annotation(annotator, clip):
    return {"clip": clip["id"], "annotator": annotator, "start": clip["start"],
            "end": clip["end"], "changes": [], "done": False, "updated": None, "format": FORMAT}


def load_annotation(annotator, clip):
    path = annotation_path(annotator, clip["id"])
    if not os.path.isfile(path):
        return empty_annotation(annotator, clip)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def clean_changes(changes, clip):
    """Validated, sorted change points. Raises ValueError with a readable reason."""
    out, seen = [], set()
    for change in changes or []:
        frame = int(change["frame"])
        label = str(change["label"]).strip().upper()
        if label not in LABELS:
            raise ValueError("label {!r} at frame {} is not one of {}".format(label, frame, LABELS))
        if not clip["start"] <= frame <= clip["end"]:
            raise ValueError("frame {} is outside clip {} ({}-{})".format(
                frame, clip["id"], clip["start"], clip["end"]))
        if frame in seen:
            raise ValueError("two change points on frame {}".format(frame))
        seen.add(frame)
        out.append({"frame": frame, "label": label, "doubt": bool(change.get("doubt", False)),
                    "note": str(change.get("note", ""))[:NOTE_MAX]})
    return sorted(out, key=lambda c: c["frame"])


def save_annotation(annotator, clip, payload):
    document = empty_annotation(annotator, clip)
    document["changes"] = clean_changes(payload.get("changes"), clip)
    document["done"] = bool(payload.get("done", False))
    document["updated"] = datetime.datetime.now().replace(microsecond=0).isoformat()
    path = annotation_path(annotator, clip["id"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=1)
    os.replace(temporary, path)  # never a half-written file, even if the server is killed
    return document


def clip_labels(changes, start, end):
    """One letter per frame of [start, end]; X before the first change point."""
    labels = np.full(end - start + 1, "X", dtype="<U1")
    ordered = sorted(changes, key=lambda c: c["frame"])
    for k, change in enumerate(ordered):
        a = max(change["frame"], start)
        b = ordered[k + 1]["frame"] if k + 1 < len(ordered) else end + 1
        labels[a - start:b - start] = change["label"]
    return labels


def full_length(labels, start, n_frames):
    out = np.full(n_frames, "X", dtype="<U1")
    out[start:start + len(labels)] = labels
    return out
