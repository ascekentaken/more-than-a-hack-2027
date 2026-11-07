"""Read the open DFL matches (IDSSE dataset) into the same layout as the challenge tracking.

    python external/dfl.py            # parses every downloaded match, caches it, prints checks

This is EXTERNAL data: seven full German Bundesliga / 2. Bundesliga matches of 2022/23, released by
the Deutsche Fussball Liga under CC BY 4.0 (Bassek, Weber, Rein & Memmert, "An integrated dataset of
spatiotemporal and event data in elite soccer", Scientific Data, 2025). The rules of the challenge
say that anything trained or tuned must be trained or tuned on external data; this is it.

The files are downloaded into external_data/dfl/ (gitignored, ~2.7 GB) from the Hugging Face
mirror pysport/idsse-data; see download() below.

What makes them useful: every ball frame carries BallStatus (1 = in play, 0 = dead) and
BallPossession (1 = home, 2 = away), set by the data provider on its corrected tracking. That is
the same kind of label as the challenge ground truth ("dead where the corrected tracking marks the
ball dead, otherwise the team it names as owning it", CHALLENGE.md), so H / A / D per frame comes
for free.

Layout returned by load(match_id), the same as the challenge ATD (see model/features.compute):
  px, py   (frames, players) positions normalised 0-1, NaN when the player is not on the pitch
  ps       (frames, players) speed in m/s (the file's S is km/h)
  team     (players,) 0 home, 1 away
  bx, by   ball, normalised 0-1
  label    (frames,) 0 H, 1 A, 2 D
  half     (frames,) 1 or 2; a single NaN frame separates the halves and has half 0
"""

import argparse
import os
import sys
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "external_data", "dfl")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
MIRROR = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"

# Params
MATCHES = {  # match id -> competition id in the file names
    "J03WMX": "000001", "J03WN1": "000001",
    "J03WOH": "000002", "J03WOY": "000002", "J03WPY": "000002", "J03WQQ": "000002", "J03WR9": "000002",
}
SECTIONS = {"firstHalf": 1, "secondHalf": 2}
KMH_TO_MS = 1 / 3.6
H, A, D = 0, 1, 2


def file_path(match_id, kind):
    prefix = {"info": "DFL_02_01_matchinformation", "events": "DFL_03_02_events_raw",
              "positions": "DFL_04_03_positions_raw_observed"}[kind]
    return os.path.join(DATA_DIR, "{}_DFL-COM-{}_DFL-MAT-{}.xml".format(prefix, MATCHES[match_id], match_id))


def download(match_id):
    """Fetch the three files of a match if they are not there yet."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for kind in ("info", "events", "positions"):
        path = file_path(match_id, kind)
        if os.path.isfile(path) and os.path.getsize(path) > 0 and _complete(path):
            continue
        request = urllib.request.Request(MIRROR + os.path.basename(path), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request) as response, open(path, "wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)


def _complete(path):
    """A finished XML file ends with its closing root tag."""
    with open(path, "rb") as handle:
        handle.seek(max(os.path.getsize(path) - 200, 0))
        return b"</PutDataRequest>" in handle.read()


def match_info(match_id):
    root = ET.parse(file_path(match_id, "info")).getroot()
    general = root.find(".//General")
    environment = root.find(".//Environment")
    team_of_person = {}
    for team in root.iter("Team"):
        for player in team.iter("Player"):
            team_of_person[player.get("PersonId")] = team.get("TeamId")
    return {"home": general.get("HomeTeamId"), "away": general.get("GuestTeamId"),
            "title": general.get("MatchTitle"), "competition": general.get("CompetitionName"),
            "length": float(environment.get("PitchX")), "width": float(environment.get("PitchY")),
            "team_of_person": team_of_person}


def parse_positions(match_id):
    """{(section, person_or_BALL): dict of arrays} streamed out of the big XML file."""
    sets = {}
    current, rows = None, None
    for event, element in ET.iterparse(file_path(match_id, "positions"), events=("start", "end")):
        tag = element.tag
        if event == "start" and tag == "FrameSet":
            section = SECTIONS.get(element.get("GameSection"))
            team = element.get("TeamId")
            current = None if section is None or team == "referee" else (
                section, "BALL" if team == "BALL" else element.get("PersonId"))
            rows = []
        elif event == "end" and tag == "Frame":
            if current is not None:
                a = element.attrib
                if current[1] == "BALL":
                    rows.append((int(a["N"]), float(a["X"]), float(a["Y"]), float(a.get("S", "nan")),
                                 int(a.get("BallPossession", "0")), int(a.get("BallStatus", "0"))))
                else:
                    rows.append((int(a["N"]), float(a["X"]), float(a["Y"]), float(a.get("S", "nan"))))
            element.clear()
        elif event == "end" and tag == "FrameSet":
            if current is not None and rows:
                sets[current] = np.array(rows, dtype=np.float64)
            current, rows = None, None
            element.clear()
    return sets


def build(match_id):
    info = match_info(match_id)
    sets = parse_positions(match_id)
    length, width = info["length"], info["width"]
    persons = sorted({person for (_, person) in sets if person != "BALL"})
    team = np.array([0 if info["team_of_person"].get(p) == info["home"] else 1 for p in persons], dtype=np.int8)
    column = {p: i for i, p in enumerate(persons)}

    parts = []
    for section in (1, 2):
        ball = sets[(section, "BALL")]
        n0, n1 = int(ball[:, 0].min()), int(ball[:, 0].max())
        n = n1 - n0 + 1
        px = np.full((n, len(persons)), np.nan, dtype=np.float32)
        py = np.full_like(px, np.nan)
        ps = np.full_like(px, np.nan)
        for (s, person), rows in sets.items():
            if s != section or person == "BALL":
                continue
            index = rows[:, 0].astype(np.int64) - n0
            keep = (index >= 0) & (index < n)
            c = column[person]
            px[index[keep], c] = (rows[keep, 1] + length / 2) / length
            py[index[keep], c] = (rows[keep, 2] + width / 2) / width
            ps[index[keep], c] = rows[keep, 3] * KMH_TO_MS
        index = ball[:, 0].astype(np.int64) - n0
        bx = np.full(n, np.nan, dtype=np.float32)
        by = np.full(n, np.nan, dtype=np.float32)
        bx[index] = (ball[:, 1] + length / 2) / length
        by[index] = (ball[:, 2] + width / 2) / width
        label = np.full(n, D, dtype=np.int8)
        possession, status = ball[:, 4].astype(int), ball[:, 5].astype(int)
        label[index] = np.where(status == 1, np.where(possession == 1, H, np.where(possession == 2, A, D)), D)
        parts.append((px, py, ps, bx, by, label, np.full(n, section, dtype=np.int8)))
        if section == 1:  # one empty frame between the halves, so nothing runs across halftime
            blank = np.full((1, len(persons)), np.nan, dtype=np.float32)
            parts.append((blank, blank, blank, np.full(1, np.nan, np.float32), np.full(1, np.nan, np.float32),
                          np.full(1, D, np.int8), np.zeros(1, np.int8)))
    out = {name: np.concatenate([p[i] for p in parts]) for i, name in
           enumerate(("px", "py", "ps", "bx", "by", "label", "half"))}
    out.update({"team": team, "length": np.array(length), "width": np.array(width)})
    return out, info


def load(match_id, rebuild=False):
    """The match in ATD layout, parsed once and cached as .npz."""
    path = os.path.join(CACHE_DIR, match_id + ".npz")
    if rebuild or not os.path.isfile(path):
        download(match_id)
        data, _ = build(match_id)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(path, **data)
    with np.load(path) as saved:
        return {k: saved[k] for k in saved.files}


def checks(match_id, data):
    """Automatic sanity checks on the external labels and units."""
    label, half = data["label"], data["half"]
    graded = half > 0
    alive = graded & (label != D)
    seen = np.isfinite(data["bx"])
    # is BallPossession 1 really the home team? The nearest player's team should agree with it
    L, W = float(data["length"]), float(data["width"])
    dist = np.sqrt(((data["px"] - data["bx"][:, None]) * L) ** 2 + ((data["py"] - data["by"][:, None]) * W) ** 2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    nearest = data["team"][dist.argmin(axis=1)]
    close = alive & seen & (dist.min(axis=1) < 1.0)
    agree = float((nearest[close] == label[close]).mean())
    # speed unit: speed from positions vs the file's speed
    dx = np.diff(data["px"], axis=0) * L
    dy = np.diff(data["py"], axis=0) * W
    from_positions = np.hypot(dx, dy) * 25
    file_speed = data["ps"][1:]
    ok = np.isfinite(from_positions) & np.isfinite(file_speed) & (file_speed > 1)
    ratio = float(np.median(from_positions[ok] / file_speed[ok]))
    codes = label[graded]
    changes = int((np.diff(codes) != 0).sum())
    return {"frames": int(graded.sum()), "minutes": graded.sum() / 1500.0,
            "H": float((codes == H).mean()), "A": float((codes == A).mean()), "D": float((codes == D).mean()),
            "changes": changes, "nearest_agrees_with_possession": agree, "speed_ratio": ratio,
            "players_on_pitch": float(np.isfinite(data["px"][graded]).sum(axis=1).mean())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("matches", nargs="*", default=sorted(MATCHES))
    args = parser.parse_args()
    for match_id in args.matches:
        data = load(match_id, args.rebuild)
        c = checks(match_id, data)
        print("{}  {:5.1f} min  H {:.0%} A {:.0%} D {:.0%}  label changes {:4d}  players {:.1f}  "
              "nearest==possession {:.0%}  speed(positions)/speed(file) {:.2f}".format(
                  match_id, c["minutes"], c["H"], c["A"], c["D"], c["changes"],
                  c["players_on_pitch"], c["nearest_agrees_with_possession"], c["speed_ratio"]))
