"""Local annotation tool: watch the match frame by frame and mark where the possession changes.

    python annotation/make_plan.py                   # once, chooses the clips
    python annotation/server.py --annotator fosco    # then open http://127.0.0.1:8765

Everything stays on this computer: the server listens on 127.0.0.1 only, reads the video from
data/, and writes the annotations to annotation/labels/<annotator>/<clip>.json after every change.
Nothing is sent anywhere else -- the terms of use forbid sharing the match data.

The page explains its own keys; README.md in this folder is the annotation protocol (what counts as
a change of possession, where exactly to put the mark). Read it before the first clip.
"""

import argparse
import datetime
import json
import os
import sys
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for sub in ("", "starter", "evaluation", "analysis"):
    sys.path.insert(0, os.path.join(ROOT, sub))

import games  # noqa: E402
import baseline as bl  # noqa: E402
import load_homography as lh  # noqa: E402
import possession_metrics as pm  # noqa: E402
from analyze_data import load_atd  # noqa: E402
import common  # noqa: E402

# Params
GAME = games.GAME
FPS = games.GAMES[GAME]["fps"]
N_FRAMES = games.source_frames("tactical", GAME)
GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
STATIC = os.path.join(HERE, "static")
LOG_PATH = os.path.join(common.LABEL_DIR, "server_errors.log")
CACHE_FRAMES = 500          # encoded frames kept in memory, for stepping back without a re-seek
KEEP_BEFORE_TARGET = 30     # when reading forward to a target, also cache the frames just before it
RESEEK_AHEAD = 150          # a jump further ahead than this re-seeks instead of decoding through
MATCH_TOLERANCE = 75        # frames (3 s) within which a mark counts as the same change as the truth
HOME_BGR, AWAY_BGR, BALL_BGR = (168, 94, 31), (47, 58, 178), (0, 214, 255)


class FrameSource(object):
    """Decodes the tactical video on demand, one capture shared behind a lock.

    Sequential reads are cheap and seeks are not, so the reader keeps its position and walks
    forward when it can, and an LRU cache of encoded frames makes stepping back instant.
    """

    def __init__(self, path, width, quality, offset):
        import cv2
        self.cv2 = cv2
        self.capture = cv2.VideoCapture(path)
        if not self.capture.isOpened():
            raise IOError("cannot open the video: " + path)
        self.width, self.quality, self.offset = width, quality, offset
        self.reported_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.next_index = None
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def _encode(self, image):
        height = int(round(image.shape[0] * self.width / float(image.shape[1])))
        small = self.cv2.resize(image, (self.width, height), interpolation=self.cv2.INTER_AREA)
        ok, data = self.cv2.imencode(".jpg", small, [self.cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return data.tobytes() if ok else None

    def jpeg(self, index):
        with self.lock:
            if index in self.cache:
                self.cache.move_to_end(index)
                return self.cache[index]
            target = index + self.offset
            if not 0 <= target < self.reported_frames:
                return None
            if self.next_index is None or target < self.next_index or target > self.next_index + RESEEK_AHEAD:
                self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, target)
                self.next_index = target
            found = None
            while self.next_index <= target:
                ok, image = self.capture.read()
                if not ok:
                    self.next_index = None
                    return None
                position = self.next_index
                self.next_index += 1
                if target - position <= KEEP_BEFORE_TARGET:
                    data = self._encode(image)
                    self.cache[position - self.offset] = data
                    if position == target:
                        found = data
            while len(self.cache) > CACHE_FRAMES:
                self.cache.popitem(last=False)
            return found


class Tracking(object):
    """The tracking and homography, only to draw an optional overlay and the ball-missing strip."""

    def __init__(self, data_dir):
        meta, raw = games.require(["atd_meta", "atd_raw"], GAME, data_dir)
        atd = load_atd(raw, N_FRAMES, os.path.join(ROOT, "analysis_out", "cache", "atd_tactical.npz"), False)
        self.px, self.py, self.bx, self.by = atd["px"], atd["py"], atd["bx"], atd["by"]
        self.team = bl.team_of_each_track(meta)
        self.ball_ok = np.isfinite(self.bx) & np.isfinite(self.by)
        self.H, _, self.valid = lh.load(GAME, data_dir)

    def missing_runs(self, start, end):
        """[[first, last], ...] stretches of the clip where the tracking has no ball."""
        missing = ~self.ball_ok[start:end + 1]
        edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
        firsts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        return [[int(start + a), int(start + b - 1)] for a, b in zip(firsts, stops)]

    def draw(self, cv2, jpeg_bytes, index, source_width):
        image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if not self.valid[index]:
            return jpeg_bytes
        scale = image.shape[1] / float(source_width)
        matrix = self.H[index]
        occupied = np.flatnonzero(np.isfinite(self.px[index]) & np.isfinite(self.py[index]))
        for s in occupied:
            u, v = lh.project(matrix, self.px[index, s], self.py[index, s])
            centre = (int(round(u * scale)), int(round(v * scale)))
            colour = HOME_BGR if self.team[s] == bl.HOME else AWAY_BGR
            cv2.circle(image, centre, 9, colour, 2)
        if self.ball_ok[index]:
            u, v = lh.project(matrix, self.bx[index], self.by[index])
            cv2.circle(image, (int(round(u * scale)), int(round(v * scale))), 6, BALL_BGR, -1)
        ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return data.tobytes() if ok else jpeg_bytes


def compare_with_truth(clip, changes, truth):
    """How an annotation of the public window scores against the real labels, and how far each
    mark sits from the true change (positive = the mark is late)."""
    start, end = clip["start"], clip["end"]
    gt = truth.copy()
    gt[:start] = pm.X
    gt[end + 1:] = pm.X
    if not (gt != pm.X).any():
        return {"error": "no public ground truth inside this clip"}
    labels = common.full_length(common.clip_labels(changes, start, end), start, N_FRAMES)
    pred = pm.label_codes(labels)
    result = pm.evaluate(gt, pred)
    runs = pm.valid_runs(gt)
    pairs, missed, spurious = pm.match_transitions(
        pm.transitions(gt, runs), pm.transitions(pm.fill_abstentions(pred, runs), runs),
        MATCH_TOLERANCE, typed="to")
    names = pm.LABELS
    rows = [{"frame": g[0], "change": "{}>{}".format(names[g[1]], names[g[2]]),
             "offset": p[0] - g[0]} for g, p in pairs]
    rows += [{"frame": g[0], "change": "{}>{}".format(names[g[1]], names[g[2]]), "offset": None}
             for g in missed]
    rows.sort(key=lambda r: r["frame"])
    offsets = [r["offset"] for r in rows if r["offset"] is not None]
    return {
        "spell_f1": round(result["spell_f1_mean"], 4),
        "macro_f1": round(result["macro_f1"], 4),
        "n_true": result["n_gt_spells"], "n_yours": result["n_pred_spells"],
        "unlabelled_frames": int((labels[start:end + 1] == "X").sum()),
        "median_offset": float(np.median(offsets)) if offsets else None,
        "n_extra_changes": len(spurious),
        "transitions": rows,
    }


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # one line per frame would drown the terminal

        def _send(self, code, body, content_type):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, code=200):
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def _error(self, message, code=400):
            self._json({"error": message}, code)

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[-1] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    with open(os.path.join(STATIC, "index.html"), "rb") as handle:
                        return self._send(200, handle.read(), "text/html; charset=utf-8")
                if url.path == "/api/config":
                    plan = common.load_plan()
                    progress = {}
                    for clip in plan["clips"]:
                        document = common.load_annotation(state["annotator"], clip)
                        progress[clip["id"]] = {"n_changes": len(document["changes"]), "done": document["done"]}
                    return self._json({"annotator": state["annotator"], "fps": FPS, "plan": plan,
                                       "progress": progress, "video_frames": state["frames"].reported_frames,
                                       "expected_frames": N_FRAMES})
                if url.path == "/api/frame":
                    index = int(query["i"])
                    data = state["frames"].jpeg(index)
                    if data is None:
                        return self._error("frame {} could not be decoded".format(index), 404)
                    if query.get("ov") == "1":
                        data = state["tracking"].draw(state["frames"].cv2, data, index, 1920)
                    return self._send(200, data, "image/jpeg")
                if url.path == "/api/clip":
                    clip = common.clip_by_id(common.load_plan(), query["clip"])
                    document = common.load_annotation(state["annotator"], clip)
                    return self._json({"clip": clip, "annotation": document,
                                       "ball_missing": state["tracking"].missing_runs(clip["start"], clip["end"])})
                if url.path == "/api/compare":
                    clip = common.clip_by_id(common.load_plan(), query["clip"])
                    document = common.load_annotation(state["annotator"], clip)
                    return self._json(compare_with_truth(clip, document["changes"], state["truth"]))
                return self._error("not found", 404)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                return None  # the browser gave up on this request (e.g. a newer frame was asked for)
            except (KeyError, ValueError, IOError) as error:
                return self._error(str(error))
            except Exception as error:  # never drop the connection without an answer
                log_failure(self.path, error)
                try:
                    return self._error("errore interno ({}): vedi {}".format(type(error).__name__, LOG_PATH), 500)
                except Exception:
                    return None

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/api/annotation":
                return self._error("not found", 404)
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                clip = common.clip_by_id(common.load_plan(), payload["clip"])
                with state["save_lock"]:
                    document = common.save_annotation(state["annotator"], clip, payload)
                return self._json({"saved": document["updated"], "n_changes": len(document["changes"])})
            except (KeyError, ValueError, IOError) as error:
                return self._error(str(error))
            except Exception as error:
                log_failure(self.path, error)
                return self._error("errore interno nel salvataggio ({}): vedi {}".format(
                    type(error).__name__, LOG_PATH), 500)

    return Handler


def log_failure(path, error):
    """Unexpected errors go to a file as well as the terminal, so they can be reported afterwards."""
    import traceback
    text = "{}  {}\n{}\n".format(datetime.datetime.now().isoformat(timespec="seconds"), path,
                                 "".join(traceback.format_exception(type(error), error, error.__traceback__)))
    sys.stderr.write(text)
    try:
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(text)
    except IOError:
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotator", required=True,
                        help="your name, lowercase: it names the folder your annotations go in")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--width", type=int, default=1280, help="width of the frames shown, in pixels")
    parser.add_argument("--quality", type=int, default=82, help="JPEG quality of the frames shown")
    parser.add_argument("--video-offset", type=int, default=0,
                        help="shift the decoded frame if the picture is visibly early or late")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    common.check_name(args.annotator, "annotator")
    common.load_plan()  # fail now, with the instruction, rather than in the browser
    video = games.require(["video"], GAME, args.data_dir)[0]
    print("loading tracking and homography ...")
    state = {
        "annotator": args.annotator,
        "frames": FrameSource(video, args.width, args.quality, args.video_offset),
        "tracking": Tracking(args.data_dir),
        "truth": pm.label_codes(pm.read_label_csv(GT_PATH)),
        "save_lock": threading.Lock(),
    }
    if state["frames"].reported_frames != N_FRAMES:
        print("WARNING: the video reports {} frames, expected {}".format(
            state["frames"].reported_frames, N_FRAMES))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    print("annotator: {}   annotations -> {}".format(
        args.annotator, os.path.join(common.LABEL_DIR, args.annotator)))
    print("open http://127.0.0.1:{}   (Ctrl+C to stop; every change is already saved)".format(args.port))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
