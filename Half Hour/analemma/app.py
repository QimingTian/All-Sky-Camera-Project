#!/usr/bin/env python3
"""Local web app for Analemma base selection, sun detection/annotation, and render."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file

from library import (
    HALF_HOUR_DIR,
    encode_preview,
    frames_at_standard_time,
    preview_scale,
    read_image,
    twilight_candidates,
)
from render import render_analemma, save_render
from sun import autodetect

APP_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = APP_DIR / "projects"
OUTPUT_DIR = APP_DIR / "output"

app = Flask(__name__, template_folder="templates", static_folder="static")
PROJECT_PATH = PROJECTS_DIR / "current.json"
FRAME_STATE_KEYS = ("x", "y", "skipped", "done", "source", "reason", "predX", "predY")


def load_project() -> dict:
    if PROJECT_PATH.exists():
        return json.loads(PROJECT_PATH.read_text())
    return {
        "name": "current",
        "basePlate": None,
        "targetTime": "0930",
        "timeBasis": "EST",
        "sunRadius": 6,
        "frames": [],
        "updatedAt": None,
    }


def save_project(project: dict) -> None:
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    project["updatedAt"] = datetime.now(timezone.utc).isoformat()
    PROJECT_PATH.write_text(json.dumps(project, indent=2))


def build_frame_list(est_hhmm: str) -> list[dict]:
    est_label = f"{est_hhmm[:2]}:{est_hhmm[2:]} EST"
    return [
        {
            "date": f.date,
            "file": f.filename,
            "timeLabel": est_label if f.time == est_hhmm else f"{est_label} (EDT {f.time_label})",
            "x": None,
            "y": None,
            "skipped": False,
            "done": False,
            "source": None,
            "reason": None,
        }
        for f in frames_at_standard_time(est_hhmm)
    ]


def merge_frames(existing: list[dict], fresh: list[dict]) -> list[dict]:
    by_file = {item["file"]: item for item in existing}
    merged: list[dict] = []
    for frame in fresh:
        old = by_file.get(frame["file"])
        if old:
            merged.append({**frame, **{k: old[k] for k in FRAME_STATE_KEYS if k in old}})
        else:
            merged.append(frame)
    return merged


def with_counts(project: dict) -> dict:
    frames = project.get("frames", [])
    project["doneCount"] = sum(1 for f in frames if f.get("done"))
    project["sunCount"] = sum(1 for f in frames if f.get("done") and not f.get("skipped"))
    project["totalCount"] = len(frames)
    return project


@app.get("/")
def index():
    from library import TWILIGHT_TIME_LABEL

    return render_template("index.html", twilight_window=TWILIGHT_TIME_LABEL)


@app.get("/api/twilight-candidates")
def api_twilight_candidates():
    return jsonify(twilight_candidates())


@app.get("/api/project")
def api_get_project():
    return jsonify(with_counts(load_project()))


@app.post("/api/project/setup")
def api_setup_project():
    data = request.get_json(force=True)
    base = data.get("basePlate")
    target_time = (data.get("targetTime") or "0930").replace(":", "")
    sun_radius = float(data.get("sunRadius") or 6)
    if not base:
        return jsonify({"error": "basePlate required"}), 400

    project = load_project()
    project.update(
        {
            "basePlate": base,
            "targetTime": target_time,
            "timeBasis": "EST",
            "sunRadius": sun_radius,
            "frames": merge_frames(project.get("frames", []), build_frame_list(target_time)),
        }
    )
    save_project(project)
    return jsonify(with_counts(project))


@app.post("/api/project/autodetect")
def api_autodetect():
    data = request.get_json(silent=True) or {}
    project = load_project()
    if not project.get("frames"):
        return jsonify({"error": "setup project first"}), 400
    stats = autodetect(project["frames"], project["targetTime"], overwrite_manual=bool(data.get("overwriteManual")))
    save_project(project)
    return jsonify({**with_counts(project), "stats": stats})


@app.post("/api/project/annotate")
def api_annotate():
    data = request.get_json(force=True)
    index = int(data["index"])
    project = load_project()
    frames = project.get("frames", [])
    if index < 0 or index >= len(frames):
        return jsonify({"error": "bad index"}), 400

    frame = frames[index]
    if data.get("skipped"):
        frame.update({"x": None, "y": None, "skipped": True, "done": True, "source": "manual", "reason": "skipped"})
    else:
        frame.update(
            {"x": float(data["x"]), "y": float(data["y"]), "skipped": False, "done": True, "source": "manual", "reason": None}
        )

    frames[index] = frame
    project["frames"] = frames
    save_project(project)
    return jsonify({"ok": True, "doneCount": with_counts(project)["doneCount"], "frame": frame})


@app.post("/api/project/reset-frame")
def api_reset_frame():
    data = request.get_json(force=True)
    index = int(data["index"])
    project = load_project()
    frames = project.get("frames", [])
    if index < 0 or index >= len(frames):
        return jsonify({"error": "bad index"}), 400
    frame = frames[index]
    frame.update({"x": None, "y": None, "skipped": False, "done": False, "source": None, "reason": None})
    frames[index] = frame
    project["frames"] = frames
    save_project(project)
    return jsonify({"ok": True, "frame": frame})


@app.get("/api/image/<path:filename>")
def api_image(filename: str):
    path = HALF_HOUR_DIR / Path(filename).name
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    max_width = int(request.args.get("w", 1280))
    img = read_image(path)
    return Response(encode_preview(img, max_width=max_width), mimetype="image/jpeg")


@app.get("/api/image-meta/<path:filename>")
def api_image_meta(filename: str):
    path = HALF_HOUR_DIR / Path(filename).name
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    img = read_image(path)
    h, w = img.shape[:2]
    max_width = int(request.args.get("w", 1280))
    scale = preview_scale(path, max_width)
    return jsonify({"width": w, "height": h, "scale": scale})


@app.post("/api/render")
def api_render():
    data = request.get_json(silent=True) or {}
    project = load_project()
    if not project.get("basePlate"):
        return jsonify({"error": "setup project first"}), 400
    annotated = [f for f in project.get("frames", []) if f.get("done") and not f.get("skipped")]
    if not annotated:
        return jsonify({"error": "no sun positions yet"}), 400

    if data.get("sunRadius"):
        project["sunRadius"] = float(data["sunRadius"])
        save_project(project)
    radius = float(project.get("sunRadius", 6))
    img = render_analemma(project["basePlate"], annotated, sun_radius=radius)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tiff, jpg = save_render(img, OUTPUT_DIR)
    preview = render_analemma(project["basePlate"], annotated, sun_radius=radius, preview_max_width=1600)
    preview_path = OUTPUT_DIR / "Analemma_preview.jpg"
    cv2.imwrite(str(preview_path), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return jsonify(
        {
            "ok": True,
            "count": len(annotated),
            "tiff": tiff.name,
            "jpg": jpg.name,
            "previewUrl": "/api/output/Analemma_preview.jpg",
        }
    )


@app.get("/api/output/<path:filename>")
def api_output(filename: str):
    path = OUTPUT_DIR / Path(filename).name
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    return send_file(path)


if __name__ == "__main__":
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Analemma tool: http://127.0.0.1:5757", flush=True)
    app.run(host="127.0.0.1", port=5757, debug=False)
