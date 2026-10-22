"""Open-set detection server: SAM 3 (Meta, SAM License) behind the same framing as servers/falcon_server.py
(8-byte big-endian header length, JSON header, 8-byte payload length, npz payload). Commands: ping | detect | shutdown.

Runs in its own virtualenv ($SAM3_DIR/.venv: torch + transformers), from the local Hugging Face cache:

    cd $SAM3_DIR && HF_HUB_OFFLINE=1 .venv/bin/python <this repo>/servers/sam3_server.py --port 5610 --device cuda:0

`detect` takes {"query": str} and an `rgb` uint8 [H, W, 3] array and returns `masks` uint8 [n, H, W] at the image
resolution, one per instance whose score passes --threshold (SAM 3's own operating point, 0.5 by default), sorted
by score, highest first, and `scores` float32 [n]. The planner-side selection rule (open_set_mask_select) is the
same as for Falcon; with this order its "first" arm picks the most confident instance.
"""
from __future__ import annotations

import argparse
import io
import json
import socketserver
import struct
import threading
import time
import traceback

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

MODEL_ID = "facebook/sam3"

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5610)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--threshold", type=float, default=0.5, help="instance score threshold")
ap.add_argument("--mask-threshold", type=float, default=0.5)
args = ap.parse_args()

processor = Sam3Processor.from_pretrained(MODEL_ID)
model = Sam3Model.from_pretrained(MODEL_ID).to(args.device).eval()
lock = threading.Lock()
n_detect = 0


def detect(rgb: np.ndarray, query: str) -> tuple[np.ndarray, np.ndarray]:
    global n_detect
    h, w = rgb.shape[:2]
    img = Image.fromarray(rgb).convert("RGB")
    with lock, torch.cuda.device(args.device), torch.inference_mode():
        inp = processor(images=img, text=query, return_tensors="pt").to(args.device)
        out = model(**inp)
        res = processor.post_process_instance_segmentation(
            out, threshold=args.threshold, mask_threshold=args.mask_threshold,
            target_sizes=inp.get("original_sizes").tolist())[0]
        n_detect += 1
    scores = res["scores"].float().cpu().numpy()
    if not len(scores):
        return np.zeros((0, h, w), np.uint8), np.zeros(0, np.float32)
    masks = res["masks"].cpu().numpy().astype(np.uint8)
    order = np.argsort(-scores, kind="stable")
    return masks[order], scores[order].astype(np.float32)


def recv_exact(f, n: int) -> bytes:
    buf = f.read(n)
    if len(buf) < n:
        raise ConnectionError("client closed")
    return buf


class Handler(socketserver.StreamRequestHandler):
    def send(self, header: dict, arrays: dict | None = None):
        payload = b""
        if arrays:
            bio = io.BytesIO()
            np.savez(bio, **arrays)
            payload = bio.getvalue()
        hb = json.dumps(header).encode()
        self.wfile.write(struct.pack(">Q", len(hb)) + hb + struct.pack(">Q", len(payload)) + payload)
        self.wfile.flush()

    def handle(self):
        try:
            while True:
                (hl,) = struct.unpack(">Q", recv_exact(self.rfile, 8))
                header = json.loads(recv_exact(self.rfile, hl).decode())
                (pl,) = struct.unpack(">Q", recv_exact(self.rfile, 8))
                arrays = dict(np.load(io.BytesIO(recv_exact(self.rfile, pl)))) if pl else {}
                cmd, t0 = header.get("cmd"), time.time()
                try:
                    if cmd == "ping":
                        self.send({"ok": True, "model": MODEL_ID, "threshold": args.threshold, "n_detect": n_detect})
                    elif cmd == "detect":
                        masks, scores = detect(np.asarray(arrays["rgb"], dtype=np.uint8), str(header["query"]))
                        self.send({"ok": True, "n_masks": int(len(masks)), "ms": (time.time() - t0) * 1000},
                                  {"masks": masks, "scores": scores})
                    elif cmd == "shutdown":
                        self.send({"ok": True})
                        threading.Thread(target=self.server.shutdown, daemon=True).start()
                        return
                    else:
                        self.send({"ok": False, "error": f"unknown cmd {cmd}"})
                except Exception as e:
                    traceback.print_exc()
                    self.send({"ok": False, "error": repr(e)})
        except (ConnectionError, OSError) as e:
            print(f"[sam3_server] connection ended: {e}", flush=True)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


with Server(("127.0.0.1", args.port), Handler) as srv:
    print(f"[sam3_server] ready on {args.port} ({args.device}, threshold {args.threshold})", flush=True)
    srv.serve_forever()
print("[sam3_server] shutting down", flush=True)
