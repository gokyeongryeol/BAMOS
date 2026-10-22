"""Open-set detection server: Falcon Perception behind the Isaac server's framing (8-byte big-endian header
length, JSON header, 8-byte payload length, npz payload). Commands: ping | detect | shutdown.

Runs in the detector's own virtualenv (torch >= 2.11), NOT in the project env:

    cd $FALCON_DIR && uv run python \
        <this repo>/servers/detector/falcon_server.py --port 5600 --device cuda:0

($FALCON_DIR: a checkout of Falcon Perception with its environment; scripts/isaac/launch_servers.sh starts the servers.)

`detect` takes {"query": str} and an `rgb` uint8 [H, W, 3] array and returns `masks` uint8 [n, H, W] at the
image resolution, one per instance the model segmented, in the model's output order. The server holds one
model and serves several clients (one per experiment worker); inference is serialized by a lock.
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
from pycocotools import mask as mask_utils

from falcon_perception import PERCEPTION_MODEL_ID, build_prompt_for_task, load_and_prepare_model, setup_torch_config
from falcon_perception.data import ImageProcessor
from falcon_perception.paged_inference import PagedInferenceEngine, SamplingParams, Sequence

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5600)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--task", default="segmentation")
args = ap.parse_args()

torch.cuda.set_device(args.device)  # Triton launches on the current device; the model lives on args.device
setup_torch_config()
# eager path: on this build the compiled path returned no masks on frames the eager path segments correctly
model, tokenizer, _ = load_and_prepare_model(hf_model_id=PERCEPTION_MODEL_ID, hf_revision="main", hf_local_dir=None,
                                             device=args.device, dtype="float32", compile=False)
engine = PagedInferenceEngine(model, tokenizer, ImageProcessor(patch_size=16, merge_size=1), max_batch_size=2,
                              max_seq_length=8192, n_pages=128, page_size=128, prefill_length_limit=8192,
                              enable_hr_cache=False, capture_cudagraph=False)
sampling = SamplingParams(stop_token_ids=[tokenizer.eos_token_id, tokenizer.end_of_query_token_id])
lock = threading.Lock()
n_detect = 0


def decode(rle, h: int, w: int) -> np.ndarray:
    m = mask_utils.decode(rle)
    if m.shape != (h, w):  # the model works at its own resolution
        import cv2

        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m.astype(np.uint8)


def detect(rgb: np.ndarray, query: str) -> np.ndarray:
    global n_detect
    h, w = rgb.shape[:2]
    seq = Sequence(text=build_prompt_for_task(query, args.task), image=Image.fromarray(rgb).convert("RGB"),
                   min_image_size=256, max_image_size=1024, task=args.task)
    # the current CUDA device is per thread, and Triton launches on it: each handler thread must set it
    with lock, torch.cuda.device(args.device), torch.inference_mode():
        engine.generate([seq], sampling_params=sampling, use_tqdm=False, print_stats=False)
        n_detect += 1
    masks = [decode(m, h, w) for m in (seq.output_aux.masks_rle or [])]
    return np.stack(masks) if masks else np.zeros((0, h, w), np.uint8)


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
                        self.send({"ok": True, "model": PERCEPTION_MODEL_ID, "task": args.task, "n_detect": n_detect})
                    elif cmd == "detect":
                        masks = detect(np.asarray(arrays["rgb"], dtype=np.uint8), str(header["query"]))
                        self.send({"ok": True, "n_masks": int(len(masks)), "ms": (time.time() - t0) * 1000}, {"masks": masks})
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
            print(f"[falcon_server] connection ended: {e}", flush=True)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


with Server(("127.0.0.1", args.port), Handler) as srv:
    print(f"[falcon_server] ready on {args.port} ({args.device})", flush=True)
    srv.serve_forever()
print("[falcon_server] shutting down", flush=True)
