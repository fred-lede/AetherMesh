from __future__ import annotations

import base64
import json
import sys
from typing import Any

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _run() -> None:
    init = json.loads(sys.stdin.readline())
    kwargs: dict[str, Any] = {
        "dtype": getattr(torch, init["dtype"]),
        "device_map": init["device_map"],
    }
    if init.get("cache_dir"):
        kwargs["cache_dir"] = init["cache_dir"]
    model = Qwen3ASRModel.from_pretrained(init["model"], **kwargs)
    _emit({"ok": True})
    for line in sys.stdin:
        try:
            job = json.loads(line)
            samples = np.frombuffer(base64.b64decode(job["samples"]), dtype=np.float32)
            results = model.transcribe(
                audio=(samples, 16000),
                language=job.get("language"),
                return_time_stamps=False,
            )
            _emit(
                {
                    "ok": True,
                    "segments": [
                        {
                            "text": str(getattr(item, "text", "") or ""),
                            "language": str(getattr(item, "language", "") or ""),
                        }
                        for item in results
                    ],
                }
            )
        except Exception as exc:
            _emit({"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)})


if __name__ == "__main__":
    _run()
