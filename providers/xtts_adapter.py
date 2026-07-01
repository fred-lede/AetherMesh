from __future__ import annotations

import io
import json
import logging
import multiprocessing
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from providers.tts_base import TTSProviderAdapter, TTSProviderError

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore[assignment]

try:
    from langdetect import detect as _detect_lang
except ImportError:
    _detect_lang = None  # type: ignore[assignment]

logger = logging.getLogger("providers.xtts_adapter")

_LANG_MAP = {
    "zh": "zh-cn",
    "zh-cn": "zh-cn",
    "zh-tw": "zh-cn",
    "en": "en", "es": "es", "fr": "fr", "de": "de",
    "it": "it", "pt": "pt", "pl": "pl", "tr": "tr",
    "ru": "ru", "nl": "nl", "cs": "cs", "ar": "ar",
    "ja": "ja", "hu": "hu", "ko": "ko", "hi": "hi",
}


def _worker_main(
    request_queue: multiprocessing.Queue,
    response_queue: multiprocessing.Queue,
    model_name: str,
    device: str,
    voices_dir: str,
    models_dir: str | None,
    dtype: str,
) -> None:
    """Run in subprocess: load TTS model, process tts/register requests."""
    import os
    import io as _io

    os.environ.setdefault("COQUI_TOS_AGREED", "1")
    if models_dir:
        os.environ["TTS_HOME"] = os.path.abspath(models_dir)

    try:
        import soundfile as _sf
    except ImportError:
        _sf = None

    import transformers.pytorch_utils as _tpu
    if not hasattr(_tpu, "isin_mps_friendly"):
        import torch as _torch_module
        def _isin_mps_friendly(elements, test_elements):
            return _torch_module.isin(elements, test_elements)
        _tpu.isin_mps_friendly = _isin_mps_friendly

    if _sf is not None:
        try:
            import torchaudio as _ta_module
            _orig_load = getattr(_ta_module, "load", None)
            if _orig_load is not None:
                def _safe_load(uri, *args, **kwargs):
                    try:
                        return _orig_load(uri, *args, **kwargs)
                    except Exception:
                        import torch as _torch_inner
                        data, sr = _sf.read(uri, dtype="float32")
                        if data.ndim == 1:
                            data = data.reshape(1, -1)
                        else:
                            data = data.T
                        return _torch_inner.from_numpy(data), sr
                _ta_module.load = _safe_load
        except Exception:
            pass

    import numpy as _np

    from TTS.api import TTS
    tts = TTS(model_name=model_name, progress_bar=False)
    tts.to(device)
    if dtype == "fp16":
        tts.half()

    # warm-up inference
    voice_entries = [p for p in Path(voices_dir).iterdir() if (p / "meta.json").exists()]
    if voice_entries:
        try:
            import torch as _torch_warm
            warm_id = voice_entries[0].name
            emb_path = Path(voices_dir) / warm_id / "speaker_embedding.pt"
            if emb_path.exists():
                w_data = _torch_warm.load(str(emb_path), map_location=device, weights_only=True)
                w_gpt = w_data["gpt_cond_latent"]
                w_spk = w_data["speaker_embedding"]
                if dtype == "fp16":
                    w_gpt = w_gpt.half()
                    w_spk = w_spk.half()
                with _torch_warm.no_grad():
                    if "cuda" in device:
                        with _torch_warm.autocast(device_type="cuda", dtype=_torch_warm.float16):
                            tts.synthesizer.tts_model.inference(
                                text="Hello", gpt_cond_latent=w_gpt,
                                speaker_embedding=w_spk, language="en", speed=1.0,
                            )
                    else:
                        tts.synthesizer.tts_model.inference(
                            text="Hello", gpt_cond_latent=w_gpt,
                            speaker_embedding=w_spk, language="en", speed=1.0,
                        )
        except Exception:
            pass

    import torch as _torch

    while True:
        req = request_queue.get()
        req_type = req.get("type", "tts")
        try:
            if req_type == "tts":
                text = req["text"]
                voice_id = req["voice_id"]
                language = req.get("language", "en")
                speed = req.get("speed", 1.0)

                vp = Path(voices_dir) / voice_id
                emb_path = vp / "speaker_embedding.pt"
                data = _torch.load(str(emb_path), map_location=device, weights_only=True)
                gpt_cond = data["gpt_cond_latent"]
                speaker_embed = data["speaker_embedding"]
                if dtype == "fp16":
                    gpt_cond = gpt_cond.half()
                    speaker_embed = speaker_embed.half()

                with _torch.no_grad():
                    if "cuda" in device:
                        with _torch.autocast(device_type="cuda", dtype=_torch.float16):
                            output = tts.synthesizer.tts_model.inference(
                                text=text, gpt_cond_latent=gpt_cond,
                                speaker_embedding=speaker_embed,
                                language=language, speed=speed,
                            )
                    else:
                        output = tts.synthesizer.tts_model.inference(
                            text=text, gpt_cond_latent=gpt_cond,
                            speaker_embedding=speaker_embed,
                            language=language, speed=speed,
                        )

                wav = output["wav"]
                if wav.dtype == _np.float16:
                    wav = wav.astype(_np.float32)
                buf = _io.BytesIO()
                if _sf is not None:
                    _sf.write(buf, wav, 24000, format="WAV", subtype="PCM_16")
                else:
                    from scipy.io.wavfile import write as _wav_write
                    _wav_write(buf, 24000, wav)
                response_queue.put(("ok", buf.getvalue()))

            elif req_type == "register":
                audio_path = req["audio_path"]
                embedding_path = req["embedding_path"]

                was_fp16 = dtype == "fp16"
                if was_fp16:
                    _mods = [tts.synthesizer.tts_model]
                    if hasattr(tts.synthesizer, "voice_encoder") and tts.synthesizer.voice_encoder is not None:
                        _mods.append(tts.synthesizer.voice_encoder)
                    if hasattr(tts.synthesizer, "vocoder_model") and tts.synthesizer.vocoder_model is not None:
                        _mods.append(tts.synthesizer.vocoder_model)
                    for _m in _mods:
                        _m.float()

                try:
                    with _torch.no_grad():
                        gpt_cond, speaker_embed = tts.synthesizer.tts_model.get_conditioning_latents(
                            audio_path=str(audio_path),
                        )
                finally:
                    if was_fp16:
                        for _m in _mods:
                            _m.half()

                gpt_cond = gpt_cond.float()
                speaker_embed = speaker_embed.float()

                has_nan = _torch.isnan(gpt_cond).any().item() or _torch.isnan(speaker_embed).any().item()
                if has_nan:
                    gpt_cond = _torch.where(
                        _torch.isnan(gpt_cond), _torch.zeros_like(gpt_cond), gpt_cond,
                    )
                    speaker_embed = _torch.where(
                        _torch.isnan(speaker_embed), _torch.zeros_like(speaker_embed), speaker_embed,
                    )

                _torch.save(
                    {"gpt_cond_latent": gpt_cond, "speaker_embedding": speaker_embed},
                    str(embedding_path),
                )
                response_queue.put(("ok",))

            else:
                response_queue.put(("error", f"Unknown request type: {req_type}"))

        except Exception as e:
            response_queue.put(("error", str(e)))
            if "cuda" in str(type(e).__name__).lower() or "cuda" in str(e).lower() or "assert" in str(e).lower():
                raise  # crash process -> triggers adapter restart


class XTTSAdapter(TTSProviderAdapter):
    provider_name = "xtts"

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cuda:0",
        voices_dir: str = "data/voices",
        models_dir: str | None = None,
        dtype: str = "fp32",
        max_ref_seconds: float = 10.0,
    ) -> None:
        self._device = device
        self._dtype = dtype
        self._model_name = model_name
        self._models_dir = models_dir
        self._max_ref_seconds = max_ref_seconds
        self._voices_dir = Path(voices_dir)
        self._voices_dir.mkdir(parents=True, exist_ok=True)

        self._process: multiprocessing.Process | None = None
        self._request_queue: multiprocessing.Queue | None = None
        self._response_queue: multiprocessing.Queue | None = None
        self._worker_last_failure: float = 0.0
        self._worker_cooldown: float = 30.0

        from runtime.gpu.vram_scheduler import vram_scheduler, GPUResource, VRAMProfile
        vram_scheduler.update_gpu(GPUResource(
            gpu_id="xtts-v2",
            node_id="localhost",
            worker_port=0,
            vram=VRAMProfile(total_mb=5120, used_mb=0, free_mb=5120),
            model_loaded="xtts-v2",
            queue_depth=0,
            healthy=True,
            tier=0,
            metadata={"type": "tts", "provider": "xtts"},
        ))

    def _voice_path(self, voice_id: str) -> Path:
        return self._voices_dir / voice_id

    def _ensure_worker(self) -> None:
        now = time.monotonic()
        if now - self._worker_last_failure < self._worker_cooldown:
            remaining = self._worker_cooldown - (now - self._worker_last_failure)
            raise TTSProviderError(
                f"TTS worker in cooldown ({remaining:.0f}s remaining)",
                status_code=503,
            )
        if self._process is not None and self._process.is_alive():
            return
        self._start_worker()
        self._worker_last_failure = 0.0

    def _start_worker(self) -> None:
        ctx = multiprocessing.get_context("spawn")
        self._request_queue = ctx.Queue()
        self._response_queue = ctx.Queue()
        self._process = ctx.Process(
            target=_worker_main,
            args=(
                self._request_queue,
                self._response_queue,
                self._model_name,
                self._device,
                str(self._voices_dir),
                self._models_dir,
                self._dtype,
            ),
            daemon=True,
        )
        self._process.start()
        time.sleep(0.5)
        if not self._process.is_alive():
            self._worker_last_failure = time.monotonic()
            self._process = None
            self._request_queue = None
            self._response_queue = None
            raise TTSProviderError(
                "TTS worker process died during startup (CUDA crash?)",
                status_code=503,
            )

    def _send_worker_request(self, request: dict[str, Any], timeout: float = 120.0) -> Any:
        self._ensure_worker()
        try:
            self._request_queue.put(request, timeout=10)
        except Exception as e:
            raise TTSProviderError(f"TTS worker request failed: {e}", status_code=503) from e
        try:
            result = self._response_queue.get(timeout=timeout)
        except Exception as e:
            if self._process is not None and not self._process.is_alive():
                self._worker_last_failure = time.monotonic()
                self._process = None
                self._request_queue = None
                self._response_queue = None
                raise TTSProviderError("TTS worker process died (CUDA crash?)", status_code=503) from e
            raise TTSProviderError(f"TTS worker response timeout: {e}", status_code=504) from e
        if not isinstance(result, tuple):
            result = (result,)
        status = result[0]
        rest = result[1:]
        if status == "error":
            msg = rest[0] if rest else "unknown error"
            raise TTSProviderError(f"TTS worker error: {msg}", status_code=500)
        return rest[0] if rest else None

    def tts(self, payload: dict[str, Any]) -> bytes:
        voice_id = payload["voice"]
        text = payload["input"]
        language = payload.get("language")
        if not language:
            try:
                meta = json.loads((self._voice_path(voice_id) / "meta.json").read_text())
                language = meta.get("language") or ""
            except Exception:
                pass
        if not language and _detect_lang is not None:
            try:
                detected = _detect_lang(text)
                language = _LANG_MAP.get(detected)
            except Exception:
                language = "en"
        language = language or "en"
        if language not in _LANG_MAP.values():
            language = "en"
        speed = payload.get("speed", 1.0)

        wav_bytes = self._send_worker_request({
            "type": "tts",
            "text": text,
            "voice_id": voice_id,
            "language": language,
            "speed": speed,
        })
        if speed != 1.0:
            wav_bytes = self._apply_speed(wav_bytes, speed)
        return wav_bytes

    def _apply_speed(self, wav_bytes: bytes, speed: float) -> bytes:
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", "pipe:0", "-filter:a", f"atempo={speed}",
                 "-f", "wav", "pipe:1"],
                input=wav_bytes, capture_output=True, check=True, timeout=30,
            )
            return result.stdout
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return wav_bytes

    def _trim_reference(self, ref_path: Path) -> float:
        duration = 0.0
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "format=duration", "-of",
                 "default=noprint_wrappers=1:nokey=1", str(ref_path)],
                capture_output=True, text=True, check=True, timeout=10,
            )
            duration = round(float(result.stdout.strip()), 2)
        except Exception:
            if sf is not None:
                try:
                    info = sf.info(str(ref_path))
                    duration = round(info.duration, 2)
                except Exception:
                    pass
        trimmed = ref_path.with_suffix(".trimmed.wav")
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(ref_path),
                "-acodec", "pcm_s16le", "-f", "wav",
            ]
            if duration > self._max_ref_seconds:
                cmd.extend(["-t", str(self._max_ref_seconds)])
                duration = self._max_ref_seconds
            cmd.append(str(trimmed))
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            ref_path.unlink()
            trimmed.rename(ref_path)
        except Exception:
            trimmed.unlink(missing_ok=True)
        return round(duration, 2)

    def register_voice(
        self,
        name: str,
        audio_data: bytes,
        language: str = "",
        content_type: str = "audio/wav",
    ) -> dict[str, Any]:
        voice_id = str(uuid4())
        vp = self._voice_path(voice_id)
        vp.mkdir(parents=True, exist_ok=True)

        ref_path = vp / "reference.wav"
        ref_path.write_bytes(audio_data)
        duration = self._trim_reference(ref_path)
        emb_path = vp / "speaker_embedding.pt"

        try:
            self._send_worker_request({
                "type": "register",
                "audio_path": str(ref_path),
                "embedding_path": str(emb_path),
            })
        except Exception as e:
            shutil.rmtree(vp, ignore_errors=True)
            raise TTSProviderError(f"Voice encoding failed: {e}", status_code=422) from e

        meta = {
            "voice_id": voice_id,
            "name": name,
            "language": language or "",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": duration,
        }
        (vp / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta

    def _get_audio_duration(self, audio_data: bytes) -> float:
        tmp = Path(tempfile.mktemp(suffix=".wav"))
        try:
            tmp.write_bytes(audio_data)
            try:
                result = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries",
                     "format=duration", "-of",
                     "default=noprint_wrappers=1:nokey=1", str(tmp)],
                    capture_output=True, text=True, check=True, timeout=10,
                )
                return round(float(result.stdout.strip()), 2)
            except Exception:
                if sf is not None:
                    try:
                        info = sf.info(str(tmp))
                        return round(info.duration, 2)
                    except Exception:
                        pass
                return 0.0
        finally:
            tmp.unlink(missing_ok=True)

    def list_voices(self) -> list[dict[str, Any]]:
        voices: list[dict[str, Any]] = []
        for entry in sorted(self._voices_dir.iterdir()):
            meta_path = entry / "meta.json"
            if meta_path.exists():
                voices.append(json.loads(meta_path.read_text()))
        return voices

    def delete_voice(self, voice_id: str) -> bool:
        vp = self._voice_path(voice_id)
        if not vp.exists():
            return False
        shutil.rmtree(vp)
        return True

    def update_voice(self, voice_id: str, name: str | None = None, language: str | None = None) -> dict[str, Any]:
        vp = self._voice_path(voice_id)
        meta_path = vp / "meta.json"
        if not meta_path.exists():
            raise TTSProviderError(f"Voice {voice_id} not found", status_code=404)
        meta = json.loads(meta_path.read_text())
        if name is not None:
            meta["name"] = name
        if language is not None:
            meta["language"] = language
        meta_path.write_text(json.dumps(meta, indent=2))
        return meta

    def voice_ref_path(self, voice_id: str) -> Path:
        vp = self._voice_path(voice_id)
        if not vp.exists():
            raise TTSProviderError(f"Voice {voice_id} not found", status_code=404)
        return vp / "reference.wav"

    def health_check(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "device": self._device,
            "worker_alive": self._process is not None and self._process.is_alive(),
            "voices_count": len(list(self._voices_dir.iterdir())),
        }
