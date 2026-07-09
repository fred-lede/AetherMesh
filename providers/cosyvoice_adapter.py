from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
import torchaudio

from providers.tts_base import TTSProviderAdapter, TTSProviderError

try:
    import soundfile as sf
except ImportError:
    sf = None

logger = logging.getLogger("providers.cosyvoice_adapter")


class CosyVoiceAdapter(TTSProviderAdapter):
    provider_name = "cosyvoice"

    def __init__(
        self,
        model_name: str = "FunAudioLLM/CosyVoice2-0.5B",
        device: str = "cuda:0",
        voices_dir: str = "data/voices",
        models_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._voices_dir = Path(voices_dir)
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        self._models_dir = models_dir
        self._model = None
        self._load_error: str | None = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2
            download_root = os.path.abspath(self._models_dir) if self._models_dir else None
            kwargs: dict[str, Any] = {
                "load_jit": False,
                "load_trt": False,
                "fp16": "cuda" in self._device,
            }
            if download_root:
                kwargs["download_root"] = download_root
            self._model = CosyVoice2(self._model_name, **kwargs)
            self._load_error = None
            logger.info("CosyVoice2 model loaded: %s on %s", self._model_name, self._device)
        except Exception as e:
            self._model = None
            self._load_error = str(e)
            logger.exception("Failed to load CosyVoice2 model: %s", e)

    def _ensure_model(self) -> None:
        if self._model is None:
            raise TTSProviderError(
                f"CosyVoice model not loaded: {self._load_error or 'unknown error'}",
                status_code=503,
            )

    def tts(self, payload: dict[str, Any]) -> bytes:
        self._ensure_model()
        text = payload["input"]
        voice_id = payload.get("voice", "")
        language = payload.get("language", "")
        speed = payload.get("speed", 1.0)

        voice_path = self._voices_dir / voice_id
        ref_wav = voice_path / "reference.wav"
        ref_text_path = voice_path / "reference.txt"

        try:
            if ref_wav.exists() and ref_text_path.exists():
                prompt_text = ref_text_path.read_text(encoding="utf-8").strip()
                prompt_speech = self._load_audio(str(ref_wav))
                output = list(self._model.inference_zero_shot(
                    text, prompt_speech, prompt_text,
                ))
            else:
                spk_id = voice_id if voice_id else "default"
                output = list(self._model.inference_sft(text, spk_id=spk_id))

            audio_data = self._cat_audio(output)
            if speed != 1.0:
                audio_data = self._apply_speed(audio_data, speed)
            return audio_data
        except TTSProviderError:
            raise
        except Exception as e:
            raise TTSProviderError(f"CosyVoice TTS failed: {e}", status_code=503) from e

    def _load_audio(self, path: str) -> Any:
        audio, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        return audio.squeeze(0)

    def _cat_audio(self, outputs: list[Any]) -> bytes:
        chunks: list[np.ndarray] = []
        for result in outputs:
            tts_speech = result.get("tts_speech")
            if tts_speech is None:
                continue
            arr = tts_speech.cpu().numpy() if isinstance(tts_speech, torch.Tensor) else tts_speech
            chunks.append(arr)
        if not chunks:
            raise TTSProviderError("CosyVoice produced no audio output", status_code=500)
        combined = np.concatenate(chunks, axis=-1)
        buf = io.BytesIO()
        if sf is not None:
            sf.write(buf, combined, 22050, format="WAV", subtype="PCM_16")
        else:
            from scipy.io.wavfile import write as wav_write
            wav_write(buf, 22050, combined)
        return buf.getvalue()

    def _apply_speed(self, wav_bytes: bytes, speed: float) -> bytes:
        import subprocess
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", "pipe:0", "-filter:a", f"atempo={speed}",
                 "-f", "wav", "pipe:1"],
                input=wav_bytes, capture_output=True, check=True, timeout=30,
            )
            return result.stdout
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return wav_bytes

    def list_voices(self) -> list[dict[str, Any]]:
        voices: list[dict[str, Any]] = []
        for entry in sorted(self._voices_dir.iterdir()):
            meta_path = entry / "meta.json"
            if meta_path.exists():
                voices.append(json.loads(meta_path.read_text()))
        return voices

    def register_voice(
        self,
        name: str,
        audio_data: bytes,
        language: str = "",
        content_type: str = "audio/wav",
    ) -> dict[str, Any]:
        voice_id = str(uuid4())
        vp = self._voices_dir / voice_id
        vp.mkdir(parents=True, exist_ok=True)

        ref_path = vp / "reference.wav"
        ref_path.write_bytes(audio_data)
        duration = self._get_audio_duration(audio_data)

        ref_text_path = vp / "reference.txt"
        ref_text_path.write_text(name, encoding="utf-8")

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

    def delete_voice(self, voice_id: str) -> bool:
        vp = self._voices_dir / voice_id
        if not vp.exists():
            return False
        shutil.rmtree(vp)
        return True

    def health_check(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model_loaded": self._model is not None,
            "device": self._device,
            "voices_count": len(list(self._voices_dir.iterdir())),
        }
