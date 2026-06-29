from __future__ import annotations

import io
import json
import shutil
import subprocess
import tempfile
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


class XTTSAdapter(TTSProviderAdapter):
    provider_name = "xtts"

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cuda:0",
        voices_dir: str = "data/voices",
        models_dir: str | None = None,
    ) -> None:
        self._device = device
        self._voices_dir = Path(voices_dir)
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        self._model = self._load_model(model_name, models_dir)

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

    def _load_model(self, model_name: str, models_dir: str | None) -> Any:
        from TTS.api import TTS
        tts = TTS(model_name=model_name, model_dir=models_dir, progress_bar=False)
        tts.to(self._device)
        return tts

    def _voice_path(self, voice_id: str) -> Path:
        return self._voices_dir / voice_id

    def _load_embedding(self, voice_id: str) -> tuple[Any, Any]:
        vp = self._voice_path(voice_id)
        emb_path = vp / "speaker_embedding.pt"
        if not emb_path.exists():
            raise TTSProviderError(f"Voice {voice_id} not found", status_code=404)
        import torch
        data = torch.load(str(emb_path), map_location=self._device, weights_only=True)
        return data["gpt_cond_latent"], data["speaker_embedding"]

    def tts(self, payload: dict[str, Any]) -> bytes:
        voice_id = payload["voice"]
        text = payload["input"]
        language = payload.get("language", "en")
        speed = payload.get("speed", 1.0)

        gpt_cond, speaker_embed = self._load_embedding(voice_id)
        wav: np.ndarray = self._model.tts(
            text=text,
            gpt_cond_latent=gpt_cond,
            speaker_embedding=speaker_embed,
            language=language,
        )
        buffer = io.BytesIO()
        if sf is not None:
            sf.write(buffer, wav, 24000, format="WAV", subtype="PCM_16")
        else:
            from scipy.io.wavfile import write as wav_write
            wav_write(buffer, 24000, wav)
        audio_bytes = buffer.getvalue()

        if speed != 1.0:
            audio_bytes = self._apply_speed(audio_bytes, speed)

        return audio_bytes

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

        gpt_cond, speaker_embed = self._model.get_conditioning_latents(
            audio_path=str(ref_path)
        )
        import torch
        torch.save(
            {"gpt_cond_latent": gpt_cond, "speaker_embedding": speaker_embed},
            str(vp / "speaker_embedding.pt"),
        )

        duration = self._get_audio_duration(audio_data)
        meta = {
            "voice_id": voice_id,
            "name": name,
            "language": language,
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
                    info = sf.info(str(tmp))
                    return round(info.duration, 2)
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

    def health_check(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model_loaded": self._model is not None,
            "device": self._device,
            "voices_count": len(list(self._voices_dir.iterdir())),
        }
