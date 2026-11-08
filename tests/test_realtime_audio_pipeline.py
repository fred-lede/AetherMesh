from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from runtime.realtime.audio_pipeline import (
    RealtimeAudioPipeline,
    language_label,
    resample_pcm16_24k_to_16k,
)


def _pcm24(samples: int = 2400) -> bytes:
    return np.full(samples, 1000, dtype=np.int16).tobytes()


def _silence24(samples: int = 24_000) -> bytes:
    return np.zeros(samples, dtype="<i2").tobytes()


def _model(text: str = "hello world") -> MagicMock:
    model = MagicMock()
    model.transcribe.return_value = ([SimpleNamespace(text=text)], SimpleNamespace(language="en"))
    return model


def test_resample_pcm16_24k_to_16k_converts_duration_without_changing_level():
    result = resample_pcm16_24k_to_16k(_pcm24())
    samples = np.frombuffer(result, dtype=np.int16)
    assert len(samples) == 1600
    assert np.all(samples == 1000)


@pytest.mark.asyncio
async def test_audio_chunk_emits_source_and_translated_text_deltas():
    model = _model()
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好世界"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    events = await pipeline.add_audio(_pcm24())
    assert [e["type"] for e in events] == [
        "input_audio_buffer.speech_started",
        "session.input_transcript.delta",
        "session.output_transcript.delta",
    ]
    assert events[1]["delta"] == "hello world" and events[1]["text"] == "hello world"
    assert events[2]["delta"] == "你好世界" and events[2]["text"] == "你好世界"
    assert service.handle_chat.call_args.args[0]["model"] == "local-chat-model"
    assert "zh" in service.handle_chat.call_args.args[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_close_flushes_audio_below_the_streaming_window():
    model = _model("last words")
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "最後一句"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=2.5,
    )
    await pipeline.add_audio(_pcm24(4800))
    events = await pipeline.close()
    assert [e["type"] for e in events] == [
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
        "session.input_transcript.delta",
        "session.output_transcript.delta",
    ]
    assert events[2]["delta"] == "last words" and events[2]["text"] == "last words"
    assert events[3]["delta"] == "最後一句" and events[3]["text"] == "最後一句"


@pytest.mark.asyncio
async def test_vad_turn_frames_carry_item_ids_in_order():
    model = _model()
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=2.5,
    )
    events = await pipeline.add_audio(_pcm24(4800) + _silence24(24_000))
    assert [e["type"] for e in events] == [
        "input_audio_buffer.speech_started",
        "input_audio_buffer.speech_stopped",
        "input_audio_buffer.committed",
    ]
    assert events[0]["item_id"] == "item_in_1"
    assert events[1]["item_id"] == "item_in_1"
    assert events[2]["item_id"] == "item_in_1"
    assert events[2]["previous_item_id"] is None


@pytest.mark.asyncio
async def test_vad_item_ids_increase_across_turns():
    model = _model()
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=2.5,
    )
    await pipeline.add_audio(_pcm24(4800) + _silence24(24_000))
    events = await pipeline.add_audio(_pcm24(4800) + _silence24(24_000))
    started = [e for e in events if e["type"] == "input_audio_buffer.speech_started"]
    assert [e["item_id"] for e in started] == ["item_in_2"]


@pytest.mark.asyncio
async def test_silent_audio_emits_no_speech_frames():
    model = _model()
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=2.5,
    )
    events = await pipeline.add_audio(_silence24(24_000))
    assert not any(e["type"].startswith("input_audio_buffer.speech") for e in events)
    events = await pipeline.close()
    assert not any(e["type"] in {"input_audio_buffer.speech_stopped", "input_audio_buffer.committed"} for e in events)


@pytest.mark.asyncio
async def test_flush_idle_emits_transcripts_when_audio_stops():
    model = _model("turn ended")
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "結束了"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=2.5,
        idle_timeout=0.05,
    )
    await pipeline.add_audio(_pcm24(4800))
    await asyncio.sleep(0.08)
    events = await pipeline.flush_idle()
    assert [e["type"] for e in events] == [
        "session.input_transcript.delta",
        "session.output_transcript.delta",
    ]
    assert events[0]["text"] == "turn ended"
    assert events[0]["is_final"] is True


@pytest.mark.asyncio
async def test_flush_idle_is_a_noop_when_audio_is_still_fresh():
    model = _model()
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=2.5,
        idle_timeout=0.05,
    )
    await pipeline.add_audio(_pcm24(4800))
    events = await pipeline.flush_idle()
    assert events == []
    service.handle_chat.assert_not_called()


@pytest.mark.asyncio
async def test_translation_without_choices_preserves_source_text_and_emits_error():
    service = MagicMock()
    service.handle_chat.return_value = {"choices": []}
    pipeline = RealtimeAudioPipeline(
        asr_model=_model(),
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    events = await pipeline.add_audio(_pcm24())
    assert events[1]["type"] == "session.input_transcript.delta"
    assert events[1]["delta"] == "hello world" and events[1]["text"] == "hello world"
    assert events[2]["type"] == "error"
    assert events[2]["error"]["message"] == "translation returned no choices"


@pytest.mark.asyncio
async def test_translation_provider_failure_preserves_source_and_emits_error():
    service = MagicMock()
    service.handle_chat.side_effect = RuntimeError("provider unavailable")
    pipeline = RealtimeAudioPipeline(
        asr_model=_model(),
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    events = await pipeline.add_audio(_pcm24())
    assert events[1]["type"] == "session.input_transcript.delta"
    assert events[2]["type"] == "error"
    assert events[2]["error"]["message"] == "provider unavailable"


@pytest.mark.asyncio
async def test_source_delta_is_emitted_before_translation_and_survives_failure():
    emitted = []

    async def emit(event):
        emitted.append(event)

    service = MagicMock()

    def fail_translation(_payload):
        assert any(e["type"] == "session.input_transcript.delta" for e in emitted)
        raise RuntimeError("provider unavailable")

    service.handle_chat.side_effect = fail_translation
    pipeline = RealtimeAudioPipeline(
        asr_model=_model(),
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    await pipeline.add_audio(_pcm24(), emit=emit)
    transcript_events = [e for e in emitted if e["type"] == "session.input_transcript.delta"]
    assert transcript_events[0]["delta"] == "hello world"
    assert emitted[-1]["type"] == "error"
    assert emitted.index(transcript_events[0]) < emitted.index(emitted[-1])


@pytest.mark.asyncio
async def test_transcript_delta_events_carry_text_and_seg_fields():
    model = _model("hello there")
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好世界"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    events = await pipeline.add_audio(_pcm24())
    delta_events = [e for e in events if e["type"] in ("session.input_transcript.delta", "session.output_transcript.delta")]
    assert delta_events, "expected transcript delta events"
    for event in delta_events:
        assert "text" in event, f"missing text field in {event['type']}"
        assert "seg" in event, f"missing seg field in {event['type']}"
        assert "is_final" in event, f"missing is_final field in {event['type']}"
        assert event["text"] == event["delta"]
    assert all(e["is_final"] is False for e in delta_events)


@pytest.mark.asyncio
async def test_close_emits_is_final_true_on_transcript_deltas():
    model = _model("last words")
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "最後一句"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=2.5,
    )
    await pipeline.add_audio(_pcm24(4800))
    events = await pipeline.close()
    delta_events = [e for e in events if e["type"] in ("session.input_transcript.delta", "session.output_transcript.delta")]
    assert delta_events, "expected transcript delta events from close"
    for event in delta_events:
        assert event["is_final"] is True, f"expected is_final=True on close batch, got {event}"


@pytest.mark.asyncio
async def test_instructions_override_translation_prompt():
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "hi"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=_model(),
        service=service,
        model="local-chat-model",
        target_language="",
        instructions="Translate the speech into English. Return only the translation.",
        window_seconds=0.1,
    )
    await pipeline.add_audio(_pcm24())
    payload = service.handle_chat.call_args.args[0]
    assert payload["messages"][0] == {
        "role": "system",
        "content": "Translate the speech into English. Return only the translation.",
    }
    # The segment is labelled, the same shape the local pipeline uses, so the model always
    # knows which part is the segment and which part is context.
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"].strip().endswith("hello world")


@pytest.mark.asyncio
async def test_transcript_events_do_not_append_trailing_whitespace():
    model = _model("hello world")
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    events = await pipeline.add_audio(_pcm24())
    for event in events:
        if event["type"] in ("session.input_transcript.delta", "session.output_transcript.delta"):
            assert not event["delta"].endswith(" "), f"delta must not end with space: {event['delta']!r}"
            assert not event["text"].endswith(" "), f"text must not end with space: {event['text']!r}"


# --- the translation prompt: a language name, the current segment, context, terminology ---


@pytest.mark.asyncio
async def test_translation_prompt_names_the_target_language_in_words():
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=_model(),
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    await pipeline.add_audio(_pcm24())
    prompt = service.handle_chat.call_args.args[0]["messages"][0]["content"]
    assert "Chinese" in prompt, f"the model must be told the language in words: {prompt}"
    assert "into zh." not in prompt, f"a bare code is too thin a target: {prompt}"


@pytest.mark.asyncio
async def test_translation_prompt_asks_for_the_current_segment_only():
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=_model(),
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    await pipeline.add_audio(_pcm24())
    prompt = service.handle_chat.call_args.args[0]["messages"][0]["content"]
    assert "current segment" in prompt, f"the prompt must scope the work: {prompt}"


@pytest.mark.asyncio
async def test_recent_source_segments_reach_the_next_translation_as_context():
    model = MagicMock()
    model.transcribe.side_effect = [
        ([SimpleNamespace(text="first thought")], SimpleNamespace(language="en")),
        ([SimpleNamespace(text="second thought")], SimpleNamespace(language="en")),
    ]
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=model,
        service=service,
        model="local-chat-model",
        target_language="zh",
        window_seconds=0.1,
    )
    await pipeline.add_audio(_pcm24())
    await pipeline.add_audio(_pcm24())
    user_text = service.handle_chat.call_args_list[1].args[0]["messages"][1]["content"]
    assert "first thought" in user_text, f"the previous segment is context: {user_text}"
    assert "second thought" in user_text, f"the current segment must be there too: {user_text}"
    assert user_text.index("first thought") < user_text.index("second thought")


@pytest.mark.asyncio
async def test_terminology_terms_reach_the_translation_prompt():
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=_model(),
        service=service,
        model="local-chat-model",
        target_language="zh",
        terms={"MIT": "麻省理工學院"},
        window_seconds=0.1,
    )
    await pipeline.add_audio(_pcm24())
    user_text = service.handle_chat.call_args.args[0]["messages"][1]["content"]
    assert "MIT => 麻省理工學院" in user_text, f"terminology must reach the model: {user_text}"


@pytest.mark.asyncio
async def test_terminology_and_context_survive_client_instructions():
    service = MagicMock()
    service.handle_chat.return_value = {"choices": [{"message": {"content": "你好"}}]}
    pipeline = RealtimeAudioPipeline(
        asr_model=_model(),
        service=service,
        model="local-chat-model",
        target_language="zh",
        instructions="Translate the speech into English. Return only the translation.",
        terms={"MIT": "麻省理工學院"},
        window_seconds=0.1,
    )
    await pipeline.add_audio(_pcm24())
    messages = service.handle_chat.call_args.args[0]["messages"]
    assert messages[0] == {
        "role": "system",
        "content": "Translate the speech into English. Return only the translation.",
    }
    assert "MIT => 麻省理工學院" in messages[1]["content"]


def test_every_language_a_client_offers_is_named_in_words():
    """A client may ask for any of the languages it lists; each needs a name, not a code."""
    codes = [
        "zh", "en", "ja", "ko", "fr", "de", "es", "pt", "ru", "it", "ar", "id", "th", "vi", "tr",
        "hi", "ms", "nl", "ur", "nb", "sv", "da", "he", "fi", "pl", "is", "cs", "fil", "fa",
        "yue", "el", "af", "ast", "az", "be", "bg", "bn", "bs", "ca", "ceb", "et", "gl", "gu",
        "hr", "hu", "jv", "kk", "kn", "ky", "lv", "mk", "ml", "mr", "pa", "ro", "sk", "sl",
        "sw", "tg", "uk",
    ]
    for code in codes:
        label = language_label(code)
        assert label != code, f"{code} has no name of its own: {label}"
        assert "Chinese" not in label or code == "zh", f"{code} must not borrow another language's name"
