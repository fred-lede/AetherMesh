from __future__ import annotations

from providers.registry import (
    CANONICAL_CAPABILITIES,
    CAPABILITY_ALIASES,
    Capability,
    parse_capabilities,
)


def test_image_alias_maps_to_image_gen():
    assert CAPABILITY_ALIASES["image"] is Capability.IMAGE_GEN


def test_image_gen_and_video_are_canonical():
    assert Capability.IMAGE_GEN.value == "image_gen"
    assert Capability.VIDEO.value == "video"
    assert "image_gen" in CANONICAL_CAPABILITIES
    assert "video" in CANONICAL_CAPABILITIES


def test_parse_capabilities_handles_new_values():
    assert parse_capabilities(["image", "video"]) == {Capability.IMAGE_GEN, Capability.VIDEO}


def test_parse_capabilities_keeps_vision_separate():
    assert parse_capabilities(["vision", "image"]) == {Capability.VISION, Capability.IMAGE_GEN}
