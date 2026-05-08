from runtime.memory.short_term import ShortTermMemory
from runtime.memory.semantic_memory import SemanticMemory
from runtime.memory.episodic_memory import EpisodicMemory
from runtime.memory.memory_manager import MemoryManager

short_term = ShortTermMemory()
semantic = SemanticMemory()
episodic = EpisodicMemory()
memory_manager = MemoryManager(short_term, semantic, episodic)

__all__ = [
    "ShortTermMemory",
    "SemanticMemory",
    "EpisodicMemory",
    "MemoryManager",
    "short_term",
    "semantic",
    "episodic",
    "memory_manager",
]
