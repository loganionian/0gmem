"""
Memory Types: Multiple memory categories for structured memory storage.

Implements seven memory types:
- Episodes: Temporal narrative sequences
- Profiles: User characteristics
- Preferences: Likes/dislikes
- Relationships: Interpersonal dynamics
- Semantic: Conceptual understanding
- Facts: Discrete factual information
- Core: High-salience important memories
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Episode:
    """A temporal narrative sequence - what happened when."""

    id: str
    event: str
    participants: list[str]
    date: str | None = None
    session_id: str = ""
    details: str = ""

    def to_text(self) -> str:
        date_str = f" on {self.date}" if self.date else ""
        return f"{', '.join(self.participants)} {self.event}{date_str}"


@dataclass
class Preference:
    """A stated or inferred like/dislike."""

    person: str
    category: str  # food, activity, music, etc.
    item: str
    sentiment: str  # like, love, dislike, hate
    confidence: float = 0.9

    def to_text(self) -> str:
        return f"{self.person} {self.sentiment}s {self.item}"


@dataclass
class Relationship:
    """Interpersonal dynamics between people."""

    person1: str
    person2: str
    relationship_type: str  # friend, spouse, child, parent, colleague
    details: str = ""

    def to_text(self) -> str:
        return f"{self.person1} is {self.person2}'s {self.relationship_type}"


@dataclass
class CoreMemory:
    """High-salience important information."""

    person: str
    memory_type: str  # identity, major_event, life_change
    content: str
    importance: float = 1.0

    def to_text(self) -> str:
        return f"{self.person}: {self.content}"


class MultiTypeMemoryStore:
    """
    Store for multiple memory types.

    Maintains separate indices for:
    - Episodes (what happened when)
    - Profiles (who is this person)
    - Preferences (what do they like/dislike)
    - Relationships (how are people connected)
    - Core memories (important high-salience info)
    """

    def __init__(self) -> None:
        self.episodes: list[Episode] = []
        self.preferences: list[Preference] = []
        self.relationships: list[Relationship] = []
        self.core_memories: list[CoreMemory] = []

        # Indices for fast lookup
        self.episodes_by_person: dict[str, list[Episode]] = defaultdict(list)
        self.preferences_by_person: dict[str, list[Preference]] = defaultdict(list)
        self.relationships_by_person: dict[str, list[Relationship]] = defaultdict(list)
        self.core_by_person: dict[str, list[CoreMemory]] = defaultdict(list)

    def add_episode(self, episode: Episode) -> None:
        """Add an episode to the store."""
        self.episodes.append(episode)
        for participant in episode.participants:
            self.episodes_by_person[participant.lower()].append(episode)

    def add_preference(self, pref: Preference) -> None:
        """Add a preference to the store."""
        self.preferences.append(pref)
        self.preferences_by_person[pref.person.lower()].append(pref)

    def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship to the store."""
        self.relationships.append(rel)
        self.relationships_by_person[rel.person1.lower()].append(rel)
        self.relationships_by_person[rel.person2.lower()].append(rel)

    def add_core_memory(self, mem: CoreMemory) -> None:
        """Add a core memory to the store."""
        self.core_memories.append(mem)
        self.core_by_person[mem.person.lower()].append(mem)

    def get_person_context(self, person: str) -> str:
        """Get all memory context for a person."""
        person_lower = person.lower()
        parts = []

        # Core memories first (most important)
        core = self.core_by_person.get(person_lower, [])
        if core:
            parts.append(f"## Core Information about {person}")
            for mem in core:
                parts.append(f"- {mem.content}")

        # Preferences
        prefs = self.preferences_by_person.get(person_lower, [])
        if prefs:
            parts.append(f"\n## {person}'s Preferences")
            for pref in prefs:
                parts.append(f"- {pref.to_text()}")

        # Relationships
        rels = self.relationships_by_person.get(person_lower, [])
        if rels:
            parts.append(f"\n## {person}'s Relationships")
            for rel in rels:
                parts.append(f"- {rel.to_text()}")

        # Recent episodes
        eps = self.episodes_by_person.get(person_lower, [])
        if eps:
            parts.append(f"\n## Recent Events involving {person}")
            for ep in eps[-10:]:  # Last 10 episodes
                parts.append(f"- {ep.to_text()}")

        return "\n".join(parts)

    def clear(self) -> None:
        """Clear all memories."""
        self.episodes.clear()
        self.preferences.clear()
        self.relationships.clear()
        self.core_memories.clear()
        self.episodes_by_person.clear()
        self.preferences_by_person.clear()
        self.relationships_by_person.clear()
        self.core_by_person.clear()


class MemoryExtractor:
    """
    Extracts multiple memory types from conversation text.

    Extracts structured memory types from conversation text.
    """

    # Preference patterns
    LIKE_PATTERNS = [
        (r"(?:i|we|she|he)\s+(?:really\s+)?(?:like|love|enjoy|adore)\s+(\w+(?:\s+\w+)?)", "like"),
        (r"(?:i|we|she|he)\s+(?:am|is|are)\s+(?:a\s+)?fan\s+of\s+(\w+(?:\s+\w+)?)", "like"),
        (r"(?:my|her|his)\s+favorite\s+(?:\w+\s+)?is\s+(\w+(?:\s+\w+)?)", "love"),
    ]

    DISLIKE_PATTERNS = [
        (
            r"(?:i|we|she|he)\s+(?:really\s+)?(?:hate|dislike|can't stand)\s+(\w+(?:\s+\w+)?)",
            "dislike",
        ),
        (r"(?:i|we|she|he)\s+(?:could\s+)?never\s+(?:\w+\s+)?(\w+(?:\s+\w+)?)", "dislike"),
    ]

    # Relationship patterns
    RELATIONSHIP_PATTERNS = [
        (r"my\s+(husband|wife|spouse|partner)\s+(\w+)?", "spouse"),
        (r"my\s+(son|daughter|child|kid)\s+(\w+)?", "child"),
        (r"my\s+(mother|father|mom|dad|parent)\s+(\w+)?", "parent"),
        (r"my\s+(friend|best friend)\s+(\w+)?", "friend"),
        (r"(\w+)\s+is\s+my\s+(husband|wife|spouse|partner)", "spouse"),
        (r"(\w+)\s+is\s+my\s+(son|daughter|child)", "child"),
    ]

    # Event patterns for episodes
    EVENT_PATTERNS = [
        r"(?:i|we|she|he)\s+(?:went|visited|attended)\s+(?:to\s+)?(?:the\s+)?(.+?)(?:\.|,|$)",
        r"(?:i|we|she|he)\s+(?:signed up|joined|participated)\s+(?:for|in)\s+(.+?)(?:\.|,|$)",
        r"(?:i|we|she|he)\s+(?:celebrated|had)\s+(.+?)(?:\.|,|$)",
    ]

    # Core memory patterns (high importance)
    CORE_PATTERNS = [
        (r"(?:i am|i'm|she is|he is)\s+(?:a\s+)?(transgender\s+\w+)", "identity"),
        (r"(?:i|she|he)\s+(?:came out|transitioned)", "identity"),
        (r"(?:i|we|she|he)\s+(?:moved|relocated)\s+(?:from|to)\s+(\w+)", "life_change"),
        (r"(?:i|we|she|he)\s+(?:got\s+)?(?:married|divorced|engaged)", "life_change"),
        (r"(?:i|we|she|he)\s+(?:had|have)\s+(\d+)\s+(?:kids?|children)", "family"),
    ]

    def __init__(self) -> None:
        self.memory_store = MultiTypeMemoryStore()
        self._episode_counter = 0

    def extract_all(self, text: str, speaker: str, session_id: str = "", date: str = "") -> None:
        """Extract all memory types from text."""
        text_lower = text.lower()

        # Extract preferences
        for pattern, sentiment in self.LIKE_PATTERNS + self.DISLIKE_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                item = match.group(1).strip()
                if len(item) > 2:
                    pref = Preference(
                        person=speaker,
                        category="general",
                        item=item,
                        sentiment=sentiment,
                    )
                    self.memory_store.add_preference(pref)

        # Extract relationships
        for pattern, rel_type in self.RELATIONSHIP_PATTERNS:
            rel_match = re.search(pattern, text_lower)
            if rel_match:
                groups = rel_match.groups()
                other_person = groups[1] if len(groups) > 1 and groups[1] else "unknown"
                rel = Relationship(
                    person1=speaker,
                    person2=other_person.title() if other_person != "unknown" else other_person,
                    relationship_type=rel_type,
                )
                self.memory_store.add_relationship(rel)

        # Extract episodes (events)
        for pattern in self.EVENT_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                event = match.group(1).strip()
                if len(event) > 3:
                    self._episode_counter += 1
                    episode = Episode(
                        id=f"ep_{self._episode_counter}",
                        event=event,
                        participants=[speaker],
                        date=date,
                        session_id=session_id,
                    )
                    self.memory_store.add_episode(episode)

        # Extract core memories
        for pattern, mem_type in self.CORE_PATTERNS:
            core_match = re.search(pattern, text_lower)
            if core_match:
                content = core_match.group(0)
                core = CoreMemory(
                    person=speaker,
                    memory_type=mem_type,
                    content=content,
                )
                self.memory_store.add_core_memory(core)

    def get_context_for_question(self, question: str, person: str) -> str:
        """Get relevant memory context for answering a question."""
        return self.memory_store.get_person_context(person)

    def clear(self) -> None:
        """Clear all extracted memories."""
        self.memory_store.clear()
        self._episode_counter = 0
