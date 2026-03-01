"""
Episodic Memory: Personal history and specific events.

Stores episodes (sequences of events) with both summaries
and detailed traces for lossless retrieval.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import numpy as np


@dataclass
class EpisodeMessage:
    """A single message/turn within an episode."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    speaker: str = ""
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    entities_mentioned: List[str] = field(default_factory=list)
    sentiment: float = 0.0  # -1 to 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """
    A complete episode representing a coherent sequence of events.

    Key design: Keeps BOTH summary AND detailed trace for lossless memory.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # High-level information
    summary: str = ""  # LLM-generated summary
    title: str = ""  # Short descriptive title

    # Detailed trace (lossless storage)
    messages: List[EpisodeMessage] = field(default_factory=list)

    # Temporal information
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    session_id: Optional[str] = None

    # Participants and context
    participants: List[str] = field(default_factory=list)  # Entity IDs
    participant_names: List[str] = field(default_factory=list)
    location: Optional[str] = None
    topics: List[str] = field(default_factory=list)

    # Emotional and importance markers
    emotional_valence: float = 0.0  # -1 (negative) to 1 (positive)
    importance: float = 0.5  # 0 to 1

    # Embeddings
    summary_embedding: Optional[np.ndarray] = None
    full_embedding: Optional[np.ndarray] = None  # Embedding of full trace

    # Access patterns for consolidation
    retrieval_count: int = 0
    last_retrieved: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)

    # Key facts extracted from this episode
    extracted_facts: List[str] = field(default_factory=list)

    # Cold storage reference (for compression)
    archived: bool = False
    archive_ref: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[timedelta]:
        """Get episode duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def message_count(self) -> int:
        """Get number of messages in episode."""
        return len(self.messages)

    def get_full_text(self) -> str:
        """Get the full conversation text."""
        lines = []
        for msg in self.messages:
            lines.append(f"{msg.speaker}: {msg.content}")
        return "\n".join(lines)

    def get_text_window(self, start_idx: int = 0, end_idx: Optional[int] = None) -> str:
        """Get a window of the conversation."""
        messages = self.messages[start_idx:end_idx]
        lines = []
        for msg in messages:
            lines.append(f"{msg.speaker}: {msg.content}")
        return "\n".join(lines)

    def add_message(self, message: EpisodeMessage) -> None:
        """Add a message to the episode."""
        self.messages.append(message)
        self.end_time = message.timestamp

        # Update participants
        if message.speaker and message.speaker not in self.participant_names:
            self.participant_names.append(message.speaker)

    def mark_retrieved(self) -> None:
        """Mark episode as retrieved (for consolidation tracking)."""
        self.retrieval_count += 1
        self.last_retrieved = datetime.now()


class EpisodicMemory:
    """
    Episodic memory store managing episodes and their lifecycle.

    Key capabilities:
    - Store episodes with full traces
    - Retrieve by time, participants, topics
    - Track access patterns for consolidation
    - Support archival of old episodes (lossless compression)
    """

    def __init__(self):
        self.episodes: Dict[str, Episode] = {}

        # Indexes for efficient retrieval
        self._time_index: Dict[str, List[str]] = {}  # date_str -> episode_ids
        self._participant_index: Dict[str, set] = {}  # participant -> episode_ids
        self._topic_index: Dict[str, set] = {}  # topic -> episode_ids
        self._session_index: Dict[str, List[str]] = {}  # session_id -> episode_ids

        # Embeddings for similarity search
        self._embeddings: List[np.ndarray] = []
        self._embedding_ids: List[str] = []

    def add_episode(self, episode: Episode) -> str:
        """Add an episode to memory."""
        self.episodes[episode.id] = episode

        # Index by date
        date_str = episode.start_time.strftime("%Y-%m-%d")
        if date_str not in self._time_index:
            self._time_index[date_str] = []
        self._time_index[date_str].append(episode.id)

        # Index by participants
        for participant in episode.participants + episode.participant_names:
            if participant not in self._participant_index:
                self._participant_index[participant] = set()
            self._participant_index[participant].add(episode.id)

        # Index by topics
        for topic in episode.topics:
            if topic not in self._topic_index:
                self._topic_index[topic] = set()
            self._topic_index[topic].add(episode.id)

        # Index by session
        if episode.session_id:
            if episode.session_id not in self._session_index:
                self._session_index[episode.session_id] = []
            self._session_index[episode.session_id].append(episode.id)

        # Add embedding
        if episode.summary_embedding is not None:
            self._embeddings.append(episode.summary_embedding)
            self._embedding_ids.append(episode.id)

        return episode.id

    def get_episode(self, episode_id: str, mark_retrieved: bool = True) -> Optional[Episode]:
        """Get an episode by ID."""
        episode = self.episodes.get(episode_id)
        if episode and mark_retrieved:
            episode.mark_retrieved()
        return episode

    def get_by_time_range(
        self,
        start: datetime,
        end: datetime,
        participants: Optional[List[str]] = None
    ) -> List[Episode]:
        """Get episodes within a time range."""
        results = []

        # Generate date range
        current = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = end.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            if date_str in self._time_index:
                for episode_id in self._time_index[date_str]:
                    episode = self.episodes.get(episode_id)
                    if episode and episode.start_time >= start:
                        if episode.end_time is None or episode.end_time <= end:
                            results.append(episode)
            current += timedelta(days=1)

        # Filter by participants if specified
        if participants:
            participant_set = set(participants)
            results = [
                ep for ep in results
                if participant_set.intersection(ep.participants + ep.participant_names)
            ]

        # Sort by time
        results.sort(key=lambda e: e.start_time)
        return results

    def get_by_participant(self, participant: str, limit: int = 50) -> List[Episode]:
        """Get episodes involving a participant."""
        episode_ids = self._participant_index.get(participant, set())
        episodes = [
            self.episodes[eid] for eid in episode_ids
            if eid in self.episodes
        ]
        # Sort by time, most recent first
        episodes.sort(key=lambda e: e.start_time, reverse=True)
        return episodes[:limit]

    def get_by_topic(self, topic: str, limit: int = 50) -> List[Episode]:
        """Get episodes about a topic."""
        episode_ids = self._topic_index.get(topic, set())
        episodes = [
            self.episodes[eid] for eid in episode_ids
            if eid in self.episodes
        ]
        episodes.sort(key=lambda e: e.start_time, reverse=True)
        return episodes[:limit]

    def get_by_session(self, session_id: str) -> List[Episode]:
        """Get all episodes from a session."""
        episode_ids = self._session_index.get(session_id, [])
        episodes = [
            self.episodes[eid] for eid in episode_ids
            if eid in self.episodes
        ]
        episodes.sort(key=lambda e: e.start_time)
        return episodes

    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[Episode, float]]:
        """Search for similar episodes by embedding."""
        if not self._embeddings:
            return []

        results = []
        for i, emb in enumerate(self._embeddings):
            sim = self._cosine_similarity(query_embedding, emb)
            if sim >= threshold:
                episode_id = self._embedding_ids[i]
                episode = self.episodes.get(episode_id)
                if episode:
                    results.append((episode, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_recent(self, limit: int = 10) -> List[Episode]:
        """Get most recent episodes."""
        episodes = list(self.episodes.values())
        episodes.sort(key=lambda e: e.start_time, reverse=True)
        return episodes[:limit]

    def get_most_accessed(self, limit: int = 10) -> List[Episode]:
        """Get most frequently accessed episodes."""
        episodes = list(self.episodes.values())
        episodes.sort(key=lambda e: e.retrieval_count, reverse=True)
        return episodes[:limit]

    def get_candidates_for_consolidation(
        self,
        min_retrievals: int = 3,
        min_age_days: int = 7
    ) -> List[Episode]:
        """
        Get episodes that are candidates for consolidation into semantic memory.

        Criteria:
        - Retrieved multiple times (indicates importance)
        - Old enough to be stable
        """
        cutoff = datetime.now() - timedelta(days=min_age_days)
        candidates = [
            ep for ep in self.episodes.values()
            if ep.retrieval_count >= min_retrievals and ep.start_time < cutoff
        ]
        return candidates

    def archive_episode(self, episode_id: str, archive_ref: str) -> bool:
        """
        Archive an episode (move detailed trace to cold storage).

        Keeps summary and metadata in memory, detailed trace in archive.
        """
        episode = self.episodes.get(episode_id)
        if not episode:
            return False

        # Mark as archived (actual archival handled by caller)
        episode.archived = True
        episode.archive_ref = archive_ref

        # Clear detailed messages to save memory
        # (In production, you'd move these to cold storage first)
        # episode.messages = []  # Uncomment for actual archival

        return True

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about episodic memory."""
        if not self.episodes:
            return {
                "total_episodes": 0,
                "total_messages": 0,
                "unique_participants": 0,
                "unique_topics": 0,
            }

        total_messages = sum(ep.message_count for ep in self.episodes.values())
        return {
            "total_episodes": len(self.episodes),
            "total_messages": total_messages,
            "unique_participants": len(self._participant_index),
            "unique_topics": len(self._topic_index),
            "unique_sessions": len(self._session_index),
            "archived_episodes": sum(1 for ep in self.episodes.values() if ep.archived),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize episodic memory.

        Note: Embeddings are NOT included. They must be saved separately
        and passed to from_dict().
        """
        return {
            "episodes": [
                {
                    "id": ep.id,
                    "summary": ep.summary,
                    "title": ep.title,
                    "start_time": ep.start_time.isoformat(),
                    "end_time": ep.end_time.isoformat() if ep.end_time else None,
                    "session_id": ep.session_id,
                    "participants": ep.participants,
                    "participant_names": ep.participant_names,
                    "location": ep.location,
                    "topics": ep.topics,
                    "emotional_valence": ep.emotional_valence,
                    "importance": ep.importance,
                    "retrieval_count": ep.retrieval_count,
                    "last_retrieved": ep.last_retrieved.isoformat() if ep.last_retrieved else None,
                    "created_at": ep.created_at.isoformat(),
                    "extracted_facts": ep.extracted_facts,
                    "archived": ep.archived,
                    "archive_ref": ep.archive_ref,
                    "metadata": ep.metadata,
                    "messages": [
                        {
                            "id": msg.id,
                            "speaker": msg.speaker,
                            "content": msg.content,
                            "timestamp": msg.timestamp.isoformat(),
                            "entities_mentioned": msg.entities_mentioned,
                            "sentiment": msg.sentiment,
                            "metadata": msg.metadata,
                        }
                        for msg in ep.messages
                    ],
                }
                for ep in self.episodes.values()
            ]
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        embeddings_map: Optional[Dict[str, np.ndarray]] = None,
    ) -> "EpisodicMemory":
        """Deserialize episodic memory from dictionary.

        Args:
            data: Output of to_dict().
            embeddings_map: Map of episode_id -> summary_embedding.
        """
        embeddings_map = embeddings_map or {}
        store = cls()

        for ep_data in data.get("episodes", []):
            messages = []
            for msg_data in ep_data.get("messages", []):
                messages.append(EpisodeMessage(
                    id=msg_data.get("id", str(uuid.uuid4())),
                    speaker=msg_data.get("speaker", ""),
                    content=msg_data.get("content", ""),
                    timestamp=datetime.fromisoformat(msg_data["timestamp"]),
                    entities_mentioned=msg_data.get("entities_mentioned", []),
                    sentiment=msg_data.get("sentiment", 0.0),
                    metadata=msg_data.get("metadata", {}),
                ))

            episode = Episode(
                id=ep_data["id"],
                summary=ep_data.get("summary", ""),
                title=ep_data.get("title", ""),
                messages=messages,
                start_time=datetime.fromisoformat(ep_data["start_time"]),
                end_time=(
                    datetime.fromisoformat(ep_data["end_time"])
                    if ep_data.get("end_time")
                    else None
                ),
                session_id=ep_data.get("session_id"),
                participants=ep_data.get("participants", []),
                participant_names=ep_data.get("participant_names", []),
                location=ep_data.get("location"),
                topics=ep_data.get("topics", []),
                emotional_valence=ep_data.get("emotional_valence", 0.0),
                importance=ep_data.get("importance", 0.5),
                summary_embedding=embeddings_map.get(ep_data["id"]),
                retrieval_count=ep_data.get("retrieval_count", 0),
                last_retrieved=(
                    datetime.fromisoformat(ep_data["last_retrieved"])
                    if ep_data.get("last_retrieved")
                    else None
                ),
                created_at=(
                    datetime.fromisoformat(ep_data["created_at"])
                    if ep_data.get("created_at")
                    else datetime.now()
                ),
                extracted_facts=ep_data.get("extracted_facts", []),
                archived=ep_data.get("archived", False),
                archive_ref=ep_data.get("archive_ref"),
                metadata=ep_data.get("metadata", {}),
            )
            store.add_episode(episode)

        return store
