"""
Session Summarizer: Compress conversation sessions into summaries.

Uses session-level compression for better context density.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class SessionSummary:
    """Summary of a conversation session."""

    session_id: str
    session_date: str
    participants: list[str]
    key_events: list[str]
    key_facts: dict[str, list[str]]  # person -> facts
    topics: list[str]
    raw_message_count: int


class SessionSummarizer:
    """
    Summarizes conversation sessions for better context compression.

    Key techniques:
    - Extract key events and facts per session
    - Track topics discussed
    - Maintain per-person fact updates
    """

    # Topic keywords
    TOPIC_KEYWORDS = {
        "family": ["kids", "children", "family", "husband", "wife", "spouse", "daughter", "son"],
        "work": ["work", "job", "career", "office", "project", "meeting"],
        "hobbies": ["hobby", "hiking", "camping", "painting", "pottery", "music", "reading"],
        "travel": ["trip", "travel", "vacation", "camping", "roadtrip", "beach", "mountains"],
        "health": ["health", "exercise", "gym", "running", "swimming", "yoga"],
        "social": ["friends", "party", "event", "celebration", "birthday", "concert"],
        "lgbtq": ["lgbtq", "transgender", "trans", "pride", "coming out", "identity"],
        "pets": ["pet", "dog", "cat", "guinea pig", "animals"],
    }

    # Event patterns
    EVENT_PATTERNS = [
        (r"(?:went|visited|attended)\s+(?:to\s+)?(?:the\s+)?(.+?)(?:\.|,|!|$)", "visited"),
        (r"(?:signed up|joined|participated)\s+(?:for|in)\s+(.+?)(?:\.|,|!|$)", "joined"),
        (r"(?:celebrated|had)\s+(?:a\s+)?(.+?)(?:\.|,|!|$)", "celebrated"),
        (r"(?:started|began)\s+(.+?)(?:\.|,|!|$)", "started"),
        (r"(?:finished|completed)\s+(.+?)(?:\.|,|!|$)", "completed"),
        (r"(?:bought|purchased|got)\s+(?:a\s+)?(.+?)(?:\.|,|!|$)", "bought"),
    ]

    def __init__(self, llm_client: Any | None = None):
        self._client = llm_client
        self.session_summaries: dict[str, SessionSummary] = {}

    def summarize_session(
        self,
        session_id: str,
        messages: list[dict[str, str]],
        session_date: str = "",
    ) -> SessionSummary:
        """
        Summarize a conversation session.

        Args:
            session_id: Unique session identifier
            messages: List of {"speaker": str, "content": str}
            session_date: Date/time of session

        Returns:
            SessionSummary with extracted information
        """
        participants = set()
        key_events = []
        key_facts: dict[str, list[str]] = {}
        topics_found = set()

        for msg in messages:
            speaker = msg.get("speaker", "Unknown")
            content = msg.get("content", msg.get("text", ""))
            participants.add(speaker)

            if speaker not in key_facts:
                key_facts[speaker] = []

            content_lower = content.lower()

            # Extract events
            for pattern, event_type in self.EVENT_PATTERNS:
                match = re.search(pattern, content_lower)
                if match:
                    event = match.group(1).strip()
                    if len(event) > 3 and len(event) < 100:
                        key_events.append(f"{speaker} {event_type} {event}")

            # Detect topics
            for topic, keywords in self.TOPIC_KEYWORDS.items():
                if any(kw in content_lower for kw in keywords):
                    topics_found.add(topic)

            # Extract key facts per person
            facts = self._extract_session_facts(content, speaker)
            key_facts[speaker].extend(facts)

        # Remove duplicates from facts
        for person in key_facts:
            key_facts[person] = list(set(key_facts[person]))

        summary = SessionSummary(
            session_id=session_id,
            session_date=session_date,
            participants=list(participants),
            key_events=key_events[:10],  # Limit events
            key_facts=key_facts,
            topics=list(topics_found),
            raw_message_count=len(messages),
        )

        self.session_summaries[session_id] = summary
        return summary

    def _extract_session_facts(self, text: str, speaker: str) -> list[str]:
        """Extract key facts from a message."""
        facts = []
        text_lower = text.lower()

        # Relationship mentions
        if "husband" in text_lower or "wife" in text_lower:
            facts.append(f"{speaker} mentioned their spouse")
        if "kids" in text_lower or "children" in text_lower:
            facts.append(f"{speaker} mentioned their children")

        # Activity mentions
        if "went camping" in text_lower:
            facts.append(f"{speaker} went camping")
        if "went hiking" in text_lower:
            facts.append(f"{speaker} went hiking")
        if "pottery" in text_lower and ("class" in text_lower or "workshop" in text_lower):
            facts.append(f"{speaker} did pottery")

        # Significant events
        if "accident" in text_lower:
            facts.append(f"{speaker} mentioned an accident")
        if "birthday" in text_lower:
            facts.append(f"{speaker} mentioned a birthday")
        if "concert" in text_lower:
            facts.append(f"{speaker} went to a concert")

        return facts

    def get_summary_text(self, session_id: str) -> str:
        """Get formatted summary text for a session."""
        summary = self.session_summaries.get(session_id)
        if not summary:
            return ""

        parts = [f"## Session: {session_id}"]
        if summary.session_date:
            parts.append(f"Date: {summary.session_date}")

        parts.append(f"Participants: {', '.join(summary.participants)}")
        parts.append(f"Topics: {', '.join(summary.topics) if summary.topics else 'general'}")

        if summary.key_events:
            parts.append("\nKey Events:")
            for event in summary.key_events[:5]:
                parts.append(f"  - {event}")

        return "\n".join(parts)

    def get_all_summaries_text(self) -> str:
        """Get formatted text of all session summaries."""
        parts = ["# Conversation Session Summaries\n"]
        for session_id, summary in sorted(self.session_summaries.items()):
            parts.append(self.get_summary_text(session_id))
            parts.append("")
        return "\n".join(parts)

    def get_relevant_sessions(
        self,
        question: str,
        target_entity: str | None = None,
        max_sessions: int = 5,
    ) -> list[SessionSummary]:
        """
        Find sessions most relevant to a question.

        Returns: List of relevant SessionSummary objects
        """
        q_lower = question.lower()
        scored_sessions = []

        for session_id, summary in self.session_summaries.items():
            score = 0

            # Check if target entity participated
            if target_entity:
                if any(target_entity.lower() in p.lower() for p in summary.participants):
                    score += 5

            # Check topic overlap
            for topic, keywords in self.TOPIC_KEYWORDS.items():
                if any(kw in q_lower for kw in keywords):
                    if topic in summary.topics:
                        score += 2

            # Check event keyword overlap
            for event in summary.key_events:
                event_words = set(event.lower().split())
                q_words = set(q_lower.split())
                overlap = len(event_words & q_words)
                score += overlap

            if score > 0:
                scored_sessions.append((score, summary))

        # Sort by score descending
        scored_sessions.sort(key=lambda x: x[0], reverse=True)

        return [s for _, s in scored_sessions[:max_sessions]]

    def clear(self) -> None:
        """Clear all summaries."""
        self.session_summaries.clear()
