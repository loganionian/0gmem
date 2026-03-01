"""
Entity Timeline: Build explicit temporal graphs for each entity.

INNOVATION: Unlike EverMemOS which uses session-level temporal resolution,
we build a complete timeline graph per entity that captures:
1. Absolute dates of events
2. Relative ordering of events
3. Duration facts (how long things lasted)
4. Temporal relationships between events

This enables answering complex temporal questions by graph traversal
rather than relying on LLM reasoning over noisy context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Set
from collections import defaultdict


@dataclass
class TimelineEvent:
    """An event in an entity's timeline."""
    event_id: str
    entity: str
    description: str
    event_type: str  # activity, milestone, state_change, recurring

    # Temporal information
    absolute_date: Optional[datetime] = None
    relative_to: Optional[str] = None  # event_id this is relative to
    relative_position: str = ""  # "before", "after", "during", "same_day"
    duration: Optional[str] = None  # "4 years", "2 weeks", etc.

    # Source tracking
    session_id: str = ""
    session_date: Optional[datetime] = None
    source_text: str = ""
    confidence: float = 0.9


@dataclass
class EntityState:
    """A state/attribute of an entity at a point in time."""
    entity: str
    attribute: str  # relationship_status, location, job, etc.
    value: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None  # None means still current
    source_event_id: str = ""


class EntityTimeline:
    """
    Timeline graph for a single entity.

    Enables temporal queries like:
    - "When did X happen?"
    - "How long has X been Y?"
    - "What happened before/after X?"
    - "What was X doing on date Y?"
    """

    def __init__(self, entity: str):
        self.entity = entity
        self.events: Dict[str, TimelineEvent] = {}
        self.states: List[EntityState] = []
        self.event_order: List[str] = []  # Ordered list of event IDs

    def add_event(self, event: TimelineEvent) -> None:
        """Add an event to the timeline."""
        self.events[event.event_id] = event

        # Insert in chronological order if we have a date
        if event.absolute_date:
            inserted = False
            for i, eid in enumerate(self.event_order):
                existing = self.events.get(eid)
                if existing and existing.absolute_date:
                    if event.absolute_date < existing.absolute_date:
                        self.event_order.insert(i, event.event_id)
                        inserted = True
                        break
            if not inserted:
                self.event_order.append(event.event_id)
        else:
            self.event_order.append(event.event_id)

    def add_state(self, state: EntityState) -> None:
        """Add a state to the entity."""
        self.states.append(state)

    def get_event_on_date(self, target_date: datetime, tolerance_days: int = 7) -> List[TimelineEvent]:
        """Find events on or near a specific date."""
        results = []
        for event in self.events.values():
            if event.absolute_date:
                diff = abs((event.absolute_date - target_date).days)
                if diff <= tolerance_days:
                    results.append(event)
        return sorted(results, key=lambda e: abs((e.absolute_date - target_date).days))

    def get_events_in_range(self, start: datetime, end: datetime) -> List[TimelineEvent]:
        """Get all events in a date range."""
        results = []
        for event in self.events.values():
            if event.absolute_date and start <= event.absolute_date <= end:
                results.append(event)
        return sorted(results, key=lambda e: e.absolute_date)

    def get_state_at_time(self, attribute: str, at_time: Optional[datetime] = None) -> Optional[EntityState]:
        """Get the state of an attribute at a specific time (or current if None)."""
        at_time = at_time or datetime.now()

        for state in self.states:
            if state.attribute != attribute:
                continue

            # Check if this state was active at the given time
            if state.start_date and state.start_date > at_time:
                continue
            if state.end_date and state.end_date < at_time:
                continue

            return state

        return None

    def get_duration(self, attribute: str) -> Optional[str]:
        """Get how long an entity has had a particular attribute."""
        current_state = self.get_state_at_time(attribute)
        if current_state and current_state.start_date:
            duration = datetime.now() - current_state.start_date
            years = duration.days // 365
            if years > 0:
                return f"{years} years"
            months = duration.days // 30
            if months > 0:
                return f"{months} months"
            return f"{duration.days} days"
        return None


class TimelineBuilder:
    """
    Builds entity timelines from conversation data.

    INNOVATION: Extracts both explicit and implicit temporal information:
    1. Explicit dates ("on May 7th")
    2. Relative dates ("yesterday", "last week")
    3. Duration markers ("for 4 years", "since 2019")
    4. Event sequences ("after the concert", "before the trip")
    """

    # Date patterns
    ABSOLUTE_DATE_PATTERNS = [
        (r'(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})', 'dmy'),
        (r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})', 'mdy'),
        (r'(\d{4})', 'year_only'),
    ]

    RELATIVE_DATE_PATTERNS = [
        (r'yesterday', -1),
        (r'today', 0),
        (r'last\s+week', -7),
        (r'last\s+weekend', -3),
        (r'last\s+month', -30),
        (r'last\s+(?:friday|saturday|sunday)', -7),
        (r'two\s+weeks?\s+ago', -14),
        (r'a\s+few\s+days?\s+ago', -3),
        (r'recently', -7),
    ]

    DURATION_PATTERNS = [
        r'for\s+(\d+)\s+(years?|months?|weeks?|days?)',
        r'(\d+)\s+(years?|months?|weeks?|days?)\s+ago',
        r'since\s+(\d{4})',
        r'been\s+(\d+)\s+(years?|months?)',
    ]

    EVENT_PATTERNS = [
        (r'(?:went|visited|attended)\s+(?:to\s+)?(?:the\s+)?(.+?)(?:\.|,|!|$)', 'activity'),
        (r'(?:signed up|joined|started)\s+(?:for|in)?\s*(.+?)(?:\.|,|!|$)', 'milestone'),
        (r'(?:celebrated|had)\s+(?:a\s+)?(.+?)(?:\.|,|!|$)', 'activity'),
        (r'(?:ran|participated in)\s+(?:a\s+)?(.+?)(?:\.|,|!|$)', 'activity'),
        (r'(?:painted|created|made)\s+(?:a\s+)?(.+?)(?:\.|,|!|$)', 'activity'),
        (r'(?:went)\s+(camping|hiking|swimming|biking)', 'activity'),
        (r'(?:took|brought)\s+(?:the\s+)?(?:kids?|children)\s+to\s+(.+?)(?:\.|,|!|$)', 'activity'),
    ]

    STATE_PATTERNS = [
        (r'(?:i am|i\'m|she is|he is)\s+(\d+)\s+years?\s+old', 'age'),
        (r'(?:married|single|divorced|dating)', 'relationship_status'),
        (r'(?:live|living|moved)\s+(?:in|to)\s+(\w+)', 'location'),
        (r'(?:work|working)\s+(?:as|at)\s+(.+?)(?:\.|,|$)', 'occupation'),
        (r'friends?\s+for\s+(\d+)\s+years?', 'friends_duration'),
    ]

    MONTH_MAP = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }

    def __init__(self):
        self.timelines: Dict[str, EntityTimeline] = {}
        self._event_counter = 0

    def get_or_create_timeline(self, entity: str) -> EntityTimeline:
        """Get or create a timeline for an entity."""
        entity_lower = entity.lower()
        if entity_lower not in self.timelines:
            self.timelines[entity_lower] = EntityTimeline(entity_lower)
        return self.timelines[entity_lower]

    def process_message(
        self,
        text: str,
        speaker: str,
        session_id: str = "",
        session_date: Optional[datetime] = None,
    ) -> List[TimelineEvent]:
        """Process a message and extract timeline events."""
        events = []
        text_lower = text.lower()

        # Get or create timeline for speaker
        timeline = self.get_or_create_timeline(speaker)

        # Extract absolute date from text
        absolute_date = self._extract_absolute_date(text_lower)

        # If no absolute date, try relative date from session
        if not absolute_date and session_date:
            relative_days = self._extract_relative_date(text_lower)
            if relative_days is not None:
                absolute_date = session_date + timedelta(days=relative_days)

        # Extract events
        for pattern, event_type in self.EVENT_PATTERNS:
            for match in re.finditer(pattern, text_lower):
                description = match.group(1).strip() if match.groups() else match.group(0)
                if len(description) > 2 and len(description) < 100:
                    self._event_counter += 1
                    event = TimelineEvent(
                        event_id=f"evt_{self._event_counter}",
                        entity=speaker.lower(),
                        description=description,
                        event_type=event_type,
                        absolute_date=absolute_date,
                        session_id=session_id,
                        session_date=session_date,
                        source_text=text[:200],
                    )
                    timeline.add_event(event)
                    events.append(event)

        # Extract duration information
        duration = self._extract_duration(text_lower)
        if duration:
            # Create a duration-based state
            for pattern, attr in self.STATE_PATTERNS:
                if attr == 'friends_duration' and 'friend' in text_lower:
                    state = EntityState(
                        entity=speaker.lower(),
                        attribute='friends_duration',
                        value=duration,
                    )
                    timeline.add_state(state)

        # Extract states
        self._extract_states(text_lower, speaker, timeline, session_date)

        return events

    def _extract_absolute_date(self, text: str) -> Optional[datetime]:
        """Extract absolute date from text."""
        for pattern, fmt in self.ABSOLUTE_DATE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                try:
                    if fmt == 'dmy':
                        day, month_name, year = match.groups()
                        month = self.MONTH_MAP.get(month_name.lower(), 1)
                        return datetime(int(year), month, int(day))
                    elif fmt == 'mdy':
                        month_name, day, year = match.groups()
                        month = self.MONTH_MAP.get(month_name.lower(), 1)
                        return datetime(int(year), month, int(day))
                    elif fmt == 'year_only':
                        year = match.group(1)
                        # Only use if it looks like a year (2020-2025 range)
                        if 2015 <= int(year) <= 2030:
                            return datetime(int(year), 6, 15)  # Mid-year default
                except (ValueError, TypeError):
                    pass
        return None

    def _extract_relative_date(self, text: str) -> Optional[int]:
        """Extract relative date offset in days."""
        for pattern, days in self.RELATIVE_DATE_PATTERNS:
            if re.search(pattern, text):
                return days
        return None

    def _extract_duration(self, text: str) -> Optional[str]:
        """Extract duration information."""
        for pattern in self.DURATION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    return f"{groups[0]} {groups[1]}"
                elif len(groups) == 1:
                    return f"since {groups[0]}"
        return None

    def _extract_states(
        self,
        text: str,
        speaker: str,
        timeline: EntityTimeline,
        session_date: Optional[datetime],
    ) -> None:
        """Extract state information from text."""
        for pattern, attr in self.STATE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                value = match.group(1) if match.groups() else match.group(0)
                state = EntityState(
                    entity=speaker.lower(),
                    attribute=attr,
                    value=value,
                    start_date=session_date,
                )
                timeline.add_state(state)

    def answer_temporal_question(
        self,
        question: str,
        target_entity: str,
    ) -> Optional[str]:
        """
        INNOVATION: Answer temporal questions by timeline graph traversal.

        This is more reliable than LLM reasoning over noisy context.
        """
        timeline = self.timelines.get(target_entity.lower())
        if not timeline:
            return None

        q_lower = question.lower()

        # "When did X happen?" questions
        when_match = re.search(r'when did (?:\w+\s+)?(\w+(?:\s+\w+){0,5})', q_lower)
        if when_match:
            activity = when_match.group(1)
            # Search events for matching activity
            for event in timeline.events.values():
                if activity in event.description.lower():
                    if event.absolute_date:
                        return event.absolute_date.strftime("%d %B %Y")

        # "How long has X been/had Y?" questions
        how_long_match = re.search(r'how long (?:has|have) (?:\w+\s+)?(?:been\s+)?(?:had\s+)?(\w+)', q_lower)
        if how_long_match:
            attr = how_long_match.group(1)
            # Check states
            for state in timeline.states:
                if attr in state.attribute.lower() or attr in state.value.lower():
                    if state.value and ('year' in state.value or 'month' in state.value):
                        return state.value

        # "How long ago was X?" questions
        ago_match = re.search(r'how long ago (?:was|did)', q_lower)
        if ago_match:
            # Search for relevant event with date
            for event in timeline.events.values():
                if event.absolute_date:
                    # Check if event matches question
                    q_words = set(q_lower.split())
                    event_words = set(event.description.lower().split())
                    if len(q_words & event_words) >= 2:
                        years_ago = (datetime.now() - event.absolute_date).days // 365
                        if years_ago > 0:
                            return f"{years_ago} years ago"

        # Check for duration states
        for state in timeline.states:
            if 'duration' in state.attribute or 'years' in str(state.value):
                # Check if question relates to this state
                if any(word in q_lower for word in state.attribute.split('_')):
                    return state.value

        return None

    def get_timeline_summary(self, entity: str) -> str:
        """Get a text summary of an entity's timeline."""
        timeline = self.timelines.get(entity.lower())
        if not timeline:
            return ""

        parts = [f"## Timeline for {entity}"]

        # Events in chronological order
        if timeline.events:
            parts.append("\n### Events:")
            for eid in timeline.event_order:
                event = timeline.events.get(eid)
                if event:
                    date_str = event.absolute_date.strftime("%d %b %Y") if event.absolute_date else "unknown date"
                    parts.append(f"- [{date_str}] {event.description}")

        # Current states
        if timeline.states:
            parts.append("\n### States:")
            for state in timeline.states:
                parts.append(f"- {state.attribute}: {state.value}")

        return "\n".join(parts)

    def clear(self) -> None:
        """Clear all timelines."""
        self.timelines.clear()
        self._event_counter = 0
