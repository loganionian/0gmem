"""
Temporal Extractor: Extracts and normalizes temporal expressions.

Critical for LoCoMo's temporal reasoning questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class TemporalType(Enum):
    """Types of temporal expressions."""
    ABSOLUTE = "absolute"      # "January 15, 2024"
    RELATIVE = "relative"      # "yesterday", "last week"
    DURATION = "duration"      # "for 3 hours"
    FREQUENCY = "frequency"    # "every Monday"
    SEQUENCE = "sequence"      # "before", "after", "then"
    RANGE = "range"           # "from Monday to Friday"


@dataclass
class TemporalExpression:
    """A detected temporal expression."""
    text: str                           # Original text
    type: TemporalType
    normalized_start: Optional[datetime] = None
    normalized_end: Optional[datetime] = None
    duration: Optional[timedelta] = None
    relation: Optional[str] = None      # before, after, during, etc.
    reference_event: Optional[str] = None  # What this is relative to
    confidence: float = 1.0
    span: Tuple[int, int] = (0, 0)     # Character positions in text


class TemporalExtractor:
    """
    Extracts temporal information from text.

    Handles:
    - Absolute dates/times
    - Relative expressions (yesterday, last week, etc.)
    - Durations (for 3 hours, over 2 days)
    - Sequences (before, after, then, first, finally)
    - Frequencies (every day, always, never)
    """

    # Patterns for temporal expressions
    RELATIVE_PATTERNS = {
        "today": (0, "day"),
        "yesterday": (-1, "day"),
        "tomorrow": (1, "day"),
        "last week": (-1, "week"),
        "next week": (1, "week"),
        "last month": (-1, "month"),
        "next month": (1, "month"),
        "last year": (-1, "year"),
        "next year": (1, "year"),
        "this morning": (0, "morning"),
        "this afternoon": (0, "afternoon"),
        "this evening": (0, "evening"),
        "tonight": (0, "night"),
        "last night": (-1, "night"),
        "the other day": (-2, "day"),
        "a few days ago": (-3, "day"),
        "a week ago": (-1, "week"),
        "recently": (-1, "week"),
    }

    DURATION_PATTERNS = [
        r"for (\d+) (second|minute|hour|day|week|month|year)s?",
        r"over (\d+) (day|week|month|year)s?",
        r"(\d+) (second|minute|hour|day|week|month|year)s? long",
        r"about (\d+) (minute|hour|day|week|month)s?",
    ]

    SEQUENCE_WORDS = {
        "before": "before",
        "after": "after",
        "then": "after",
        "later": "after",
        "earlier": "before",
        "first": "first",
        "finally": "last",
        "next": "after",
        "previously": "before",
        "subsequently": "after",
        "meanwhile": "during",
        "during": "during",
        "while": "during",
        "when": "during",
        "since": "since",
        "until": "until",
    }

    FREQUENCY_PATTERNS = [
        (r"every (day|week|month|year)", "recurring"),
        (r"every (\w+day)", "weekly"),  # every Monday
        (r"always", "always"),
        (r"never", "never"),
        (r"sometimes", "sometimes"),
        (r"usually", "usually"),
        (r"often", "often"),
        (r"rarely", "rarely"),
        (r"once a (day|week|month|year)", "recurring"),
        (r"twice a (day|week|month|year)", "recurring"),
        (r"(\d+) times a (day|week|month|year)", "recurring"),
    ]

    ABSOLUTE_PATTERNS = [
        # ISO format
        r"\d{4}-\d{2}-\d{2}",
        # Common formats
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,?\s+\d{4})?",
        r"\d{1,2}/\d{1,2}/\d{2,4}",
        r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}",
        # Times
        r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?",
    ]

    def __init__(self, reference_time: Optional[datetime] = None):
        """
        Initialize the temporal extractor.

        Args:
            reference_time: Base time for resolving relative expressions.
                          Defaults to current time.
        """
        self.reference_time = reference_time or datetime.now()

    def set_reference_time(self, reference_time: datetime) -> None:
        """Update the reference time for relative expressions."""
        self.reference_time = reference_time

    def extract(self, text: str) -> List[TemporalExpression]:
        """
        Extract all temporal expressions from text.

        Args:
            text: Input text to analyze

        Returns:
            List of TemporalExpression objects
        """
        expressions = []

        # Extract absolute dates/times
        expressions.extend(self._extract_absolute(text))

        # Extract relative expressions
        expressions.extend(self._extract_relative(text))

        # Extract durations
        expressions.extend(self._extract_durations(text))

        # Extract sequence markers
        expressions.extend(self._extract_sequences(text))

        # Extract frequency expressions
        expressions.extend(self._extract_frequencies(text))

        # Sort by position in text
        expressions.sort(key=lambda e: e.span[0])

        return expressions

    def _extract_absolute(self, text: str) -> List[TemporalExpression]:
        """Extract absolute date/time expressions."""
        expressions = []
        text_lower = text.lower()

        for pattern in self.ABSOLUTE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                expr = TemporalExpression(
                    text=match.group(),
                    type=TemporalType.ABSOLUTE,
                    span=(match.start(), match.end()),
                )
                # Try to parse the date
                normalized = self._parse_absolute_date(match.group())
                if normalized:
                    expr.normalized_start = normalized
                expressions.append(expr)

        return expressions

    def _extract_relative(self, text: str) -> List[TemporalExpression]:
        """Extract relative time expressions."""
        expressions = []
        text_lower = text.lower()

        for phrase, (offset, unit) in self.RELATIVE_PATTERNS.items():
            pattern = r'\b' + re.escape(phrase) + r'\b'
            for match in re.finditer(pattern, text_lower):
                normalized = self._resolve_relative(offset, unit)
                expr = TemporalExpression(
                    text=match.group(),
                    type=TemporalType.RELATIVE,
                    normalized_start=normalized,
                    span=(match.start(), match.end()),
                )
                expressions.append(expr)

        return expressions

    def _extract_durations(self, text: str) -> List[TemporalExpression]:
        """Extract duration expressions."""
        expressions = []

        for pattern in self.DURATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) >= 2:
                    amount = int(groups[0])
                    unit = groups[1].lower()
                    duration = self._create_duration(amount, unit)

                    expr = TemporalExpression(
                        text=match.group(),
                        type=TemporalType.DURATION,
                        duration=duration,
                        span=(match.start(), match.end()),
                    )
                    expressions.append(expr)

        return expressions

    def _extract_sequences(self, text: str) -> List[TemporalExpression]:
        """Extract sequence/ordering expressions."""
        expressions = []
        text_lower = text.lower()

        for word, relation in self.SEQUENCE_WORDS.items():
            pattern = r'\b' + re.escape(word) + r'\b'
            for match in re.finditer(pattern, text_lower):
                expr = TemporalExpression(
                    text=match.group(),
                    type=TemporalType.SEQUENCE,
                    relation=relation,
                    span=(match.start(), match.end()),
                )
                expressions.append(expr)

        return expressions

    def _extract_frequencies(self, text: str) -> List[TemporalExpression]:
        """Extract frequency expressions."""
        expressions = []

        for pattern, freq_type in self.FREQUENCY_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                expr = TemporalExpression(
                    text=match.group(),
                    type=TemporalType.FREQUENCY,
                    relation=freq_type,
                    span=(match.start(), match.end()),
                )
                expressions.append(expr)

        return expressions

    def _resolve_relative(self, offset: int, unit: str) -> datetime:
        """Resolve a relative time expression to absolute datetime."""
        base = self.reference_time

        if unit == "day":
            return base + timedelta(days=offset)
        elif unit == "week":
            return base + timedelta(weeks=offset)
        elif unit == "month":
            # Approximate month as 30 days
            return base + timedelta(days=offset * 30)
        elif unit == "year":
            return base.replace(year=base.year + offset)
        elif unit == "morning":
            return base.replace(hour=9, minute=0, second=0)
        elif unit == "afternoon":
            return base.replace(hour=14, minute=0, second=0)
        elif unit == "evening":
            return base.replace(hour=18, minute=0, second=0)
        elif unit == "night":
            result = base + timedelta(days=offset)
            return result.replace(hour=21, minute=0, second=0)
        else:
            return base

    def _create_duration(self, amount: int, unit: str) -> timedelta:
        """Create a timedelta from amount and unit."""
        unit = unit.lower().rstrip('s')  # Remove plural

        if unit == "second":
            return timedelta(seconds=amount)
        elif unit == "minute":
            return timedelta(minutes=amount)
        elif unit == "hour":
            return timedelta(hours=amount)
        elif unit == "day":
            return timedelta(days=amount)
        elif unit == "week":
            return timedelta(weeks=amount)
        elif unit == "month":
            return timedelta(days=amount * 30)
        elif unit == "year":
            return timedelta(days=amount * 365)
        else:
            return timedelta()

    def _parse_absolute_date(self, text: str) -> Optional[datetime]:
        """Parse an absolute date string to datetime."""
        formats = [
            "%Y-%m-%d",
            "%B %d, %Y",
            "%B %d %Y",
            "%d %B %Y",
            "%d %b %Y",
            "%m/%d/%Y",
            "%m/%d/%y",
            "%d/%m/%Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(text.strip(), fmt)
            except ValueError:
                continue

        return None

    def get_temporal_context(self, expressions: List[TemporalExpression]) -> Dict[str, Any]:
        """
        Analyze extracted expressions to build temporal context.

        Returns structured temporal information for memory encoding.
        """
        context = {
            "has_temporal_info": len(expressions) > 0,
            "absolute_times": [],
            "relative_times": [],
            "durations": [],
            "sequences": [],
            "frequencies": [],
            "earliest_time": None,
            "latest_time": None,
        }

        times = []

        for expr in expressions:
            if expr.type == TemporalType.ABSOLUTE:
                context["absolute_times"].append({
                    "text": expr.text,
                    "normalized": expr.normalized_start.isoformat() if expr.normalized_start else None,
                })
                if expr.normalized_start:
                    times.append(expr.normalized_start)

            elif expr.type == TemporalType.RELATIVE:
                context["relative_times"].append({
                    "text": expr.text,
                    "normalized": expr.normalized_start.isoformat() if expr.normalized_start else None,
                })
                if expr.normalized_start:
                    times.append(expr.normalized_start)

            elif expr.type == TemporalType.DURATION:
                context["durations"].append({
                    "text": expr.text,
                    "seconds": expr.duration.total_seconds() if expr.duration else 0,
                })

            elif expr.type == TemporalType.SEQUENCE:
                context["sequences"].append({
                    "text": expr.text,
                    "relation": expr.relation,
                })

            elif expr.type == TemporalType.FREQUENCY:
                context["frequencies"].append({
                    "text": expr.text,
                    "type": expr.relation,
                })

        # Compute time bounds
        if times:
            context["earliest_time"] = min(times).isoformat()
            context["latest_time"] = max(times).isoformat()

        return context
