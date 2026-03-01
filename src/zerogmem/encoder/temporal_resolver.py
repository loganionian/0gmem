"""
Temporal Resolver: Resolves relative time expressions to absolute dates.

Critical for LoCoMo temporal questions that require date calculation.
Example: Message says "yesterday", session timestamp is "8 May 2023"
         → Resolved date is "7 May 2023"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta


@dataclass
class ResolvedDate:
    """A resolved absolute date from a relative expression."""
    original_text: str
    resolved_date: datetime
    confidence: float
    resolution_method: str  # "relative", "absolute", "inferred"
    context: Optional[str] = None


@dataclass
class TemporalContext:
    """Temporal context for a message or conversation."""
    session_date: Optional[datetime] = None
    message_dates: List[ResolvedDate] = field(default_factory=list)
    event_dates: Dict[str, datetime] = field(default_factory=dict)
    date_references: List[str] = field(default_factory=list)


class TemporalResolver:
    """
    Resolves relative time expressions using session timestamps.

    Handles:
    - "yesterday", "today", "tomorrow"
    - "last week", "next week"
    - "X days ago", "in X days"
    - "the week before [date]"
    - "the sunday before [date]"
    """

    # Relative patterns with their offsets
    RELATIVE_PATTERNS = [
        # Days
        (r"\byesterday\b", -1, "day"),
        (r"\btoday\b", 0, "day"),
        (r"\btomorrow\b", 1, "day"),
        (r"\bthe day before yesterday\b", -2, "day"),
        (r"\bthe other day\b", -2, "day"),
        (r"\ba few days ago\b", -3, "day"),
        (r"\bseveral days ago\b", -4, "day"),

        # Weeks
        (r"\blast week\b", -1, "week"),
        (r"\bthis week\b", 0, "week"),
        (r"\bnext week\b", 1, "week"),
        (r"\bthe week before\b", -1, "week"),
        (r"\ba week ago\b", -1, "week"),
        (r"\btwo weeks ago\b", -2, "week"),

        # Months
        (r"\blast month\b", -1, "month"),
        (r"\bthis month\b", 0, "month"),
        (r"\bnext month\b", 1, "month"),
        (r"\ba month ago\b", -1, "month"),

        # Years
        (r"\blast year\b", -1, "year"),
        (r"\bthis year\b", 0, "year"),
        (r"\bnext year\b", 1, "year"),
    ]

    # N units ago patterns
    N_UNITS_AGO = [
        (r"(\d+)\s+days?\s+ago", "day"),
        (r"(\d+)\s+weeks?\s+ago", "week"),
        (r"(\d+)\s+months?\s+ago", "month"),
        (r"(\d+)\s+years?\s+ago", "year"),
    ]

    # Day of week patterns
    DAY_OF_WEEK_MAP = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }

    # Month patterns
    MONTH_MAP = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }

    def __init__(self, default_year: int = 2023):
        """Initialize with default year for incomplete dates."""
        self.default_year = default_year

    def parse_session_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """
        Parse session timestamp from LoCoMo format.

        Examples:
        - "8 May 2023"
        - "25 May 2023"
        - "9 June 2023"
        """
        if not timestamp_str:
            return None

        timestamp_str = timestamp_str.strip()

        # Try common formats
        formats = [
            "%d %B %Y",      # 8 May 2023
            "%d %b %Y",      # 8 May 2023 (short month)
            "%B %d, %Y",     # May 8, 2023
            "%Y-%m-%d",      # 2023-05-08
            "%m/%d/%Y",      # 05/08/2023
            "%d/%m/%Y",      # 08/05/2023
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        # Try dateutil parser as fallback
        try:
            return date_parser.parse(timestamp_str, dayfirst=True)
        except:
            pass

        return None

    def resolve(
        self,
        text: str,
        reference_date: datetime,
        context: Optional[str] = None,
    ) -> List[ResolvedDate]:
        """
        Resolve all temporal expressions in text.

        Args:
            text: Text containing temporal expressions
            reference_date: The reference date (e.g., session timestamp)
            context: Optional context for logging

        Returns:
            List of resolved dates
        """
        resolved = []
        text_lower = text.lower()

        # 1. Check for simple relative patterns
        for pattern, offset, unit in self.RELATIVE_PATTERNS:
            if re.search(pattern, text_lower):
                resolved_date = self._apply_offset(reference_date, offset, unit)
                resolved.append(ResolvedDate(
                    original_text=pattern.replace(r"\b", ""),
                    resolved_date=resolved_date,
                    confidence=0.95,
                    resolution_method="relative",
                    context=context,
                ))

        # 2. Check for "N units ago" patterns
        for pattern, unit in self.N_UNITS_AGO:
            for match in re.finditer(pattern, text_lower):
                n = int(match.group(1))
                resolved_date = self._apply_offset(reference_date, -n, unit)
                resolved.append(ResolvedDate(
                    original_text=match.group(0),
                    resolved_date=resolved_date,
                    confidence=0.95,
                    resolution_method="relative",
                    context=context,
                ))

        # 3. Check for "the [day] before [date]" pattern
        day_before_pattern = r"the\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s+before\s+(\d{1,2}\s+\w+\s+\d{4})"
        for match in re.finditer(day_before_pattern, text_lower):
            target_day = match.group(1)
            date_str = match.group(2)

            ref_date = self.parse_session_timestamp(date_str)
            if ref_date:
                resolved_date = self._find_previous_day(ref_date, target_day)
                resolved.append(ResolvedDate(
                    original_text=match.group(0),
                    resolved_date=resolved_date,
                    confidence=0.9,
                    resolution_method="computed",
                    context=context,
                ))

        # 4. Check for "the week before [date]" pattern
        week_before_pattern = r"the\s+week\s+before\s+(\d{1,2}\s+\w+\s+\d{4})"
        for match in re.finditer(week_before_pattern, text_lower):
            date_str = match.group(1)
            ref_date = self.parse_session_timestamp(date_str)
            if ref_date:
                resolved_date = ref_date - timedelta(weeks=1)
                resolved.append(ResolvedDate(
                    original_text=match.group(0),
                    resolved_date=resolved_date,
                    confidence=0.9,
                    resolution_method="computed",
                    context=context,
                ))

        # 5. Check for absolute dates in text
        absolute_patterns = [
            r"(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})",
            r"(\d{4})-(\d{2})-(\d{2})",
        ]

        for pattern in absolute_patterns:
            for match in re.finditer(pattern, text_lower):
                try:
                    parsed = self.parse_session_timestamp(match.group(0))
                    if parsed:
                        resolved.append(ResolvedDate(
                            original_text=match.group(0),
                            resolved_date=parsed,
                            confidence=1.0,
                            resolution_method="absolute",
                            context=context,
                        ))
                except:
                    pass

        # 6. Check for years (e.g., "in 2022")
        year_pattern = r"\b(19|20)\d{2}\b"
        for match in re.finditer(year_pattern, text_lower):
            year = int(match.group(0))
            # Only add if it's a standalone year reference
            if match.start() > 0 and text_lower[match.start()-1:match.start()+1].strip().isdigit():
                continue
            resolved.append(ResolvedDate(
                original_text=match.group(0),
                resolved_date=datetime(year, 6, 15),  # Mid-year approximation
                confidence=0.7,
                resolution_method="year_only",
                context=context,
            ))

        return resolved

    def _apply_offset(self, base: datetime, offset: int, unit: str) -> datetime:
        """Apply a time offset to a base date."""
        if unit == "day":
            return base + timedelta(days=offset)
        elif unit == "week":
            return base + timedelta(weeks=offset)
        elif unit == "month":
            return base + relativedelta(months=offset)
        elif unit == "year":
            return base + relativedelta(years=offset)
        else:
            return base

    def _find_previous_day(self, reference: datetime, day_name: str) -> datetime:
        """Find the previous occurrence of a day of week before reference."""
        target_weekday = self.DAY_OF_WEEK_MAP.get(day_name.lower(), 0)

        # Start from reference and go backwards
        current = reference - timedelta(days=1)

        while current.weekday() != target_weekday:
            current -= timedelta(days=1)

        return current

    def build_temporal_context(
        self,
        messages: List[Dict[str, Any]],
        session_timestamp: Optional[str] = None,
    ) -> TemporalContext:
        """
        Build temporal context for a conversation session.

        Args:
            messages: List of message dicts with 'content' key
            session_timestamp: Optional session timestamp string

        Returns:
            TemporalContext with resolved dates
        """
        context = TemporalContext()

        # Parse session date
        if session_timestamp:
            context.session_date = self.parse_session_timestamp(session_timestamp)

        if not context.session_date:
            context.session_date = datetime.now()

        # Process each message
        for i, msg in enumerate(messages):
            content = msg.get("content", msg.get("text", ""))
            if not content:
                continue

            # Resolve dates in this message
            resolved = self.resolve(
                content,
                context.session_date,
                context=f"message_{i}",
            )

            context.message_dates.extend(resolved)

            # Store significant event dates
            for r in resolved:
                if r.confidence >= 0.8:
                    # Use part of original text as key
                    key = r.original_text[:30]
                    context.event_dates[key] = r.resolved_date
                    context.date_references.append(f"{key}: {r.resolved_date.strftime('%Y-%m-%d')}")

        return context

    def answer_temporal_question(
        self,
        question: str,
        temporal_context: TemporalContext,
        retrieved_content: str,
    ) -> Optional[str]:
        """
        Try to answer a temporal question using resolved dates.

        Args:
            question: The question asking about a date
            temporal_context: Resolved temporal context
            retrieved_content: Retrieved memory content

        Returns:
            Formatted date answer if found, None otherwise
        """
        question_lower = question.lower()

        # Check for questions asking "when"
        if not any(word in question_lower for word in ["when", "what date", "what time"]):
            return None

        # Look for relevant dates in context
        for original, resolved_date in temporal_context.event_dates.items():
            # Check if the original expression or related keywords appear
            if any(word in retrieved_content.lower() for word in original.split()):
                return resolved_date.strftime("%d %B %Y")

        # Check message dates
        for rd in temporal_context.message_dates:
            if rd.confidence >= 0.8:
                # Check relevance to question
                question_words = set(question_lower.split())
                if question_words.intersection(set(rd.original_text.lower().split())):
                    return rd.resolved_date.strftime("%d %B %Y")

        return None
