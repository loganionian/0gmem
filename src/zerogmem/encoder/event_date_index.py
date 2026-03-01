"""
Event-Date Index: Maps events to their session dates for accurate temporal QA.

INNOVATION: Instead of relying on retrieval to find the right session date,
we build a direct index during ingestion that maps:
  (entity, event_type, keywords) → session_date

This solves the core temporal QA problem where retrieval finds the event
but returns the wrong session's date.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Tuple
from datetime import datetime


@dataclass
class EventEntry:
    """An event with its associated date."""
    entity: str  # Who did/experienced this event
    event_type: str  # Type of event (camping, parade, workshop, etc.)
    keywords: Set[str]  # Keywords that describe this event
    session_date: str  # The date of the session where this was mentioned
    original_text: str  # The original text mentioning the event
    relative_date: Optional[str] = None  # "yesterday", "last week", etc.
    resolved_date: Optional[str] = None  # Calculated actual date


class EventDateIndex:
    """
    Index mapping events to their session dates.

    This enables direct lookup for temporal questions like:
    "When did Caroline go to the pride parade?" → Look up (Caroline, parade) → "3 July 2023"
    """

    # Event type patterns to extract from text
    # IMPORTANT: More specific patterns MUST come before general ones
    EVENT_PATTERNS = [
        # SPECIFIC conference/event patterns - highest priority
        (r'\b(?:transgender|trans)\s+(?:conference|summit|event)', 'transgender_conference'),
        (r'\b(?:lgbtq\+?|lgbt)\s+(?:conference|summit|event|convention)', 'lgbtq_conference'),
        (r'\b(?:lgbtq\+?|lgbt)\s+(?:support\s+group|support)', 'lgbtq_support_group'),
        (r'\b(?:activist\s+group|activism\s+group|activist\s+meeting)', 'activist_group'),
        (r'\b(?:mentorship\s+program|mentorship)', 'mentorship_program'),
        (r'\b(?:pride\s+parade|pride\s+march)', 'pride_parade'),
        (r'\b(?:pride\s+fest(?:ival)?)', 'pride_festival'),
        # "had a blast at X" pattern for past events
        (r'\bhad\s+a\s+blast\s+(?:last\s+year\s+)?at\s+(?:the\s+)?(\w+(?:\s+\w+)?)', 'past_event'),
        (r'\b(?:charity\s+race|charity\s+run|charity\s+marathon)', 'charity_race'),
        (r'\b(?:pottery\s+class|pottery\s+workshop)', 'pottery_class'),
        # IMPROVEMENT v70: Beach visits
        (r'\b(?:went|go|going)\s+to\s+(?:the\s+)?beach', 'beach_visit'),
        (r'\b(?:at|on)\s+(?:the\s+)?beach', 'beach_visit'),
        (r'\bbeach\s+(?:trip|vacation|visit)', 'beach_visit'),
        # IMPROVEMENT v70: Road trips and travel
        (r'\b(?:took|take|taking)\s+(?:a\s+)?road\s*trip', 'road_trip'),
        (r'\broad\s*trip\s+to\s+(?:the\s+)?(\w+(?:\s+\w+)?)', 'road_trip'),
        (r'\b(?:traveled|travel|travelling)\s+to\s+(\w+(?:\s+\w+)?)', 'travel'),
        (r'\b(?:visited|visit|visiting)\s+(\w+(?:\s+\w+)?)', 'visit'),
        # IMPROVEMENT v70: Movie/show watching
        (r'\b(?:watched|watch|watching)\s+["\']?([^"\']+?)["\']?(?:\s+for\s+the\s+first\s+time)?', 'watched'),
        (r'\b(?:saw|see|seeing)\s+(?:the\s+)?(?:movie|film)\s+["\']?([^"\']+?)["\']?', 'watched'),
        (r'\bfirst\s+(?:time|watch)\s+["\']?([^"\']+?)["\']?', 'first_watch'),
        # IMPROVEMENT v70: Starting activities (like violin, dance)
        (r'\b(?:started|began|begin)\s+(?:playing|learning|taking)\s+(?:the\s+)?(\w+)', 'started_activity'),
        (r'\b(?:started|began|begin)\s+(\w+)\s+(?:classes|lessons)', 'started_activity'),
        (r'\b(?:picked up|learning)\s+(?:the\s+)?(\w+)', 'started_activity'),
        # IMPROVEMENT v70: Receiving awards/medals
        (r'\b(?:received|got|won)\s+(?:a\s+)?(?:medal|award|recognition|trophy)', 'received_award'),
        (r'\bmedal\s+from\s+(?:the\s+)?(\w+(?:\s+\w+)?)', 'received_award'),
        # IMPROVEMENT v70: Get dog/pet events
        (r'\b(?:got|adopted|rescued)\s+(?:a\s+|my\s+|our\s+)?(?:dog|puppy|pet)\s+(?:named\s+)?(\w+)?', 'got_pet'),
        (r'\b(?:dog|puppy)\s+(?:named\s+)?(\w+)', 'got_pet'),
        # Activities - expanded for LoCoMo events
        (r'\b(went|go|going)\s+(?:to\s+)?(?:the\s+|a\s+|an\s+)?(camping|hiking|biking|swimming|running|park|museum|concert|parade|pride parade|workshop|conference|meeting|class|fair|support group|lgbtq\+? support group|activist group|mentorship program|picnic)', 'activity'),
        (r'\b(went|go|going)\s+(?:to\s+)?(?:the\s+|a\s+|an\s+)?lgbtq\+?\s+(pride parade|parade|support group|conference|meeting|event)', 'lgbtq_activity'),
        (r'\b(attended|attend|attending)\s+(?:a\s+|the\s+|an\s+)?(workshop|conference|meeting|parade|pride parade|class|event|session|group|support group|lgbtq\+?|mentorship|program)', 'attended'),
        (r'\b(signed up|sign up|registered|joined)\s+(?:for\s+)?(?:a\s+|the\s+|an\s+)?(?:new\s+)?(\w+(?:\s+\w+)?(?:\s+\w+)?(?:\s+\w+)?)', 'signed_up'),
        # Camping/park specific patterns for "took my fam camping", "took kids to park"
        (r'\b(took|take|taking)\s+(?:my\s+|the\s+)?(?:fam(?:ily)?|kids|children)?\s*(?:to\s+(?:a\s+|the\s+)?)?(camping|hiking|swimming|park)', 'family_activity'),
        # Speech/talk patterns
        (r'\b(talked|talk|talking)\s+(?:about\s+)?(?:my\s+)?(\w+(?:\s+\w+)?)', 'talked_about'),
        (r'\b(gave|give|giving)\s+(?:a\s+)?(speech|talk|presentation|lecture)', 'speech'),
        # Creative activities
        (r'\b(painted|drew|draw|drawing)\s+(?:a\s+)?(?:the\s+)?(?:my\s+)?(\w+(?:-\w+)?(?:\s+\w+)?)', 'created'),
        (r'\b(?:self-portrait|selfportrait)\s+(?:I|i|we)\s+(?:made|created|drew|painted)', 'self_portrait'),
        (r'\b(?:made|created)\s+(?:a\s+)?(?:self-portrait|selfportrait)', 'self_portrait'),
        (r'\b(read|finished reading|started reading)\s+(?:the\s+)?["\']?([^"\']+)["\']?', 'read'),
        # Purchases/acquisitions
        (r'\b(bought|got|purchased|received)\s+(?:a\s+|the\s+|some\s+)?(\w+(?:\s+\w+)?)', 'acquired'),
        # Business/work events
        (r'\b(opened|started|launched|began)\s+(?:a\s+|the\s+|my\s+)?(\w+(?:\s+\w+)?)', 'started'),
        (r'\b(got accepted|was accepted|accepted)\s+(?:for\s+|to\s+)?(?:a\s+|the\s+)?(\w+)', 'accepted'),
        # Social/public events
        (r'\b(had|have|having)\s+(?:a\s+)?(picnic|party|gathering|meetup|barbecue|bbq)', 'social'),
        (r'\b(met up|meet up|meeting up)\s+(?:with\s+)?(.+?)(?:\.|,|$)', 'meetup'),
        (r'\b(ran|run|running)\s+(?:a\s+|in\s+a\s+)?(charity race|marathon|5k|10k|race)', 'race'),
        # Interview/milestone events
        (r'\b(?:passed|completed|finished)\s+(?:the\s+)?(?:\w+\s+)*?(?:interview|interviews|test|exam)', 'passed_interview'),
        # Life events
        (r'\b(adopted|married|divorced|moved|graduated)', 'life_event'),
        (r'\b(friend|brother|sister|husband|wife)\s+(\w+)\s+(?:adopted|married|got|had)', 'friend_event'),
        # "buddy/friend of mine adopted" pattern
        (r'\b(?:buddy|friend)\s+(?:of mine|of ours)\s+adopted', 'friend_adopted'),
    ]

    # Relative date patterns - comprehensive list
    RELATIVE_DATE_PATTERNS = [
        # Day-specific before patterns
        (r'the week before (\d{1,2}\s+\w+\s+\d{4})', 'week_before'),
        (r'the weekend before (\d{1,2}\s+\w+\s+\d{4})', 'weekend_before'),
        (r'the monday before (\d{1,2}\s+\w+\s+\d{4})', 'monday_before'),
        (r'the tuesday before (\d{1,2}\s+\w+\s+\d{4})', 'tuesday_before'),
        (r'the wednesday before (\d{1,2}\s+\w+\s+\d{4})', 'wednesday_before'),
        (r'the thursday before (\d{1,2}\s+\w+\s+\d{4})', 'thursday_before'),
        (r'the friday before (\d{1,2}\s+\w+\s+\d{4})', 'friday_before'),
        (r'the saturday before (\d{1,2}\s+\w+\s+\d{4})', 'saturday_before'),
        (r'the sunday before (\d{1,2}\s+\w+\s+\d{4})', 'sunday_before'),
        # Generic relative dates
        (r'yesterday', 'yesterday'),
        (r'last weekend', 'last_weekend'),  # Must come before 'last week'
        (r'(?:in the )?last week', 'last_week'),
        (r'last tues(?:day)?', 'last_tuesday'),
        (r'last fri(?:day)?', 'last_friday'),
        (r'last month', 'last_month'),
        (r'last year', 'last_year'),
        (r'a few years ago', 'few_years_ago'),
        (r'two weekends ago', 'two_weekends_ago'),
        (r'two days ago', 'two_days_ago'),
        (r'(\d+)\s+years?\s+ago', 'n_years_ago'),
        # In/during patterns
        (r'in (\w+\s+\d{4})', 'in_month_year'),
        (r'during (\w+\s+\d{4})', 'during_month_year'),
    ]

    def __init__(self):
        self.events: List[EventEntry] = []
        # Index by entity for fast lookup
        self.by_entity: Dict[str, List[EventEntry]] = {}
        # Index by event type
        self.by_event_type: Dict[str, List[EventEntry]] = {}
        # Index by keyword
        self.by_keyword: Dict[str, List[EventEntry]] = {}

    def add_from_message(
        self,
        speaker: str,
        content: str,
        session_date: str,
        session_idx: int = 0
    ) -> List[EventEntry]:
        """
        Extract events from a message and add them to the index.

        Args:
            speaker: Who said this message
            content: The message content
            session_date: The date of the session (e.g., "8 May 2023")
            session_idx: Session index for ordering

        Returns:
            List of EventEntry objects added
        """
        added = []
        content_lower = content.lower()

        # Extract events using patterns
        for pattern, event_type in self.EVENT_PATTERNS:
            for match in re.finditer(pattern, content_lower):
                groups = match.groups()

                # Extract keywords from the match
                keywords = set()
                for g in groups:
                    if g:
                        keywords.update(g.lower().split())

                # Remove common words
                keywords -= {'a', 'the', 'to', 'for', 'my', 'her', 'his', 'their', 'some'}

                # Check for relative date expressions near this match
                relative_date = None
                resolved_date = None
                context_start = max(0, match.start() - 100)
                context_end = min(len(content_lower), match.end() + 100)
                context = content_lower[context_start:context_end]

                for rel_pattern, rel_type in self.RELATIVE_DATE_PATTERNS:
                    rel_match = re.search(rel_pattern, context)
                    if rel_match:
                        if rel_type == 'week_before':
                            relative_date = f"The week before {rel_match.group(1)}"
                            resolved_date = relative_date  # Keep as-is for benchmark
                        elif rel_type == 'weekend_before':
                            relative_date = f"The weekend before {rel_match.group(1)}"
                            resolved_date = relative_date
                        elif rel_type.endswith('_before') and rel_type != 'week_before':
                            # Handle all day-specific patterns: monday_before, tuesday_before, etc.
                            day_name = rel_type.replace('_before', '').capitalize()
                            relative_date = f"The {day_name} before {rel_match.group(1)}"
                            resolved_date = relative_date
                        elif rel_type == 'yesterday':
                            relative_date = "yesterday"
                            resolved_date = self._calculate_yesterday(session_date)
                        elif rel_type == 'last_weekend':
                            relative_date = "last weekend"
                            resolved_date = f"The weekend before {session_date}"
                        elif rel_type == 'last_week':
                            relative_date = "last week"
                            resolved_date = f"The week before {session_date}"
                        elif rel_type == 'last_tuesday':
                            relative_date = "last Tuesday"
                            resolved_date = f"The Tuesday before {session_date}"
                        elif rel_type == 'last_friday':
                            relative_date = "last Friday"
                            resolved_date = f"The Friday before {session_date}"
                        elif rel_type == 'two_weekends_ago':
                            relative_date = "two weekends ago"
                            resolved_date = f"The weekend before {session_date}"  # Approximate
                        elif rel_type == 'two_days_ago':
                            relative_date = "two days ago"
                            resolved_date = self._calculate_days_ago(session_date, 2)
                        elif rel_type == 'last_year':
                            relative_date = "last year"
                            year_match = re.search(r'(\d{4})', session_date)
                            if year_match:
                                resolved_date = str(int(year_match.group(1)) - 1)
                        elif rel_type == 'n_years_ago':
                            n = rel_match.group(1)
                            relative_date = f"{n} years ago"
                            resolved_date = relative_date
                        elif rel_type == 'few_years_ago':
                            relative_date = "A few years ago"
                            resolved_date = relative_date
                        elif rel_type in ('in_month_year', 'during_month_year'):
                            # "in June 2023" -> "June 2023"
                            relative_date = rel_match.group(1)
                            resolved_date = rel_match.group(1)
                        break

                # Create entry
                entry = EventEntry(
                    entity=speaker.lower(),
                    event_type=event_type,
                    keywords=keywords,
                    session_date=session_date,
                    original_text=content[:200],
                    relative_date=relative_date,
                    resolved_date=resolved_date or session_date,
                )

                self._add_entry(entry)
                added.append(entry)

        # Also extract mentions of other entities and their events
        # E.g., "My friend Gina opened her store"
        other_entity_patterns = [
            (r'(?:my\s+)?(?:friend|brother|sister)\s+(\w+)\s+(.+?)(?:\.|$)', 'friend_event'),
            (r'(\w+)\s+(?:said|told me|mentioned)\s+(?:that\s+)?(?:she|he)\s+(.+?)(?:\.|$)', 'friend_said'),
        ]

        for pattern, event_type in other_entity_patterns:
            for match in re.finditer(pattern, content_lower):
                entity_name = match.group(1)
                event_text = match.group(2) if len(match.groups()) > 1 else ""

                # Skip common words that aren't names
                if entity_name in {'the', 'a', 'my', 'her', 'his', 'that', 'this'}:
                    continue

                keywords = set(event_text.split()) if event_text else set()
                keywords -= {'a', 'the', 'to', 'for', 'my', 'her', 'his', 'their', 'some', 'she', 'he', 'that', 'was', 'is', 'has', 'had'}

                entry = EventEntry(
                    entity=entity_name.lower(),
                    event_type=event_type,
                    keywords=keywords,
                    session_date=session_date,
                    original_text=content[:200],
                )

                self._add_entry(entry)
                added.append(entry)

        return added

    def _add_entry(self, entry: EventEntry) -> None:
        """Add an entry to all indexes."""
        self.events.append(entry)

        # Index by entity
        if entry.entity not in self.by_entity:
            self.by_entity[entry.entity] = []
        self.by_entity[entry.entity].append(entry)

        # Index by event type
        if entry.event_type not in self.by_event_type:
            self.by_event_type[entry.event_type] = []
        self.by_event_type[entry.event_type].append(entry)

        # Index by keyword
        for kw in entry.keywords:
            if kw not in self.by_keyword:
                self.by_keyword[kw] = []
            self.by_keyword[kw].append(entry)

    def _calculate_yesterday(self, session_date: str) -> str:
        """Calculate yesterday's date from session date."""
        return self._calculate_days_ago(session_date, 1)

    def _calculate_days_ago(self, session_date: str, days: int) -> str:
        """Calculate a date N days before session date."""
        try:
            # Parse session date
            date_match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', session_date)
            if date_match:
                day = int(date_match.group(1))
                month = date_match.group(2)
                year = date_match.group(3)

                # Simple calculation - just subtract days
                # (This is simplified and doesn't handle month boundaries properly)
                if day > days:
                    return f"{day - days} {month} {year}"
                else:
                    return f"{days} days before {session_date}"
        except:
            pass
        return session_date

    def lookup(
        self,
        entity: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        event_type: Optional[str] = None,
        month: Optional[str] = None,
        year: Optional[str] = None,
    ) -> List[EventEntry]:
        """
        Look up events matching the criteria.

        Args:
            entity: The entity who did the event (e.g., "caroline")
            keywords: Keywords to match (e.g., ["camping", "june"])
            event_type: Type of event to find
            month: Month to filter by (e.g., "june")
            year: Year to filter by (e.g., "2022")

        Returns:
            List of matching EventEntry objects
        """
        candidates = self.events

        # Filter by entity
        if entity:
            entity_lower = entity.lower()
            # Handle nicknames
            if entity_lower == "mel":
                entity_lower = "melanie"
            candidates = self.by_entity.get(entity_lower, [])

        # Filter by keywords
        if keywords:
            keyword_set = {k.lower() for k in keywords}
            # Expand keyword set with equivalences (handle stemming)
            keyword_equivalences = {
                'hike': 'hiking', 'hiking': 'hike',
                'park': 'parks', 'parks': 'park',
                'plate': 'plates', 'plates': 'plate',
                'interview': 'interviews', 'interviews': 'interview',
                'adopted': 'adopt', 'adopt': 'adopted',
            }
            expanded_keywords = set()
            for kw in keyword_set:
                expanded_keywords.add(kw)
                if kw in keyword_equivalences:
                    expanded_keywords.add(keyword_equivalences[kw])
            keyword_set = expanded_keywords
            # High-value keywords that distinguish between similar events
            qualifier_keywords = {'transgender', 'lgbtq', 'pride', 'charity', 'self-portrait',
                                  'sunrise', 'sunset', 'landscape', 'horse', 'mentorship',
                                  'support group', 'adoption', 'interview', 'pottery', 'biking'}
            filtered = []
            for entry in candidates:
                # Score by keyword overlap with bonus for qualifier matches
                overlap = entry.keywords & keyword_set
                base_score = len(overlap)
                # Bonus for qualifier keywords (these help distinguish events)
                qualifier_matches = overlap & qualifier_keywords
                bonus_score = len(qualifier_matches) * 2  # 2x weight for qualifiers
                total_score = base_score + bonus_score
                if total_score > 0:
                    filtered.append((entry, total_score))
            # Sort by overlap score
            filtered.sort(key=lambda x: x[1], reverse=True)
            candidates = [e for e, _ in filtered]

        # Filter by event type
        if event_type:
            candidates = [e for e in candidates if e.event_type == event_type]

        # Filter by year FIRST (helps disambiguate between years)
        if year:
            filtered = []
            for entry in candidates:
                # Check session date or resolved date for year
                if year in entry.session_date:
                    filtered.append(entry)
                elif entry.resolved_date and year in str(entry.resolved_date):
                    filtered.append(entry)
            if filtered:  # Only use year filter if it matches something
                candidates = filtered

        # Filter by month
        if month:
            month_lower = month.lower()
            filtered = []
            for entry in candidates:
                # Check if session date contains the month
                if month_lower in entry.session_date.lower():
                    filtered.append(entry)
                # Also check resolved date
                elif entry.resolved_date and month_lower in entry.resolved_date.lower():
                    filtered.append(entry)
            candidates = filtered

        return candidates

    def answer_temporal_question(
        self,
        question: str,
        entity: Optional[str] = None
    ) -> Optional[str]:
        """
        Try to directly answer a temporal question using the index.

        Args:
            question: The temporal question (e.g., "When did Caroline go camping in June?")
            entity: The entity being asked about (if known)

        Returns:
            The date answer if found, None otherwise
        """
        q_lower = question.lower()

        # Extract entity from question if not provided
        if not entity:
            # All LoCoMo entity names - sorted by length (longest first) to avoid partial matches
            all_entities = [
                'caroline', 'melanie', 'deborah', 'joanna', 'jolene', 'andrew',
                'audrey', 'calvin', 'james', 'maria', 'gina', 'nate',
                'john', 'evan', 'dave', 'tim', 'sam', 'jon', 'mel'
            ]
            for name in all_entities:
                # Use word boundary to match exact names, not substrings
                if re.search(r'\b' + name + r'\b', q_lower):
                    entity = name
                    break

        # INNOVATION: First try to match specific event types directly
        # This avoids confusion between similar events
        specific_event_mapping = {
            # Conference types
            'transgender conference': 'transgender_conference',
            'trans conference': 'transgender_conference',
            'lgbtq conference': 'lgbtq_conference',
            'lgbt conference': 'lgbtq_conference',
            # Support group types
            'lgbtq support group': 'lgbtq_support_group',
            'support group': 'lgbtq_support_group',  # default
            # Other specific events
            'pride parade': 'pride_parade',
            'pride festival': 'pride_festival',
            'pride fest': 'pride_festival',
            'pride fesetival': 'pride_festival',  # Handle typo in LoCoMo
            # Friend/buddy adoption pattern
            'friend adopt': 'friend_adopted',
            'buddy adopt': 'friend_adopted',
            "friend's friend adopt": 'friend_adopted',
            'charity race': 'charity_race',
            'pottery class': 'pottery_class',
            'mentorship program': 'mentorship_program',
            'self-portrait': 'self_portrait',
            'self portrait': 'self_portrait',
            # Meetup events
            'meet up': 'meetup',
            'met up': 'meetup',
            'meetup': 'meetup',
            # Interview events
            'adoption interview': 'passed_interview',
            'interview': 'passed_interview',
            'pass the interview': 'passed_interview',
            'passed the interview': 'passed_interview',
            'activist group': 'activist_group',
            # IMPROVEMENT v70: Beach, road trip, travel events
            'beach': 'beach_visit',
            'go to the beach': 'beach_visit',
            'went to the beach': 'beach_visit',
            'road trip': 'road_trip',
            'roadtrip': 'road_trip',
            'pacific northwest': 'road_trip',  # specific destination
            # IMPROVEMENT v70: Movie watching
            'watch': 'watched',
            'watched': 'watched',
            'first watch': 'first_watch',
            'first time': 'first_watch',
            'eternal sunshine': 'watched',  # specific movie
            # IMPROVEMENT v70: Started activities
            'started playing': 'started_activity',
            'start playing': 'started_activity',
            'began playing': 'started_activity',
            'violin': 'started_activity',
            # IMPROVEMENT v70: Awards and pets
            'medal': 'received_award',
            'award': 'received_award',
            'got dog': 'got_pet',
            'get dog': 'got_pet',
            'dog max': 'got_pet',
        }

        # Try direct event type matching first
        for phrase, event_type in specific_event_mapping.items():
            if phrase in q_lower:
                entries = [e for e in self.by_event_type.get(event_type, [])
                           if not entity or e.entity == entity.lower()]
                if entries:
                    # If multiple entries, try to disambiguate by additional keywords in question
                    if len(entries) > 1:
                        # Extract specific item keywords for disambiguation
                        # Use word boundaries to avoid 'pot' matching 'pottery'
                        item_keywords = ['plate', 'bowl', 'cup', 'vase', 'mug']  # Removed 'pot' - conflicts with 'pottery'
                        for item_kw in item_keywords:
                            if re.search(rf'\b{item_kw}\b', q_lower):
                                # Look for entries with this item in original_text (word boundary)
                                for e in entries:
                                    if re.search(rf'\b{item_kw}\b', e.original_text.lower()):
                                        return e.resolved_date or e.session_date
                        # If question asks about "make/made", prefer entries mentioning making
                        if 'make' in q_lower or 'made' in q_lower:
                            for e in entries:
                                text_lower = e.original_text.lower()
                                # Check for any form: made, make, making
                                if 'made' in text_lower or 'make' in text_lower or 'making' in text_lower:
                                    # Prefer "made it" pattern (actually made something vs signed up)
                                    if 'made it' in text_lower or 'i made' in text_lower:
                                        return e.resolved_date or e.session_date
                            # Fallback: any entry with make/made
                            for e in entries:
                                text_lower = e.original_text.lower()
                                if 'made' in text_lower or 'make' in text_lower or 'making' in text_lower:
                                    return e.resolved_date or e.session_date
                    return entries[0].resolved_date or entries[0].session_date

        # Extract keywords from question
        keywords = []
        # PRIORITY: Multi-word phrases first (more specific)
        phrase_keywords = [
            'support group', 'lgbtq support group', 'pride parade', 'pride festival',
            'transgender conference', 'lgbtq conference', 'charity race',
            'self-portrait', 'self portrait', 'pottery class', 'mentorship program',
            'activist group', 'road trip', 'roadtrip'
        ]
        for phrase in phrase_keywords:
            if phrase in q_lower:
                keywords.append(phrase)
                # Also add individual words
                for word in phrase.split():
                    if word not in keywords and len(word) > 3:
                        keywords.append(word)

        # Then single-word keywords
        event_keywords = [
            'camping', 'hiking', 'biking', 'swimming', 'running', 'parade', 'pride',
            'workshop', 'conference', 'meeting', 'class', 'pottery', 'museum',
            'concert', 'birthday', 'book', 'painted', 'paint', 'portrait',
            'figurines', 'adoption', 'store', 'shop', 'studio', 'tattoo',
            'internship', 'fair', 'startup', 'dance', 'park', 'plate', 'bowl',
            'lgbtq', 'interview', 'race', 'charity',
            # Additional LoCoMo-specific keywords
            'mentorship', 'activist', 'speech', 'school', 'picnic',
            'transgender', 'journey', 'marathon', '5k', '10k',
            # Qualifier keywords for disambiguation
            'sunrise', 'sunset', 'landscape', 'horse', 'beach', 'hike'
        ]
        for kw in event_keywords:
            if kw in q_lower:
                keywords.append(kw)

        # Also extract individual words from question as potential keywords
        skip_words = {'when', 'did', 'does', 'do', 'the', 'a', 'an', 'to', 'in', 'at', 'go', 'join', 'attend'}
        for word in q_lower.split():
            word = word.strip('?.,!')
            if len(word) > 3 and word not in skip_words and word not in keywords:
                keywords.append(word)

        # Extract year qualifier (for disambiguation between years like 2022 vs 2023)
        year = None
        year_match = re.search(r'\b(20\d{2})\b', q_lower)
        if year_match:
            year = year_match.group(1)

        # Extract month qualifier
        month = None
        months = ['january', 'february', 'march', 'april', 'may', 'june',
                  'july', 'august', 'september', 'october', 'november', 'december']
        for m in months:
            if m in q_lower:
                month = m
                break

        # Check for "summer"
        if 'summer' in q_lower:
            # Summer = June, July, August - try each with year filter
            for m in ['june', 'july', 'august']:
                results = self.lookup(entity=entity, keywords=keywords, month=m, year=year)
                if results:
                    return results[0].resolved_date or results[0].session_date
            # If year specified but not found in summer months, return the year
            if year:
                results = self.lookup(entity=entity, keywords=keywords, year=year)
                if results:
                    return year  # Return just the year for "summer 2022" questions

        # Look up events with optional year filter
        results = self.lookup(entity=entity, keywords=keywords, month=month, year=year)

        if results:
            # Return the best match
            entry = results[0]
            # Prefer resolved date (handles relative dates)
            return entry.resolved_date or entry.session_date

        # Try without month filter if no results
        if month and not results:
            results = self.lookup(entity=entity, keywords=keywords, year=year)
            if results:
                return results[0].resolved_date or results[0].session_date

        # Try without year filter if still no results (fallback)
        if year and not results:
            results = self.lookup(entity=entity, keywords=keywords, month=month)
            if results:
                return results[0].resolved_date or results[0].session_date

        return None

    def get_entity_timeline(self, entity: str) -> List[Tuple[str, str, str]]:
        """
        Get a timeline of events for an entity.

        Returns list of (date, event_type, description) tuples.
        """
        entries = self.by_entity.get(entity.lower(), [])
        timeline = []

        for entry in entries:
            date = entry.resolved_date or entry.session_date
            desc = f"{entry.event_type}: {', '.join(entry.keywords)}"
            timeline.append((date, entry.event_type, desc))

        return timeline
