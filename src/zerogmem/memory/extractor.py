"""
MemCell Extractor: Extract atomic memory units from conversation messages.

Uses both rule-based and LLM-based extraction to identify:
- Episodes (activities, events)
- Facts (static information)
- Preferences (likes, dislikes)
- Relations (relationships with other people)
- Plans (future intentions)
- Achievements (accomplishments)
"""

from __future__ import annotations

import re
import json
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from .memcell import MemCell, MemScene, CellType, SceneType, MemoryStore


class MemCellExtractor:
    """
    Extract MemCells from conversation messages.

    Combines rule-based patterns with optional LLM extraction
    for comprehensive memory extraction.
    """

    # Patterns for different cell types
    EPISODE_PATTERNS = [
        # Activities with "went/go/going" - expanded to catch more variations
        (r'\b(i|we)\s+(went|go|going)\s+(?:to\s+)?(?:the\s+|a\s+|an\s+)?(.+?)(?:\.|,|!|\?|$)', 'activity'),
        # "went to the beach/park/etc." with location
        (r'\b(i|we)\s+(went)\s+to\s+(?:the\s+)?(\w+(?:\s+\w+)?)(?:\s+(?:recently|yesterday|last|this|with))?', 'visit'),
        # Activities with "attended"
        (r'\b(i|we)\s+(attended|attend|attending)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\.|,|!|\?|$)', 'attended'),
        # Activities with "joined"
        (r'\b(i|we)\s+(joined|join|joining)\s+(?:the\s+|a\s+|an\s+)?(.+?)(?:\.|,|!|\?|$)', 'joined'),
        # Activities with "started"
        (r'\b(i|we)\s+(started|start|starting)\s+(?:to\s+)?(.+?)(?:\.|,|!|\?|$)', 'started'),
        # Activities with verbs - expanded
        (r'\b(i|we)\s+(painted|drew|made|created|read|finished|bought|got|received|visited|saw|watched)\s+(.+?)(?:\.|,|!|\?|$)', 'created'),
        # Past activities
        (r'\b(i|we)\s+(had|have|having)\s+(?:a\s+)?(.+?)(?:\.|,|!|\?|$)', 'had'),
        # Took/take activities
        (r'\b(i|we)\s+(took|take|taking)\s+(?:my\s+|the\s+)?(.+?)(?:\.|,|!|\?|$)', 'took'),
        # Gave/give
        (r'\b(i|we)\s+(gave|give|giving)\s+(?:a\s+)?(.+?)(?:\.|,|!|\?|$)', 'gave'),
        # Ran/run
        (r'\b(i|we)\s+(ran|run|running)\s+(?:a\s+|in\s+a\s+)?(.+?)(?:\.|,|!|\?|$)', 'ran'),
        # Met/meet
        (r'\b(i|we)\s+(met|meet|meeting)\s+(?:up\s+)?(?:with\s+)?(.+?)(?:\.|,|!|\?|$)', 'met'),
        # "it was" activity descriptions
        (r'\b(it|that)\s+(was)\s+(?:really\s+|so\s+|very\s+)?(\w+)(?:\.|,|!|\?|$)', 'experience'),
        # "kids had a blast" type activities
        (r'\b(kids?|children|family)\s+(had|have)\s+(?:a\s+|such\s+a\s+)?(\w+(?:\s+\w+)?)(?:\.|,|!|\?|$)', 'family_activity'),
    ]

    FACT_PATTERNS = [
        # "I am/I'm" statements
        (r"\b(i'?m|i am)\s+(?:a\s+|an\s+)?(\w+(?:\s+\w+)?)", 'identity'),
        # "I have" statements - extended match for longer possessions
        (r'\b(i have|i\'ve got)\s+(?:a\s+|an\s+|two\s+|three\s+|lots of\s+|many\s+)?(.+?)(?:\.|,|!|\?|$)', 'possession'),
        # "I've got lots of X" - specific pattern for collections
        (r"\b(i'?ve got|i have)\s+(?:lots of|many|some)\s+(.+?)(?:\.|,|!|\?|-|$)", 'collection'),
        # "My X is/are"
        (r'\bmy\s+(\w+(?:\s+\w+)?)\s+(?:is|are)\s+(.+?)(?:\.|,|!|\?|$)', 'attribute'),
        # "I live in/at"
        (r'\bi\s+live\s+(?:in|at)\s+(.+?)(?:\.|,|!|\?|$)', 'location'),
        # "I work at/for/as"
        (r'\bi\s+work\s+(?:at|for|as)\s+(?:a\s+|an\s+)?(.+?)(?:\.|,|!|\?|$)', 'occupation'),
        # "I studied/study"
        (r'\bi\s+(?:studied|study|studied at)\s+(.+?)(?:\.|,|!|\?|$)', 'education'),
        # "Since YEAR"
        (r'\bsince\s+(\d{4})', 'since'),
        # "for X years"
        (r'\bfor\s+(\d+)\s+years?', 'duration'),
        # "I collect/have collected"
        (r'\bi\s+(collect|have collected|like to collect)\s+(.+?)(?:\.|,|!|\?|$)', 'hobby_collection'),
    ]

    PREFERENCE_PATTERNS = [
        # "I love/like/enjoy"
        (r'\bi\s+(love|like|enjoy|adore)\s+(.+?)(?:\.|,|!|\?|$)', 'likes'),
        # "I hate/dislike/don't like"
        (r"\bi\s+(hate|dislike|don't like|can't stand)\s+(.+?)(?:\.|,|!|\?|$)", 'dislikes'),
        # "My favorite"
        (r'\bmy\s+favorite\s+(\w+)\s+(?:is|are)\s+(.+?)(?:\.|,|!|\?|$)', 'favorite'),
        # "I prefer"
        (r'\bi\s+prefer\s+(.+?)(?:\s+(?:to|over)\s+.+)?(?:\.|,|!|\?|$)', 'prefers'),
    ]

    RELATION_PATTERNS = [
        # "My friend/brother/sister X"
        (r'\bmy\s+(friend|brother|sister|husband|wife|mom|dad|mother|father|son|daughter|cousin|aunt|uncle)\s+(\w+)', 'family_friend'),
        # "X is my friend"
        (r'\b(\w+)\s+is\s+my\s+(friend|brother|sister|husband|wife|colleague|coworker)', 'relation'),
        # Possessive relations
        (r"\b(\w+)'s\s+(husband|wife|friend|brother|sister)", 'third_party_relation'),
    ]

    PLAN_PATTERNS = [
        # "I'm planning to"
        (r"\bi'?m\s+planning\s+(?:to|on)\s+(.+?)(?:\.|,|!|\?|$)", 'planning'),
        # "I want to"
        (r'\bi\s+want\s+to\s+(.+?)(?:\.|,|!|\?|$)', 'want'),
        # "I'm hoping to"
        (r"\bi'?m\s+hoping\s+to\s+(.+?)(?:\.|,|!|\?|$)", 'hoping'),
        # "I'm going to"
        (r"\bi'?m\s+going\s+to\s+(.+?)(?:\.|,|!|\?|$)", 'going_to'),
        # "Next month/year/week"
        (r'\bnext\s+(month|year|week|summer|winter)\s+(.+?)(?:\.|,|!|\?|$)', 'future'),
    ]

    # Relative date patterns for temporal resolution
    RELATIVE_DATE_PATTERNS = [
        (r'the week before (\d{1,2}\s+\w+(?:,?\s+\d{4})?)', 'week_before'),
        (r'the weekend before (\d{1,2}\s+\w+(?:,?\s+\d{4})?)', 'weekend_before'),
        (r'the (\w+day) before (\d{1,2}\s+\w+(?:,?\s+\d{4})?)', 'day_before'),
        (r'yesterday', 'yesterday'),
        (r'last (\w+)', 'last'),
        (r'(\d+)\s+(days?|weeks?|months?|years?)\s+ago', 'n_ago'),
        (r'in (\w+\s+\d{4})', 'in_date'),
        (r'on (\d{1,2}\s+\w+(?:,?\s+\d{4})?)', 'on_date'),
    ]

    def __init__(self, llm_client=None):
        """
        Initialize the extractor.

        Args:
            llm_client: Optional OpenAI client for LLM-based extraction
        """
        self.llm_client = llm_client
        self.memory_store = MemoryStore()

    def extract_from_message(
        self,
        speaker: str,
        content: str,
        session_id: str,
        session_date: str,
        session_idx: int = 0,
        use_llm: bool = False
    ) -> List[MemCell]:
        """
        Extract MemCells from a single message.

        Args:
            speaker: Who said the message
            content: Message content
            session_id: ID of the conversation session
            session_date: Date of the session (e.g., "8 May 2023")
            session_idx: Session index for ordering
            use_llm: Whether to use LLM for extraction (slower but more accurate)

        Returns:
            List of extracted MemCells
        """
        cells = []
        content_lower = content.lower()

        # Extract relative date context for the message
        event_date = self._extract_event_date(content_lower, session_date)

        # Rule-based extraction
        cells.extend(self._extract_episodes(speaker, content, content_lower, session_id, session_date, session_idx, event_date))
        cells.extend(self._extract_facts(speaker, content, content_lower, session_id, session_date, session_idx))
        cells.extend(self._extract_preferences(speaker, content, content_lower, session_id, session_date, session_idx))
        cells.extend(self._extract_relations(speaker, content, content_lower, session_id, session_date, session_idx))
        cells.extend(self._extract_plans(speaker, content, content_lower, session_id, session_date, session_idx))

        # LLM-based extraction for complex cases
        if use_llm and self.llm_client:
            llm_cells = self._extract_with_llm(speaker, content, session_id, session_date, session_idx)
            cells.extend(llm_cells)

        # Add all cells to memory store
        for cell in cells:
            self.memory_store.add_cell(cell)

        return cells

    def _extract_event_date(self, content_lower: str, session_date: str) -> Optional[str]:
        """Extract the actual event date from relative expressions."""
        for pattern, pattern_type in self.RELATIVE_DATE_PATTERNS:
            match = re.search(pattern, content_lower)
            if match:
                if pattern_type == 'week_before':
                    return f"The week before {match.group(1)}"
                elif pattern_type == 'weekend_before':
                    return f"The weekend before {match.group(1)}"
                elif pattern_type == 'day_before':
                    return f"The {match.group(1).capitalize()} before {match.group(2)}"
                elif pattern_type == 'yesterday':
                    return self._calculate_yesterday(session_date)
                elif pattern_type == 'last':
                    period = match.group(1)
                    if period == 'week':
                        return f"The week before {session_date}"
                    elif period == 'weekend':
                        return f"The weekend before {session_date}"
                    elif period in ['tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'monday']:
                        return f"The {period.capitalize()} before {session_date}"
                    elif period == 'year':
                        year_match = re.search(r'(\d{4})', session_date)
                        if year_match:
                            return str(int(year_match.group(1)) - 1)
                elif pattern_type == 'n_ago':
                    n, unit = match.groups()
                    return f"{n} {unit} ago"
                elif pattern_type in ('in_date', 'on_date'):
                    return match.group(1)
        return None

    def _calculate_yesterday(self, session_date: str) -> str:
        """Calculate yesterday's date from session date."""
        try:
            date_match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', session_date)
            if date_match:
                day = int(date_match.group(1))
                month = date_match.group(2)
                year = date_match.group(3)
                if day > 1:
                    return f"{day - 1} {month} {year}"
        except:
            pass
        return session_date

    def _extract_episodes(
        self, speaker: str, content: str, content_lower: str,
        session_id: str, session_date: str, session_idx: int,
        event_date: Optional[str]
    ) -> List[MemCell]:
        """Extract episode cells (activities, events)."""
        cells = []

        for pattern, action_type in self.EPISODE_PATTERNS:
            for match in re.finditer(pattern, content_lower):
                groups = match.groups()

                # Extract the activity description
                activity = groups[-1] if groups else ""
                if not activity or len(activity) < 3:
                    continue

                # Clean up the activity
                activity = activity.strip()
                activity = re.sub(r'\s+', ' ', activity)

                # Extract keywords
                keywords = set(activity.split())
                keywords -= {'a', 'the', 'to', 'for', 'my', 'her', 'his', 'their', 'some', 'an', 'in', 'at', 'on'}

                # Add action type as keyword
                keywords.add(action_type)

                cell = MemCell(
                    id="",
                    content=f"{speaker} {groups[1] if len(groups) > 1 else 'did'} {activity}",
                    cell_type=CellType.EPISODE,
                    entity=speaker.lower(),
                    session_date=session_date,
                    event_date=event_date or session_date,
                    session_id=session_id,
                    session_idx=session_idx,
                    speaker=speaker,
                    original_text=content[:200],
                    keywords=keywords,
                )
                cells.append(cell)

        return cells

    def _extract_facts(
        self, speaker: str, content: str, content_lower: str,
        session_id: str, session_date: str, session_idx: int
    ) -> List[MemCell]:
        """Extract fact cells (static information)."""
        cells = []

        for pattern, fact_type in self.FACT_PATTERNS:
            for match in re.finditer(pattern, content_lower):
                groups = match.groups()

                if fact_type == 'identity':
                    fact_content = f"{speaker} is {groups[1] if len(groups) > 1 else groups[0]}"
                elif fact_type == 'possession':
                    fact_content = f"{speaker} has {groups[1] if len(groups) > 1 else groups[0]}"
                elif fact_type == 'collection':
                    fact_content = f"{speaker} has lots of {groups[1] if len(groups) > 1 else groups[0]}"
                elif fact_type == 'hobby_collection':
                    fact_content = f"{speaker} collects {groups[1] if len(groups) > 1 else groups[0]}"
                elif fact_type == 'attribute':
                    fact_content = f"{speaker}'s {groups[0]} is {groups[1]}"
                elif fact_type == 'location':
                    fact_content = f"{speaker} lives in {groups[0]}"
                elif fact_type == 'occupation':
                    fact_content = f"{speaker} works as/at {groups[0]}"
                elif fact_type in ('since', 'duration'):
                    fact_content = f"{speaker} has been doing something since/for {groups[0]}"
                else:
                    fact_content = f"{speaker}: {' '.join(groups)}"

                # Extract keywords
                keywords = set()
                for g in groups:
                    if g:
                        keywords.update(g.split())
                keywords -= {'a', 'the', 'to', 'for', 'my', 'is', 'are', 'i', 'we'}
                keywords.add(fact_type)

                cell = MemCell(
                    id="",
                    content=fact_content,
                    cell_type=CellType.FACT,
                    entity=speaker.lower(),
                    session_date=session_date,
                    session_id=session_id,
                    session_idx=session_idx,
                    speaker=speaker,
                    original_text=content[:200],
                    keywords=keywords,
                )
                cells.append(cell)

        return cells

    def _extract_preferences(
        self, speaker: str, content: str, content_lower: str,
        session_id: str, session_date: str, session_idx: int
    ) -> List[MemCell]:
        """Extract preference cells (likes, dislikes)."""
        cells = []

        for pattern, pref_type in self.PREFERENCE_PATTERNS:
            for match in re.finditer(pattern, content_lower):
                groups = match.groups()

                if pref_type == 'likes':
                    pref_content = f"{speaker} loves/likes {groups[1]}"
                elif pref_type == 'dislikes':
                    pref_content = f"{speaker} dislikes {groups[1]}"
                elif pref_type == 'favorite':
                    pref_content = f"{speaker}'s favorite {groups[0]} is {groups[1]}"
                elif pref_type == 'prefers':
                    pref_content = f"{speaker} prefers {groups[0]}"
                else:
                    pref_content = f"{speaker}: {' '.join(groups)}"

                keywords = set()
                for g in groups:
                    if g:
                        keywords.update(g.split())
                keywords.add(pref_type)

                cell = MemCell(
                    id="",
                    content=pref_content,
                    cell_type=CellType.PREFERENCE,
                    entity=speaker.lower(),
                    session_date=session_date,
                    session_id=session_id,
                    session_idx=session_idx,
                    speaker=speaker,
                    original_text=content[:200],
                    keywords=keywords,
                )
                cells.append(cell)

        return cells

    def _extract_relations(
        self, speaker: str, content: str, content_lower: str,
        session_id: str, session_date: str, session_idx: int
    ) -> List[MemCell]:
        """Extract relation cells (relationships)."""
        cells = []

        for pattern, rel_type in self.RELATION_PATTERNS:
            for match in re.finditer(pattern, content_lower):
                groups = match.groups()

                if rel_type == 'family_friend':
                    relation_type = groups[0]
                    related_name = groups[1]
                    rel_content = f"{speaker}'s {relation_type} is {related_name}"
                    related_entities = {related_name.lower()}
                elif rel_type == 'relation':
                    related_name = groups[0]
                    relation_type = groups[1]
                    rel_content = f"{related_name} is {speaker}'s {relation_type}"
                    related_entities = {related_name.lower()}
                else:
                    rel_content = f"{speaker}: {' '.join(groups)}"
                    related_entities = set()

                keywords = set(groups)
                keywords.add(rel_type)

                cell = MemCell(
                    id="",
                    content=rel_content,
                    cell_type=CellType.RELATION,
                    entity=speaker.lower(),
                    session_date=session_date,
                    session_id=session_id,
                    session_idx=session_idx,
                    speaker=speaker,
                    original_text=content[:200],
                    keywords=keywords,
                    related_entities=related_entities,
                )
                cells.append(cell)

        return cells

    def _extract_plans(
        self, speaker: str, content: str, content_lower: str,
        session_id: str, session_date: str, session_idx: int
    ) -> List[MemCell]:
        """Extract plan cells (future intentions)."""
        cells = []

        for pattern, plan_type in self.PLAN_PATTERNS:
            for match in re.finditer(pattern, content_lower):
                groups = match.groups()

                plan_content = f"{speaker} is planning/wants to {groups[-1] if groups else 'something'}"

                keywords = set()
                for g in groups:
                    if g:
                        keywords.update(g.split())
                keywords.add(plan_type)
                keywords.add('plan')
                keywords.add('future')

                cell = MemCell(
                    id="",
                    content=plan_content,
                    cell_type=CellType.PLAN,
                    entity=speaker.lower(),
                    session_date=session_date,
                    session_id=session_id,
                    session_idx=session_idx,
                    speaker=speaker,
                    original_text=content[:200],
                    keywords=keywords,
                )
                cells.append(cell)

        return cells

    def _extract_with_llm(
        self, speaker: str, content: str,
        session_id: str, session_date: str, session_idx: int
    ) -> List[MemCell]:
        """Use LLM to extract complex memories."""
        if not self.llm_client:
            return []

        prompt = f"""Extract atomic facts from this message. Return a JSON array.

Message from {speaker}: "{content}"
Session date: {session_date}

For each fact, provide:
- content: The fact in a complete sentence
- type: One of [episode, fact, preference, relation, plan, achievement, emotion]
- keywords: List of key words
- event_date: When this happened (if mentioned), null otherwise
- related_entities: Names of other people mentioned

Example output:
[
  {{"content": "Caroline went camping", "type": "episode", "keywords": ["camping", "outdoor"], "event_date": "last week", "related_entities": []}}
]

Extract ALL facts, even small ones. Return empty array [] if no facts found.

JSON:"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0,
            )

            result = response.choices[0].message.content.strip()

            # Parse JSON
            if result.startswith('['):
                facts = json.loads(result)
            else:
                # Try to find JSON in the response
                json_match = re.search(r'\[.*\]', result, re.DOTALL)
                if json_match:
                    facts = json.loads(json_match.group())
                else:
                    return []

            cells = []
            for fact in facts:
                cell_type_map = {
                    'episode': CellType.EPISODE,
                    'fact': CellType.FACT,
                    'preference': CellType.PREFERENCE,
                    'relation': CellType.RELATION,
                    'plan': CellType.PLAN,
                    'achievement': CellType.ACHIEVEMENT,
                    'emotion': CellType.EMOTION,
                }

                cell = MemCell(
                    id="",
                    content=fact.get('content', ''),
                    cell_type=cell_type_map.get(fact.get('type', 'fact'), CellType.FACT),
                    entity=speaker.lower(),
                    session_date=session_date,
                    event_date=fact.get('event_date'),
                    session_id=session_id,
                    session_idx=session_idx,
                    speaker=speaker,
                    original_text=content[:200],
                    keywords=set(fact.get('keywords', [])),
                    related_entities=set(fact.get('related_entities', [])),
                )
                cells.append(cell)

            return cells

        except Exception as e:
            print(f"LLM extraction error: {e}")
            return []


class MemSceneBuilder:
    """
    Build MemScenes by grouping related MemCells.

    Groups cells based on:
    - Same entity + same topic/activity
    - Same session (temporal proximity)
    - Related entities
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.scenes: Dict[str, MemScene] = {}

    def build_scenes(self, cells: List[MemCell], memory_store: MemoryStore) -> List[MemScene]:
        """
        Build scenes from a collection of cells.

        Groups cells into coherent scenes and generates summaries.
        """
        scenes = []

        # Group by entity
        cells_by_entity: Dict[str, List[MemCell]] = {}
        for cell in cells:
            if cell.entity not in cells_by_entity:
                cells_by_entity[cell.entity] = []
            cells_by_entity[cell.entity].append(cell)

        # For each entity, group by topic/activity
        for entity, entity_cells in cells_by_entity.items():
            entity_scenes = self._cluster_cells_by_topic(entity, entity_cells)
            scenes.extend(entity_scenes)

        # Generate summaries for scenes
        for scene in scenes:
            scene.summary = self._generate_scene_summary(scene)
            memory_store.add_scene(scene)

        return scenes

    def _cluster_cells_by_topic(self, entity: str, cells: List[MemCell]) -> List[MemScene]:
        """Cluster cells into topic-based scenes."""
        scenes = []

        # Group by cell type first
        by_type: Dict[CellType, List[MemCell]] = {}
        for cell in cells:
            if cell.cell_type not in by_type:
                by_type[cell.cell_type] = []
            by_type[cell.cell_type].append(cell)

        # Create scenes for each type
        type_to_scene_type = {
            CellType.EPISODE: SceneType.ACTIVITY,
            CellType.FACT: SceneType.CONVERSATION_TOPIC,
            CellType.PREFERENCE: SceneType.CONVERSATION_TOPIC,
            CellType.RELATION: SceneType.RELATIONSHIP,
            CellType.PLAN: SceneType.LIFE_EVENT,
            CellType.ACHIEVEMENT: SceneType.LIFE_EVENT,
        }

        for cell_type, type_cells in by_type.items():
            if not type_cells:
                continue

            # Further cluster by keywords
            keyword_clusters = self._cluster_by_keywords(type_cells)

            for cluster_name, cluster_cells in keyword_clusters.items():
                scene = MemScene(
                    id="",
                    cells=cluster_cells,
                    scene_type=type_to_scene_type.get(cell_type, SceneType.CONVERSATION_TOPIC),
                    title=f"{entity.title()}'s {cluster_name}",
                    entities={entity},
                )

                for cell in cluster_cells:
                    scene.keywords.update(cell.keywords)
                    scene.entities.update(cell.related_entities)
                    if cell.session_id:
                        scene.session_ids.add(cell.session_id)
                    cell.scene_id = scene.id

                scenes.append(scene)

        return scenes

    def _cluster_by_keywords(self, cells: List[MemCell]) -> Dict[str, List[MemCell]]:
        """Cluster cells by keyword similarity."""
        if not cells:
            return {}

        # Simple clustering: group by most common keyword
        keyword_counts: Dict[str, int] = {}
        for cell in cells:
            for kw in cell.keywords:
                keyword_counts[kw] = keyword_counts.get(kw, 0) + 1

        # Get top keywords
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        clusters: Dict[str, List[MemCell]] = {}
        assigned = set()

        for kw, count in top_keywords:
            if count < 2:
                continue
            cluster_cells = [c for c in cells if kw in c.keywords and c.id not in assigned]
            if cluster_cells:
                clusters[kw] = cluster_cells
                assigned.update(c.id for c in cluster_cells)

        # Add remaining cells to "other"
        remaining = [c for c in cells if c.id not in assigned]
        if remaining:
            clusters['other'] = remaining

        return clusters

    def _generate_scene_summary(self, scene: MemScene) -> str:
        """Generate a summary for a scene."""
        if not scene.cells:
            return ""

        # If LLM available, use it
        if self.llm_client:
            return self._generate_summary_with_llm(scene)

        # Rule-based summary
        contents = [c.content for c in scene.cells[:5]]
        entity = list(scene.entities)[0] if scene.entities else "Someone"

        if scene.scene_type == SceneType.ACTIVITY:
            return f"{entity.title()} activities: " + "; ".join(contents[:3])
        elif scene.scene_type == SceneType.RELATIONSHIP:
            return f"{entity.title()}'s relationships and social connections"
        else:
            return f"Information about {entity.title()}: " + "; ".join(contents[:3])

    def _generate_summary_with_llm(self, scene: MemScene) -> str:
        """Use LLM to generate scene summary."""
        if not self.llm_client:
            return ""

        contents = [c.content for c in scene.cells[:10]]
        entity = list(scene.entities)[0] if scene.entities else "the person"

        prompt = f"""Summarize these related memories about {entity} in 1-2 sentences:

{chr(10).join(f'- {c}' for c in contents)}

Summary:"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except:
            return ""
