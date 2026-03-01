"""
MemCell and MemScene: Hierarchical Memory Structures

Inspired by EverMemOS's engram-based memory organization:
- MemCell: Atomic memory unit (single fact, event, preference, or relation)
- MemScene: Collection of related MemCells forming coherent context

This enables:
1. Fine-grained fact storage and retrieval
2. Context-aware grouping of related memories
3. Progressive profile building from aggregated cells
4. Efficient retrieval through scene-level indexing
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Set, Any, Tuple
from enum import Enum


class CellType(Enum):
    """Types of atomic memory cells."""
    EPISODE = "episode"         # Specific event or activity (e.g., "went camping")
    FACT = "fact"               # Static fact (e.g., "has two cats")
    PREFERENCE = "preference"   # Like/dislike (e.g., "loves hiking")
    RELATION = "relation"       # Relationship (e.g., "friend Gina")
    PLAN = "plan"               # Future intention (e.g., "planning to adopt")
    EMOTION = "emotion"         # Emotional state (e.g., "felt empowered")
    ACHIEVEMENT = "achievement" # Accomplishment (e.g., "got promoted")


class SceneType(Enum):
    """Types of memory scenes."""
    ACTIVITY = "activity"           # Group of related activities
    LIFE_EVENT = "life_event"       # Major life event (adoption, marriage, etc.)
    CONVERSATION_TOPIC = "topic"    # Discussion about a specific topic
    RELATIONSHIP = "relationship"   # Interactions about a person
    HOBBY = "hobby"                 # Hobby-related memories
    WORK = "work"                   # Work/career related
    TEMPORAL_CLUSTER = "temporal"   # Same time period


@dataclass
class MemCell:
    """
    Atomic memory unit - one fact, event, or statement.

    This is the smallest unit of memory that can be independently
    stored, retrieved, and reasoned about.
    """
    id: str
    content: str                    # The actual memory content
    cell_type: CellType             # Type of memory
    entity: str                     # Primary entity this is about

    # Temporal information
    session_date: str               # Date of the session where mentioned
    timestamp: Optional[datetime] = None  # Exact timestamp if available
    event_date: Optional[str] = None      # When the event actually happened

    # Source information
    session_id: str = ""            # Which session this came from
    session_idx: int = 0            # Session index for ordering
    speaker: str = ""               # Who said this
    original_text: str = ""         # Original message text

    # Extraction metadata
    confidence: float = 1.0         # Confidence in extraction
    keywords: Set[str] = field(default_factory=set)

    # Relations
    related_entities: Set[str] = field(default_factory=set)
    related_cells: List[str] = field(default_factory=list)  # IDs of related cells

    # Scene membership
    scene_id: Optional[str] = None  # Which scene this belongs to

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if isinstance(self.cell_type, str):
            self.cell_type = CellType(self.cell_type)
        if isinstance(self.keywords, list):
            self.keywords = set(self.keywords)
        if isinstance(self.related_entities, list):
            self.related_entities = set(self.related_entities)

    def matches_query(self, keywords: List[str], entity: Optional[str] = None) -> float:
        """
        Score how well this cell matches a query.

        Returns a score from 0 to 1.
        """
        score = 0.0

        # Entity match
        if entity:
            if entity.lower() == self.entity.lower():
                score += 0.3
            elif entity.lower() in [e.lower() for e in self.related_entities]:
                score += 0.15

        # Keyword match
        if keywords:
            cell_keywords = self.keywords | set(self.content.lower().split())
            query_keywords = set(k.lower() for k in keywords)
            overlap = len(cell_keywords & query_keywords)
            if overlap > 0:
                score += min(0.7, overlap * 0.2)

        return min(1.0, score)

    def to_context_string(self, include_date: bool = True) -> str:
        """Convert to a string suitable for LLM context."""
        parts = []
        if self.speaker:
            parts.append(f"[{self.speaker}]")
        if include_date and self.session_date:
            parts.append(f"(Session: {self.session_date})")
        parts.append(f": {self.content}")
        return " ".join(parts)


@dataclass
class MemScene:
    """
    Collection of related MemCells forming coherent context.

    Scenes group memories that:
    - Relate to the same topic/activity
    - Involve the same entities
    - Occurred in the same time period
    """
    id: str
    cells: List[MemCell] = field(default_factory=list)
    scene_type: SceneType = SceneType.CONVERSATION_TOPIC

    # Scene metadata
    title: str = ""                 # Short title (e.g., "Caroline's camping trips")
    summary: str = ""               # LLM-generated summary
    entities: Set[str] = field(default_factory=set)
    keywords: Set[str] = field(default_factory=set)

    # Temporal bounds
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    session_ids: Set[str] = field(default_factory=set)

    # For retrieval
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())[:8]
        if isinstance(self.scene_type, str):
            self.scene_type = SceneType(self.scene_type)
        if isinstance(self.entities, list):
            self.entities = set(self.entities)
        if isinstance(self.keywords, list):
            self.keywords = set(self.keywords)
        if isinstance(self.session_ids, list):
            self.session_ids = set(self.session_ids)

    def add_cell(self, cell: MemCell) -> None:
        """Add a cell to this scene."""
        self.cells.append(cell)
        cell.scene_id = self.id

        # Update scene metadata
        self.entities.add(cell.entity)
        self.entities.update(cell.related_entities)
        self.keywords.update(cell.keywords)

        if cell.session_id:
            self.session_ids.add(cell.session_id)

        # Update temporal bounds
        if cell.session_date:
            if not self.start_date or cell.session_date < self.start_date:
                self.start_date = cell.session_date
            if not self.end_date or cell.session_date > self.end_date:
                self.end_date = cell.session_date

    def matches_query(self, keywords: List[str], entity: Optional[str] = None) -> float:
        """Score how well this scene matches a query."""
        score = 0.0

        # Entity match
        if entity:
            if entity.lower() in [e.lower() for e in self.entities]:
                score += 0.3

        # Keyword match at scene level
        if keywords:
            query_keywords = set(k.lower() for k in keywords)
            scene_keywords = self.keywords | set(self.summary.lower().split())
            overlap = len(scene_keywords & query_keywords)
            if overlap > 0:
                score += min(0.4, overlap * 0.15)

        # Aggregate cell scores
        if self.cells and keywords:
            cell_scores = [c.matches_query(keywords, entity) for c in self.cells]
            max_cell_score = max(cell_scores) if cell_scores else 0
            score += max_cell_score * 0.3

        return min(1.0, score)

    def get_best_cells(self, keywords: List[str], entity: Optional[str] = None, top_k: int = 5) -> List[MemCell]:
        """Get the most relevant cells for a query."""
        scored = [(c, c.matches_query(keywords, entity)) for c in self.cells]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, s in scored[:top_k] if s > 0]

    def to_context_string(self, max_cells: int = 10, include_summary: bool = True) -> str:
        """Convert to a string suitable for LLM context."""
        parts = []

        if include_summary and self.summary:
            parts.append(f"## {self.title or 'Memory Scene'}")
            parts.append(f"Summary: {self.summary}")
            parts.append("")

        parts.append("Details:")
        for cell in self.cells[:max_cells]:
            parts.append(f"- {cell.to_context_string()}")

        if len(self.cells) > max_cells:
            parts.append(f"  ... and {len(self.cells) - max_cells} more memories")

        return "\n".join(parts)


class MemoryStore:
    """
    Storage and indexing for MemCells and MemScenes.

    Provides:
    - Cell and scene storage
    - Multi-index retrieval (by entity, keyword, date, type)
    - Scene-based context composition
    """

    def __init__(self):
        # Primary storage
        self.cells: Dict[str, MemCell] = {}
        self.scenes: Dict[str, MemScene] = {}

        # Indexes
        self.cells_by_entity: Dict[str, List[str]] = {}      # entity -> [cell_ids]
        self.cells_by_type: Dict[CellType, List[str]] = {}   # type -> [cell_ids]
        self.cells_by_session: Dict[str, List[str]] = {}     # session_id -> [cell_ids]
        self.scenes_by_entity: Dict[str, List[str]] = {}     # entity -> [scene_ids]
        self.scenes_by_type: Dict[SceneType, List[str]] = {} # type -> [scene_ids]

        # Keyword index (inverted index)
        self.keyword_index: Dict[str, Set[str]] = {}  # keyword -> {cell_ids}

    def add_cell(self, cell: MemCell) -> None:
        """Add a cell to storage and indexes."""
        self.cells[cell.id] = cell

        # Index by entity
        entity_lower = cell.entity.lower()
        if entity_lower not in self.cells_by_entity:
            self.cells_by_entity[entity_lower] = []
        self.cells_by_entity[entity_lower].append(cell.id)

        # Index by type
        if cell.cell_type not in self.cells_by_type:
            self.cells_by_type[cell.cell_type] = []
        self.cells_by_type[cell.cell_type].append(cell.id)

        # Index by session
        if cell.session_id:
            if cell.session_id not in self.cells_by_session:
                self.cells_by_session[cell.session_id] = []
            self.cells_by_session[cell.session_id].append(cell.id)

        # Keyword index
        for kw in cell.keywords:
            kw_lower = kw.lower()
            if kw_lower not in self.keyword_index:
                self.keyword_index[kw_lower] = set()
            self.keyword_index[kw_lower].add(cell.id)

    def add_scene(self, scene: MemScene) -> None:
        """Add a scene to storage and indexes."""
        self.scenes[scene.id] = scene

        # Index by entities
        for entity in scene.entities:
            entity_lower = entity.lower()
            if entity_lower not in self.scenes_by_entity:
                self.scenes_by_entity[entity_lower] = []
            self.scenes_by_entity[entity_lower].append(scene.id)

        # Index by type
        if scene.scene_type not in self.scenes_by_type:
            self.scenes_by_type[scene.scene_type] = []
        self.scenes_by_type[scene.scene_type].append(scene.id)

    def get_cells_by_entity(self, entity: str) -> List[MemCell]:
        """Get all cells about an entity."""
        cell_ids = self.cells_by_entity.get(entity.lower(), [])
        return [self.cells[cid] for cid in cell_ids]

    def get_cells_by_keywords(self, keywords: List[str]) -> List[MemCell]:
        """Get cells matching any of the keywords."""
        cell_ids = set()
        for kw in keywords:
            cell_ids.update(self.keyword_index.get(kw.lower(), set()))
        return [self.cells[cid] for cid in cell_ids]

    def get_scenes_by_entity(self, entity: str) -> List[MemScene]:
        """Get all scenes involving an entity."""
        scene_ids = self.scenes_by_entity.get(entity.lower(), [])
        return [self.scenes[sid] for sid in scene_ids]

    def search(
        self,
        query_keywords: List[str],
        entity: Optional[str] = None,
        cell_types: Optional[List[CellType]] = None,
        top_k: int = 20
    ) -> List[Tuple[MemCell, float]]:
        """
        Search for relevant cells.

        Returns list of (cell, score) tuples.
        """
        # Get candidate cells
        if entity:
            candidates = set(self.cells_by_entity.get(entity.lower(), []))
        else:
            candidates = set(self.cells.keys())

        # Filter by type if specified
        if cell_types:
            type_cells = set()
            for ct in cell_types:
                type_cells.update(self.cells_by_type.get(ct, []))
            candidates &= type_cells

        # Expand with keyword matches
        for kw in query_keywords:
            candidates.update(self.keyword_index.get(kw.lower(), set()))

        # Score candidates
        scored = []
        for cell_id in candidates:
            cell = self.cells[cell_id]
            score = cell.matches_query(query_keywords, entity)
            if score > 0:
                scored.append((cell, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_scenes(
        self,
        query_keywords: List[str],
        entity: Optional[str] = None,
        top_k: int = 5
    ) -> List[Tuple[MemScene, float]]:
        """
        Search for relevant scenes.

        Returns list of (scene, score) tuples.
        """
        if entity:
            candidates = self.scenes_by_entity.get(entity.lower(), [])
        else:
            candidates = list(self.scenes.keys())

        scored = []
        for scene_id in candidates:
            scene = self.scenes[scene_id]
            score = scene.matches_query(query_keywords, entity)
            if score > 0:
                scored.append((scene, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def compose_context(
        self,
        query_keywords: List[str],
        entity: Optional[str] = None,
        max_cells: int = 20,
        include_scene_summaries: bool = True
    ) -> str:
        """
        Compose retrieval context from matching scenes and cells.

        This is the main retrieval interface.
        """
        parts = []

        # Get relevant scenes
        scenes = self.search_scenes(query_keywords, entity, top_k=3)

        if scenes and include_scene_summaries:
            parts.append("## Relevant Memory Scenes")
            for scene, score in scenes:
                if scene.summary:
                    parts.append(f"\n### {scene.title or 'Scene'}")
                    parts.append(scene.summary)

        # Get relevant cells (including from scenes)
        cells = self.search(query_keywords, entity, top_k=max_cells)

        if cells:
            parts.append("\n## Specific Memories")
            for cell, score in cells:
                parts.append(f"- {cell.to_context_string()}")

        return "\n".join(parts) if parts else ""

    def get_entity_profile_context(self, entity: str) -> str:
        """Get all memories about an entity as profile context."""
        cells = self.get_cells_by_entity(entity)

        # Group by type
        by_type: Dict[CellType, List[MemCell]] = {}
        for cell in cells:
            if cell.cell_type not in by_type:
                by_type[cell.cell_type] = []
            by_type[cell.cell_type].append(cell)

        parts = [f"## Profile: {entity.title()}"]

        type_labels = {
            CellType.FACT: "Facts",
            CellType.PREFERENCE: "Preferences",
            CellType.RELATION: "Relationships",
            CellType.EPISODE: "Activities/Events",
            CellType.PLAN: "Plans/Goals",
            CellType.ACHIEVEMENT: "Achievements",
            CellType.EMOTION: "Emotional States",
        }

        for cell_type, label in type_labels.items():
            type_cells = by_type.get(cell_type, [])
            if type_cells:
                parts.append(f"\n### {label}")
                for cell in type_cells[:10]:  # Limit per type
                    parts.append(f"- {cell.content}")

        return "\n".join(parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_cells": len(self.cells),
            "total_scenes": len(self.scenes),
            "entities": list(self.cells_by_entity.keys()),
            "cell_types": {ct.value: len(ids) for ct, ids in self.cells_by_type.items()},
            "keywords": len(self.keyword_index),
        }

    def compose_multihop_context(
        self,
        query_keywords: List[str],
        entity: Optional[str] = None,
        question_type: str = "general",
        max_cells: int = 25
    ) -> str:
        """
        INNOVATION: Scene-guided context composition for multi-hop questions.

        This method retrieves coherent context by:
        1. Finding relevant scenes (episodic clusters)
        2. Expanding to related scenes via entity connections
        3. Including scene summaries for high-level context
        4. Gathering specific cells for detailed evidence

        Args:
            query_keywords: Keywords from the question
            entity: Target entity if known
            question_type: Type of question (inference, comparison, etc.)
            max_cells: Maximum cells to include

        Returns:
            Composed context string
        """
        parts = []

        # Step 1: Find directly relevant scenes
        direct_scenes = self.search_scenes(query_keywords, entity, top_k=3)

        # Step 2: Find related scenes via entity connections
        related_scenes = []
        if entity:
            entity_scenes = self.get_scenes_by_entity(entity)
            for scene in entity_scenes:
                if scene.id not in [s.id for s, _ in direct_scenes]:
                    # Score by keyword overlap
                    score = scene.matches_query(query_keywords, entity)
                    if score > 0.1:
                        related_scenes.append((scene, score))
            related_scenes.sort(key=lambda x: x[1], reverse=True)
            related_scenes = related_scenes[:2]

        # Step 3: Build context from scenes
        all_scenes = direct_scenes + related_scenes
        seen_cell_ids = set()

        if all_scenes:
            parts.append("## Relevant Memory Scenes")
            for scene, score in all_scenes:
                parts.append(f"\n### {scene.title or 'Episode'}")
                if scene.summary:
                    parts.append(f"Summary: {scene.summary}")

                # Get best cells from this scene for the query
                best_cells = scene.get_best_cells(query_keywords, entity, top_k=5)
                if best_cells:
                    parts.append("Key details:")
                    for cell in best_cells:
                        if cell.id not in seen_cell_ids:
                            seen_cell_ids.add(cell.id)
                            parts.append(f"  - {cell.to_context_string()}")

        # Step 4: Add additional relevant cells not in scenes
        remaining_cells = max_cells - len(seen_cell_ids)
        if remaining_cells > 0:
            additional_cells = self.search(query_keywords, entity, top_k=remaining_cells + 10)
            additional = [(c, s) for c, s in additional_cells if c.id not in seen_cell_ids][:remaining_cells]

            if additional:
                parts.append("\n## Additional Evidence")
                for cell, score in additional:
                    parts.append(f"- {cell.to_context_string()}")
                    seen_cell_ids.add(cell.id)

        # Step 5: For inference questions, add entity profile summary
        if question_type == "inference" and entity:
            # Get preference and fact cells for inference
            pref_cells = [self.cells[cid] for cid in self.cells_by_type.get(CellType.PREFERENCE, [])
                          if self.cells[cid].entity.lower() == entity.lower()]
            fact_cells = [self.cells[cid] for cid in self.cells_by_type.get(CellType.FACT, [])
                          if self.cells[cid].entity.lower() == entity.lower()]

            if pref_cells or fact_cells:
                parts.append(f"\n## {entity}'s Profile")
                if pref_cells:
                    parts.append("Preferences:")
                    for cell in pref_cells[:5]:
                        if cell.id not in seen_cell_ids:
                            parts.append(f"  - {cell.content}")
                if fact_cells:
                    parts.append("Facts:")
                    for cell in fact_cells[:5]:
                        if cell.id not in seen_cell_ids:
                            parts.append(f"  - {cell.content}")

        return "\n".join(parts) if parts else ""
