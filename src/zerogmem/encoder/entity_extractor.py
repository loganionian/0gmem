"""
Entity Extractor: Extracts entities and relationships from text.

Uses pattern matching and optional LLM for enhanced extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

from zerogmem.graph.entity import EntityType


@dataclass
class ExtractedEntity:
    """An extracted entity mention."""
    text: str                        # As mentioned in text
    normalized: str                  # Normalized form
    type: EntityType
    span: Tuple[int, int]           # Character positions
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedRelation:
    """An extracted relation between entities."""
    subject: str                     # Subject entity text
    predicate: str                   # Relation type
    object: str                      # Object entity text
    negated: bool = False           # Is this relation negated?
    confidence: float = 1.0
    span: Tuple[int, int] = (0, 0)


class EntityExtractor:
    """
    Extracts entities and relationships from text.

    Uses:
    - Pattern matching for common entity types
    - Title case detection for names
    - Relationship pattern matching
    - Negation detection for adversarial robustness
    """

    # Common name prefixes/titles
    NAME_TITLES = ["mr", "mrs", "ms", "dr", "prof", "sir", "lady"]

    # Relation patterns: (pattern, relation_type)
    RELATION_PATTERNS = [
        # Likes/Preferences
        (r"(\w+(?:\s\w+)?)\s+(?:really\s+)?(?:like|love|enjoy|prefer)s?\s+(.+?)(?:\.|,|$)", "likes"),
        (r"(\w+(?:\s\w+)?)\s+(?:hate|dislike|can't stand)s?\s+(.+?)(?:\.|,|$)", "dislikes"),

        # Relationships
        (r"(\w+(?:\s\w+)?)\s+is\s+(?:a\s+)?friend\s+(?:of|with)\s+(\w+(?:\s\w+)?)", "friend_of"),
        (r"(\w+(?:\s\w+)?)\s+knows?\s+(\w+(?:\s\w+)?)", "knows"),
        (r"(\w+(?:\s\w+)?)\s+(?:is\s+)?married\s+to\s+(\w+(?:\s\w+)?)", "married_to"),
        (r"(\w+(?:\s\w+)?)\s+works?\s+(?:at|for)\s+(.+?)(?:\.|,|$)", "works_at"),
        (r"(\w+(?:\s\w+)?)\s+lives?\s+(?:in|at)\s+(.+?)(?:\.|,|$)", "lives_in"),

        # Possession
        (r"(\w+(?:\s\w+)?)'s\s+(\w+)", "has"),
        (r"(\w+(?:\s\w+)?)\s+(?:has|have|own)s?\s+(?:a\s+)?(.+?)(?:\.|,|$)", "has"),

        # Attributes
        (r"(\w+(?:\s\w+)?)\s+is\s+(?:a\s+)?(\w+)", "is_a"),
    ]

    # Negative preference patterns - CRITICAL for adversarial questions
    NEGATIVE_PREFERENCE_PATTERNS = [
        # "I could never eat X" / "I would never eat X"
        (r"(I|we|he|she|they|\w+)\s+(?:could|would|can|will)\s+never\s+(\w+)\s+(.+?)(?:\.|,|!|$)", "cannot"),
        # "I don't like X" / "I don't eat X"
        (r"(I|we|he|she|they|\w+)\s+(?:don't|doesn't|do not|does not)\s+(?:like|love|enjoy|eat|want)\s+(.+?)(?:\.|,|!|$)", "dislikes"),
        # "I hate X" / "I can't stand X"
        (r"(I|we|he|she|they|\w+)\s+(?:hate|detest|loathe|can't stand|cannot stand)\s+(.+?)(?:\.|,|!|$)", "dislikes"),
        # "I'm not a fan of X"
        (r"(I|we|he|she|they|\w+)(?:'m|\s+am|\s+is|\s+are)\s+not\s+(?:a\s+)?fan\s+of\s+(.+?)(?:\.|,|!|$)", "dislikes"),
        # "X is not for me"
        (r"(.+?)\s+(?:is|are)\s+not\s+for\s+(me|us|him|her|them|\w+)", "dislikes"),
        # "I never eat/like X"
        (r"(I|we|he|she|they|\w+)\s+never\s+(?:eat|like|enjoy|want)s?\s+(.+?)(?:\.|,|!|$)", "never_does"),
    ]

    # Negation patterns - expanded for better detection
    NEGATION_PATTERNS = [
        r"(?:do|does|did)n't\s+",
        r"(?:do|does|did)\s+not\s+",
        r"never\s+",
        r"no\s+longer\s+",
        r"not\s+",
        r"isn't\s+",
        r"aren't\s+",
        r"wasn't\s+",
        r"weren't\s+",
        r"(?:could|would|can|will)\s+never\s+",  # Added: "could never", "would never"
        r"(?:could|would|can|will)\s+not\s+",    # Added: "could not", "would not"
        r"(?:could|would|can|will)n't\s+",       # Added: "couldn't", "wouldn't"
    ]

    # Location indicators
    LOCATION_PREPOSITIONS = ["in", "at", "from", "to", "near", "by"]

    # Organization indicators
    ORG_SUFFIXES = ["inc", "corp", "llc", "ltd", "company", "co", "org", "foundation"]

    def __init__(self, use_llm: bool = False, llm_client: Optional[Any] = None):
        """
        Initialize the entity extractor.

        Args:
            use_llm: Whether to use LLM for enhanced extraction
            llm_client: LLM client for extraction (if use_llm=True)
        """
        self.use_llm = use_llm
        self.llm_client = llm_client

    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities from text.

        Returns list of ExtractedEntity objects.
        """
        entities = []

        # Extract names (title case words that aren't sentence starts)
        entities.extend(self._extract_names(text))

        # Extract organizations
        entities.extend(self._extract_organizations(text))

        # Extract locations
        entities.extend(self._extract_locations(text))

        # Deduplicate by normalized form
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity.normalized.lower(), entity.type)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)

        return unique_entities

    def extract_relations(self, text: str) -> List[ExtractedRelation]:
        """
        Extract relationships between entities from text.

        Returns list of ExtractedRelation objects.
        """
        relations = []

        # Extract positive relations
        for pattern, relation_type in self.RELATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) >= 2:
                    # Check for negation
                    negated = self._is_negated(text, match.start())

                    relation = ExtractedRelation(
                        subject=groups[0].strip(),
                        predicate=relation_type,
                        object=groups[1].strip(),
                        negated=negated,
                        span=(match.start(), match.end()),
                    )
                    relations.append(relation)

        # Extract NEGATIVE preference relations - critical for adversarial questions
        for pattern, relation_type in self.NEGATIVE_PREFERENCE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) >= 2:
                    subject = groups[0].strip()
                    # Handle "could never eat X" -> object is the last group
                    obj = groups[-1].strip()

                    relation = ExtractedRelation(
                        subject=subject,
                        predicate=relation_type,
                        object=obj,
                        negated=True,  # These are always negated preferences
                        confidence=0.95,  # High confidence for explicit negations
                        span=(match.start(), match.end()),
                    )
                    relations.append(relation)

        return relations

    def extract_negations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract explicit negations for adversarial robustness.

        Returns list of negated statements with structured information.
        """
        negations = []

        # Combined negation pattern
        neg_pattern = "|".join(self.NEGATION_PATTERNS)
        full_pattern = f"({neg_pattern})(.+?)(?:\\.|,|!|$)"

        for match in re.finditer(full_pattern, text, re.IGNORECASE):
            negation_word = match.group(1)
            negated_content = match.group(2).strip()

            negations.append({
                "negation_marker": negation_word.strip(),
                "content": negated_content,
                "full_text": match.group().strip(),
                "span": (match.start(), match.end()),
                "type": "general",
            })

        # Extract preference-specific negations with more structure
        for pattern, neg_type in self.NEGATIVE_PREFERENCE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                subject = groups[0].strip()
                obj = groups[-1].strip()

                # Determine the specific action/preference being negated
                full_match = match.group()

                negations.append({
                    "negation_marker": neg_type,
                    "subject": subject,
                    "object": obj,
                    "content": f"{subject} does not {neg_type} {obj}",
                    "full_text": full_match.strip(),
                    "span": (match.start(), match.end()),
                    "type": "preference",
                    "is_preference_negation": True,
                })

        return negations

    def _extract_names(self, text: str) -> List[ExtractedEntity]:
        """Extract person names from text."""
        entities = []

        # Pattern for names with titles
        title_pattern = r'\b(' + '|'.join(self.NAME_TITLES) + r')\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        for match in re.finditer(title_pattern, text, re.IGNORECASE):
            name = match.group(2)
            entities.append(ExtractedEntity(
                text=match.group(),
                normalized=name,
                type=EntityType.PERSON,
                span=(match.start(), match.end()),
            ))

        # Pattern for capitalized names (2-3 words)
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b'
        for match in re.finditer(name_pattern, text):
            # Skip if at sentence start
            if match.start() > 0 and text[match.start() - 1] not in '.!?\n':
                name = match.group(1)
                # Skip common words that might be capitalized
                if name.lower() not in ['the', 'this', 'that', 'these', 'those']:
                    entities.append(ExtractedEntity(
                        text=name,
                        normalized=name,
                        type=EntityType.PERSON,
                        span=(match.start(), match.end()),
                        confidence=0.7,  # Lower confidence without title
                    ))

        # First person references (I, me, my)
        first_person = re.findall(r'\b(I|me|my|mine|myself)\b', text, re.IGNORECASE)
        if first_person:
            entities.append(ExtractedEntity(
                text="I",
                normalized="USER",  # Normalized as USER
                type=EntityType.PERSON,
                span=(0, 0),  # Multiple occurrences
                confidence=1.0,
            ))

        return entities

    def _extract_organizations(self, text: str) -> List[ExtractedEntity]:
        """Extract organization names from text."""
        entities = []

        # Organizations with suffixes
        suffix_pattern = r'\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(' + '|'.join(self.ORG_SUFFIXES) + r')\.?\b'
        for match in re.finditer(suffix_pattern, text, re.IGNORECASE):
            entities.append(ExtractedEntity(
                text=match.group(),
                normalized=match.group(1),
                type=EntityType.ORGANIZATION,
                span=(match.start(), match.end()),
            ))

        # All-caps acronyms (likely organizations)
        acronym_pattern = r'\b([A-Z]{2,5})\b'
        for match in re.finditer(acronym_pattern, text):
            # Skip common acronyms that aren't organizations
            if match.group(1) not in ['AM', 'PM', 'TV', 'OK', 'US', 'UK']:
                entities.append(ExtractedEntity(
                    text=match.group(1),
                    normalized=match.group(1),
                    type=EntityType.ORGANIZATION,
                    span=(match.start(), match.end()),
                    confidence=0.6,
                ))

        return entities

    def _extract_locations(self, text: str) -> List[ExtractedEntity]:
        """Extract location names from text."""
        entities = []

        # Locations after prepositions
        prep_pattern = r'\b(?:' + '|'.join(self.LOCATION_PREPOSITIONS) + r')\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        for match in re.finditer(prep_pattern, text):
            entities.append(ExtractedEntity(
                text=match.group(1),
                normalized=match.group(1),
                type=EntityType.LOCATION,
                span=(match.start(), match.end()),
                confidence=0.8,
            ))

        return entities

    def _is_negated(self, text: str, position: int) -> bool:
        """Check if content at position is negated."""
        # Look back for negation words
        window_start = max(0, position - 30)
        window = text[window_start:position]

        for pattern in self.NEGATION_PATTERNS:
            if re.search(pattern, window, re.IGNORECASE):
                return True

        return False

    def get_extraction_summary(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive extraction summary.

        Returns structured data about all extractions.
        """
        entities = self.extract_entities(text)
        relations = self.extract_relations(text)
        negations = self.extract_negations(text)

        return {
            "entities": [
                {
                    "text": e.text,
                    "normalized": e.normalized,
                    "type": e.type.value,
                    "confidence": e.confidence,
                }
                for e in entities
            ],
            "relations": [
                {
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "negated": r.negated,
                }
                for r in relations
            ],
            "negations": negations,
            "entity_count": len(entities),
            "relation_count": len(relations),
            "has_negations": len(negations) > 0,
        }
