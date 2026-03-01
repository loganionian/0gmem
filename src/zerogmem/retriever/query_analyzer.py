"""
Query Analyzer: Understands user queries to route to appropriate retrieval.

Classifies query intent, extracts entities, and determines reasoning type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from zerogmem.encoder.temporal_extractor import TemporalExtractor, TemporalExpression


class QueryIntent(Enum):
    """Types of query intents."""
    FACTUAL = "factual"           # "What is X's favorite color?"
    TEMPORAL = "temporal"         # "When did X happen?"
    CAUSAL = "causal"            # "Why did X happen?"
    RELATIONAL = "relational"    # "Who does X know?"
    PREFERENCE = "preference"     # "Does X like Y?"
    EVENT = "event"              # "What happened at/during X?"
    COMPARISON = "comparison"    # "How is X different from Y?"
    LIST = "list"                # "List all X that Y"
    VERIFICATION = "verification" # "Is it true that X?"


class ReasoningType(Enum):
    """Types of reasoning required."""
    SINGLE_HOP = "single_hop"    # Direct fact lookup
    MULTI_HOP = "multi_hop"      # Connecting multiple facts
    TEMPORAL = "temporal"        # Time-based reasoning
    CAUSAL = "causal"           # Cause-effect reasoning
    COMMONSENSE = "commonsense"  # World knowledge integration
    ADVERSARIAL = "adversarial"  # Testing for false information


class TemporalScope(Enum):
    """Temporal scope of the query."""
    POINT = "point"              # Specific moment
    RANGE = "range"              # Time range
    RELATIVE = "relative"        # Before/after something
    NONE = "none"                # No temporal constraint


@dataclass
class QueryAnalysis:
    """Result of analyzing a query."""
    original_query: str
    intent: QueryIntent
    reasoning_type: ReasoningType
    entities: List[str]
    temporal_scope: TemporalScope
    temporal_expressions: List[TemporalExpression]
    keywords: List[str]
    is_negation_check: bool = False  # Is this checking if something is NOT true?
    expected_answer_type: str = "text"  # text, yes_no, list, date, number
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryAnalyzer:
    """
    Analyzes queries to determine intent and retrieval strategy.

    Critical for routing queries to the right retrieval method,
    especially for LoCoMo's different question types.
    """

    # Intent classification patterns
    INTENT_PATTERNS = {
        QueryIntent.TEMPORAL: [
            r"\bwhen\b",
            r"\bwhat time\b",
            r"\bwhat date\b",
            r"\bhow long\b",
            r"\bafter\b.*\bhappen",
            r"\bbefore\b.*\bhappen",
            r"\bduring\b",
        ],
        QueryIntent.CAUSAL: [
            r"\bwhy\b",
            r"\breason\b",
            r"\bcause\b",
            r"\bbecause\b",
            r"\bdue to\b",
            r"\bresult\b",
            r"\blead to\b",
        ],
        QueryIntent.RELATIONAL: [
            r"\bwho\b.*\bknow",
            r"\bwho\b.*\bfriend",
            r"\brelation",
            r"\bconnect",
            r"\bwho\b.*\bwork",
            r"\bwho\b.*\blive",
        ],
        QueryIntent.PREFERENCE: [
            r"\blike\b",
            r"\blove\b",
            r"\bhate\b",
            r"\bprefer\b",
            r"\bfavorite\b",
            r"\benjoy\b",
            r"\bdislike\b",
        ],
        QueryIntent.EVENT: [
            r"\bwhat happen",
            r"\bwhat did\b.*\bdo\b",
            r"\bdescribe\b",
            r"\btell.*about\b",
            r"\bwhat.*event\b",
        ],
        QueryIntent.COMPARISON: [
            r"\bcompare\b",
            r"\bdifferent\b",
            r"\bsimilar\b",
            r"\bsame\b.*\bas\b",
            r"\bversus\b",
            r"\bvs\b",
        ],
        QueryIntent.LIST: [
            r"\blist\b",
            r"\ball\b.*\bthat\b",
            r"\bevery\b",
            r"\beach\b",
            r"\bhow many\b",
        ],
        QueryIntent.VERIFICATION: [
            r"\bis it true\b",
            r"\bdid\b.*\breally\b",
            r"\bconfirm\b",
            r"\bverify\b",
            r"\bcorrect\b.*\bthat\b",
        ],
    }

    # Reasoning type indicators
    MULTI_HOP_INDICATORS = [
        r"\band\b",
        r"\balso\b",
        r"\bboth\b",
        r"\bwho\b.*\bwho\b",  # Nested questions
        r"\bthen\b.*\bwhat\b",
        r"\bafter that\b",
        r"\bbased on\b",       # Causal connection
        r"\baccording to\b",   # Reference to source
        r"\bexperience\b",     # Personal experience connection
        r"\brecommend",        # Recommendation (often multi-hop)
        r"\bcommon\b",         # Comparing multiple things
    ]

    TEMPORAL_REASONING_INDICATORS = [
        r"\bbefore\b",
        r"\bafter\b",
        r"\bduring\b",
        r"\bwhile\b",
        r"\bsince\b",
        r"\buntil\b",
        r"\bfirst\b.*\bthen\b",
        r"\bchronolog",
        r"\bsequence\b",
        r"\border\b",
    ]

    # Negation patterns for adversarial
    NEGATION_CHECK_PATTERNS = [
        r"\bnot\b.*\btrue\b",
        r"\bnever\b",
        r"\bfalse\b",
        r"\bwrong\b",
        r"\bincorrect\b",
        r"\bdoesn't\b",
        r"\bdoes not\b",
        r"\bdidn't\b",
        r"\bdid not\b",
    ]

    def __init__(self):
        self.temporal_extractor = TemporalExtractor()

    def analyze(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryAnalysis:
        """
        Analyze a query to determine intent and retrieval strategy.

        Args:
            query: The user's query
            context: Optional context (e.g., conversation history)

        Returns:
            QueryAnalysis with intent, entities, and routing information
        """
        query_lower = query.lower()

        # Determine intent
        intent = self._classify_intent(query_lower)

        # Determine reasoning type
        reasoning_type = self._classify_reasoning(query_lower)

        # Extract entities (simple pattern-based)
        entities = self._extract_entities(query)

        # Analyze temporal aspects
        temporal_expressions = self.temporal_extractor.extract(query)
        temporal_scope = self._determine_temporal_scope(query_lower, temporal_expressions)

        # Extract keywords
        keywords = self._extract_keywords(query_lower)

        # Check for negation/adversarial
        is_negation_check = self._is_negation_check(query_lower)

        # Determine expected answer type
        expected_answer_type = self._determine_answer_type(query_lower, intent)

        # Override reasoning type based on patterns
        if reasoning_type == ReasoningType.SINGLE_HOP:
            if temporal_scope != TemporalScope.NONE:
                reasoning_type = ReasoningType.TEMPORAL
            elif is_negation_check:
                reasoning_type = ReasoningType.ADVERSARIAL

        return QueryAnalysis(
            original_query=query,
            intent=intent,
            reasoning_type=reasoning_type,
            entities=entities,
            temporal_scope=temporal_scope,
            temporal_expressions=temporal_expressions,
            keywords=keywords,
            is_negation_check=is_negation_check,
            expected_answer_type=expected_answer_type,
            metadata={
                "query_length": len(query.split()),
                "has_temporal": len(temporal_expressions) > 0,
            },
        )

    def _classify_intent(self, query_lower: str) -> QueryIntent:
        """Classify the primary intent of the query."""
        # Check patterns in priority order
        intent_scores = {}

        for intent, patterns in self.INTENT_PATTERNS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            if score > 0:
                intent_scores[intent] = score

        if intent_scores:
            # Return highest scoring intent
            return max(intent_scores, key=intent_scores.get)

        # Default to factual
        return QueryIntent.FACTUAL

    def _classify_reasoning(self, query_lower: str) -> ReasoningType:
        """Classify the type of reasoning required."""
        # Check for multi-hop indicators
        for pattern in self.MULTI_HOP_INDICATORS:
            if re.search(pattern, query_lower):
                return ReasoningType.MULTI_HOP

        # Check for temporal reasoning
        for pattern in self.TEMPORAL_REASONING_INDICATORS:
            if re.search(pattern, query_lower):
                return ReasoningType.TEMPORAL

        # Check for causal reasoning
        if re.search(r"\bwhy\b|\bcause\b|\breason\b", query_lower):
            return ReasoningType.CAUSAL

        # Check for commonsense reasoning (comparing/generalizing across contexts)
        if re.search(r"\bcommon\b|\btypical\b|\busual\b|\bgenerally\b|\bboth\b.*\bcit", query_lower):
            return ReasoningType.COMMONSENSE

        return ReasoningType.SINGLE_HOP

    def _extract_entities(self, query: str) -> List[str]:
        """Extract entity mentions from query (simplified)."""
        entities = []

        # Look for capitalized words (potential names)
        name_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        for match in re.finditer(name_pattern, query):
            # Skip common question words
            if match.group(1).lower() not in ['what', 'when', 'where', 'who', 'why', 'how', 'which']:
                entities.append(match.group(1))

        # Look for quoted terms
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        quoted = re.findall(r"'([^']+)'", query)
        entities.extend(quoted)

        return list(set(entities))

    def _extract_keywords(self, query_lower: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
            'until', 'while', 'about', 'what', 'which', 'who', 'whom',
            'this', 'that', 'these', 'those', 'am', 'it', 'its',
        }

        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query_lower)
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def _determine_temporal_scope(
        self,
        query_lower: str,
        temporal_expressions: List[TemporalExpression]
    ) -> TemporalScope:
        """Determine the temporal scope of the query."""
        if not temporal_expressions:
            # Check for temporal keywords without specific expressions
            if re.search(r'\bwhen\b|\btime\b|\bdate\b', query_lower):
                return TemporalScope.POINT
            return TemporalScope.NONE

        # Analyze the expressions
        has_point = False
        has_range = False
        has_relative = False

        for expr in temporal_expressions:
            if expr.normalized_start and expr.normalized_end:
                has_range = True
            elif expr.normalized_start:
                has_point = True
            if expr.relation in ['before', 'after', 'during']:
                has_relative = True

        if has_relative:
            return TemporalScope.RELATIVE
        if has_range:
            return TemporalScope.RANGE
        if has_point:
            return TemporalScope.POINT

        return TemporalScope.NONE

    def _is_negation_check(self, query_lower: str) -> bool:
        """Check if query is verifying a negation."""
        for pattern in self.NEGATION_CHECK_PATTERNS:
            if re.search(pattern, query_lower):
                return True
        return False

    def _determine_answer_type(self, query_lower: str, intent: QueryIntent) -> str:
        """Determine expected answer type."""
        # Yes/No questions
        if re.match(r'^(is|are|was|were|do|does|did|can|could|will|would|has|have|had)\b', query_lower):
            return "yes_no"

        # Date/time questions
        if re.search(r'\bwhen\b|\bwhat time\b|\bwhat date\b', query_lower):
            return "date"

        # Count questions
        if re.search(r'\bhow many\b|\bhow much\b', query_lower):
            return "number"

        # List questions
        if intent == QueryIntent.LIST:
            return "list"

        return "text"

    def get_retrieval_strategy(self, analysis: QueryAnalysis) -> Dict[str, Any]:
        """
        Determine the retrieval strategy based on query analysis.

        Returns configuration for the retriever.
        """
        strategy = {
            "use_semantic_search": True,
            "use_entity_search": len(analysis.entities) > 0,
            "use_temporal_search": analysis.temporal_scope != TemporalScope.NONE,
            "use_graph_traversal": analysis.reasoning_type == ReasoningType.MULTI_HOP,
            "check_negations": analysis.is_negation_check,
            "max_hops": 1,
            "top_k": 10,
        }

        # Adjust based on reasoning type
        if analysis.reasoning_type == ReasoningType.MULTI_HOP:
            strategy["max_hops"] = 3
            strategy["top_k"] = 20  # Increased for better coverage
            strategy["use_graph_traversal"] = True

        elif analysis.reasoning_type == ReasoningType.TEMPORAL:
            strategy["use_temporal_search"] = True
            strategy["temporal_priority"] = True

        elif analysis.reasoning_type == ReasoningType.CAUSAL:
            strategy["use_causal_graph"] = True
            strategy["max_hops"] = 2

        elif analysis.reasoning_type == ReasoningType.COMMONSENSE:
            strategy["top_k"] = 20  # Increased for diverse context
            strategy["use_graph_traversal"] = True
            strategy["max_hops"] = 2

        elif analysis.reasoning_type == ReasoningType.ADVERSARIAL:
            strategy["check_negations"] = True
            strategy["verify_facts"] = True

        return strategy
