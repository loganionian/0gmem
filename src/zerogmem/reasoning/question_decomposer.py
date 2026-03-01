"""
Question Decomposer: Break complex questions into atomic sub-questions.

Decomposes questions into sub-questions that can be answered independently,
then synthesizes the final answer (rather than multi-query retrieval for the same question).

This is especially powerful for multi-hop questions like:
"Would Caroline enjoy classical music based on her hobbies?"
-> Sub-Q1: "What are Caroline's hobbies?"
-> Sub-Q2: "Do any of her hobbies relate to music?"
-> Synthesis: "Yes/No, because [hobbies] suggest [conclusion]"
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SubQuestion:
    """A decomposed sub-question."""

    id: str
    question: str
    question_type: str  # factual, preference, temporal, existence
    depends_on: list[str] = field(default_factory=list)  # IDs of questions this depends on
    answer: str | None = None
    confidence: float = 0.0


@dataclass
class DecompositionResult:
    """Result of question decomposition."""

    original_question: str
    sub_questions: list[SubQuestion]
    reasoning_chain: str
    synthesis_template: str


class QuestionDecomposer:
    """
    Decomposes complex questions into answerable sub-questions.

    INNOVATION: Pattern-based decomposition with synthesis templates
    that enable step-by-step reasoning.
    """

    # Question type patterns
    MULTI_HOP_PATTERNS = [
        # "Would X do Y based on Z?"
        (r"would (\w+) (.+?) based on (.+?)(?:\?|$)", "inference_from_facts"),
        # "Would X like/enjoy Y?"
        (r"would (\w+) (?:like|enjoy|prefer) (.+?)(?:\?|$)", "preference_inference"),
        # "Would X be considered Y?" (religious, patriotic, member of)
        (
            r"would (\w+) be (?:considered|thought of as) (?:a |an )?(.+?)(?:\?|$)",
            "identity_inference",
        ),
        # "Would X be more interested in A or B?"
        (r"would (\w+) be (?:more )?interested in (.+?) or (.+?)(?:\?|$)", "preference_choice"),
        # "What would X recommend?"
        (r"what (?:would|might|could) (\w+) (.+?)(?:\?|$)", "recommendation"),
        # "What might X's Y be?" (degree, status, financial status)
        (r"what (?:might|could|would) (\w+)\'?s? (.+?) be(?:\?|$)", "attribute_inference"),
        # "What fields/traits/job might X pursue/have?"
        (
            r"what (\w+) (?:might|would|could) (\w+) (?:have|pursue|say|describe)(?:\?|$)",
            "attribute_list",
        ),
        # "Based on X, would Y?"
        (r"based on (.+?), (?:would|could|might) (\w+) (.+?)(?:\?|$)", "conditional"),
        # "Would X be open to Y?"
        (r"would (\w+) be (?:open|willing|likely) to (.+?)(?:\?|$)", "likelihood_inference"),
        # "Does X live close to Y?"
        (r"does (\w+) live (?:close|near) to (.+?)(?:\?|$)", "location_inference"),
    ]

    COMPARISON_PATTERNS = [
        # "What do X and Y both like?"
        (r"what (?:do|does) (\w+) and (\w+) (?:both )?(like|enjoy|have)", "intersection"),
        # "How is X different from Y?"
        (r"how (?:is|are) (\w+) different from (\w+)", "difference"),
    ]

    TEMPORAL_CHAIN_PATTERNS = [
        # "What happened before X?"
        (r"what happened (?:before|after) (.+?)(?:\?|$)", "temporal_chain"),
        # "When did X happen relative to Y?"
        (
            r"(?:when|what time) did (.+?) "
            r"(?:happen|occur) "
            r"(?:relative to|compared to) (.+?)(?:\?|$)",
            "temporal_relative",
        ),
    ]

    def decompose(self, question: str, target_entity: str | None = None) -> DecompositionResult:
        """
        Decompose a question into sub-questions.

        Returns DecompositionResult with sub-questions and synthesis template.
        """
        q_lower = question.lower()

        # Check for multi-hop patterns
        for pattern, q_type in self.MULTI_HOP_PATTERNS:
            match = re.search(pattern, q_lower)
            if match:
                return self._decompose_multi_hop(question, match, q_type, target_entity)

        # Check for comparison patterns
        for pattern, q_type in self.COMPARISON_PATTERNS:
            match = re.search(pattern, q_lower)
            if match:
                return self._decompose_comparison(question, match, q_type)

        # Check for temporal chain patterns
        for pattern, q_type in self.TEMPORAL_CHAIN_PATTERNS:
            match = re.search(pattern, q_lower)
            if match:
                return self._decompose_temporal(question, match, q_type, target_entity)

        # Default: no decomposition needed
        return DecompositionResult(
            original_question=question,
            sub_questions=[
                SubQuestion(
                    id="q1",
                    question=question,
                    question_type="direct",
                )
            ],
            reasoning_chain="Direct question - no decomposition",
            synthesis_template="{q1}",
        )

    def _decompose_multi_hop(
        self,
        question: str,
        match: re.Match,
        q_type: str,
        target_entity: str | None,
    ) -> DecompositionResult:
        """Decompose multi-hop inference questions."""
        groups = match.groups()

        if q_type == "inference_from_facts":
            entity, action, basis = groups
            return DecompositionResult(
                original_question=question,
                sub_questions=[
                    SubQuestion(
                        id="q1",
                        question=f"What is {entity}'s {basis}?",
                        question_type="factual",
                    ),
                    SubQuestion(
                        id="q2",
                        question=f"Based on this information, would {entity} {action}?",
                        question_type="inference",
                        depends_on=["q1"],
                    ),
                ],
                reasoning_chain=f"1. Find {entity}'s {basis}\n2. Infer if they would {action}",
                synthesis_template="Based on {entity}'s {basis} ({q1}), {inference}",
            )

        elif q_type == "preference_inference":
            entity, item = groups
            return DecompositionResult(
                original_question=question,
                sub_questions=[
                    SubQuestion(
                        id="q1",
                        question=f"What are {entity}'s hobbies and interests?",
                        question_type="factual",
                    ),
                    SubQuestion(
                        id="q2",
                        question=f"What are {entity}'s stated preferences?",
                        question_type="factual",
                    ),
                    SubQuestion(
                        id="q3",
                        question=f"Do any of these suggest {entity} would like {item}?",
                        question_type="inference",
                        depends_on=["q1", "q2"],
                    ),
                ],
                reasoning_chain=(
                    f"1. Find {entity}'s hobbies\n"
                    f"2. Find stated preferences\n"
                    f"3. Infer preference for {item}"
                ),
                synthesis_template=(
                    "Given {entity}'s interests" " ({q1}) and preferences" " ({q2}), {conclusion}"
                ),
            )

        elif q_type == "recommendation":
            entity, action = groups
            return DecompositionResult(
                original_question=question,
                sub_questions=[
                    SubQuestion(
                        id="q1",
                        question=f"What does {entity} have experience with?",
                        question_type="factual",
                    ),
                    SubQuestion(
                        id="q2",
                        question=f"What does {entity} like or prefer?",
                        question_type="factual",
                    ),
                ],
                reasoning_chain=(
                    f"1. Find {entity}'s experiences\n"
                    "2. Find preferences\n"
                    "3. Synthesize recommendation"
                ),
                synthesis_template=(
                    "Based on {entity}'s experience"
                    " ({q1}) and preferences"
                    " ({q2}), they would likely"
                    " {action}"
                ),
            )

        elif q_type == "identity_inference":
            # "Would X be considered religious/patriotic/member of Y?"
            entity, identity = groups
            return DecompositionResult(
                original_question=question,
                sub_questions=[
                    SubQuestion(
                        id="q1",
                        question=f"What activities, beliefs, or statements has {entity} made?",
                        question_type="factual",
                    ),
                    SubQuestion(
                        id="q2",
                        question=f"Based on these, would {entity} be considered {identity}?",
                        question_type="inference",
                        depends_on=["q1"],
                    ),
                ],
                reasoning_chain=(
                    f"1. Find {entity}'s activities"
                    f" and beliefs\n"
                    f"2. Infer if they would be"
                    f" {identity}"
                ),
                synthesis_template="Based on {entity}'s activities ({q1}), {conclusion}",
            )

        elif q_type == "preference_choice":
            # "Would X be more interested in A or B?"
            entity, option_a, option_b = groups
            return DecompositionResult(
                original_question=question,
                sub_questions=[
                    SubQuestion(
                        id="q1",
                        question=f"What are {entity}'s hobbies and interests?",
                        question_type="factual",
                    ),
                    SubQuestion(
                        id="q2",
                        question=f"Does {entity} prefer outdoor or indoor activities?",
                        question_type="factual",
                    ),
                ],
                reasoning_chain=(
                    f"1. Find {entity}'s interests\n"
                    f"2. Determine preference between"
                    f" {option_a} and {option_b}"
                ),
                synthesis_template="Given {entity}'s interests ({q1}), they would prefer {choice}",
            )

        elif q_type == "attribute_inference":
            # "What might X's degree/status be?"
            entity, attribute = groups
            return DecompositionResult(
                original_question=question,
                sub_questions=[
                    SubQuestion(
                        id="q1",
                        question=f"What has {entity} studied or worked on?",
                        question_type="factual",
                    ),
                    SubQuestion(
                        id="q2",
                        question=f"What are {entity}'s career goals or interests?",
                        question_type="factual",
                    ),
                ],
                reasoning_chain=f"1. Find {entity}'s education/work\n2. Infer their {attribute}",
                synthesis_template=(
                    "Based on {entity}'s background"
                    " ({q1}), their {attribute}"
                    " is likely {conclusion}"
                ),
            )

        elif q_type == "likelihood_inference":
            # "Would X be open to Y?"
            entity, action = groups
            return DecompositionResult(
                original_question=question,
                sub_questions=[
                    SubQuestion(
                        id="q1",
                        question=f"What are {entity}'s goals and priorities?",
                        question_type="factual",
                    ),
                    SubQuestion(
                        id="q2",
                        question=f"Has {entity} expressed interest in {action}?",
                        question_type="factual",
                    ),
                ],
                reasoning_chain=f"1. Find {entity}'s goals\n2. Determine likelihood of {action}",
                synthesis_template="Given {entity}'s priorities ({q1}), {conclusion}",
            )

        elif q_type == "location_inference":
            # "Does X live close to beach or mountains?"
            entity, location = groups
            return DecompositionResult(
                original_question=question,
                sub_questions=[
                    SubQuestion(
                        id="q1",
                        question=f"Where does {entity} live or mention living?",
                        question_type="factual",
                    ),
                    SubQuestion(
                        id="q2",
                        question=f"What outdoor activities does {entity} do?",
                        question_type="factual",
                    ),
                ],
                reasoning_chain=(
                    f"1. Find {entity}'s location\n" f"2. Determine proximity to" f" {location}"
                ),
                synthesis_template=(
                    "Based on {entity}'s location" " ({q1}), they live near" " {conclusion}"
                ),
            )

        elif q_type == "conditional":
            condition, entity, action = groups
            return DecompositionResult(
                original_question=question,
                sub_questions=[
                    SubQuestion(
                        id="q1",
                        question=f"What do we know about {condition}?",
                        question_type="factual",
                    ),
                    SubQuestion(
                        id="q2",
                        question=f"Given this, would {entity} {action}?",
                        question_type="inference",
                        depends_on=["q1"],
                    ),
                ],
                reasoning_chain=(
                    f"1. Establish facts about"
                    f" {condition}\n"
                    f"2. Apply to determine if"
                    f" {entity} would {action}"
                ),
                synthesis_template="Given {condition} ({q1}), {conclusion}",
            )

        # Default fallback
        return self._default_decomposition(question)

    def _decompose_comparison(
        self,
        question: str,
        match: re.Match,
        q_type: str,
    ) -> DecompositionResult:
        """Decompose comparison questions."""
        groups = match.groups()

        if q_type == "intersection":
            entity1, entity2, attribute = groups
            return DecompositionResult(
                original_question=question,
                sub_questions=[
                    SubQuestion(
                        id="q1",
                        question=f"What does {entity1} {attribute}?",
                        question_type="factual",
                    ),
                    SubQuestion(
                        id="q2",
                        question=f"What does {entity2} {attribute}?",
                        question_type="factual",
                    ),
                    SubQuestion(
                        id="q3",
                        question=f"What is common between {entity1} and {entity2}'s {attribute}?",
                        question_type="comparison",
                        depends_on=["q1", "q2"],
                    ),
                ],
                reasoning_chain=(
                    f"1. Find {entity1}'s"
                    f" {attribute}\n"
                    f"2. Find {entity2}'s"
                    f" {attribute}\n"
                    "3. Find intersection"
                ),
                synthesis_template="Both {entity1} and {entity2} {attribute}: {intersection}",
            )

        elif q_type == "difference":
            entity1, entity2 = groups
            return DecompositionResult(
                original_question=question,
                sub_questions=[
                    SubQuestion(
                        id="q1",
                        question=f"What are {entity1}'s characteristics?",
                        question_type="factual",
                    ),
                    SubQuestion(
                        id="q2",
                        question=f"What are {entity2}'s characteristics?",
                        question_type="factual",
                    ),
                ],
                reasoning_chain=(
                    f"1. Find {entity1}'s traits\n" f"2. Find {entity2}'s traits\n" "3. Compare"
                ),
                synthesis_template="{entity1} differs from {entity2} in: {differences}",
            )

        return self._default_decomposition(question)

    def _decompose_temporal(
        self,
        question: str,
        match: re.Match,
        q_type: str,
        target_entity: str | None,
    ) -> DecompositionResult:
        """Decompose temporal chain questions."""
        groups = match.groups()

        if q_type == "temporal_chain":
            event = groups[0]
            entity = target_entity or "the person"
            return DecompositionResult(
                original_question=question,
                sub_questions=[
                    SubQuestion(
                        id="q1",
                        question=f"When did {event} happen?",
                        question_type="temporal",
                    ),
                    SubQuestion(
                        id="q2",
                        question=f"What events happened around that time for {entity}?",
                        question_type="temporal_context",
                        depends_on=["q1"],
                    ),
                ],
                reasoning_chain=f"1. Find when {event} happened\n2. Find surrounding events",
                synthesis_template="Before/after {event}, {related_events}",
            )

        return self._default_decomposition(question)

    def _default_decomposition(self, question: str) -> DecompositionResult:
        """Default decomposition for unrecognized patterns."""
        return DecompositionResult(
            original_question=question,
            sub_questions=[
                SubQuestion(
                    id="q1",
                    question=question,
                    question_type="direct",
                )
            ],
            reasoning_chain="Direct question",
            synthesis_template="{q1}",
        )


class ReasoningChainExecutor:
    """
    INNOVATION: Execute decomposed questions and synthesize final answer.

    This enables explicit step-by-step reasoning rather than
    relying on LLM to implicitly connect facts.
    """

    def __init__(
        self,
        answer_fn: Callable[[str], str],
        llm_client: Any | None = None,
        model: str | None = None,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
    ):
        """
        Initialize executor.

        Args:
            answer_fn: Function to answer a single question
            llm_client: Optional LLM for synthesis
        """
        self.answer_fn = answer_fn
        self._client = llm_client
        self._model = (
            model or os.getenv("OPENAI_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini"
        )
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

    def _chat_completion(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> str | None:
        if not self._client:
            return None

        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                result: str = response.choices[0].message.content.strip()
                return result
            except Exception:
                sleep_s = min(30.0, self._retry_backoff**attempt)
                time.sleep(sleep_s)

        return None

    def execute(self, decomposition: DecompositionResult) -> str:
        """Execute the reasoning chain and return final answer."""
        answers: dict[str, str] = {}

        # Answer sub-questions in dependency order
        for sq in decomposition.sub_questions:
            # Check dependencies are satisfied
            for dep_id in sq.depends_on:
                if dep_id not in answers:
                    # Dependency not yet answered, skip for now
                    continue

            # Build context from dependencies
            context = ""
            for dep_id in sq.depends_on:
                if dep_id in answers:
                    context += f"\n{dep_id}: {answers[dep_id]}"

            # If this question has context from dependencies, include it
            if context:
                modified_q = f"{sq.question}\n\nContext from previous answers:{context}"
                answers[sq.id] = self.answer_fn(modified_q)
            else:
                answers[sq.id] = self.answer_fn(sq.question)

            sq.answer = answers[sq.id]

        # Synthesize final answer
        return self._synthesize(decomposition, answers)

    def _synthesize(
        self,
        decomposition: DecompositionResult,
        answers: dict[str, str],
    ) -> str:
        """Synthesize final answer from sub-answers."""
        # If only one question, return its answer directly
        if len(decomposition.sub_questions) == 1:
            return answers.get("q1", "None")

        # Try to use synthesis template
        template = decomposition.synthesis_template
        for sq_id, answer in answers.items():
            template = template.replace(f"{{{sq_id}}}", answer)

        # If LLM available, do proper synthesis
        if self._client:
            sub_answers = "\n".join(
                f"- {sq.question}: {sq.answer}" for sq in decomposition.sub_questions
            )

            prompt = f"""Synthesize a final answer from these sub-answers:

Original Question: {decomposition.original_question}

Sub-questions and answers:
{sub_answers}

Reasoning approach: {decomposition.reasoning_chain}

Provide a concise final answer that combines the sub-answers appropriately.
For yes/no questions, start with Yes/No.

Final Answer:"""

            try:
                synthesis_result = self._chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0,
                )
                if synthesis_result:
                    return synthesis_result
            except Exception:
                pass

        # Fallback: return first answer or template result
        return template if "{" not in template else answers.get("q1", "None")
