"""
Reasoning Module: Novel innovations for superior performance.

This module contains innovative approaches that go beyond EverMemOS:
- Answer Verification: Generate-verify-refine loop
- Question Decomposition: Break complex questions into sub-questions
"""

from zerogmem.reasoning.answer_verifier import AnswerVerifier, VerificationResult, ConsistencyChecker
from zerogmem.reasoning.question_decomposer import QuestionDecomposer, ReasoningChainExecutor, SubQuestion

__all__ = [
    "AnswerVerifier",
    "VerificationResult",
    "ConsistencyChecker",
    "QuestionDecomposer",
    "ReasoningChainExecutor",
    "SubQuestion",
]
