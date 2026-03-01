#!/usr/bin/env python3
"""Test profile-based answering."""
import sys
sys.path.insert(0, 'src')
import os
import openai

# Initialize with real LLM client
llm_client = openai.OpenAI()

from zerogmem.evaluation.locomo import LoCoMoEvaluator, LoCoMoQuestion

evaluator = LoCoMoEvaluator(
    data_path='data/locomo/locomo10.json',
    llm_client=llm_client,
    use_cache=True,
    use_bm25=True,
)
evaluator.load_dataset()

# Ingest first conversation
conv = list(evaluator.conversations.values())[0]
evaluator.ingest_conversation(conv)

# Check profile
print('=== Caroline Profile ===')
profile = evaluator.llm_fact_extractor.get_profile('caroline')
for k, v in list(profile.items())[:5]:
    print(f'  {k}: {v}')

# Test identity question
q = LoCoMoQuestion(
    id='test',
    question="What is Caroline's identity?",
    answer='Transgender woman',
    category='single_hop',
    conversation_id=conv.id,
)

print()
print('=== Testing Question ===')
print(f'Question: {q.question}')

# Check profile answer
profile_answer = evaluator.llm_fact_extractor.answer_from_profile(q.question, 'caroline')
print(f'profile_answer: {profile_answer}')

# Now call actual answer_question
answer, context = evaluator.answer_question(q)
print()
print(f'Final answer: {answer}')
print(f'Context type: {"Profile-based" if "Profile-based" in context else "Retrieved"}')
