#!/usr/bin/env python3
"""
Run 0GMem evaluation on LoCoMo benchmark.

Usage:
    python scripts/run_evaluation.py --data-path data/locomo/sample_locomo.json
    python scripts/run_evaluation.py --data-path data/locomo/locomo10.json --use-llm --use-cache --use-bm25
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from zerogmem.evaluation.locomo import LoCoMoEvaluator
from zerogmem.memory.manager import MemoryConfig
from zerogmem.encoder.encoder import EncoderConfig
from zerogmem.retriever.retriever import RetrieverConfig


def main():
    parser = argparse.ArgumentParser(description="Run 0GMem LoCoMo evaluation")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/locomo/sample_locomo.json",
        help="Path to LoCoMo dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/locomo_results.json",
        help="Path to save results"
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=None,
        help="Maximum conversations to evaluate"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum questions to evaluate"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Categories to evaluate (single_hop, multi_hop, temporal, commonsense, adversarial)"
    )
    parser.add_argument(
        "--partial-output",
        type=str,
        default=None,
        help="Path to save intermediate results after each conversation"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to partial results file to resume from"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for answer generation (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.getenv("OPENAI_MODEL") or os.getenv("OPENAI_CHAT_MODEL") or "gpt-4o-mini",
        help="Chat model to use for LLM calls"
    )
    parser.add_argument(
        "--use-evidence-reranker",
        action="store_true",
        help="Use LLM to rerank evidence context before answering"
    )
    parser.add_argument(
        "--no-evidence-reranker",
        action="store_true",
        help="Disable evidence reranking"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=True,
        help="Use embedding cache for faster evaluation (default: True)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable embedding cache"
    )
    parser.add_argument(
        "--use-bm25",
        action="store_true",
        default=True,
        help="Use BM25 hybrid search (default: True)"
    )
    parser.add_argument(
        "--no-bm25",
        action="store_true",
        help="Disable BM25 hybrid search"
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Use cross-encoder reranker for retrieval candidates"
    )
    parser.add_argument(
        "--reranker-model",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder reranker model name"
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=30,
        help="Number of candidates to rerank"
    )
    parser.add_argument(
        "--rerank-weight",
        type=float,
        default=0.6,
        help="Weight for reranker score blending (0-1)"
    )

    args = parser.parse_args()

    # Handle negation flags
    use_cache = args.use_cache and not args.no_cache
    use_bm25 = args.use_bm25 and not args.no_bm25

    # Check data path
    if not Path(args.data_path).exists():
        print(f"Data path not found: {args.data_path}")
        print("Run 'python scripts/download_locomo.py' first to download/create data.")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize LLM client if requested
    llm_client = None
    if args.use_llm:
        try:
            import openai
            llm_client = openai.OpenAI()
            print("LLM client initialized successfully.")
        except Exception as e:
            print(f"Could not initialize LLM client: {e}")
            print("Proceeding without LLM (using rule-based answers).")

    # Configure components
    memory_config = MemoryConfig(
        working_memory_capacity=30,
        embedding_dim=1536,
        auto_consolidate=True,
    )

    encoder_config = EncoderConfig(
        embedding_model=args.embedding_model,
        embedding_dim=1536,
    )

    retriever_config = RetrieverConfig(
        top_k=20,  # Increased for better retrieval
        use_position_aware_composition=True,
        check_negations=True,
        use_reranker=args.use_reranker,
        rerank_top_n=args.rerank_top_n,
        rerank_weight=args.rerank_weight,
        reranker_model=args.reranker_model,
    )

    # Initialize evaluator with new features
    print("\nInitializing 0GMem evaluator...")
    print(f"  Embedding cache: {'enabled' if use_cache else 'disabled'}")
    print(f"  BM25 hybrid search: {'enabled' if use_bm25 else 'disabled'}")

    use_evidence_reranker = args.use_evidence_reranker and not args.no_evidence_reranker

    evaluator = LoCoMoEvaluator(
        data_path=args.data_path,
        memory_config=memory_config,
        encoder_config=encoder_config,
        retriever_config=retriever_config,
        llm_client=llm_client,
        llm_model=args.llm_model,
        use_evidence_reranker=use_evidence_reranker,
        use_cache=use_cache,
        use_bm25=use_bm25,
    )

    # Load dataset
    print(f"\nLoading dataset from {args.data_path}...")
    num_convs = evaluator.load_dataset()
    print(f"Loaded {num_convs} conversations.")

    # Run evaluation
    print("\nRunning evaluation...")
    try:
        results = evaluator.run_evaluation(
            max_conversations=args.max_conversations,
            max_questions=args.max_questions,
            categories=args.categories,
            resume_from=args.resume_from,
            partial_path=args.partial_output,
        )
    finally:
        # Always save cache on exit
        if use_cache:
            evaluator.save_cache()

    # Print results
    evaluator.print_results(results)

    # Save results
    evaluator.save_results(results, str(output_path))

    return results


if __name__ == "__main__":
    main()
