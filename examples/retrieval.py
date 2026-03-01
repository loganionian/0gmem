"""
Advanced retrieval example for 0GMem.

This example demonstrates:
- Different retrieval strategies
- Query analysis
- Working with retrieval results
"""

from zerogmem import (
    MemoryManager,
    Encoder,
    Retriever,
    RetrieverConfig,
    QueryAnalyzer,
)


def main():
    # Initialize with custom configuration
    print("Initializing 0GMem with custom config...")
    memory = MemoryManager()
    encoder = Encoder()
    memory.set_embedding_function(encoder.get_embedding)

    # Configure retriever with custom settings
    retriever_config = RetrieverConfig(
        semantic_weight=0.5,
        temporal_weight=0.3,
        entity_weight=0.2,
        top_k=5,
        use_reranking=True,
    )
    retriever = Retriever(
        memory,
        embedding_fn=encoder.get_embedding,
        config=retriever_config,
    )

    # Initialize query analyzer
    query_analyzer = QueryAnalyzer()

    # Add some conversations with temporal information
    print("\nAdding conversations with temporal context...")

    # Session 1: Last week
    memory.start_session()
    memory.add_message("Alice", "I just got back from my trip to Paris!")
    memory.add_message("Bob", "How was it?")
    memory.add_message("Alice", "Amazing! I visited the Louvre and saw the Mona Lisa.")
    memory.add_message("Alice", "The Eiffel Tower at night was breathtaking.")
    memory.end_session()

    # Session 2: Yesterday
    memory.start_session()
    memory.add_message("Alice", "I'm thinking about my next trip.")
    memory.add_message("Bob", "Where are you considering?")
    memory.add_message("Alice", "Maybe Tokyo. I've always wanted to try authentic sushi.")
    memory.add_message("Alice", "Also want to visit the temples in Kyoto.")
    memory.end_session()

    # Demonstrate query analysis
    print("\n--- Query Analysis ---")
    queries = [
        "What did Alice see in Paris?",
        "Where does Alice want to travel next?",
        "Has Alice been to any museums?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")

        # Analyze the query
        analysis = query_analyzer.analyze(query)
        print(f"  Intent: {analysis.intent}")
        print(f"  Entities: {analysis.entities}")
        print(f"  Temporal: {analysis.temporal_context}")

        # Retrieve with the analyzed query
        result = retriever.retrieve(query)
        print(f"  Retrieved {len(result.results)} results")

        if result.results:
            print(f"  Top result score: {result.results[0].score:.3f}")

    # Demonstrate working with retrieval results
    print("\n--- Detailed Retrieval Results ---")
    query = "Tell me about Alice's Paris trip"
    result = retriever.retrieve(query)

    print(f"Query: {query}")
    print(f"Composed context:\n{result.composed_context}")

    print("\nIndividual results:")
    for i, r in enumerate(result.results[:3], 1):
        print(f"  {i}. Score: {r.score:.3f}")
        print(f"     Content: {r.content[:100]}...")
        print(f"     Source: {r.source}")


if __name__ == "__main__":
    main()
