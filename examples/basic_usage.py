"""
Basic usage example for 0GMem.

This example demonstrates:
- Initializing the memory system
- Adding conversation messages
- Querying memories
"""

from zerogmem import MemoryManager, Encoder, Retriever


def main():
    # Initialize components
    print("Initializing 0GMem...")
    memory = MemoryManager()
    encoder = Encoder()

    # Connect encoder's embedding function to memory
    memory.set_embedding_function(encoder.get_embedding)

    # Create retriever
    retriever = Retriever(memory, embedding_fn=encoder.get_embedding)

    # Start a conversation session
    print("\nStarting conversation session...")
    session_id = memory.start_session()
    print(f"Session ID: {session_id}")

    # Add messages from a conversation
    print("\nAdding conversation messages...")
    memory.add_message("Alice", "I love hiking in the mountains.")
    memory.add_message("Bob", "Which mountains have you visited?")
    memory.add_message("Alice", "I've been to the Alps last summer and Rocky Mountains in 2022.")
    memory.add_message("Bob", "The Alps sound amazing! What was your favorite part?")
    memory.add_message("Alice", "The view from the Matterhorn was incredible. I'd love to go back.")

    # End the session
    print("Ending session...")
    memory.end_session()

    # Query the memory
    print("\n--- Querying Memories ---")

    queries = [
        "When did Alice visit the Alps?",
        "What mountains has Alice been to?",
        "What was Alice's favorite part of the Alps?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = retriever.retrieve(query)
        print(f"Response:\n{result.composed_context[:500]}...")

    # Get memory statistics
    print("\n--- Memory Statistics ---")
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
