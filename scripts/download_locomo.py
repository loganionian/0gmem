#!/usr/bin/env python3
"""
Download the LoCoMo benchmark dataset.
"""

import os
import json
import subprocess
from pathlib import Path


def download_locomo(data_dir: str = "data/locomo") -> None:
    """Download LoCoMo dataset from GitHub."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # Clone the LoCoMo repository
    repo_url = "https://github.com/snap-research/locomo.git"
    repo_path = data_path / "locomo_repo"

    if not repo_path.exists():
        print("Cloning LoCoMo repository...")
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
            check=True
        )
        print("Repository cloned successfully.")
    else:
        print("Repository already exists, skipping clone.")

    # Copy relevant data files
    source_data = repo_path / "data"
    if source_data.exists():
        print("Found data directory in repository.")

        # List available files
        for file_path in source_data.glob("**/*.json"):
            print(f"  Found: {file_path}")

            # Copy to data directory
            dest = data_path / file_path.name
            if not dest.exists():
                import shutil
                shutil.copy(file_path, dest)
                print(f"  Copied to: {dest}")

    print(f"\nLoCoMo data downloaded to: {data_path}")
    return str(data_path)


def create_sample_data(data_dir: str = "data/locomo") -> str:
    """Create sample LoCoMo-format data for testing."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    sample_data = {
        "conversations": [
            {
                "id": "conv_1",
                "sessions": [
                    {
                        "session_id": "session_1",
                        "messages": [
                            {"speaker": "Alice", "content": "Hi Bob! How was your trip to Paris last week?"},
                            {"speaker": "Bob", "content": "It was amazing! I visited the Eiffel Tower and the Louvre."},
                            {"speaker": "Alice", "content": "That sounds wonderful! Did you try any French food?"},
                            {"speaker": "Bob", "content": "Yes! I had croissants every morning and tried escargot."},
                            {"speaker": "Alice", "content": "Escargot? That's adventurous! I could never eat snails."},
                            {"speaker": "Bob", "content": "It was actually delicious! I also bought some macarons from Ladurée."},
                        ]
                    },
                    {
                        "session_id": "session_2",
                        "messages": [
                            {"speaker": "Alice", "content": "Hey Bob, I'm planning my own trip to Europe."},
                            {"speaker": "Bob", "content": "That's exciting! Where are you thinking of going?"},
                            {"speaker": "Alice", "content": "I'm considering Italy. I've always wanted to see Rome."},
                            {"speaker": "Bob", "content": "Rome is beautiful! You should definitely visit the Colosseum."},
                            {"speaker": "Alice", "content": "I heard the pizza there is incredible too."},
                            {"speaker": "Bob", "content": "Absolutely! Much better than what we have here."},
                        ]
                    },
                    {
                        "session_id": "session_3",
                        "messages": [
                            {"speaker": "Alice", "content": "I booked my trip to Rome! Going next month."},
                            {"speaker": "Bob", "content": "Congratulations! How long will you be there?"},
                            {"speaker": "Alice", "content": "Just five days, but I'll try to see as much as possible."},
                            {"speaker": "Bob", "content": "Make sure to visit the Vatican too. The Sistine Chapel is breathtaking."},
                            {"speaker": "Alice", "content": "Good idea! By the way, did you keep in touch with anyone from Paris?"},
                            {"speaker": "Bob", "content": "Yes, I met a tour guide named Pierre. We still chat sometimes."},
                        ]
                    }
                ],
                "questions": [
                    {
                        "id": "q1",
                        "question": "Where did Bob travel to?",
                        "answer": "Paris",
                        "category": "single_hop"
                    },
                    {
                        "id": "q2",
                        "question": "What landmarks did Bob visit in Paris?",
                        "answer": "The Eiffel Tower and the Louvre",
                        "category": "single_hop"
                    },
                    {
                        "id": "q3",
                        "question": "Where is Alice planning to travel?",
                        "answer": "Rome, Italy",
                        "category": "single_hop"
                    },
                    {
                        "id": "q4",
                        "question": "Did Bob visit Paris before or after Alice started planning her trip?",
                        "answer": "Before",
                        "category": "temporal"
                    },
                    {
                        "id": "q5",
                        "question": "Who did Bob meet during his trip that he still keeps in touch with?",
                        "answer": "Pierre, a tour guide",
                        "category": "multi_hop"
                    },
                    {
                        "id": "q6",
                        "question": "What food did Bob say Alice should try, based on his own experience with local cuisine?",
                        "answer": "Bob recommended pizza in Rome, having experienced good local food in Paris",
                        "category": "multi_hop"
                    },
                    {
                        "id": "q7",
                        "question": "Does Alice like escargot?",
                        "answer": "No, she said she could never eat snails",
                        "category": "adversarial"
                    },
                    {
                        "id": "q8",
                        "question": "What is a common activity tourists do in both cities mentioned?",
                        "answer": "Visit famous landmarks and try local food",
                        "category": "commonsense"
                    },
                    {
                        "id": "q9",
                        "question": "How long will Alice's trip to Rome be?",
                        "answer": "Five days",
                        "category": "single_hop"
                    },
                    {
                        "id": "q10",
                        "question": "What happened first: Bob buying macarons or Alice booking her trip?",
                        "answer": "Bob buying macarons",
                        "category": "temporal"
                    }
                ]
            }
        ]
    }

    # Save sample data
    sample_path = data_path / "sample_locomo.json"
    with open(sample_path, "w") as f:
        json.dump(sample_data, f, indent=2)

    print(f"Sample data created at: {sample_path}")
    return str(sample_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download LoCoMo dataset")
    parser.add_argument("--data-dir", default="data/locomo", help="Directory to store data")
    parser.add_argument("--sample-only", action="store_true", help="Only create sample data")

    args = parser.parse_args()

    if args.sample_only:
        create_sample_data(args.data_dir)
    else:
        try:
            download_locomo(args.data_dir)
        except Exception as e:
            print(f"Could not download full dataset: {e}")
            print("Creating sample data instead...")
            create_sample_data(args.data_dir)
