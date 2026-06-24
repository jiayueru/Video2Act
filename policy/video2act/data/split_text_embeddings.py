"""
Script to split text embeddings into two folders:
- text_embeddings: first 100 embeddings (for training)
- unseen_text_embeddings: embeddings after 100 (for evaluation)

Usage:
    python split_text_embeddings.py <episode_dir>

Example:
    python split_text_embeddings.py /path/to/episode_0
"""

import os
import shutil
import sys


def split_text_embeddings(episode_dir):
    """
    Split text_embeddings folder into two folders:
    - text_embeddings: first 100 .pkl files
    - unseen_text_embeddings: remaining .pkl files after 100

    Args:
        episode_dir: Path to the episode directory containing text_embeddings folder
    """
    text_embeddings_path = os.path.join(episode_dir, 'text_embeddings')

    if not os.path.exists(text_embeddings_path):
        print(f"Error: text_embeddings folder not found at {text_embeddings_path}")
        return

    # Get all .pkl files and sort them
    pkl_files = sorted([f for f in os.listdir(text_embeddings_path) if f.endswith('.pkl')])

    if len(pkl_files) <= 100:
        print(f"Only {len(pkl_files)} embeddings found, no need to split (<=100)")
        return

    # Create unseen_text_embeddings folder
    unseen_path = os.path.join(episode_dir, 'unseen_text_embeddings')
    os.makedirs(unseen_path, exist_ok=True)

    # Move files after index 100 to unseen_text_embeddings
    moved_count = 0
    for i, filename in enumerate(pkl_files):
        if i >= 100:  # Keep first 100 in text_embeddings, move the rest
            src = os.path.join(text_embeddings_path, filename)
            dst = os.path.join(unseen_path, filename)
            shutil.move(src, dst)
            moved_count += 1

    print(f"✓ Split completed for {episode_dir}")
    print(f"  - text_embeddings: {100} files (kept)")
    print(f"  - unseen_text_embeddings: {moved_count} files (moved)")


def split_all_episodes(data_dir):
    """
    Split text embeddings for all episodes in a data directory

    Args:
        data_dir: Path to the root data directory containing multiple episodes
    """
    episode_count = 0
    for root, dirs, files in os.walk(data_dir):
        if 'text_embeddings' in dirs:
            split_text_embeddings(root)
            episode_count += 1

    print(f"\n✓ Processed {episode_count} episodes in total")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_text_embeddings.py <episode_dir_or_data_dir>")
        print("\nExamples:")
        print("  Single episode: python split_text_embeddings.py /path/to/episode_0")
        print("  All episodes:   python split_text_embeddings.py /path/to/data_root")
        sys.exit(1)

    target_path = sys.argv[1]

    if not os.path.exists(target_path):
        print(f"Error: Path not found: {target_path}")
        sys.exit(1)

    # Check if it's a single episode or a data directory
    if os.path.exists(os.path.join(target_path, 'text_embeddings')):
        # Single episode
        split_text_embeddings(target_path)
    else:
        # Data directory with multiple episodes
        split_all_episodes(target_path)
