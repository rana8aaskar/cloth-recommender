"""
Build Annoy index from pre-computed embeddings for fast similarity search.
Run this once after generating embeddings with app.py
"""
import pickle
from annoy import AnnoyIndex

# Load pre-computed embeddings
print("Loading embeddings...")
feature_list = pickle.load(open('embedding.pkl', 'rb'))
filenames = pickle.load(open('filenames.pkl', 'rb'))

print(f"Loaded {len(feature_list)} embeddings")
print(f"Feature dimension: {len(feature_list[0])}")

# Build Annoy index
# Parameters:
# - f: dimension of feature vectors (ResNet50 with GlobalMaxPooling = 2048)
# - metric: 'angular' (cosine), 'euclidean', 'manhattan', 'hamming', 'dot'
f = len(feature_list[0])  # Should be 2048 for ResNet50
annoy_index = AnnoyIndex(f, 'euclidean')

print("Building Annoy index...")
for i, vector in enumerate(feature_list):
    annoy_index.add_item(i, vector)
    if (i + 1) % 1000 == 0:
        print(f"  Added {i + 1}/{len(feature_list)} items...")

# Build index with n_trees
# More trees = better accuracy but larger index size and slower build time
# Recommended: 10-100 trees for most use cases
n_trees = 50
print(f"Building {n_trees} trees (this may take a minute)...")
annoy_index.build(n_trees)

# Save index to disk
print("Saving Annoy index...")
annoy_index.save('annoy_index.ann')

print("✅ Done! Index saved to 'annoy_index.ann'")
print(f"   Index size on disk: ~{annoy_index.get_n_items() * f * 4 / (1024*1024):.1f} MB")
print(f"   Ready for fast similarity search!")
