import os
import sys
import uuid
from tqdm import tqdm
import numpy as np
from numpy.linalg import norm
import tensorflow
import keras
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input

import chromadb

# ✅ Hide TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Initializing ResNet50...")
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)
    return normalized_result.tolist()  # ChromaDB needs standard Python lists, not numpy arrays

# 🚀 Initialize ChromaDB
print("Initializing ChromaDB Database...")
# This creates a folder called 'chroma_storage' in your current directory.
client = chromadb.PersistentClient(path="./chroma_storage")

# Create or load a collection (a "table" for vectors)
# ResNet outputs 2048 dimensions, so we use cosine similarity (which handles normalized vectors perfectly)
collection = client.get_or_create_collection(
    name="fashion_items",
    metadata={"hnsw:space": "cosine"}
)

IMAGE_DIR = os.path.join('archive', 'images')

# Get all images in the folder
all_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Found {len(all_files)} images in {IMAGE_DIR}.")

# 🚀 FAILSAVE: Check what's already in the DB so we can resume if it crashes!
existing_count = collection.count()
print(f"Database currently holds {existing_count} items.")

# We batch insert because inserting 1 by 1 is slow.
BATCH_SIZE = 100
current_batch_embeddings = []
current_batch_ids = []

# ✅ Loop through and extract directly into ChromaDB
for i, file in enumerate(tqdm(all_files, desc="Processing images", unit="image", file=sys.stdout, dynamic_ncols=True)):
    file_id = file  # e.g. "1163.jpg"
    
    # Check if this image was already processed in a previous run
    # (This makes it 100% crash-proof and resumable!)
    result = collection.get(ids=[file_id])
    if result and result['ids']:
        continue  # Skip! We already did this one!

    img_path = os.path.join(IMAGE_DIR, file)
    try:
        features = extract_features(img_path, model)
        current_batch_embeddings.append(features)
        current_batch_ids.append(file_id)
        
        # When we hit our batch size, push to ChromaDB and clear the batch
        if len(current_batch_ids) >= BATCH_SIZE:
            collection.add(
                embeddings=current_batch_embeddings,
                ids=current_batch_ids
            )
            current_batch_embeddings = []
            current_batch_ids = []
            
    except Exception as e:
        print(f"\nSkipping {file} due to error: {e}")

# Push any leftovers
if len(current_batch_ids) > 0:
    collection.add(
        embeddings=current_batch_embeddings,
        ids=current_batch_ids
    )

print("\n✅ Extraction Complete! All images safely stored in ChromaDB ('chroma_storage' folder).")
print(f"Final Database Count: {collection.count()}")
