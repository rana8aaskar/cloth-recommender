import pickle
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import tensorflow
from annoy import AnnoyIndex

# Load filenames (we'll use Annoy index instead of feature_list)
filenames = pickle.load(open('filenames.pkl','rb'))

# Load pre-built Annoy index
# If annoy_index.ann doesn't exist, run: python build_annoy_index.py
print("Loading Annoy index...")
f = 2048  # ResNet50 feature dimension
annoy_index = AnnoyIndex(f, 'euclidean')
annoy_index.load('annoy_index.ann')
print(f"✅ Loaded index with {annoy_index.get_n_items()} items")

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img('sample/kurti.jpeg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img, verbose=0).flatten()  # ✅ no spam output
normalized_result = result / norm(result)

# Use Annoy for fast similarity search
# get_nns_by_vector returns (indices, distances) or just indices
# search_k: controls accuracy/speed tradeoff (higher = more accurate but slower)
# -1 means use default (n_trees * n_neighbors)
n_neighbors = 5
indices = annoy_index.get_nns_by_vector(normalized_result, n_neighbors, search_k=-1, include_distances=False)

print(f"Top {n_neighbors} similar items:")
print(indices)
for idx in indices:
    print(filenames[idx])