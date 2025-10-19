import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import sys
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ✅ hide TensorFlow logs

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()  # ✅ no spam output
    normalized_result = result / norm(result)
    return normalized_result

filenames = []
for file in os.listdir('New Images'):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # ✅ avoid non-images
        filenames.append(os.path.join('New Images', file))

feature_list = []

# ✅ tqdm fixed to work in PyCharm console
for file in tqdm(filenames, desc="Extracting features", unit="image", file=sys.stdout, dynamic_ncols=True):
    feature_list.append(extract_features(file, model))

pickle.dump(feature_list,open('embedding.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))
