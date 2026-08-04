import os
import io
import numpy as np
from numpy.linalg import norm
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

import keras
from keras.preprocessing import image as keras_image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
import chromadb

# 1. Initialize FastAPI App
app = FastAPI(title="Fashion Recommender API", version="1.0.0")

# Allow frontend (like Vercel/React) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the Machine Learning Model (runs once when server starts)
print("Loading ResNet50 Model into Memory...")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# 3. Connect to the Dynamic Vector Database
print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./chroma_storage")
collection = chroma_client.get_collection(name="fashion_items")

print(f"API Ready! Connected to database with {collection.count()} items.")

# --- Helper Function to Extract Features from raw bytes ---
def extract_features_from_bytes(image_bytes):
    # Convert raw uploaded bytes into a PIL Image, then resize to 224x224
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    
    # Standard Keras processing
    img_array = keras_image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    
    # Get 2048 numbers
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)
    return normalized_result.tolist()

# --- THE API ENDPOINTS ---

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fashion AI Microservice!", "status": "Online"}

@app.post("/recommend")
async def recommend_fashion(file: UploadFile = File(...), num_results: int = 5):
    """
    Receives an image, extracts features, and queries ChromaDB for the closest matches.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read the image the user uploaded
        image_bytes = await file.read()
        
        # 1. Convert Image to Numbers (2048 Vector)
        query_vector = extract_features_from_bytes(image_bytes)
        
        # 2. Search ChromaDB (Instant Similarity Search!)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=num_results
        )
        
        # 3. Format the Response
        # ChromaDB returns a dictionary with 'ids' and 'distances'
        recommended_filenames = results['ids'][0]
        distances = results['distances'][0]
        
        # Prepend the AWS S3 Bucket URL to return fully working image links
        s3_base_url = "https://aaskar-fashion-images-2026.s3.ap-southeast-2.amazonaws.com/images/"
        
        final_recommendations = []
        for i in range(len(recommended_filenames)):
            # Convert cosine distance to a "Match Percentage" for the UI
            distance = distances[i]
            match_percentage = max(0, 100 - (distance * 50))
            
            final_recommendations.append({
                "url": s3_base_url + recommended_filenames[i],
                "match": f"{match_percentage:.1f}%"
            })
        
        return {
            "success": True,
            "query_filename": file.filename,
            "recommendations": final_recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
