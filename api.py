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
        import time
        start_time = time.time()
        
        # Read the image the user uploaded
        image_bytes = await file.read()
        file_size_kb = len(image_bytes) / 1024
        
        # 1. Convert Image to Numbers (2048 Vector)
        vector_start = time.time()
        query_vector = extract_features_from_bytes(image_bytes)
        vector_time = time.time() - vector_start
        
        # 2. Search ChromaDB (Instant Similarity Search!)
        search_start = time.time()
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=num_results
        )
        search_time = time.time() - search_start
        
        # 3. Format the Response
        recommended_filenames = results['ids'][0]
        distances = results['distances'][0]
        s3_base_url = "https://aaskar-fashion-images-2026.s3.ap-southeast-2.amazonaws.com/images/"
        
        final_recommendations = []
        for i in range(len(recommended_filenames)):
            distance = distances[i]
            match_percentage = max(0, 100 - (distance * 50))
            final_recommendations.append({
                "url": s3_base_url + recommended_filenames[i],
                "match": f"{match_percentage:.1f}%"
            })
            
        total_time = time.time() - start_time
        
        # 4. Generate Real Telemetry Logs
        real_logs = [
            f"[EC2] Received image payload: {file_size_kb:.2f} KB",
            f"[EC2] ResNet50 preprocessing complete (shape: 1x224x224x3)",
            f"[EC2] Extracted 2048-dimensional feature vector in {vector_time:.3f}s",
            f"[EC2] ChromaDB vector search completed in {search_time:.3f}s",
            f"[EC2] Mapped {num_results} UUIDs to AWS S3 Object URLs",
            f"[EC2] Pipeline execution finished in {total_time:.3f}s"
        ]
        
        return {
            "success": True,
            "query_filename": file.filename,
            "recommendations": final_recommendations,
            "logs": real_logs
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
