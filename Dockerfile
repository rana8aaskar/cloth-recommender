# 1. THE FOUNDATION: Use a lightweight version of Python 3.12
FROM python:3.12-slim

# 2. THE FOLDER: Create a folder called /app inside the virtual box
WORKDIR /app

# 3. COPY PACKAGES: Copy your requirements list from Windows into the box
COPY requirements.txt .

# 4. INSTALL: Run pip install inside the box to install TensorFlow, ResNet, etc.
# We add the extra FastAPI packages here too just to be safe!
RUN pip install --no-cache-dir -r requirements.txt
# 5. COPY CODE & DATABASE: Copy your FastAPI script AND your built database!
COPY api.py .
COPY chroma_storage/ ./chroma_storage/

# Expose port 80 so AWS allows web traffic to hit your API
EXPOSE 80

# 6. THE IGNITION SWITCH: Tell the box to start FastAPI when it turns on
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
