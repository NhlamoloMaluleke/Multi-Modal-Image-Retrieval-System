import torch
import open_clip
import faiss
import numpy as np
from PIL import Image

# Initialize and load the CLIP model
import open_clip
import torch

# Set up the device to use GPU if available, otherwise default to CPU

device = "cuda" if torch.cuda.is_available() else "cpu"
model = open_clip.create_model("ViT-B-32", pretrained="openai").to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
preprocess = open_clip.image_transform(model.visual.image_size, is_train=False) 


# Retrieve the FAISS index and the stored image file paths
index = faiss.read_index("image_index.faiss")
image_paths = np.load("image_paths.npy")

def search_images(query, top_k=5):
    # Convert the input text query into an embedding using the CLIP model
    with torch.no_grad():
        text_embedding = model.encode_text(tokenizer([query])).cpu().numpy()

     # Retrieve the most relevant images based on similarity search
    _, indices = index.search(text_embedding, top_k)
    
    return [image_paths[i] for i in indices[0]]

# Example search query

query = "a flower"
results = search_images(query)

print("Matching images:", results)
