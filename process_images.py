import os
from PIL import Image
import torch
import open_clip
from torchvision import transforms
import faiss
import numpy as np
from tqdm import tqdm
import open_clip
import torch

#Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = open_clip.create_model("ViT-B-32", pretrained="openai").to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
preprocess = open_clip.image_transform(model.visual.image_size, is_train=False) 


# Define the directory containing images and load the file names
image_folder = r"C:\Users\Nhlamolom\Downloads\image_retrieval_project\static\test_data_v2"
if not os.path.exists(image_folder):
    raise FileNotFoundError(f"Error: The directory '{image_folder}' does not exist. Check the path.")
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

image_embeddings = []
image_paths = []

# Process each image by loading, converting to RGB, and encoding it
for img_path in tqdm(image_files, desc="Processing images"):
    image = Image.open(img_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        img_embedding = model.encode_image(image).cpu().numpy()
    
    image_embeddings.append(img_embedding)
    image_paths.append(img_path)

# Construct a FAISS index for efficient image similarity search
image_embeddings = np.vstack(image_embeddings)
index = faiss.IndexFlatL2(image_embeddings.shape[1])
index.add(image_embeddings)

# Store the FAISS index and corresponding image file paths
faiss.write_index(index, "image_index.faiss")
np.save("image_paths.npy", image_paths)

print("Image embeddings saved successfully!")


