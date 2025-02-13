import faiss
import numpy as np
import os
from PIL import Image
import open_clip
import torch
from flask import Flask, render_template, request

# Initialize the CLIP model (adjusted to use the "-quickgelu" variant for compatibility)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
model.to(device)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

# Define the directory containing images
image_folder = os.path.join("static", "test_data_v2")
if not os.path.exists(image_folder):
    raise FileNotFoundError(f"Error: The directory '{image_folder}' does not exist. Check the path.")

# Retrieve only image files from the specified directory
image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) 
               if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]

# Extract image embeddings and create a FAISS index
image_features = []
for image_path in image_paths:
    try:
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = model.encode_image(image_input).cpu().numpy()
        image_features.append(image_feature)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
# Create and save the FAISS index if image features are available
if image_features:
    image_features = np.vstack(image_features)
    index = faiss.IndexFlatL2(image_features.shape[1])
    index.add(image_features)
    faiss.write_index(index, "faiss_index.bin")
    print("FAISS index created and saved as 'faiss_index.bin'.")
else:
    raise ValueError("No valid images found for indexing.")

#Set up the Flask web application
app = Flask(__name__)
index = faiss.read_index("faiss_index.bin")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query = request.form["query"]
        with torch.no_grad():
            text_features = model.encode_text(tokenizer([query])).cpu().numpy()
        D, I = index.search(text_features, 5)  # Retrieve top 5 matches
        results = [image_paths[i] for i in I[0] if i < len(image_paths)]
        return render_template("index.html", query=query, results=results)
    return render_template("index.html", query=None, results=[])

# Run the Flask application in debug mode
if __name__ == "__main__":
    app.run(debug=True)
