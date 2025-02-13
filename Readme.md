# Multi-Modal Image Retrieval System

Project Summary
This project is a Multi-Modal Image Retrieval System that leverages CLIP (Contrastive Language–Image Pretraining) and FAISS (Facebook AI Similarity Search) to retrieve images based on text queries. Users can input natural language descriptions, and the system efficiently finds the most relevant images from a dataset.

Key Features
Text-to-Image Search: Users can search for images using descriptive text.
CLIP Model: Utilizes OpenAI’s CLIP (ViT-B-32-quickgelu) for encoding text and images.
FAISS for Fast Search: Implements FAISS for high-speed similarity matching.
Flask Web Interface: Provides an easy-to-use frontend for search and display.
Robust Processing: Handles image indexing and query processing effective

Project Structure
```
📂 Multi-Modal Image Retrieval System
│── 📂 static
│   │── 📂 test_data_v2  # Image dataset
│── 📂 templates
│   │── index.html  # Frontend UI
│── app.py  # Main application script (Flask)
     search.py # search engine
     process.py # image embedded
|
│─faiss_index.bin  # FAISS index file
│── requirements.txt  # Dependencies
│── README.md  # Project Documentation
```


1.Clone the Repository
```sh
git clone https://github.com/your-username/multi-modal-image-retrieval.git
cd multi-modal-image-retrieval
```

2.Create a Virtual Environment (Optional but Recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Dependencies
```sh
pip install -r requirements.txt
```

4. Run the Application
```sh
python app.py
```
Go to **http://127.0.0.1:5000/** in your browser.

 How It Works
1. **Indexing Phase**:
   - Loads all images from `static/test_data_v2`.
   - Uses CLIP to extract image embeddings.
   - Stores embeddings in a FAISS index for fast retrieval.
2. **Query Phase**:
   - User enters a text query.
   - CLIP converts the query into an embedding.
   - FAISS retrieves the top 5 most relevant images.
   - Results are displayed on the webpage.

 Example Usage
1. **Input Query:** _"a red sports car on the road"_
2. **Output:** Displays relevant images from the dataset.

Tech Stack
- **Python** (Backend)
- **Flask** (Web Framework)
- **FAISS** (Efficient Image Retrieval)
- **CLIP (OpenAI)** (Text-Image Embedding)
- **PIL & NumPy** (Image Processing)
- **HTML, CSS** (Frontend)

 To-Do / Future Enhancements
- 🔹 Implement a larger dataset for testing.
- 🔹 Add user-uploaded images for retrieval.
- 🔹 Deploy on a cloud server (AWS/GCP/Heroku).

## 👨‍💻 Author
[Your Name](https://github.com/NhlamoloMaluleke)

## 📜 License
This project is licensed under the **MIT License**.

---
⭐ **If you find this project helpful, give it a star on GitHub!** ⭐

