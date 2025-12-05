# 📸 Image Captioning with Visual Attention

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c)
![Deep Learning](https://img.shields.io/badge/Task-Image%20Captioning-green)

A Deep Learning model that generates descriptive captions for images. This project utilizes an **Encoder-Decoder architecture** with **Attention Mechanism**, combining **EfficientNet-B3** for visual feature extraction and an **LSTM** for text generation.

## 🌟 Project Overview

Image captioning requires a model to understand the visual content of an image and translate it into a natural language sentence. This project solves this using a CNN-RNN architecture enhanced with Bahdanau Attention to focus on specific image regions during caption generation.

### Key Features
* **Encoder:** Pre-trained **EfficientNet-B3** (frozen) for robust feature extraction.
* **Decoder:** **LSTM** (Long Short-Term Memory) network to handle sequence generation.
* **Attention Mechanism:** "Soft" Attention (Bahdanau) allows the model to focus on relevant parts of the image for each word it generates.
* **Inference:** Implements **Beam Search** (k=3) for higher quality caption generation compared to standard greedy search.

## 🏗️ Architecture

1.  **Input Image:** resized to `(224, 224, 3)`.
2.  **Feature Extractor:** EfficientNet-B3 outputs a `14x14` grid of `1536` feature channels.
3.  **Attention Layer:** Computes a context vector based on the current LSTM state and image features.
4.  **Sequence Generator:** LSTM takes the context vector + previous word embedding to predict the next word.

## 📊 Dataset

* **Dataset Used:** [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
* **Size:** 8,091 Images.
* **Vocab Size:** ~3,000 words (Threshold = 5).
* **Splits:** 80% Training, 20% Validation/Test.

## 🛠️ Tech Stack & Hyperparameters

* **Framework:** PyTorch
* **Image Processing:** Torchvision, PIL
* **Text Processing:** NLTK
* **Hyperparameters:**
    * `BATCH_SIZE`: 64
    * `EPOCHS`: 5
    * `LEARNING_RATE`: 3e-4 (Adam Optimizer)
    * `EMBED_DIM`: 512
    * `HIDDEN_DIM`: 512
    * `ATTENTION_DIM`: 512

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/image-captioning-attention.git](https://github.com/your-username/image-captioning-attention.git)
    cd image-captioning-attention
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Dataset Setup:**
    * Download the Flickr8k dataset from Kaggle.
    * Update the `DATA_DIR` path in the notebook to point to your local folder.

4.  **Run the Notebook:**
    * Open `Image_Captioning.ipynb` in Jupyter Notebook or Google Colab.
    * Run all cells to train the model or use the inference block.

## 📈 Results

* **Loss Curve:** Shows steady convergence over 5 epochs (Final Loss ~2.29).
* **Evaluation:** The model is evaluated using **BLEU Scores** to measure n-gram overlap with human references.

### Example Output
*(You can insert a screenshot of a prediction here)*
> **Predicted:** "A dog runs through the green grass."

## 🤝 Acknowledgements
* Common architectures based on "Show, Attend and Tell" paper.
* EfficientNet implementation from `torchvision`.

---
