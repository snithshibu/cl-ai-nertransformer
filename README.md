# Named Entity Recognition with BERT (Sequence Labeling)

This project fine-tunes a **BERT** transformer model for **Named Entity Recognition (NER)**, a sequence labeling task where each token in a sentence is assigned a tag such as person, location, organization, or time using the BIO scheme. 

Author: Snith Shibu

---

## 1. Project Overview

The goal of this project is to:

- Explain and implement a **sequence labeling** task (NER) using a pre-trained transformer model (BERT).
- Train a model to recognize entities such as names, locations, organizations, and dates.
- Evaluate the model with precision, recall, and F1-score.
- Provide a simple GUI demo and a Medium article describing the approach.
  
This work is based on a labeled English NER dataset and on standard BERT fine-tuning practices for token classification. 

---

## 2. Dataset

- Format: A CSV file `ner.csv` with the following columns:
  - `Sentence #`: Sentence identifier.
  - `Sentence`: Full sentence text.
  - `POS`: List of POS tags for each token in the sentence.
  - `Tag`: List of NER tags (BIO format) for each token. 

- Label scheme:
  - 17 BIO tags, including:
    - `B-geo`, `I-geo` (geographical entities)
    - `B-gpe`, `I-gpe` (countries/cities as political entities)
    - `B-org`, `I-org` (organizations)
    - `B-per`, `I-per` (persons)
    - `B-tim`, `I-tim` (time expressions)
    - `B-art`, `I-art`, `B-eve`, `I-eve`, `B-nat`, `I-nat`
    - `O` (outside any entity).
      
- Preprocessing steps:
  - Convert each row into `(tokens, labels)` for a full sentence.
  - Map label strings to integer IDs (`label2id`) and back (`id2label`).
  - Use the BIO scheme to preserve entity boundaries. 

---

## 3. Model and Training Setup

### 3.1 Base model

- Architecture: **BERT** transformer encoder. 
- Checkpoint: `bert-base-cased` from Hugging Face. 
- Task head: Token classification layer for 17 labels. 

### 3.2 Tokenization and label alignment

- Tokenizer: `AutoTokenizer` for `bert-base-cased` with WordPiece subword tokenization. 
- Special handling:
  - Input is split into words and passed with `is_split_into_words=True`.
  - For each word, only the **first subword** receives the NER label; subsequent subwords and special tokens (`[CLS]`, `[SEP]`, padding) are assigned label `-100` so they are ignored in the loss. 

### 3.3 Training configuration

- Framework: **PyTorch** + Hugging Face `transformers`. 
- Loss function: Token-level cross-entropy (built into `AutoModelForTokenClassification`). 
- Optimizer: **AdamW**, learning rate `2e-5`. 
- Max sequence length: 128 tokens.
- Batch size: 16.
- Epochs: 2 (fine-tuning from pre-trained weights).
- Device: Trained on GPU using Google Colab. 

### 3.4 Data split

- Train/test split:
  - 90% of sentences for training.
  - 10% of sentences for testing.
- A few noisy rows with mismatched token/tag lengths are skipped during preprocessing. 

---

## 4. Evaluation

Evaluation is performed on the held-out test set using the **seqeval** library, which computes entity-level precision, recall, and F1-score. 

- Overall metrics (micro-averaged across entity types):

  - **Precision**: 0.8378  
  - **Recall**: 0.8380  
  - **F1-score**: 0.8379  

- Entity-wise performance (examples):

  - `geo` (geographical entities): F1 ≈ 0.88  
  - `gpe` (political entities): F1 ≈ 0.95  
  - `tim` (time expressions): F1 ≈ 0.87  
  - `org` (organizations): F1 ≈ 0.70  
  - `per` (persons): F1 ≈ 0.77  

These scores are typical for BERT-based NER models on CoNLL-style data and show strong performance on common entity types. 

---

## 5. Repository Structure

Suggested layout of this repository:

```
.
├── ner_finetuning.ipynb    # Jupyter/Colab notebook with full training & evaluation pipeline
├── ner.csv                 # Labeled NER dataset (if allowed to include) or instructions to download
├── app.py                  # Simple Gradio GUI for interactive NER demo
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .gitignore              # Ignore large model files, checkpoints, etc.
```

Large model weight files (e.g., `pytorch_model.bin`) are not stored in this GitHub repository due to size limits; the model is used directly in the notebook and can be hosted externally if needed. 

---

## 6. Installation and Usage

### 6.1 Local setup

1. Clone the repository:

```
git clone <GITHUB_REPO_URL>
cd <REPO_FOLDER_NAME>
```

2. (Optional but recommended) Create and activate a virtual environment.

3. Install dependencies:

```
pip install -r requirements.txt
```

### 6.2 Running the notebook

1. Open `ner_finetuning.ipynb` in VS Code, Jupyter, or Google Colab. 
2. Make sure `ner.csv` is available in the same folder (or adjust the path in the notebook).
3. Run the cells in order to:
   - Load and preprocess the dataset.
   - Fine-tune BERT for token classification.
   - Evaluate the model with seqeval classification report.

### 6.3 Running the Gradio demo locally

1. Ensure you have a fine-tuned model available (either locally or by adjusting `app.py` to load from Hugging Face Hub). 
2. In the project directory:

```
python app.py
```

3. Open the local URL printed in the terminal and enter a sentence (e.g., “Barack Obama visited London in 2010.”) to see detected entities and their labels. 

---

## 7. Online Demo (Temporary Gradio Link)

During development and evaluation, the model is exposed via a Gradio app running in Google Colab, using a temporary public URL that is valid for several days. 

- Temporary demo URL (may expire):  
  `https://3d78449908a570f0e4.gradio.live/`

If the link has expired, you can recreate the Gradio demo by running the last section of `ner_finetuning.ipynb` in Google Colab.

---

## 8. Medium Article

A Medium article accompanies this repository and explains:

- What sequence labeling and NER are.
- Traditional approaches to NER (feature-based models, HMMs/CRFs, BiLSTM-CRF). 
- Why transformer-based models like BERT work well for NER. 
- The full pipeline of this project:
  - Dataset and BIO tagging.
  - Tokenization and label alignment.
  - Fine-tuning BERT.
  - Evaluation and demo.

- Medium article: `https://medium.com/@snithshibu/from-tokens-to-entities-building-a-named-entity-recognition-system-with-bert-5fe262795481`

---
