# LayoutLM Document NER Pipeline

## Overview
This project focuses on extracting named entities like amount, date, etc from documents using LayoutLM model, given the spatial information along with OCR & Labels. For this implementation, [LayoutLM base-uncased model](https://huggingface.co/microsoft/layoutlm-base-uncased) is used which operates directly on tokens and bounding box without requiring raw images. 

The model incorporates the layout information in the token representation itself, treating the structure as another feature of each token, which enables the model to capture both textual and structural information at once.

---

## Key Components

### 1. Data Preparation
- Normalize dataset-specific schemas using adapters
- Standardize (scaling, rounding, clipping, and conversion to integer coordinates)

### 2. Tokenization & Encoding
- Map labels to its numerical IDs
- Tokenize using LayoutLM tokenizer
- Assign same bounding boxes values to subword tokens as its original word token

### 3. Chunking
- Split each tokenized document into 512-token segments
- Assign chunk index to preserve the order
- Pad final segment for consistency

### 4. Training
- Wrap dataset using Dataset class for trainer compatibility
- Fine-tune LayoutLM and track evaluation metrics
- Monitor F1 score for convergence using an Early Stopping Callback

### 5. Inference & Reconstruction
- Group prediction chunks belonging to same file by document ID
- Merge the chunk predictions in sorted order using chunk indices
- Remove padding tokens using attention masks
- Reconstruct original OCR words using explicit token-to-word alignment IDs

### 6. Validation
- Ensure alignment between raw data and predictions by checking for length mismatch

---

## Design Decisions

### Chunking before padding
- Padding was initially applied before chunking, which made all documents to a fixed maximum length. This created entire chunks filled with padded tokens & ignored labels(-100) that resulted in NaN validation loss.
- To avoid such fake data chunks, documents are chunked first and only the last segment was padded with zeros to meet the 512- token length. This ensures every training sample(chunk) has valid tokens.

### Merging the predictions
- Recombining fixed number of chunks per document meant combine three consecutive chunks to form the original document, but this logic breaks when every document had variable number of chunks.
- So while breaking the document, each chunk is tagged with the original document ID and an index which later enables to group the chunks by document ID and merge the chunks in the sorted order of chunk indices.
- Thus removing dependency on fixed chunk counts and makes the process efficient for handling arbitrarily long or short documents.

### Partial functions for computing metrics
- Since the Trainer expects a fixed compute_metrics signature, the entity_labels defined in config.py couldn't directly be passed in as an argument.
- Therefore the compute_metrics function was wrapped using a partial function in order to inject the label mapping cleanly into the Trainer.
- This parameterization allows passing the entity_labels while maintaining separation between configuration & evaluation logics.

### Unified preprocessing for train & test
- Separate preprocessing logics for training and testing data led to code duplication because labels existed only in train data.
- This was refactored into a single pipeline in which labels were treated as optional, allowing both train & inference data to pass through the same sequence of steps.

### Explicit token-to-word alignment
- Earlier reconstruction logic relied on bounding box deduplication to merge subword predictions back into original OCR words.
- However, this could incorrectly merge actual OCR duplicate rows because identical bounding boxes were treated as the same token.
- To avoid this, word alignment IDs are now tracked during tokenization.
- Each subword token stores the index of the original OCR word it came from.
- This allows reconstruction using token-word alignment instead of spatial heuristics.

---

## Project Structure

```
layoutlm-ner-project/
│
├── src/
│   ├── adapters/          # Dataset normalizers
│   │   ├── w2_adapter.py
│   │
│   ├── configs/           # Labels and mappings
│   │   ├── w2_config.py
│   │
│   ├── train.py           # Training pipeline
│   ├── predict.py         # Inference pipeline
│   ├── preprocess.py      # TSV → tokens + bbox processing
│   ├── metrics.py         # Evaluation metrics
│   ├── chunking.py        # Chunking logic (512 tokens)
│   ├── dataset.py         # PyTorch Dataset wrapper
│
├── notebooks/             # Notebook version of the full pipeline
│   └── named_entity_recognition_layoutlm.ipynb
│
├── dataset/               # (Not included)
│   └── train/
│       ├── boxes_transcripts_labels/
│       └── images/
│   └── test/
│       ├── boxes_transcripts/
│       └── images/
│
├── models/                # (Ignored in Git)
├── results/               # (Ignored in Git)
│
├── requirements.txt
├── .gitignore
├── README.md
```

---

## Dataset Format

This project expects the dataset in the following structure:

```
dataset/
  train/
    boxes_transcripts_labels/
      doc1.tsv
      doc2.tsv
    images/
      doc1.jpg
      doc2.jpg
  test/
    boxes_transcripts/
      doc1.tsv
      doc2.tsv
    images/
      doc1.jpg
      doc2.jpg
```

### TSV Format

Each `.tsv` file should contain:

```
[start_index, end_index, x1, y1, x2, y2, text, label]
```

- `x1, y1, x2, y2` → bounding box coordinates  
- `text` → token  
- `label` → entity label (optional during inference)

---

## Workflow

### Training Flow

```
TSV → adapter → preprocess → chunk → dataset → LayoutLM → training
```

### Inference Flow

```
TSV → adapter → preprocess → chunk → dataset → predict → recombine → reconstruct → output
```

---

## Running the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

### 3. Run inference

```bash
python src/predict.py
```

---

## Output Format

Predictions are returned as:

```json
[
  {
    "file_name": "doc1",
    "predictions": ["O", "O", "InvoiceNo", "..."]
  }
]
```

Reconstructed key-value output:

```json
[
  {
    "file_name": "doc1",
    "entities": {
      "InvoiceNo": ["INV-1024"],
      "InvoiceDate": ["2024-05-01"],
      "VendorName": ["ABC Pvt Ltd"]
    }
  }
]
```
---

## Important Notes

- The dataset used in development is **not included** due to confidentiality.
- Replace the dataset paths with your own data following the expected format.
- The model assumes OCR tokens and bounding boxes are already available.
- Input documents should follow a consistent structure (e.g., invoices) for reliable performance.
- File naming consistency between images and corresponding annotation files is expected.

---

## Limitations

- Requires OCR output (tokens + bounding boxes)  
- Currently supports `.tsv` format only  
- No built-in OCR pipeline 
- Performance depends on OCR quality and annotation consistency  
- Limited dataset size may affect generalization 

---

## Future Work

### Integrate OCR
- Extract the text fields and their bounding boxes directly from images.

### Multi-dataset support
- Extend the adapter-based pipeline to support datasets like SROIE and raw OCR outputs.

### Improved chunking strategies
- Sliding window enhances better boundary context handling, when an entity gets split across chunks.

### Handling class imbalance
- To enhance performance on underrepresented labels, explore techniques like weighted loss, data augmentation.

### Device optimization
- Enable GPU acceleration (CUDA/MPS) if available for faster training and inference.

### Custom training loop
- Define a customized training function and remove dependency on Trainer API to gain more control over training and evaluation.

### Output formatting
- Structure the output in various expressive formats like key-value extraction, summarize the frequency of labels, and pictorial representation.

### Upgrade to LayoutLM v2
- Visually rich features like logos/seals from the documents can be captured by upgrading to image aware models.

### Containerization
- Dockerize the whole project to easily deploy and reproduce.

---

## Acknowledgements

- HuggingFace Transformers  
- Microsoft LayoutLM  