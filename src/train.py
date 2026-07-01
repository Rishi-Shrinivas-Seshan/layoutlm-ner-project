import os
from adapters.fatura_adapter import load_fatura_document
from preprocess import process_document
from transformers import LayoutLMTokenizer
from dataset import LayoutLMDataset
from sklearn.model_selection import train_test_split
from transformers import LayoutLMForTokenClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from chunking import split_data_into_chunks
from transformers import DataCollatorForTokenClassification
from functools import partial
from metrics import compute_metrics
from configs.fatura_config import entity_labels, label2id, id2label

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths for directories containing the TSV files and corresponding images of train data
TRAIN_TSV_DIR = os.path.join(BASE_DIR, "dataset", "train", "boxes_transcripts_labels")
TRAIN_IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "train", "images")

tokenizer = LayoutLMTokenizer.from_pretrained(
    "microsoft/layoutlm-base-uncased"
)

# Process all documents in the tsv directory
all_documents = []

for tsv_file in os.listdir(TRAIN_TSV_DIR):
    if tsv_file.endswith('.tsv'):
        tsv_path = os.path.join(TRAIN_TSV_DIR, tsv_file)
        image_name = tsv_file.replace('.tsv', '.jpg')  # for images that are .jpg, adjust if necessary
        image_path = os.path.join(TRAIN_IMAGE_DIR, image_name)

        if os.path.exists(image_path):
            df = load_fatura_document(tsv_path, include_labels=True)
            document = process_document(df, tsv_path, image_path, tokenizer, label2id)
            all_documents.append(document)
        else:
            print(f"Image for {tsv_file} not found.")

print(f"Loaded {len(all_documents)} documents")

train_data, val_data = train_test_split(
    all_documents,
    test_size=0.2,
    random_state=42
)

chunk_size = 512

train_data_split = split_data_into_chunks(train_data, chunk_size=chunk_size)

val_data_split = split_data_into_chunks(val_data, chunk_size=chunk_size)

train_dataset = LayoutLMDataset(train_data_split)
val_dataset = LayoutLMDataset(val_data_split)

print("Dataset ready")

model = LayoutLMForTokenClassification.from_pretrained(
    "microsoft/layoutlm-base-uncased",
    num_labels=len(entity_labels),
    id2label=id2label,
    label2id=label2id
)

print("Model loaded")

# Data collator ensures correct formatting for token classification tasks
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True) # data collator to handle padding

compute_metrics_fn = partial(compute_metrics, entity_labels=entity_labels)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=20,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    eval_strategy="epoch",  # Evaluate after every epoch
    save_strategy="epoch",  # Save model after every epoch
    load_best_model_at_end=True,
    logging_dir="./logs",
    report_to=["tensorboard"]          # Report to TensorBoard
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics_fn,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer.train()

MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "layoutlm_invoice_model")

trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
