import os
import torch
import pandas as pd
from collections import defaultdict
from transformers import LayoutLMForTokenClassification, LayoutLMTokenizer, Trainer
import json
from dataset import LayoutLMDataset
from adapters.w2_adapter import load_w2_document
from preprocess import process_document
from chunking import split_data_into_chunks
from configs.w2_config import id2label

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "layoutlm_invoice_model")

tokenizer = LayoutLMTokenizer.from_pretrained(MODEL_PATH)
model = LayoutLMForTokenClassification.from_pretrained(MODEL_PATH)

model.eval()

# Function to load the documents for testing
def load_test_documents(tsv_dir, image_dir):
    documents =[]

    for tsv_file in os.listdir(tsv_dir):
        if tsv_file.endswith('.tsv'):
            tsv_path = os.path.join(tsv_dir, tsv_file)
            image_name = tsv_file.replace('.tsv', '.jpg')  # for images that are .jpg, adjust if necessary
            image_path = os.path.join(image_dir, image_name)

            if os.path.exists(image_path):
                df = load_w2_document(tsv_path)
                document = process_document(df, tsv_path, image_path, tokenizer=tokenizer, label2id=None)
                documents.append(document)
            else:
                print(f"Image for {tsv_file} not found.")

    return documents

# Function to obtain predictions using trainer
def run_prediction(test_dataset):
    trainer = Trainer(model=model)

    # Inference using trainer.predict()
    predictions, _, _ = trainer.predict(test_dataset)

    # Convert predictions from numpy to torch tensor
    predictions_tensor = torch.tensor(predictions)

    # Convert logits to predicted label IDs (take the argmax over num_labels)
    predicted_label_ids = torch.argmax(predictions_tensor, dim=-1)

    # Map predicted label IDs to actual entity labels
    predicted_entities = [[id2label[label_id.item()] for label_id in example] for example in predicted_label_ids]

    return predicted_entities

# Function to reconstruct the predictions
def recombine_predictions(documents, chunked_docs, predicted_entities):

    grouped_preds = defaultdict(list)

    for item, pred in zip(chunked_docs, predicted_entities):
        grouped_preds[item['id']].append({
            "chunk_idx": item['chunk_idx'],
            "predictions": pred,
            "attention_mask": item['attention_mask']
        })

    final_output = []

    # Create a Dictionary converting the List of Documents into a lookup
    doc_map = {doc["id"]: doc for doc in documents}

    for doc_id in grouped_preds:

        token_word_ids = doc_map[doc_id]["word_ids"]
        chunks = sorted(grouped_preds[doc_id], key=lambda x: x['chunk_idx'])

        combined_preds = []
        combined_masks = []

        for chunk in chunks:
            combined_preds.extend(chunk["predictions"])
            combined_masks.extend(chunk["attention_mask"].tolist())

        # Strip the padded zeros in the merged documents using attention mask
        cleaned_preds = [prediction for prediction, mask in zip(combined_preds, combined_masks) if mask==1]

        # Detokenize - tokenization splits the words into subwords
        # Dropping the duplicates using the word IDs will match the length of the original list of words
        df = pd.DataFrame({"pred": cleaned_preds, "word_id": token_word_ids})
        df = df.drop_duplicates(subset=["word_id"])

        # Format the final output by associating the predictions with its corresponding file name
        final_preds = df["pred"].tolist()

        final_output.append({"file_name": doc_id, "predictions": final_preds})

    return final_output

# Function to verify if number of predictions matched the length original data
def validate_predictions(output, tsv_dir):

    pred_map = {item["file_name"]: item["predictions"] for item in output}

    for tsv_file in os.listdir(tsv_dir):
        doc_id = tsv_file.split('.')[0]

        tsv_path = os.path.join(tsv_dir, tsv_file)
        df = pd.read_csv(tsv_path, header=None)

        preds = pred_map.get(doc_id)

        assert len(df) == len(preds), f"Length mismatch in {doc_id}"

# Function to summarize entity predictions
def summarize_output(output):

    summarized = []

    for item in output:

        entity_counts = {}

        for label in item["predictions"]:

            if label != "OTHER":

                if label not in entity_counts:
                    entity_counts[label] = 1
                else:
                    entity_counts[label] += 1

        summarized.append({
            "file_name": item["file_name"],
            "entities_found": entity_counts
        })

    return summarized

def extract_entities(documents, final_output):

    extracted = []

    for doc, pred in zip(documents, final_output):

        original_words = doc["words"]
        preds = pred["predictions"]

        entity_dict = {}

        for word, label in zip(original_words, preds):

            if label != "OTHER":

                if label not in entity_dict:
                    entity_dict[label] = word
                else:
                    entity_dict[label] += " " + word

        extracted.append({
            "file_name": doc["id"],
            "extracted_entities": entity_dict
        })

    return extracted

if __name__ == "__main__":

    # Dataset paths for directories containing the TSV files and corresponding images of test data
    TEST_TSV_DIR = os.path.join(BASE_DIR, "dataset", "test", "boxes_transcripts")
    TEST_IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "test", "images")

    print("Loading test documents...")
    documents = load_test_documents(TEST_TSV_DIR, TEST_IMAGE_DIR)

    print("Transforming the documents...")
    chunked_docs = split_data_into_chunks(documents, chunk_size=512)

    print("Preparing the model inputs...")
    dataset = LayoutLMDataset(chunked_docs)

    print("Running inference...")
    predictions = run_prediction(dataset)

    print("Reconstructing predictions...")
    output = recombine_predictions(documents, chunked_docs, predictions)

    print("\nInference complete")

    print("Validating predictions...")
    validate_predictions(output, TEST_TSV_DIR)
    
    print("\nSample Predictions:")
    print(json.dumps(output[:1], indent=2))

    summary = summarize_output(output)

    print("\nSummarized Predictions:\n")
    print(json.dumps(summary[:1], indent=4))

    recognized_entities = extract_entities(documents, output)
    print("\nExtracted Entities:")
    print(json.dumps(recognized_entities[:1], indent=4))

    print("\nFinal Output: Successfully computed")