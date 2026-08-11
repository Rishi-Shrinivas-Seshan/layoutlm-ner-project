import json
from configs.funsd_config import background_label

# --------------------------------------------------
# Function to summarize entity predictions
# --------------------------------------------------

def summarize_output(output):

    summarized = []

    for item in output:

        entity_counts = {}

        for label in item["predictions"]:

            if label != background_label:

                if label not in entity_counts:
                    entity_counts[label] = 1
                else:
                    entity_counts[label] += 1

        summarized.append({
            "file_name": item["file_name"],
            "entities_found": entity_counts
        })

    return summarized

# --------------------------------------------------
# Function to reconstruct the classified document
# --------------------------------------------------

def extract_entities(documents, output):

    extracted = []

    for doc, pred in zip(documents, output):

        original_words = doc["words"]
        preds = pred["predictions"]

        rows = []

        for word, label in zip(original_words, preds):

            rows.append({
                "text": word,
                "label": label
            })

        extracted.append({
            "file_name": doc["id"],
            "classified_document": rows
        })

    return extracted

# --------------------------------------------------
# Function to present FUNSD predictions
# --------------------------------------------------

def format_funsd_entities(documents, output):

    extracted_entities = extract_entities(documents, output)

    print("\nReconstructed Document:\n")
    print(json.dumps(extracted_entities[:1], indent=4))

    return extracted_entities