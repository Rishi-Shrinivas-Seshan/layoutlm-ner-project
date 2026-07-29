import json
from configs.w2_config import background_label

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
# Function to Reconstruct entities
# --------------------------------------------------

def extract_entities(documents, output):

    extracted = []

    for doc, pred in zip(documents, output):

        original_words = doc["words"]
        preds = pred["predictions"]

        entity_dict = {}

        for word, label in zip(original_words, preds):

            if label != background_label:

                if label not in entity_dict:
                    entity_dict[label] = word
                else:
                    entity_dict[label] += " " + word

        extracted.append({
            "file_name": doc["id"],
            "extracted_entities": entity_dict
        })

    return extracted

def format_w2_entities(documents, output):

    summarized_output = summarize_output(output)

    print("\nSummarized Predictions:\n")
    print(json.dumps(summarized_output[:1], indent=4))

    extracted_entities = extract_entities(documents, output)

    print("\nExtracted Output:")
    print(json.dumps(extracted_entities[:1], indent=4))

    return extracted_entities