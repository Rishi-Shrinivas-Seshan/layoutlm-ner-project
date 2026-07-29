import json
import re
from configs.sroie_config import background_label

# --------------------------------------------------
# SROIE-specific cleaning functions
# --------------------------------------------------

# Function to clean out repeated text
def clean_total(text):

    words = text.split()

    cleaned = []

    for word in words:
        if word not in cleaned:
            cleaned.append(word)

    return " ".join(cleaned)

# Function to clean out repeated pattern like dates
def clean_date(text):

    match = re.search(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", text)

    if match:
        return match.group()

    return text

# Function scope to clean company
def clean_company(text):

    return text

# Function scope to clean address
def clean_address(text):

    return text

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

# --------------------------------------------------
# Function to Clean the reconstructed entities
# --------------------------------------------------

def clean_entities(results):

    for doc in results:

        entities = doc["extracted_entities"]

        # Clean total
        if "total" in entities:
            entities["total"] = clean_total(entities["total"])

        # Clean date
        if "date" in entities:
            entities["date"] = clean_date(entities["date"])

        # Clean company
        if "company" in entities:
            entities["company"] = clean_company(entities["company"])

        # Clean address
        if "address" in entities:
            entities["address"] = clean_address(entities["address"])

    return results

def format_sroie_entities(documents, output):

    summarized_output = summarize_output(output)

    print("\nSummarized Predictions:\n")
    print(json.dumps(summarized_output[:1], indent=4))

    extracted_entities = extract_entities(documents, output)

    cleaned_entities = clean_entities(extracted_entities)

    print("\nFormatted Output:")
    print(json.dumps(cleaned_entities[:1], indent=4))

    return cleaned_entities