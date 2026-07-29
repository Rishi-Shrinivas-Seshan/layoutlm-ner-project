import os
import json
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SROIE_DIR = os.path.join(BASE_DIR, "SROIE_dataset")

OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_BOX_DIR = os.path.join(SROIE_DIR, "train", "box")

TRAIN_ENTITY_DIR = os.path.join(SROIE_DIR, "train", "entities")

TRAIN_IMAGE_DIR = os.path.join(SROIE_DIR, "train", "img")

OUTPUT_TRAIN_TSV_DIR = os.path.join(OUTPUT_DATASET_DIR, "train", "boxes_transcripts_labels")

OUTPUT_TRAIN_IMAGE_DIR = os.path.join(OUTPUT_DATASET_DIR, "train", "images")

TEST_BOX_DIR = os.path.join(SROIE_DIR, "test", "box")

TEST_ENTITY_DIR = os.path.join(SROIE_DIR, "test", "entities")

TEST_IMAGE_DIR = os.path.join(SROIE_DIR, "test", "img")

OUTPUT_TEST_TSV_DIR = os.path.join(OUTPUT_DATASET_DIR, "test", "boxes_transcripts")

OUTPUT_TEST_IMAGE_DIR = os.path.join(OUTPUT_DATASET_DIR, "test", "images")

# Create the directory without crashes
def create_directory(path):

    os.makedirs(path, exist_ok=True)

def normalize_text(text):

    text = str(text)
    
    text = text.strip()

    text = text.lower()

    text = "".join(text.split())

    text = text.replace(",", "")

    text = text.replace(".", "")

    return text

# Assign token level labels by matching field level entities
def assign_label(index, ocr_texts, entities):

    text = ocr_texts[index]
    normalized_text = normalize_text(text)
    
    for label, value in entities.items():

        if value is None:
            continue

        if isinstance(value, list):
            values = value
        else:
            values = [value]

        for entity_text in values:
            
            normalized_entity = normalize_text(entity_text)

            # -----------------------------
            # ADDRESS
            # -----------------------------
            if label == "address":

                # Ignore tiny OCR fragments
                if len(normalized_text) < 5:
                    continue

                # Address GT often spans multiple OCR rows
                if normalized_text in normalized_entity:
                    return label

            # -----------------------------
            # COMPANY
            # -----------------------------
            elif label == "company":

                # Ignore tiny OCR fragments
                if len(normalized_text) < 5:
                    continue
                
                # If the company was a single OCR line
                if normalized_entity in normalized_text:
                    return label
                
                # Current OCR line looks like part of the company
                if normalized_text in normalized_entity:

                    current = normalize_text(ocr_texts[index])

                    # -----------------------------
                    # (i-1, i)
                    # -----------------------------
                    if index >= 1:

                        combined_text = (
                            normalize_text(ocr_texts[index - 1]) +
                            current
                        )

                        if combined_text == normalized_entity:
                            return label
                        
                    # -----------------------------
                    # (i, i+1)
                    # -----------------------------
                    if index + 1 < len(ocr_texts):

                        combined_text = (
                            current +
                            normalize_text(ocr_texts[index + 1])
                        )

                        if combined_text == normalized_entity:
                            return label
                        
                    # -----------------------------
                    # (i-2, i-1, i)
                    # -----------------------------
                    if index >= 2:

                        combined_text = (
                            normalize_text(ocr_texts[index - 2]) +
                            normalize_text(ocr_texts[index - 1]) +
                            current
                        )

                        if combined_text == normalized_entity:
                            return label

                    # -----------------------------
                    # (i-1, i, i+1)
                    # -----------------------------
                    if index >= 1 and index + 1 < len(ocr_texts):

                        combined_text = (
                            normalize_text(ocr_texts[index - 1]) +
                            current +
                            normalize_text(ocr_texts[index + 1])
                        )

                        if combined_text == normalized_entity:
                            return label

                    # -----------------------------
                    # (i, i+1, i+2)
                    # -----------------------------
                    if index + 2 < len(ocr_texts):

                        combined_text = (
                            current +
                            normalize_text(ocr_texts[index + 1]) +
                            normalize_text(ocr_texts[index + 2])
                        )

                        if combined_text == normalized_entity:
                            return label

            # -----------------------------
            # DATE
            # -----------------------------
            elif label == "date":

                if normalized_entity in normalized_text:
                    return label

            # -----------------------------
            # TOTAL
            # -----------------------------
            elif label == "total":

                if normalized_text == normalized_entity:
                    return label

            # -----------------------------
            # FUTURE ENTITY TYPES
            # -----------------------------
            else:

                if normalized_text == normalized_entity:
                    return label

    return "OTHER"

# Preapre the dataset 
def prepare_dataset(box_dir, entity_dir, image_dir, output_tsv_dir, output_image_dir):

    for file_name in os.listdir(box_dir):

        if not file_name.endswith(".txt"):
            continue

        file_id = file_name.replace(".txt", "")

        box_path = os.path.join(box_dir, file_name)

        entity_path = os.path.join(entity_dir, f"{file_id}.txt")

        image_path = os.path.join(image_dir, f"{file_id}.jpg")

        output_tsv_path = os.path.join(output_tsv_dir, f"{file_id}.tsv")

        output_image_path = os.path.join(output_image_dir, f"{file_id}.jpg")

        # Loading field-level labels from entities folder
        with open(entity_path, "r", encoding="utf-8") as f:

            entities = json.load(f)
        
        ocr_rows = []
        ocr_texts = []

        # Adding the token level label to each OCR extracted text
        with open(box_path, "r", encoding="latin-1") as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                # Since all features of a OCR output are comma separated in the same line
                # Features are polygon coordinates followed by the OCR text inside it
                features = line.split(",")

                # OCR text itself may contain commas
                # Join everything after first 8 values
                # Since the first 8 values correspond to polygon coordinates and the rest is OCR text
                text = ",".join(features[8:]).strip()

                ocr_rows.append(features)
                ocr_texts.append(text)

        output_lines = []

        for i, features in enumerate(ocr_rows):

            text = ocr_texts[i]

            label = assign_label(i, ocr_texts, entities)

            row = features[:8] + [text, label]

            output_lines.append(row)
            
        # Convert the updated data into .tsv extension
        with open(output_tsv_path, "w", encoding="utf-8") as f:

            for row in output_lines:
                f.write("\t".join(map(str, row)) + "\n")
        
        # Copy corresponding images from the raw dataset
        if os.path.exists(image_path):

            shutil.copy(image_path, output_image_path)

def convert_sroie_dataset():

    create_directory(OUTPUT_TRAIN_TSV_DIR)
    create_directory(OUTPUT_TRAIN_IMAGE_DIR)

    create_directory(OUTPUT_TEST_TSV_DIR)
    create_directory(OUTPUT_TEST_IMAGE_DIR)

    prepare_dataset(TRAIN_BOX_DIR, TRAIN_ENTITY_DIR, TRAIN_IMAGE_DIR, OUTPUT_TRAIN_TSV_DIR, OUTPUT_TRAIN_IMAGE_DIR)
    print("Train Dataset transformation complete")
    
    prepare_dataset(TEST_BOX_DIR, TEST_ENTITY_DIR, TEST_IMAGE_DIR, OUTPUT_TEST_TSV_DIR, OUTPUT_TEST_IMAGE_DIR)
    print("Test Dataset transformation complete")

if __name__ == "__main__":

    convert_sroie_dataset()

    print("SROIE dataset successfully transformed")