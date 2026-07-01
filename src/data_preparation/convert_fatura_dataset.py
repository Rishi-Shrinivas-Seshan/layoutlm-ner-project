import os
import json
import shutil

from src.configs.fatura_config import id2label

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FATURA_DIR = os.path.join(BASE_DIR, "FATURA_dataset")

HF_ANNOTATION_DIR = os.path.join(FATURA_DIR, "Annotations", "layoutlm_HF_format")

IMAGE_DIR = os.path.join(FATURA_DIR, "images")

OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_BOX_DIR = os.path.join(OUTPUT_DATASET_DIR, "train", "boxes_transcripts_labels")

TRAIN_IMAGE_DIR = os.path.join(OUTPUT_DATASET_DIR, "train", "images")

TEST_BOX_DIR = os.path.join(OUTPUT_DATASET_DIR, "test", "boxes_transcripts")

TEST_IMAGE_DIR = os.path.join(OUTPUT_DATASET_DIR, "test", "images")

# Create the directory without crashes
def create_directory(path):

    os.makedirs(path, exist_ok=True)

# Function to create .tsv files and structure the dataset folders as per pipeline requirements
def convert_json_to_tsv(json_path, output_tsv_path):

    # Loading labels from the annotations folder
    with open(json_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    words = annotation["words"]
    bboxes = annotation["bboxes"]
    ner_tags = annotation["ner_tags"]

    # Check for any discreapencies in the raw dataset
    assert len(words) == len(bboxes), (
        f"Length mismatch in {json_path}"
    )

    # Check for any discreapencies in the raw dataset
    assert len(words) == len(ner_tags), (
        f"Length mismatch in {json_path}"
    )

    # Convert the updated data into .tsv extension
    with open(output_tsv_path, "w", encoding="utf-8") as f:

        for idx in range(len(words)):

            x1, y1, x2, y2 = bboxes[idx]

            word = str(words[idx])

            # Convert the ner_tags into it's corresponding labels
            # Since the preprocess from the pipeline expects labels
            label = id2label[ner_tags[idx]]
            
            row = [ str(x1), str(y1), str(x2), str(y2), word, label]

            f.write("\t".join(row) + "\n")

# Function to copy corresponding images from the raw dataset
def copy_image(file_id, destination_dir):

    image_path = os.path.join(IMAGE_DIR, f"{file_id}.jpg")

    if not os.path.exists(image_path):

        print(f"Image not found for {file_id}")

        return
    
    shutil.copy2(image_path, os.path.join(destination_dir, f"{file_id}.jpg"))

# Function to prepare the dataset as per the pipeline requirements
# FATURA annotations exclude the contents within the table
def create_fatura_dataset():

    create_directory(TRAIN_BOX_DIR)
    create_directory(TRAIN_IMAGE_DIR)

    create_directory(TEST_BOX_DIR)
    create_directory(TEST_IMAGE_DIR)

    train_count = 0
    test_count = 0
    
    for json_file in os.listdir(HF_ANNOTATION_DIR):

        if not json_file.endswith(".json"):
            continue

        json_path = os.path.join(HF_ANNOTATION_DIR, json_file)

        file_id = json_file.split("_hugg_")[0]

        # -------------------------
        # TEST FILES
        # -------------------------
        if json_file.endswith("_hugg_test.json"):

            output_tsv_path = os.path.join(TEST_BOX_DIR, f"{file_id}.tsv")
            
            convert_json_to_tsv(json_path, output_tsv_path)

            copy_image(file_id, TEST_IMAGE_DIR)

            test_count += 1
        # -------------------------
        # TRAIN + DEV FILES
        # -------------------------
        else:

            output_tsv_path = os.path.join(TRAIN_BOX_DIR, f"{file_id}.tsv")

            convert_json_to_tsv(json_path, output_tsv_path)

            copy_image(file_id, TRAIN_IMAGE_DIR)

            train_count += 1

    print(f"Train/Dev files processed: {train_count}")

    print(f"Test files processed: {test_count}")

    print("FATURA dataset successfully transformed")

if __name__=="__main__":

    create_fatura_dataset()