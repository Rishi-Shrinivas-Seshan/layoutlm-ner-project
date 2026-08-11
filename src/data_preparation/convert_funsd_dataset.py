import os
import json
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FUNSD_DIR = os.path.join(BASE_DIR, "FUNSD_dataset")

OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Create the directory without crashes
def create_directory(path):

    os.makedirs(path, exist_ok=True)

# Function to create .tsv files and structure the dataset folders as per pipeline requirements
def prepare_dataset(annotation_dir, image_dir, output_tsv_dir, output_image_dir):

    create_directory(output_tsv_dir)
    create_directory(output_image_dir)

    for json_file in os.listdir(annotation_dir):

        if not json_file.endswith(".json"):
            continue

        file_id = json_file.replace(".json", "")

        annotation_path = os.path.join(annotation_dir, json_file)

        image_path = os.path.join(image_dir, f"{file_id}.png")

        output_tsv_path = os.path.join(output_tsv_dir, f"{file_id}.tsv")

        # Loading labels from the annotations folder
        with open(annotation_path, "r", encoding="utf-8") as f:

            annotation = json.load(f)

        rows = []

        # Since original FUNSD dataset is structured as a form containing entities
        for entity in annotation["form"]:

            entity_label = entity["label"]
            entity_id = entity["id"]
            entity_links = entity["linking"]

            # Multiple OCR words are grouped into a single semantic entity
            # Preserve every word and propagate labels, links & id to each word
            for word in entity["words"]:

                # Individual word from a collection of meaningful words
                text = word.get("text")

                x1, y1, x2, y2 = word["box"]

                rows.append([x1, y1, x2, y2, text, entity_label, entity_id, entity_links])

        # Convert the data file into .tsv extension
        with open(output_tsv_path, "w", encoding="utf-8") as f:

            for row in rows:

                f.write("\t".join(map(str, row)) + "\n")

        # Copy corresponding images from the raw dataset
        if os.path.exists(image_path):

            shutil.copy(image_path, os.path.join(output_image_dir, f"{file_id}.png"))

if __name__ == "__main__":

    raw_train_annotations = os.path.join(FUNSD_DIR, "training_data", "annotations")
    raw_train_images = os.path.join(FUNSD_DIR, "training_data", 'images')
    converted_train_tsv_files = os.path.join(OUTPUT_DATASET_DIR, "train", "boxes_transcripts_labels")
    converted_train_images = os.path.join(OUTPUT_DATASET_DIR, "train", "images")

    prepare_dataset(raw_train_annotations, raw_train_images, converted_train_tsv_files, converted_train_images)

    raw_test_annotations = os.path.join(FUNSD_DIR, "testing_data", "annotations")
    raw_test_images = os.path.join(FUNSD_DIR, "testing_data", "images")
    converted_test_tsv_files = os.path.join(OUTPUT_DATASET_DIR, "test", "boxes_transcripts")
    converted_test_images = os.path.join(OUTPUT_DATASET_DIR, "test", "images")

    prepare_dataset(raw_test_annotations, raw_test_images, converted_test_tsv_files, converted_test_images)

    print("FUNSD dataset successfully transformed")