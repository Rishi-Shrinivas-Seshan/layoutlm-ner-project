import csv
import pandas as pd

def load_funsd_document(tsv_file, include_labels=True):

    # Preserve OCR strings such as "N/A" by disabling Pandas' default NA parsing (i.e keep_default_na=False)
    df = pd.read_csv(tsv_file, sep="\t", header=None, quoting=csv.QUOTE_NONE, keep_default_na=False)

    # Converted TSV schema
    df.columns = [ "x1", "y1", "x2", "y2", "text", "entity_label", "entity_id", "entity_links"]

    # Handling missing values from dataset
    # Replacing None with blank "" which will be later removed
    df["text"] = df["text"].fillna("")

    # Convert the text column into string and remove whitespaces
    df["text"] = df["text"].astype(str)
    df["text"] = df["text"].str.strip()

    # Remove the blank strings like ""
    df = df[df["text"] != ""]

    # The current pipeline does token classfication
    # Metadata related to relation extraction is removed
    df = df.drop(columns=["entity_id", "entity_links"])

    # Rename 'entity_label' to 'label' as per pipeline's canonical schema
    df = df.rename(columns={"entity_label": "label"})

    # Converted dataset contains labels
    # The pipeline removes them for testing data
    if not include_labels:

        # Testing data TSV
        df = df.drop(columns=["label"])

    return df