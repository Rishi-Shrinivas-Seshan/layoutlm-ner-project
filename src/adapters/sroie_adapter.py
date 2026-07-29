import pandas as pd
import csv

# Convert the coordinates as per pipeline requirement
# SROIE contains coordinates of all the four vertices
# This pipeline needs just top right and bottom left coordinates

def load_sroie_document(tsv_file, include_labels=True):

    df = pd.read_csv(tsv_file, sep="\t", header=None, quoting=csv.QUOTE_NONE,keep_default_na=False)

    # SROIE TSV always contains:
    # (x1,y1) (x2,y2) (x3,y3) (x4,y4) OCR-text Label
    df.columns = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4", "text", "label"]

    # Convert polygon -> bounding box
    df = df[["x1", "y1", "x3", "y3", "text", "label"]]

    # Training data TSV [rename (x3,y3) as (x2,y2)]
    df.columns = ["x1", "y1", "x2", "y2", "text", "label"]
    
    if not include_labels:

        # Testing data TSV
        df = df.drop(columns=["label"])

    return df