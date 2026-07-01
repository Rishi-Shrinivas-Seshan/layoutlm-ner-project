import csv
import pandas as pd

def load_fatura_document(tsv_file, include_labels=True):

    df = pd.read_csv(tsv_file, sep="\t", header=None, quoting=csv.QUOTE_NONE, keep_default_na=False)

    # Training data TSV
    df.columns = [ "x1", "y1", "x2", "y2", "text", "label"]

    if not include_labels:

        # Testing data TSV
        df = df.drop(columns=["label"])

    return df