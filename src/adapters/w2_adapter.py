import pandas as pd
import csv

# Transform the dataset into canonical schema of the pipeline
def load_w2_document(tsv_file, include_labels=True):

    df = pd.read_csv(tsv_file, header=None, keep_default_na=False)

    # Add a header (column names)
    if len(df.columns) == 8:
        df.columns = ['start_index', 'end_index', 'x1', 'y1', 'x2', 'y2', 'text', 'label']
    else:
        df.columns = ['start_index', 'end_index', 'x1', 'y1', 'x2', 'y2', 'text']
    
    # Convert into canonical schema
    canonical_columns = ['x1', 'y1', 'x2', 'y2', 'text']

    # Check if labels have to be added
    if include_labels:

        # Check if the original dataset files have labels provided to add for train data
        if 'label' not in df.columns:
            raise ValueError("No labels found for training")
        
        # Append labels as a column for training
        canonical_columns.append('label')

    df = df[canonical_columns]

    return df