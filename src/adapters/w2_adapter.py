import pandas as pd

# Transform the dataset into canonical schema of the pipeline
def load_w2_document(tsv_file):

    df = pd.read_csv(tsv_file, header=None)

    # Add a header (column names)
    if len(df.columns) == 8:
        df.columns = ['start_index', 'end_index', 'x1', 'y1', 'x2', 'y2', 'text', 'label']
    else:
        df.columns = ['start_index', 'end_index', 'x1', 'y1', 'x2', 'y2', 'text']
    
    # Convert into canonical schema
    canonical_columns = ['x1', 'y1', 'x2', 'y2', 'text']

    if 'label' in df.columns:
        canonical_columns.append('label')

    df = df[canonical_columns]

    return df