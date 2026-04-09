import os
import pandas as pd
from PIL import Image
import torch

# Function to process a single document of TRAINING
def process_document(tsv_file, image_file, tokenizer, label2id=None):
    # Read the TSV file into a pandas dataframe
    df = pd.read_csv(tsv_file, header=None)

    # Read the image to get its dimensions (optional for LayoutLM v1)
    image = Image.open(image_file)
    width, height = image.size

    x_scale = 1000 / width
    y_scale = 1000 / height

    # Add a header (column names)
    if len(df.columns) == 8:
        df.columns = ['start_index', 'end_index', 'x1', 'y1', 'x2', 'y2', 'text', 'label']
    else:
        df.columns = ['start_index', 'end_index', 'x1', 'y1', 'x2', 'y2', 'text']
    
    # Drop the 'start_index' and 'end_index' columns
    df = df.drop(columns=['start_index', 'end_index'])

    # Scaling the bounding boxes to fit the image of size 1000x1000
    df[['x1', 'x2']] = df[['x1', 'x2']] * x_scale
    df[['y1', 'y2']] = df[['y1', 'y2']] * y_scale

    # If the scale values were in floating points, the coordinate values will also become floating when multiplied
    # Since LayoutLM expects only integers as the bounding boxes values, round off the figure and cast it as 'int'
    df[['x1', 'x2', 'y1', 'y2']] = df[['x1', 'x2', 'y1', 'y2']].round().astype(int)

    # Clip the values to enfore 0-1000 range
    # Even after standardising, due to noisy data some values may overflow
    df[['x1', 'x2', 'y1', 'y2']] = df[['x1', 'x2', 'y1', 'y2']].clip(0, 1000)

    # Convert the 'text' column to string type for tokenizer to work without error
    df['text'] = df['text'].astype(str)

    # Drop duplicates based on bounding boxes columns
    df = df.drop_duplicates(subset=['x1', 'y1', 'x2', 'y2', 'text'], keep='first')

    words = df['text'].tolist()
    bboxes = df[['x1', 'y1', 'x2', 'y2']].values.tolist()
    
    if label2id is not None and 'label' in df.columns:
        raw_labels = df['label'].tolist()
        # Converting the labels into label_ids
        labels = [label2id[label] for label in raw_labels]
    else:
        labels = None

    tokenized_words = []
    token_bboxes = []
    token_labels = [] if labels is not None else None

    # Process each word in the document
    for i, (word, bbox) in enumerate(zip(words, bboxes)):
        word = str(word)
        # Tokenize the word
        tokenized_word = tokenizer.tokenize(word)
        tokenized_words.extend(tokenized_word)

        # Add the same bounding box for each subword
        token_bboxes.extend([bbox] * len(tokenized_word))

        # Add the same label for each subword if present
        if labels is not None:
            label = labels[i]
            token_labels.extend([label] * len(tokenized_word))

    # Convert tokenized words to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_words)

    # Create attention masks (1 for tokens, 0 for padding, if any)
    attention_mask = [1] * len(input_ids)

    # Format the document into the required structure
    document = {
        "id": os.path.basename(tsv_file).split('.')[0],  # Use the file name as the document ID
        "input_ids": torch.tensor(input_ids),
        "bbox": torch.tensor(token_bboxes),
        "attention_mask": torch.tensor(attention_mask)
    }

    if token_labels is not None:
        document["labels"] = torch.tensor(token_labels)

    return document

