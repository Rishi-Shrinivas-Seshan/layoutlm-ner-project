import torch

# function to split the data as per model requirements
def split_data_into_chunks(full_dataset, chunk_size=512):
    new_dataset = []

    for item in full_dataset:
        input_ids = item['input_ids']
        bbox = item['bbox']
        attention_mask = item['attention_mask']
        labels = item['labels'] if 'labels' in item else None
        id_value = item['id']  # Keep the ID the same

        total_len =len(input_ids)

        # Iterate over the sequence and split into num_chunks of chunk_size
        for chunk_idx, i in enumerate(range(0, total_len, chunk_size)):
            
            chunk_input_ids = input_ids[i:i+chunk_size]
            chunk_bbox = bbox[i:i+chunk_size]
            chunk_attention_mask = attention_mask[i:i+chunk_size]
            
            if labels is not None:
                chunk_labels = labels[i:i+chunk_size]

            # Padding
            pad_len = chunk_size - len(chunk_input_ids)

            if pad_len > 0:
                chunk_input_ids = torch.cat([chunk_input_ids, torch.zeros(pad_len, dtype=torch.long)])
                chunk_bbox = torch.cat([chunk_bbox, torch.zeros((pad_len,4), dtype=torch.long)])
                chunk_attention_mask = torch.cat([chunk_attention_mask, torch.zeros(pad_len, dtype=torch.long)])

                if labels is not None:
                    chunk_labels = torch.cat([chunk_labels, torch.full((pad_len,), -100, dtype=torch.long)])

            new_item = {
                'id': id_value,
                'chunk_idx': chunk_idx,
                'input_ids': chunk_input_ids,
                'bbox': chunk_bbox,
                'attention_mask': chunk_attention_mask,
            }
            if labels is not None:
                new_item['labels'] = chunk_labels

            new_dataset.append(new_item)

    return new_dataset