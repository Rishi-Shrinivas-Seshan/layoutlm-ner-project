from torch.utils.data import Dataset

# Define your dataset
class LayoutLMDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        output =  {
            'input_ids': item['input_ids'],
            'bbox': item['bbox'],
            'attention_mask': item['attention_mask'],
        }

        if 'labels' in item:
            output['labels'] = item['labels']

        return output
        