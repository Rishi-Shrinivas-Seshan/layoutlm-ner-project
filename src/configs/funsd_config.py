entity_labels = [
    "other",
    "question",
    "answer",
    "header"
]

background_label = entity_labels[0]

label2id = {label: idx for idx, label in enumerate(entity_labels)}
id2label = {idx: label for label, idx in label2id.items()}