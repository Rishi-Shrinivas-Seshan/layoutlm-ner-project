entity_labels = [
    'SELLER_SITE',      # 0
    'TOTAL',            # 1
    'TOTAL_WORDS',      # 2
    'DATE',             # 3
    'DUE_DATE',         # 4
    'BUYER',            # 5
    'SELLER_NAME',      # 6
    'UNUSED_7',         # 7
    'BILL_TO',          # 8
    'SHIP_TO',          # 9
    'TABLE',            # 10
    'LOGO',             # 11
    'INVOICE_NUMBER',   # 12
    'OTHER',            # 13
]

background_label = 'OTHER'

label2id = {label: idx for idx, label in enumerate(entity_labels)}
id2label = {idx: label for label, idx in label2id.items()}