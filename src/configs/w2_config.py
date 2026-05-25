entity_labels = [
    'OTHER',
    'employerName',
    'employerAddressStreet_name',
    'employerAddressCity',
    'employerAddressState',
    'employerAddressZip',
    'einEmployerIdentificationNumber',
    'employeeName',
    'ssnOfEmployee',
    'box1WagesTipsAndOtherCompensations',
    'box2FederalIncomeTaxWithheld',
    'box3SocialSecurityWages',
    'box4SocialSecurityTaxWithheld',
    'box16StateWagesTips',
    'box17StateIncomeTax',
    'taxYear',
]

label2id = {label: idx for idx, label in enumerate(entity_labels)}
id2label = {idx: label for label, idx in label2id.items()}