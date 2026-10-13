def extract(row):
    return {'prompt': row['instruction'] + ': ' + row['input'], 'source': 'parts', 'category': None}
