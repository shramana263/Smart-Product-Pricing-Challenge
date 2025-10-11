import pandas as pd
import re

def parse_catalog_content(catalog_content):
    """Test version of the parsing function"""
    if pd.isna(catalog_content):
        return set()
    
    patterns = {
        'Item Name': r'Item Name(\s+\d+)?:',
        'Product Description': r'Product Description:',
        'Bullet Point': r'Bullet Point\s*\d*:',
        'Value': r'Value:',
        'Unit': r'Unit:'
    }
    
    attributes = set()
    
    for attr_name, pattern in patterns.items():
        if re.search(pattern, catalog_content, re.IGNORECASE):
            attributes.add(attr_name)
    
    return attributes

# Test with a sample that has multiple bullet points
df = pd.read_csv('try2/dataset/train1.csv')
sample = df['catalog_content'].iloc[1]

print("Sample content (first 300 chars):")
print(sample[:300] + "...")
print(f"\nTotal 'Bullet Point' occurrences in text: {sample.count('Bullet Point')}")
print(f"Parsed attributes: {parse_catalog_content(sample)}")
print(f"'Bullet Point' in parsed set: {'Bullet Point' in parse_catalog_content(sample)}")