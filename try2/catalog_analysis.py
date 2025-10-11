import pandas as pd
import re
from itertools import combinations
from collections import Counter, defaultdict

def parse_catalog_content(catalog_content):
    """
    Parse the catalog_content string to extract unique attributes present
    """
    if pd.isna(catalog_content):
        return set()
    
    # Define attribute patterns
    patterns = {
        'Item Name': r'Item Name(\s+\d+)?:',
        'Product Description': r'Product Description:',
        'Bullet Point': r'Bullet Point\s*\d*:',
        'Value': r'Value:',
        'Unit': r'Unit:'
    }
    
    attributes = set()
    
    # Check for each pattern in the catalog content
    for attr_name, pattern in patterns.items():
        if re.search(pattern, catalog_content, re.IGNORECASE):
            attributes.add(attr_name)
    
    return attributes

def analyze_catalog_attributes(train1_path, train2_path):
    """
    Analyze catalog attributes and their combinations
    """
    print("Loading datasets...")
    
    # Load both training datasets
    df1 = pd.read_csv(train1_path)
    df2 = pd.read_csv(train2_path)
    
    # Combine the datasets
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    print(f"Combined dataset shape: {df_combined.shape}")
    print(f"Total rows: {len(df_combined)}")
    
    # Extract attributes for each row
    print("Extracting attributes from catalog content...")
    attribute_sets = []
    
    for idx, row in df_combined.iterrows():
        if idx % 50000 == 0:
            print(f"Processed {idx} rows...")
        
        catalog_content = row['catalog_content']
        attributes = parse_catalog_content(catalog_content)
        attribute_sets.append(attributes)
    
    print("Calculating individual attribute frequencies...")
    
    # Count individual attributes
    individual_counts = Counter()
    for attr_set in attribute_sets:
        for attr in attr_set:
            individual_counts[attr] += 1
    
    print("Calculating combination frequencies...")
    
    # Count combinations
    combination_counts = Counter()
    
    # Generate all possible combinations (2 to max length)
    for attr_set in attribute_sets:
        if len(attr_set) > 1:
            # Generate combinations of different sizes
            for r in range(2, len(attr_set) + 1):
                for combo in combinations(sorted(attr_set), r):
                    combo_str = " and ".join(combo)
                    combination_counts[combo_str] += 1
    
    print("Creating results dataframe...")
    
    # Create results dataframe
    results = []
    
    # Add individual attributes
    for attr, count in individual_counts.items():
        results.append({
            'attribute': attr,
            'count': count,
            'type': 'individual'
        })
    
    # Add combinations
    for combo, count in combination_counts.items():
        results.append({
            'attribute': combo,
            'count': count,
            'type': 'combination'
        })
    
    # Convert to DataFrame and sort by count
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('count', ascending=False).reset_index(drop=True)
    
    return results_df, df_combined

def main():
    # File paths
    train1_path = r"c:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing\try2\dataset\train1.csv"
    train2_path = r"c:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing\try2\dataset\train2.csv"
    
    # Analyze the data
    results_df, combined_df = analyze_catalog_attributes(train1_path, train2_path)
    
    # Display results
    print("\n" + "="*80)
    print("CATALOG ATTRIBUTE FREQUENCY ANALYSIS")
    print("="*80)
    
    print(f"\nTotal unique attributes found: {len(results_df[results_df['type'] == 'individual'])}")
    print(f"Total combinations found: {len(results_df[results_df['type'] == 'combination'])}")
    
    print("\nINDIVIDUAL ATTRIBUTE FREQUENCIES:")
    print("-" * 50)
    individual_results = results_df[results_df['type'] == 'individual'].copy()
    for _, row in individual_results.iterrows():
        print(f"{row['attribute']:<25}: {row['count']:>10,}")
    
    print("\nTOP 20 COMBINATION FREQUENCIES:")
    print("-" * 70)
    combination_results = results_df[results_df['type'] == 'combination'].head(20)
    for _, row in combination_results.iterrows():
        print(f"{row['attribute']:<50}: {row['count']:>10,}")
    
    # Save results to CSV
    output_path = r"c:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing\try2\catalog_attribute_analysis.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Create separate files for individual and combinations
    individual_path = r"c:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing\try2\individual_attributes.csv"
    combination_path = r"c:\Users\User\Desktop\hackathons\Amazon-Ml-Challange2025\Smart-Product-Pricing\try2\combination_attributes.csv"
    
    individual_results.drop('type', axis=1).to_csv(individual_path, index=False)
    results_df[results_df['type'] == 'combination'].drop('type', axis=1).to_csv(combination_path, index=False)
    
    print(f"Individual attributes saved to: {individual_path}")
    print(f"Combination attributes saved to: {combination_path}")
    
    return results_df

if __name__ == "__main__":
    results = main()