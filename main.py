import pandas as pd
import os

# Define source and destination directories
source_dir = 'try1/Material/student_resource/dataset'
dest_dir = 'try2/dataset'

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Chunk size
chunk_size = 37500

# Process train.csv
# print("Processing train.csv...")
# df_train = pd.read_csv(os.path.join(source_dir, 'train.csv'))
# num_chunks_train = (len(df_train) + chunk_size - 1) // chunk_size  # Ceiling division
# for i in range(num_chunks_train):
#     start_idx = i * chunk_size
#     end_idx = min((i + 1) * chunk_size, len(df_train))
#     df_chunk = df_train.iloc[start_idx:end_idx]
#     output_file = os.path.join(dest_dir, f'train{i+1}.csv')
#     df_chunk.to_csv(output_file, index=False)
#     print(f"Created {output_file} with {len(df_chunk)} rows")

# # Process test.csv
# print("Processing test.csv...")
# df_test = pd.read_csv(os.path.join(source_dir, 'test.csv'))
# num_chunks_test = (len(df_test) + chunk_size - 1) // chunk_size  # Ceiling division
# for i in range(num_chunks_test):
#     start_idx = i * chunk_size
#     end_idx = min((i + 1) * chunk_size, len(df_test))
#     df_chunk = df_test.iloc[start_idx:end_idx]
#     output_file = os.path.join(dest_dir, f'test{i+1}.csv')
#     df_chunk.to_csv(output_file, index=False)
#     print(f"Created {output_file} with {len(df_chunk)} rows")

# Sample 50 random rows from train.csv
print("Creating sample train dataset...")
df_train_full = pd.read_csv(os.path.join(dest_dir, 'train1.csv'))
sample_train = df_train_full.sample(n=50, random_state=42)  # random_state for reproducibility
sample_train.to_csv(os.path.join(dest_dir, 'sample_train.csv'), index=False)
print("Created sample_train.csv with 50 random rows")

print("Dataset processing completed!")