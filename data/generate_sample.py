"""
Run this once to generate the sample dataset:
    python data/generate_sample.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import generate_sample_dataset

if __name__ == "__main__":
    df = generate_sample_dataset(n=1000)
    output_path = os.path.join(os.path.dirname(__file__), "hr_sample.csv")
    df.to_csv(output_path, index=False)
    print(f"Sample dataset saved to {output_path}")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Treatment rate: {df['treatment'].mean():.2%}")
