"""
Example client for sending code and files to RunPod Cloud PC

This script demonstrates how to use the FileUploader to automatically detect
dependencies and send them along with your code to RunPod.
"""

import requests
import json
from file_uploader import FileUploader


# Your RunPod API endpoint
RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
RUNPOD_API_KEY = "YOUR_API_KEY"


def execute_on_runpod(code: str, base_dir: str = '.', manual_files: list = None):
    """
    Execute code on RunPod with automatic file dependency detection.

    Args:
        code: Python code to execute
        base_dir: Base directory for resolving file paths
        manual_files: Optional list of glob patterns for additional files

    Returns:
        Response from RunPod
    """
    # Create uploader and prepare payload
    uploader = FileUploader(base_dir=base_dir)

    print("=" * 60)
    print("Preparing code execution on RunPod Cloud PC")
    print("=" * 60)

    # Prepare payload with automatic dependency detection
    payload = uploader.prepare_payload(code, analyze_dependencies=True)

    # Add manual files if specified
    if manual_files:
        print("\n" + "=" * 60)
        print("Adding manual files")
        print("=" * 60)
        payload = uploader.add_manual_files(payload, manual_files)

    print("\n" + "=" * 60)
    print("Sending request to RunPod")
    print("=" * 60)
    print(f"Code size: {len(code)} bytes")
    print(f"Files to upload: {len(payload['input']['files'])}")

    # Send request to RunPod
    headers = {
        'Authorization': f'Bearer {RUNPOD_API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(RUNPOD_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()

        print("\n" + "=" * 60)
        print("Execution Results")
        print("=" * 60)

        if 'output' in result:
            output = result['output']

            if output.get('stdout'):
                print("\n--- STDOUT ---")
                print(output['stdout'])

            if output.get('stderr'):
                print("\n--- STDERR ---")
                print(output['stderr'])

            if output.get('result') is not None:
                print("\n--- RESULT ---")
                print(output['result'])

            if output.get('error'):
                print("\n--- ERROR ---")
                print(output['error'])

        return result

    except requests.exceptions.RequestException as e:
        print(f"\nError sending request: {e}")
        return None


# Example 1: Simple code with data file
def example_1():
    """Example with pandas CSV file."""
    code = """
import pandas as pd

# Load and process data
df = pd.read_csv('data.csv')
print(f"Loaded {len(df)} rows")
print(df.head())

result = df.describe().to_dict()
"""

    execute_on_runpod(
        code=code,
        manual_files=['data.csv']  # Manually add CSV file
    )


# Example 2: Code with local module imports
def example_2():
    """Example with local module imports."""
    code = """
from utils import process_data
import pandas as pd

df = pd.read_csv('dataset.csv')
result = process_data(df)
print(f"Processed result: {result}")
"""

    # The analyzer will automatically detect utils.py
    execute_on_runpod(
        code=code,
        manual_files=['dataset.csv']
    )


# Example 3: PyTorch model with weights
def example_3():
    """Example with PyTorch model."""
    code = """
import torch
import torch.nn as nn

# Load model
model = torch.load('model.pth')
print(f"Model loaded: {model}")

# Run inference
with torch.no_grad():
    output = model(torch.randn(1, 3, 224, 224))
    print(f"Output shape: {output.shape}")

result = output.tolist()
"""

    execute_on_runpod(
        code=code,
        manual_files=['model.pth']
    )


# Example 4: Multiple files and directories
def example_4():
    """Example with multiple data files."""
    code = """
import pandas as pd
import os

# List all CSV files
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
print(f"Found {len(csv_files)} CSV files: {csv_files}")

# Load and concatenate
dfs = [pd.read_csv(f) for f in csv_files]
combined = pd.concat(dfs, ignore_index=True)

print(f"Total rows: {len(combined)}")
result = len(combined)
"""

    execute_on_runpod(
        code=code,
        manual_files=['data/*.csv']  # Upload all CSVs in data folder
    )


if __name__ == '__main__':
    print("RunPod Cloud PC - Example Client")
    print("\nBefore running, please update:")
    print("1. RUNPOD_ENDPOINT with your endpoint URL")
    print("2. RUNPOD_API_KEY with your API key")
    print("\nUncomment an example to run it:")
    print("  - example_1(): Simple CSV processing")
    print("  - example_2(): Local module imports")
    print("  - example_3(): PyTorch model loading")
    print("  - example_4(): Multiple data files")

    # Uncomment to run an example:
    # example_1()
