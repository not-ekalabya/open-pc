"""
Test script for the file uploader - demonstrates dependency detection
"""

from file_uploader import FileUploader, FileDependencyAnalyzer
import json


def test_dependency_detection():
    """Test the dependency analyzer with sample code."""

    print("=" * 60)
    print("Testing File Dependency Analyzer")
    print("=" * 60)

    # Sample code with various dependencies
    sample_code = """
import pandas as pd
import numpy as np
from my_utils import helper_function
import torch

# Load data files
df = pd.read_csv('data/dataset.csv')
excel_data = pd.read_excel('data/report.xlsx')
model = torch.load('models/model.pth')

# Open text file
with open('config.txt') as f:
    config = f.read()

# NumPy operations
arr = np.load('arrays/data.npy')

# Custom module usage
result = helper_function(df)
print(result)
"""

    print("\nSample Code:")
    print("-" * 60)
    print(sample_code)
    print("-" * 60)

    # Analyze dependencies
    analyzer = FileDependencyAnalyzer(sample_code, base_dir='.')
    dependencies = analyzer.analyze()

    print(f"\nDetected {len(dependencies)} dependencies:")
    for dep in sorted(dependencies):
        print(f"  - {dep}")

    print("\n" + "=" * 60)
    print("Building File Tree")
    print("=" * 60)

    uploader = FileUploader(base_dir='.')
    payload = uploader.prepare_payload(sample_code, analyze_dependencies=True)

    print(f"\nPayload structure:")
    print(f"  Code size: {len(payload['input']['code'])} bytes")
    print(f"  Files to upload: {len(payload['input']['files'])}")

    # Show file list
    if payload['input']['files']:
        print("\nFiles in payload:")
        for file_info in payload['input']['files']:
            content_size = len(file_info['content'])
            original_size = len(file_info['content']) * 3 // 4  # Approximate original size
            print(f"  - {file_info['path']} (~{original_size} bytes, encoded: {content_size} bytes)")

    return payload


def test_manual_files():
    """Test manual file addition."""

    print("\n" + "=" * 60)
    print("Testing Manual File Addition")
    print("=" * 60)

    code = "import pandas as pd\nprint('Hello')"

    uploader = FileUploader(base_dir='.')
    payload = uploader.prepare_payload(code, analyze_dependencies=False)

    # Add files manually (these patterns won't match anything in the test, but show usage)
    print("\nAdding files with glob patterns...")
    payload = uploader.add_manual_files(payload, [
        '*.py',        # All Python files in current directory
        'data/*.csv',  # All CSV files in data folder
    ])

    print(f"\nTotal files: {len(payload['input']['files'])}")


def test_payload_generation():
    """Generate a complete test payload."""

    print("\n" + "=" * 60)
    print("Generating Complete Test Payload")
    print("=" * 60)

    code = """
import torch
import numpy as np

print('Testing file upload system')
print(f'PyTorch version: {torch.__version__}')
print(f'NumPy version: {np.__version__}')

result = {'status': 'success', 'message': 'All files loaded correctly'}
"""

    uploader = FileUploader(base_dir='.')
    payload = uploader.prepare_payload(code, analyze_dependencies=True)

    # Save to file for inspection
    output_file = 'test_payload.json'
    with open(output_file, 'w') as f:
        # Don't include full base64 content in saved file to keep it readable
        readable_payload = {
            'input': {
                'code': payload['input']['code'],
                'files': [
                    {
                        'path': f['path'],
                        'content': f['content'][:50] + '...' if len(f['content']) > 50 else f['content']
                    }
                    for f in payload['input']['files']
                ]
            }
        }
        json.dump(readable_payload, f, indent=2)

    print(f"\nTest payload saved to: {output_file}")
    print("\nYou can now send this payload to RunPod:")
    print("""
import requests

response = requests.post(
    'https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json=payload
)

print(response.json())
""")


if __name__ == '__main__':
    print("RunPod File Uploader - Test Suite\n")

    # Run tests
    test_dependency_detection()
    test_manual_files()
    test_payload_generation()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Create some test files (e.g., my_utils.py, data/dataset.csv)")
    print("2. Run this test again to see actual file detection")
    print("3. Use example_client.py to send requests to RunPod")
