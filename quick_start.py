"""
Quick Start - Minimal example to get you started with RunPod Cloud PC file uploads
"""

from file_uploader import FileUploader
import json


def quick_example():
    """
    Simplest possible example - analyze code and see what files would be uploaded.
    """

    # Your Python code
    code = """
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('sales_data.csv')

# Analyze
total_sales = df['amount'].sum()
avg_sales = df['amount'].mean()

print(f'Total Sales: ${total_sales:,.2f}')
print(f'Average Sale: ${avg_sales:,.2f}')

# Return result
result = {
    'total': float(total_sales),
    'average': float(avg_sales),
    'count': len(df)
}
"""

    print("Code to execute:")
    print("-" * 60)
    print(code)
    print("-" * 60)

    # Create uploader
    uploader = FileUploader(base_dir='.')

    # Prepare payload - this will automatically detect 'sales_data.csv'
    print("\nAnalyzing dependencies...")
    payload = uploader.prepare_payload(code, analyze_dependencies=True)

    # If you need to add more files manually:
    # payload = uploader.add_manual_files(payload, ['data/*.csv', 'models/*.pth'])

    # Show what will be sent
    print(f"\nReady to send:")
    print(f"  - Code: {len(code)} bytes")
    print(f"  - Files: {len(payload['input']['files'])}")

    # Save payload for inspection
    with open('quick_start_payload.json', 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nPayload saved to: quick_start_payload.json")

    # To actually send to RunPod, uncomment and configure:
    """
    import requests

    RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
    RUNPOD_API_KEY = "YOUR_API_KEY"

    response = requests.post(
        RUNPOD_ENDPOINT,
        headers={
            'Authorization': f'Bearer {RUNPOD_API_KEY}',
            'Content-Type': 'application/json'
        },
        json=payload
    )

    result = response.json()
    print("\nResult:")
    print(result['output']['stdout'])
    print(f"\nReturned value: {result['output']['result']}")
    """

    return payload


if __name__ == '__main__':
    print("=" * 60)
    print("RunPod Cloud PC - Quick Start")
    print("=" * 60)
    print()

    payload = quick_example()

    print("\n" + "=" * 60)
    print("What happens next:")
    print("=" * 60)
    print("""
1. The code is sent to RunPod along with any detected files
2. RunPod creates a temporary directory
3. All files are saved in that directory
4. Your code runs with access to those files
5. Results (stdout, stderr, result) are sent back
6. The temporary directory is cleaned up

To send this to RunPod:
1. Update RUNPOD_ENDPOINT and RUNPOD_API_KEY in the code above
2. Uncomment the requests section
3. Run this script
""")
