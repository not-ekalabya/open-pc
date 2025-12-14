"""
Function Executor for RunPod Cloud PC

This module provides an easy way to execute functions (including void functions)
on RunPod with automatic file dependency detection and full logging.
"""

import requests
import json
from file_uploader import FileUploader


class FunctionExecutor:
    """Execute Python functions on RunPod Cloud PC."""

    def __init__(self, endpoint_url: str, api_key: str, base_dir: str = '.'):
        """
        Initialize the function executor.

        Args:
            endpoint_url: RunPod endpoint URL
            api_key: RunPod API key
            base_dir: Base directory for resolving file paths
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.uploader = FileUploader(base_dir=base_dir)

    def execute_function(self, code: str, function_name: str = None,
                        function_args: list = None, function_kwargs: dict = None,
                        files: list = None, analyze_dependencies: bool = True):
        """
        Execute a function on RunPod.

        Args:
            code: Python code containing the function definition
            function_name: Name of the function to call (optional)
            function_args: Positional arguments for the function
            function_kwargs: Keyword arguments for the function
            files: Manual file patterns to upload
            analyze_dependencies: Whether to analyze and upload dependencies

        Returns:
            dict: Execution results with stdout, stderr, and return value
        """
        # Prepare payload
        payload = self.uploader.prepare_payload(code, analyze_dependencies=analyze_dependencies)

        # Add function execution parameters
        if function_name:
            payload['input']['function_name'] = function_name
            payload['input']['function_args'] = function_args or []
            payload['input']['function_kwargs'] = function_kwargs or {}

        # Add manual files if specified
        if files:
            payload = self.uploader.add_manual_files(payload, files)

        # Send request
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        print("=" * 60)
        print("Executing function on RunPod Cloud PC")
        print("=" * 60)
        print(f"Code size: {len(code)} bytes")
        print(f"Files to upload: {len(payload['input']['files'])}")
        if function_name:
            print(f"Function to call: {function_name}()")
        print()

        try:
            response = requests.post(self.endpoint_url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()

            # Display results
            self._display_results(result)

            return result

        except requests.exceptions.RequestException as e:
            print(f"\nError sending request: {e}")
            return None

    def _display_results(self, result: dict):
        """Display execution results in a formatted way."""
        print("=" * 60)
        print("Execution Results")
        print("=" * 60)

        if 'output' in result:
            output = result['output']

            if output.get('stdout'):
                print("\n--- STDOUT (All terminal logs) ---")
                print(output['stdout'])

            if output.get('stderr'):
                print("\n--- STDERR ---")
                print(output['stderr'])

            if output.get('result') is not None:
                print("\n--- RETURN VALUE ---")
                print(f"Type: {type(output['result']).__name__}")
                print(f"Value: {output['result']}")
            elif output.get('result') is None and not output.get('error'):
                print("\n--- RETURN VALUE ---")
                print("None (void function - no return value)")

            if output.get('error'):
                print("\n--- ERROR ---")
                print(output['error'])
        else:
            print(f"\nRaw result: {result}")


# Example usage
if __name__ == '__main__':
    # Configuration
    RUNPOD_ENDPOINT = "https://api.runpod.ai/v2/yv68hhady2hgii/runsync"
    RUNPOD_API_KEY = "API_KEY_HERE"

    # Create executor
    executor = FunctionExecutor(
        endpoint_url=RUNPOD_ENDPOINT,
        api_key=RUNPOD_API_KEY,
        base_dir='.'
    )

    # Example 1: Void function (no return value, just logs)
    print("\n" + "=" * 60)
    print("Example 1: Void Function with Logging")
    print("=" * 60)

    code_void = """
import pandas as pd

def process_data():
    '''Void function that processes data and logs everything'''
    print("Loading data...")
    df = pd.read_csv('data/data.csv')

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    print("\\nCalculating statistics...")
    print(f"Mean: {df.select_dtypes(include='number').mean().to_dict()}")

    print("\\nProcessing complete!")
    # No return statement - this is a void function
"""

    executor.execute_function(
        code=code_void,
        function_name='process_data',
        files=['data/*.csv']
    )

    # Example 2: Function with return value
    print("\n" + "=" * 60)
    print("Example 2: Function with Return Value")
    print("=" * 60)

    code_return = """
import pandas as pd

def analyze_sales(min_amount=0):
    '''Function that returns analysis results'''
    print("Starting sales analysis...")

    df = pd.read_csv('data/data.csv')
    print(f"Loaded {len(df)} records")

    # Filter by minimum amount
    filtered = df[df['amount'] >= min_amount]
    print(f"Filtered to {len(filtered)} records with amount >= {min_amount}")

    # Calculate statistics
    total = filtered['amount'].sum()
    average = filtered['amount'].mean()
    count = len(filtered)

    print(f"\\nResults:")
    print(f"  Total: ${total:,.2f}")
    print(f"  Average: ${average:,.2f}")
    print(f"  Count: {count}")

    # Return the results
    return {
        'total': float(total),
        'average': float(average),
        'count': count,
        'min_amount_filter': min_amount
    }
"""

    executor.execute_function(
        code=code_return,
        function_name='analyze_sales',
        function_kwargs={'min_amount': 100},
        files=['data/*.csv']
    )

    # Example 3: Function with multiple arguments
    print("\n" + "=" * 60)
    print("Example 3: Function with Multiple Arguments")
    print("=" * 60)

    code_args = """
def calculate_metrics(data_list, multiplier=1.0, verbose=True):
    '''Function with positional and keyword arguments'''
    if verbose:
        print(f"Calculating metrics for {len(data_list)} items")
        print(f"Using multiplier: {multiplier}")

    total = sum(data_list) * multiplier
    avg = total / len(data_list)

    if verbose:
        print(f"\\nResults:")
        print(f"  Total: {total}")
        print(f"  Average: {avg}")

    return {'total': total, 'average': avg}
"""

    executor.execute_function(
        code=code_args,
        function_name='calculate_metrics',
        function_args=[[10, 20, 30, 40, 50]],
        function_kwargs={'multiplier': 2.0, 'verbose': True},
        analyze_dependencies=False
    )

    # Example 4: Function with file operations
    print("\n" + "=" * 60)
    print("Example 4: Function with File Operations")
    print("=" * 60)

    code_files = """
import torch
import numpy as np

def load_and_process_model(model_path):
    '''Function that loads files and processes them'''
    print(f"Loading model from {model_path}...")

    try:
        model = torch.load(model_path)
        print(f"Model loaded successfully")
        print(f"Model type: {type(model)}")

        # Process model
        print("\\nProcessing model...")
        # Your processing logic here

        return {'status': 'success', 'model_type': str(type(model))}
    except Exception as e:
        print(f"Error loading model: {e}")
        return {'status': 'error', 'message': str(e)}
"""

    executor.execute_function(
        code=code_files,
        function_name='load_and_process_model',
        function_args=['model.pth'],
        files=['models/*.pth']
    )
