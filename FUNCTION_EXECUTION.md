# Function Execution on RunPod Cloud PC

## Overview

The RunPod Cloud PC now supports executing Python functions (including void functions) with full terminal logging and return value capture. This feature allows you to:

- ✅ Execute void functions (functions that don't return anything)
- ✅ Execute functions with return values
- ✅ Capture ALL terminal logs (print statements)
- ✅ Pass arguments to functions (positional and keyword)
- ✅ Upload files and dependencies automatically
- ✅ Get structured responses with stdout, stderr, and return values

## What's New

### Previous Behavior
Previously, you could only execute code directly:
```python
payload = {
    'input': {
        'code': 'print("Hello")\nresult = 42'
    }
}
```

### New Behavior
Now you can define and call functions:
```python
payload = {
    'input': {
        'code': 'def my_func():\n    print("Hello")\n    return 42',
        'function_name': 'my_func',
        'function_args': [],
        'function_kwargs': {}
    }
}
```

## Why Use Function Execution?

### 1. **Better Code Organization**
Functions make your code modular and reusable:
```python
def process_data(filename):
    df = pd.read_csv(filename)
    # ... processing logic
    return results
```

### 2. **Void Function Support**
Execute functions that only log output without returning values:
```python
def log_system_info():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    # No return statement - all output is captured in stdout
```

### 3. **Full Terminal Logging**
ALL print statements and logs are captured and returned:
```python
def analyze():
    print("Step 1: Loading data...")  # ✅ Captured
    print("Step 2: Processing...")    # ✅ Captured
    print("Step 3: Complete!")        # ✅ Captured
    return results
```

### 4. **Argument Passing**
Pass data to your functions:
```python
def calculate(values, multiplier=1.0):
    total = sum(values) * multiplier
    print(f"Total: {total}")
    return total
```

## Usage

### Method 1: Using FunctionExecutor Class (Recommended)

```python
from function_executor import FunctionExecutor

# Initialize
executor = FunctionExecutor(
    endpoint_url="https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync",
    api_key="YOUR_API_KEY",
    base_dir='.'
)

# Define your function
code = """
import pandas as pd

def process_sales_data(min_amount=0):
    print("Loading sales data...")
    df = pd.read_csv('sales.csv')

    print(f"Total records: {len(df)}")

    filtered = df[df['amount'] >= min_amount]
    print(f"Records above ${min_amount}: {len(filtered)}")

    total = filtered['amount'].sum()
    print(f"Total sales: ${total:,.2f}")

    return {
        'total_sales': float(total),
        'record_count': len(filtered)
    }
"""

# Execute the function
result = executor.execute_function(
    code=code,
    function_name='process_sales_data',
    function_kwargs={'min_amount': 100},
    files=['sales.csv']
)

# Access results
print(result['output']['stdout'])     # All print statements
print(result['output']['result'])     # Return value
```

### Method 2: Direct API Call

```python
import requests
import json

payload = {
    'input': {
        'code': '''
def hello(name):
    print(f"Hello, {name}!")
    print("Welcome to RunPod Cloud PC")
    return f"Greeted {name}"
''',
        'function_name': 'hello',
        'function_args': ['Alice'],
        'function_kwargs': {}
    }
}

response = requests.post(
    'https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json=payload
)

result = response.json()
```

## Response Format

All responses follow this structure:

```json
{
  "output": {
    "stdout": "All print statements and terminal logs",
    "stderr": "Error messages if any",
    "result": "Return value from the function (or null for void functions)",
    "error": "Error message if execution failed (or null)"
  }
}
```

### Example Responses

#### Void Function (No Return Value)
```json
{
  "output": {
    "stdout": "Processing data...\nStep 1 complete\nStep 2 complete\nDone!\n",
    "stderr": "",
    "result": null,
    "error": null
  }
}
```

#### Function with Return Value
```json
{
  "output": {
    "stdout": "Calculating...\nTotal: 150.00\n",
    "stderr": "",
    "result": {
      "total": 150.0,
      "count": 10
    },
    "error": null
  }
}
```

## Complete Examples

### Example 1: Void Function with Data Processing

```python
code = """
import pandas as pd
import numpy as np

def analyze_dataset():
    '''Void function that analyzes data and logs everything'''

    print("=" * 60)
    print("Data Analysis Report")
    print("=" * 60)

    # Load data
    print("\\n1. Loading dataset...")
    df = pd.read_csv('data.csv')
    print(f"   ✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Basic statistics
    print("\\n2. Basic Statistics:")
    print(f"   - Shape: {df.shape}")
    print(f"   - Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Numerical analysis
    print("\\n3. Numerical Columns:")
    for col in df.select_dtypes(include=np.number).columns:
        print(f"   - {col}:")
        print(f"     Mean: {df[col].mean():.2f}")
        print(f"     Std:  {df[col].std():.2f}")
        print(f"     Min:  {df[col].min():.2f}")
        print(f"     Max:  {df[col].max():.2f}")

    print("\\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

    # No return statement - this is a void function
"""

executor.execute_function(
    code=code,
    function_name='analyze_dataset',
    files=['data.csv']
)

# Output:
# ============================================================
# Data Analysis Report
# ============================================================
#
# 1. Loading dataset...
#    ✓ Loaded 1000 rows, 5 columns
# ...
```

### Example 2: Function with Arguments and Return Value

```python
code = """
import pandas as pd

def filter_and_aggregate(csv_file, column, min_value, operation='sum'):
    '''Filter data and perform aggregation'''

    print(f"Processing {csv_file}...")
    print(f"Filter: {column} >= {min_value}")
    print(f"Operation: {operation}")
    print()

    # Load and filter
    df = pd.read_csv(csv_file)
    filtered = df[df[column] >= min_value]

    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(filtered)}")
    print()

    # Aggregate
    if operation == 'sum':
        result_value = filtered[column].sum()
    elif operation == 'mean':
        result_value = filtered[column].mean()
    elif operation == 'count':
        result_value = len(filtered)
    else:
        result_value = None

    print(f"Result ({operation}): {result_value}")

    return {
        'operation': operation,
        'column': column,
        'min_value': min_value,
        'original_count': len(df),
        'filtered_count': len(filtered),
        'result': float(result_value) if result_value is not None else None
    }
"""

result = executor.execute_function(
    code=code,
    function_name='filter_and_aggregate',
    function_args=['sales.csv', 'amount', 100],
    function_kwargs={'operation': 'sum'},
    files=['sales.csv']
)

# Access the return value
print(result['output']['result'])
# {'operation': 'sum', 'column': 'amount', 'min_value': 100, ...}
```

### Example 3: Machine Learning Model Inference

```python
code = """
import torch
import torch.nn as nn

def run_inference(model_path, input_shape):
    '''Load model and run inference'''

    print(f"Loading model from {model_path}...")
    model = torch.load(model_path)
    model.eval()
    print("✓ Model loaded")

    print(f"\\nCreating input tensor with shape {input_shape}...")
    input_tensor = torch.randn(*input_shape)
    print(f"✓ Input tensor created: {input_tensor.shape}")

    print("\\nRunning inference...")
    with torch.no_grad():
        output = model(input_tensor)

    print(f"✓ Inference complete")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    return {
        'output_shape': list(output.shape),
        'output_mean': float(output.mean()),
        'output_std': float(output.std())
    }
"""

result = executor.execute_function(
    code=code,
    function_name='run_inference',
    function_args=['model.pth', [1, 3, 224, 224]],
    files=['models/*.pth']
)
```

### Example 4: Multi-File Processing

```python
code = """
import pandas as pd
import os

def merge_csv_files(output_name='merged.csv'):
    '''Merge all CSV files in current directory'''

    print("Scanning for CSV files...")
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files:")

    dfs = []
    for csv_file in csv_files:
        print(f"  - Loading {csv_file}...")
        df = pd.read_csv(csv_file)
        print(f"    {len(df)} rows, {len(df.columns)} columns")
        dfs.append(df)

    print(f"\\nMerging {len(dfs)} dataframes...")
    merged = pd.concat(dfs, ignore_index=True)

    print(f"✓ Merged dataframe:")
    print(f"  Total rows: {len(merged)}")
    print(f"  Columns: {list(merged.columns)}")

    # Note: Can't save files back (yet), but we can return the data
    return {
        'total_rows': len(merged),
        'columns': list(merged.columns),
        'sample': merged.head().to_dict()
    }
"""

result = executor.execute_function(
    code=code,
    function_name='merge_csv_files',
    files=['data/*.csv']
)
```

## Input Parameters

### Required
- `code` (str): Python code containing the function definition

### Optional
- `function_name` (str): Name of the function to call
  - If omitted, code is executed directly (original behavior)

- `function_args` (list): Positional arguments for the function
  - Default: `[]`
  - Example: `['value1', 'value2']`

- `function_kwargs` (dict): Keyword arguments for the function
  - Default: `{}`
  - Example: `{'param1': 'value1', 'param2': 100}`

- `files` (list): Files to upload (base64 encoded)
  - Auto-detected from code when using FileUploader
  - Can also be added manually

## Best Practices

### 1. **Use Descriptive Logging**
```python
def process():
    print("Step 1/3: Loading data...")
    # ... load data

    print("Step 2/3: Processing...")
    # ... process

    print("Step 3/3: Finalizing...")
    # ... finalize

    print("✓ Complete!")
```

### 2. **Return Structured Data**
```python
def analyze():
    # ... analysis code

    return {
        'status': 'success',
        'results': results_dict,
        'metadata': {
            'rows_processed': count,
            'duration': elapsed_time
        }
    }
```

### 3. **Handle Errors Gracefully**
```python
def safe_process():
    try:
        # ... processing code
        print("✓ Success")
        return {'status': 'success', 'data': data}
    except Exception as e:
        print(f"✗ Error: {e}")
        return {'status': 'error', 'message': str(e)}
```

### 4. **Use Type Hints**
```python
def calculate_metrics(values: list, threshold: float = 0.5) -> dict:
    '''Calculate metrics with type hints for clarity'''
    # ... code
    return results
```

## Comparison: Code Execution vs Function Execution

### Direct Code Execution (Original)
```python
payload = {
    'input': {
        'code': '''
import pandas as pd
df = pd.read_csv('data.csv')
print(f"Rows: {len(df)}")
result = len(df)
'''
    }
}
```

**Use when:**
- Simple scripts
- One-time execution
- No need for arguments

### Function Execution (New)
```python
payload = {
    'input': {
        'code': '''
def count_rows(filename):
    df = pd.read_csv(filename)
    print(f"Rows: {len(df)}")
    return len(df)
''',
        'function_name': 'count_rows',
        'function_args': ['data.csv']
    }
}
```

**Use when:**
- Reusable logic
- Need to pass arguments
- Better code organization
- Void functions with logging

## Limitations

1. **Execution Environment**
   - Each execution runs in an isolated temporary directory
   - No persistent state between executions
   - Files are uploaded each time

2. **Return Values**
   - Must be JSON serializable
   - Complex objects are converted to strings
   - Use `make_json_safe` for automatic conversion

3. **File Operations**
   - You can read uploaded files
   - Cannot save files back (response only contains stdout/result)
   - For file outputs, return data in the result dict

## Troubleshooting

### Function Not Found Error
```
ValueError: Function 'my_func' not found in code
```

**Solution:** Ensure the function is defined in the code:
```python
code = """
def my_func():  # ✓ Function defined
    print("Hello")
"""
```

### Function Not Callable Error
```
ValueError: 'my_var' is not a callable function
```

**Solution:** Make sure you're calling a function, not a variable:
```python
function_name='my_func'  # ✓ Function
# not
function_name='my_var'   # ✗ Variable
```

### No Return Value
If you expect a return value but get `null`:
- Check that your function has a `return` statement
- Void functions return `null` - this is expected behavior
- All output is still in `stdout`

## Migration Guide

### Migrating from Direct Execution

**Before:**
```python
code = "print('Hello')\nresult = 42"
payload = {'input': {'code': code}}
```

**After:**
```python
code = "def main():\n    print('Hello')\n    return 42"
payload = {
    'input': {
        'code': code,
        'function_name': 'main'
    }
}
```

### Keeping Both Styles

You can still use direct execution - it's fully backward compatible:
```python
# This still works (no function_name specified)
payload = {'input': {'code': 'print("Hello")\nresult = 42'}}
```

## Summary

The function execution feature provides:

✅ **Void Function Support** - Execute functions without return values
✅ **Full Terminal Logging** - All print statements captured
✅ **Return Value Capture** - Get function results back
✅ **Argument Passing** - Send data to your functions
✅ **File Upload** - Automatic dependency detection
✅ **Backward Compatible** - Original code execution still works

Use `function_executor.py` for the easiest experience, or call the API directly for more control.

# RunPod Cloud PC - Update Notes

## 🎉 New Feature: Function Execution with Full Logging

We've added comprehensive support for executing Python functions on RunPod Cloud PC, including void functions (functions with no return value) with complete terminal logging.

---

## What's New

### ✅ Execute Void Functions
Functions that don't return anything now work perfectly. All print statements are captured and returned:

```python
def process_data():
    print("Loading data...")
    df = pd.read_csv('data.csv')
    print(f"Loaded {len(df)} rows")
    print("Processing complete!")
    # No return statement - this is a void function
```

### ✅ Full Terminal Logging
**ALL** terminal output is captured and returned in the response:

```python
def analyze():
    print("Step 1: Loading...")     # ✅ Captured
    print("Step 2: Processing...")   # ✅ Captured
    print("Step 3: Complete!")       # ✅ Captured
    return results
```

### ✅ Function Return Values
Return values are automatically captured and JSON-serialized:

```python
def calculate():
    result = sum([1, 2, 3, 4, 5])
    return {
        'total': result,
        'count': 5
    }
```

### ✅ Pass Arguments to Functions
Support for both positional and keyword arguments:

```python
def process(filename, threshold=100, verbose=True):
    # ... your code
    return results

# Call with arguments
function_args=['data.csv']
function_kwargs={'threshold': 200, 'verbose': True}
```

---

## Updated Files

### 1. Handler (`rp_handler.py`)
**Changes:**
- Added `function_name` parameter to specify which function to call
- Added `function_args` parameter for positional arguments
- Added `function_kwargs` parameter for keyword arguments
- Function execution logic that calls the function and captures return value
- All terminal logs (stdout/stderr) are still captured

**New Input Format:**
```json
{
  "input": {
    "code": "def my_func(): ...",
    "function_name": "my_func",
    "function_args": [],
    "function_kwargs": {}
  }
}
```

### 2. Function Executor (`function_executor.py`) - NEW
**Purpose:** Easy-to-use Python class for executing functions on RunPod

**Features:**
- Simple API for function execution
- Automatic file dependency detection
- Formatted result display
- Error handling

**Usage:**
```python
from function_executor import FunctionExecutor

executor = FunctionExecutor(
    endpoint_url="https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync",
    api_key="YOUR_API_KEY"
)

result = executor.execute_function(
    code=your_code,
    function_name='process_data',
    function_kwargs={'threshold': 100}
)
```

### 3. Documentation (`FUNCTION_EXECUTION.md`) - NEW
**Complete guide covering:**
- Overview and benefits
- Usage examples (void functions, return values, arguments)
- Response format
- Best practices
- Troubleshooting
- Migration guide

### 4. Quick Start (`quick_start.py`)
**Updated with:**
- Function execution example
- Both file upload and function execution demos

---

## How It Works

### Architecture

```
┌─────────────────┐
│  Your Code      │
│  + Function     │
└────────┬────────┘
         │
         │ Send via API
         ▼
┌─────────────────┐
│  RunPod Handler │
│  1. Save files  │
│  2. Execute code│
│  3. Call func   │
│  4. Capture all │
└────────┬────────┘
         │
         │ Return
         ▼
┌─────────────────┐
│  Response       │
│  - stdout       │ ◄─── All print statements
│  - stderr       │ ◄─── All errors
│  - result       │ ◄─── Return value
│  - error        │ ◄─── Exception if failed
└─────────────────┘
```

### Execution Flow

1. **Code Upload**: Your function code and files are sent to RunPod
2. **Environment Setup**: Temporary directory created, files saved
3. **Code Execution**: Code is executed in isolated namespace
4. **Function Call**: If `function_name` specified, function is called with arguments
5. **Output Capture**: ALL stdout/stderr captured during execution
6. **Return Value**: Function return value is JSON-serialized
7. **Cleanup**: Temporary directory removed
8. **Response**: Complete response with logs and results sent back

---

## Response Format

All responses follow this structure:

```json
{
  "output": {
    "stdout": "All terminal logs and print statements",
    "stderr": "Error messages if any",
    "result": "Return value or null for void functions",
    "error": "Error message if execution failed"
  }
}
```

### Example: Void Function Response
```json
{
  "output": {
    "stdout": "Loading data...\nLoaded 1000 rows\nProcessing complete!\n",
    "stderr": "",
    "result": null,
    "error": null
  }
}
```

### Example: Function with Return Value
```json
{
  "output": {
    "stdout": "Calculating...\nTotal: 150.00\n",
    "stderr": "",
    "result": {"total": 150.0, "count": 10},
    "error": null
  }
}
```

---

## Examples

### Example 1: Void Function with Logging

```python
code = """
import pandas as pd

def analyze_data():
    print("Starting analysis...")
    df = pd.read_csv('data.csv')
    print(f"Loaded {len(df)} rows")

    print("\\nCalculating statistics...")
    print(f"Mean: {df['value'].mean():.2f}")
    print(f"Max: {df['value'].max():.2f}")

    print("\\nAnalysis complete!")
    # No return - void function
"""

executor.execute_function(
    code=code,
    function_name='analyze_data',
    files=['data.csv']
)

# Output (in stdout):
# Starting analysis...
# Loaded 1000 rows
#
# Calculating statistics...
# Mean: 45.32
# Max: 99.87
#
# Analysis complete!
```

### Example 2: Function with Arguments

```python
code = """
def calculate_total(numbers, multiplier=1.0):
    print(f"Calculating total for {len(numbers)} numbers")
    print(f"Using multiplier: {multiplier}")

    total = sum(numbers) * multiplier
    print(f"Total: {total}")

    return {
        'total': total,
        'count': len(numbers),
        'multiplier': multiplier
    }
"""

result = executor.execute_function(
    code=code,
    function_name='calculate_total',
    function_args=[[10, 20, 30, 40]],
    function_kwargs={'multiplier': 2.0}
)

# stdout: "Calculating total for 4 numbers\nUsing multiplier: 2.0\nTotal: 200.0\n"
# result: {'total': 200.0, 'count': 4, 'multiplier': 2.0}
```

### Example 3: Data Processing with Files

```python
code = """
import pandas as pd

def process_sales(min_amount=0):
    print(f"Loading sales data...")
    df = pd.read_csv('sales.csv')
    print(f"✓ Loaded {len(df)} records")

    print(f"\\nFiltering sales >= ${min_amount}...")
    filtered = df[df['amount'] >= min_amount]
    print(f"✓ {len(filtered)} records match")

    total = filtered['amount'].sum()
    avg = filtered['amount'].mean()

    print(f"\\nResults:")
    print(f"  Total Sales: ${total:,.2f}")
    print(f"  Average Sale: ${avg:,.2f}")

    return {
        'total_sales': float(total),
        'average_sale': float(avg),
        'record_count': len(filtered)
    }
"""

result = executor.execute_function(
    code=code,
    function_name='process_sales',
    function_kwargs={'min_amount': 100},
    files=['sales.csv']
)
```

---

## Key Benefits

### 🎯 Better Code Organization
Functions make your code modular and reusable instead of writing inline scripts.

### 📊 Complete Visibility
All terminal output is captured, giving you full visibility into execution.

### 🔄 Void Function Support
Finally, you can execute functions that only log output without returning values.

### 📦 Argument Flexibility
Pass any data to your functions using positional or keyword arguments.

### 🔒 Type Safety
Return values are automatically JSON-serialized, ensuring safe data transmission.

### 🔁 Backward Compatible
Original code execution (without functions) still works exactly as before.

---

## Migration Guide

### From Direct Code Execution

**Before:**
```python
payload = {
    'input': {
        'code': '''
import pandas as pd
df = pd.read_csv('data.csv')
print(f"Rows: {len(df)}")
result = len(df)
'''
    }
}
```

**After (as function):**
```python
payload = {
    'input': {
        'code': '''
def count_rows(filename):
    import pandas as pd
    df = pd.read_csv(filename)
    print(f"Rows: {len(df)}")
    return len(df)
''',
        'function_name': 'count_rows',
        'function_args': ['data.csv']
    }
}
```

### Both Styles Work!

You don't have to migrate - both execution styles are supported:
- **Direct execution**: Set only `code`
- **Function execution**: Set `code` + `function_name`

---

## Quick Reference

### Direct API Call
```python
import requests

response = requests.post(
    'https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json={
        'input': {
            'code': 'def func(): print("Hi")\n',
            'function_name': 'func',
            'function_args': [],
            'function_kwargs': {}
        }
    }
)
```

### Using FunctionExecutor
```python
from function_executor import FunctionExecutor

executor = FunctionExecutor(endpoint_url=..., api_key=...)
result = executor.execute_function(
    code=code,
    function_name='my_func',
    function_args=[arg1, arg2],
    function_kwargs={'param': value}
)
```

---

## Documentation Files

📚 **FUNCTION_EXECUTION.md** - Complete feature documentation
- Overview and benefits
- Detailed usage examples
- Response format
- Best practices
- Troubleshooting

🚀 **function_executor.py** - Python client library
- FunctionExecutor class
- Helper methods
- Example usage

⚡ **quick_start.py** - Quick start examples
- Basic file upload
- Function execution demo

📖 **README_FILE_UPLOAD.md** - File upload system docs

---

## Summary

This update adds powerful function execution capabilities to RunPod Cloud PC:

✅ **Void functions** - Execute functions with no return value
✅ **Full logging** - Capture all terminal output
✅ **Return values** - Get function results back
✅ **Arguments** - Pass data to functions
✅ **Backward compatible** - Original code execution still works

Start using it today with the `FunctionExecutor` class or by adding `function_name` to your API calls!

For complete documentation, see `FUNCTION_EXECUTION.md`.
