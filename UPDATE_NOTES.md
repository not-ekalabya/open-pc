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
