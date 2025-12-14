# RunPod Cloud PC - File Upload System

This system allows you to execute Python code on RunPod with automatic file dependency detection and upload.

## Features

- **Automatic Dependency Detection**: Analyzes your code to find local imports, data files, and other dependencies
- **File Upload**: Uploads all detected files to the RunPod cloud PC
- **Isolated Execution**: Each execution runs in a temporary directory with all uploaded files available
- **Support for**: Local Python modules, CSV/Excel files, PyTorch models, NumPy arrays, and more

## Files

- `rp_handler.py` - RunPod serverless handler that executes code with uploaded files
- `file_uploader.py` - Analyzes code dependencies and prepares upload payload
- `example_client.py` - Example client showing how to use the system

## How It Works

### 1. Handler (rp_handler.py)

The handler accepts requests with this format:

```json
{
  "input": {
    "code": "import pandas as pd\ndf = pd.read_csv('data.csv')\nprint(df.head())",
    "files": [
      {
        "path": "data.csv",
        "content": "base64_encoded_file_content"
      }
    ]
  }
}
```

The handler:
1. Creates a temporary working directory
2. Saves all uploaded files maintaining their directory structure
3. Changes to the working directory
4. Executes the code
5. Returns stdout, stderr, and results
6. Cleans up the temporary directory

### 2. File Uploader (file_uploader.py)

The `FileDependencyAnalyzer` automatically detects:

- **Local module imports**: `from my_module import func` → finds `my_module.py`
- **File operations**: `open('file.txt')` → finds `file.txt`
- **Data loading**:
  - `pd.read_csv('data.csv')` → finds `data.csv`
  - `pd.read_excel('data.xlsx')` → finds `data.xlsx`
  - `np.load('array.npy')` → finds `array.npy`
  - `torch.load('model.pth')` → finds `model.pth`

The `FileUploader` class:
- Builds a file dependency tree
- Encodes files in base64
- Prepares the complete payload for RunPod

## Usage

### Basic Example

```python
from file_uploader import FileUploader

# Your code
code = """
import pandas as pd
df = pd.read_csv('data/sales.csv')
print(f'Total sales: {df["amount"].sum()}')
result = df["amount"].sum()
"""

# Create uploader
uploader = FileUploader(base_dir='.')

# Prepare payload (automatically detects data/sales.csv)
payload = uploader.prepare_payload(code, analyze_dependencies=True)

# Send to RunPod
import requests
response = requests.post(
    'https://api.runpod.ai/v2/YOUR_ENDPOINT/runsync',
    headers={'Authorization': 'Bearer YOUR_API_KEY'},
    json=payload
)

result = response.json()
print(result['output']['stdout'])
```

### Adding Files Manually

```python
# Prepare payload with auto-detection
payload = uploader.prepare_payload(code)

# Add additional files using glob patterns
payload = uploader.add_manual_files(payload, [
    'data/*.csv',      # All CSV files in data folder
    'models/*.pth',    # All PyTorch models
    'config.json'      # Specific file
])
```

### Example with Local Modules

```python
code = """
from my_utils import preprocess, analyze
import pandas as pd

df = pd.read_csv('input.csv')
df = preprocess(df)
results = analyze(df)
print(results)
"""

# Automatically detects and uploads:
# - my_utils.py (local module)
# - input.csv (data file)
payload = uploader.prepare_payload(code)
```

## Supported File Patterns

The analyzer automatically detects:

### Import Statements
- `import my_module` → `my_module.py`
- `from my_package import func` → `my_package/__init__.py`
- `from my_package.submodule import func` → `my_package/submodule.py`

### File Operations
- `open('file.txt')`
- `Path('file.txt')`

### Data Loading Functions
- `pd.read_csv('data.csv')`
- `pd.read_excel('data.xlsx')`
- `pd.read_json('data.json')`
- `pd.read_parquet('data.parquet')`
- `np.load('array.npy')`
- `np.loadtxt('data.txt')`
- `torch.load('model.pth')`

## Deployment

1. **Build Docker image**:
   ```bash
   docker build -t your-runpod-image .
   ```

2. **Push to registry**:
   ```bash
   docker push your-runpod-image
   ```

3. **Deploy on RunPod**:
   - Create a new serverless endpoint
   - Use your Docker image
   - Set the handler to `rp_handler.handler`

4. **Test**:
   ```python
   from example_client import execute_on_runpod

   code = "print('Hello from RunPod!')"
   execute_on_runpod(code)
   ```

## API Response Format

```json
{
  "output": {
    "stdout": "Output printed to console",
    "stderr": "Error messages if any",
    "result": "Value of 'result' variable if set",
    "error": "Error message if execution failed"
  }
}
```

## Examples

See `example_client.py` for complete examples including:
1. CSV data processing
2. Local module imports
3. PyTorch model loading
4. Multiple file uploads

## Limitations

- Files are uploaded on each request (no persistent storage)
- Binary files are base64 encoded (increases payload size)
- Large files may hit API payload limits
- Execution is stateless (no state between requests)

## Tips

- Keep uploaded files small (< 10MB total recommended)
- For large datasets, consider pre-installing them in the Docker image
- Use `manual_files` patterns to upload entire directories
- The working directory is cleaned up after each execution
