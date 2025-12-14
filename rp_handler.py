import runpod
import io
import sys
import traceback
import json
import base64
import os
import tempfile
import shutil

def make_json_safe(obj):
    """Convert an object to a JSON-safe format."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    else:
        # For non-serializable objects, return their string representation
        return str(obj)

def save_uploaded_files(files, work_dir):
    """
    Save uploaded files to the working directory.

    Args:
        files (list): List of file dictionaries with 'path' and 'content' (base64 encoded)
        work_dir (str): Working directory path

    Returns:
        list: List of saved file paths
    """
    saved_files = []

    for file_info in files:
        file_path = file_info.get('path', '')
        file_content_b64 = file_info.get('content', '')

        if not file_path or not file_content_b64:
            continue

        # Create full path in working directory
        full_path = os.path.join(work_dir, file_path)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # Decode and save file
        try:
            file_content = base64.b64decode(file_content_b64)
            with open(full_path, 'wb') as f:
                f.write(file_content)
            saved_files.append(file_path)
            print(f"Saved file: {file_path}")
        except Exception as e:
            print(f"Error saving file {file_path}: {e}")

    return saved_files

def handler(event):
    """
    This function processes incoming requests to your Serverless endpoint.

    Args:
        event (dict): Contains the input data and request metadata

    Returns:
        dict: Contains the execution output, stdout, stderr, and any errors

    Input format:
        - code (str): Python code to execute
        - files (list): Optional list of files to upload
        - function_name (str): Optional function name to call after executing code
        - function_args (list): Optional positional arguments for the function
        - function_kwargs (dict): Optional keyword arguments for the function
    """

    print(f"Worker Start")
    input_data = event['input']

    # Get the Python code to execute
    code = input_data.get('code', '')

    # Get uploaded files (list of {path, content} dicts)
    files = input_data.get('files', [])

    # Get function execution parameters
    function_name = input_data.get('function_name', None)
    function_args = input_data.get('function_args', [])
    function_kwargs = input_data.get('function_kwargs', {})

    if not code:
        return {
            'error': 'No code provided',
            'stdout': '',
            'stderr': '',
            'result': None
        }

    # Create a temporary working directory for this execution
    work_dir = tempfile.mkdtemp(prefix='runpod_exec_')
    print(f"Working directory: {work_dir}")

    try:
        # Save uploaded files
        if files:
            print(f"Uploading {len(files)} file(s)...")
            saved_files = save_uploaded_files(files, work_dir)
            print(f"Saved {len(saved_files)} file(s)")

        print(f"Executing code...")

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        # Store original stdout/stderr and cwd
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        original_cwd = os.getcwd()

        result = None
        error = None

        try:
            # Change to working directory
            os.chdir(work_dir)

            # Add working directory to Python path
            if work_dir not in sys.path:
                sys.path.insert(0, work_dir)

            # Redirect stdout and stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Create a namespace for execution with common imports
            exec_namespace = {
                '__name__': '__main__',
                '__file__': os.path.join(work_dir, 'main.py')
            }

            # Execute the code
            exec(code, exec_namespace)

            # If function_name is specified, call the function
            if function_name:
                if function_name not in exec_namespace:
                    raise ValueError(f"Function '{function_name}' not found in code")

                func = exec_namespace[function_name]
                if not callable(func):
                    raise ValueError(f"'{function_name}' is not a callable function")

                print(f"Calling function: {function_name}")
                # Call the function with provided arguments
                func_result = func(*function_args, **function_kwargs)
                result = make_json_safe(func_result)
            # Otherwise, check if there's a 'result' variable in the namespace
            elif 'result' in exec_namespace:
                result = make_json_safe(exec_namespace['result'])

        except Exception as e:
            error = str(e)
            stderr_capture.write(traceback.format_exc())
        finally:
            # Restore original stdout/stderr and cwd
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            os.chdir(original_cwd)

            # Remove working directory from path
            if work_dir in sys.path:
                sys.path.remove(work_dir)

        # Get the captured output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        print(f"Execution complete")

        response = {
            'stdout': stdout_output,
            'stderr': stderr_output,
            'result': result,
            'error': error
        }

        # Ensure the response is JSON serializable
        try:
            json.dumps(response)
        except (TypeError, ValueError) as e:
            # If still not serializable, make everything safe
            response = make_json_safe(response)

        return response

    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(work_dir)
            print(f"Cleaned up working directory")
        except Exception as e:
            print(f"Error cleaning up working directory: {e}") 

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })