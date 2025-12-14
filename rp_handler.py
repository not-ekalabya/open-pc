import runpod
import io
import sys
import traceback
import json

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

def handler(event):
    """
    This function processes incoming requests to your Serverless endpoint.

    Args:
        event (dict): Contains the input data and request metadata

    Returns:
        dict: Contains the execution output, stdout, stderr, and any errors
    """

    print(f"Worker Start")
    input_data = event['input']

    # Get the Python code to execute
    code = input_data.get('code', '')

    if not code:
        return {
            'error': 'No code provided',
            'stdout': '',
            'stderr': '',
            'result': None
        }

    print(f"Executing code...")

    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Store original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    result = None
    error = None

    try:
        # Redirect stdout and stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        # Create a namespace for execution
        exec_namespace = {}

        # Execute the code
        exec(code, exec_namespace)

        # If there's a 'result' variable in the namespace, use it as the result
        if 'result' in exec_namespace:
            result = make_json_safe(exec_namespace['result'])

    except Exception as e:
        error = str(e)
        stderr_capture.write(traceback.format_exc())
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

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

# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })