"""
File Dependency Analyzer and Uploader for RunPod Cloud PC

This script analyzes Python code to find file dependencies and uploads them to RunPod.
"""

import ast
import os
import re
import base64
import json
from pathlib import Path
from typing import Set, List, Dict, Any


class FileDependencyAnalyzer:
    """Analyzes Python code to find file dependencies."""

    def __init__(self, code: str, base_dir: str = '.'):
        self.code = code
        self.base_dir = Path(base_dir).resolve()
        self.dependencies: Set[Path] = set()

    def analyze(self) -> Set[Path]:
        """Analyze code and return all file dependencies."""
        self._find_imports()
        self._find_file_operations()
        self._find_data_file_patterns()
        return self.dependencies

    def _find_imports(self):
        """Find local module imports."""
        try:
            tree = ast.parse(self.code)
            for node in ast.walk(tree):
                # Handle 'import module' and 'from module import ...'
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self._resolve_import(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self._resolve_import(node.module)
        except SyntaxError:
            # If code has syntax errors, skip AST parsing
            pass

    def _resolve_import(self, module_name: str):
        """Resolve import to file path if it's a local module."""
        # Skip standard library and installed packages
        if '.' in module_name:
            parts = module_name.split('.')
        else:
            parts = [module_name]

        # Try to find as a package or module file
        current_path = self.base_dir
        for part in parts:
            # Check for package
            package_path = current_path / part
            if package_path.is_dir() and (package_path / '__init__.py').exists():
                self.dependencies.add(package_path / '__init__.py')
                current_path = package_path
            # Check for module file
            elif (current_path / f"{part}.py").exists():
                self.dependencies.add(current_path / f"{part}.py")
                return
            else:
                # Not a local module
                return

    def _find_file_operations(self):
        """Find file operations like open(), read, etc."""
        # Patterns for file operations
        patterns = [
            r'open\s*\(\s*["\']([^"\']+)["\']',  # open('file.txt')
            r'Path\s*\(\s*["\']([^"\']+)["\']',  # Path('file.txt')
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, self.code)
            for match in matches:
                file_path = match.group(1)
                self._add_file_if_exists(file_path)

    def _find_data_file_patterns(self):
        """Find data file operations from pandas, numpy, etc."""
        # Patterns for data loading functions
        patterns = [
            r'read_csv\s*\(\s*["\']([^"\']+)["\']',  # pd.read_csv('file.csv')
            r'read_excel\s*\(\s*["\']([^"\']+)["\']',  # pd.read_excel('file.xlsx')
            r'read_json\s*\(\s*["\']([^"\']+)["\']',  # pd.read_json('file.json')
            r'read_parquet\s*\(\s*["\']([^"\']+)["\']',  # pd.read_parquet('file.parquet')
            r'load\s*\(\s*["\']([^"\']+)["\']',  # np.load('file.npy')
            r'loadtxt\s*\(\s*["\']([^"\']+)["\']',  # np.loadtxt('file.txt')
            r'torch\.load\s*\(\s*["\']([^"\']+)["\']',  # torch.load('model.pth')
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, self.code)
            for match in matches:
                file_path = match.group(1)
                self._add_file_if_exists(file_path)

    def _add_file_if_exists(self, file_path: str):
        """Add file to dependencies if it exists."""
        path = Path(file_path)

        # Try absolute path
        if path.is_absolute() and path.exists():
            self.dependencies.add(path)
            return

        # Try relative to base directory
        abs_path = (self.base_dir / path).resolve()
        if abs_path.exists():
            self.dependencies.add(abs_path)


class FileUploader:
    """Uploads files to RunPod endpoint."""

    def __init__(self, base_dir: str = '.'):
        self.base_dir = Path(base_dir).resolve()

    def build_file_tree(self, file_paths: Set[Path]) -> List[Dict[str, Any]]:
        """
        Build file tree with base64 encoded content.

        Args:
            file_paths: Set of file paths to upload

        Returns:
            List of file dictionaries with 'path' and 'content' (base64)
        """
        files = []

        for file_path in file_paths:
            try:
                # Get relative path from base directory
                try:
                    rel_path = file_path.relative_to(self.base_dir)
                except ValueError:
                    # File is outside base directory, use filename only
                    rel_path = file_path.name

                # Read and encode file
                with open(file_path, 'rb') as f:
                    content = f.read()
                    content_b64 = base64.b64encode(content).decode('utf-8')

                files.append({
                    'path': str(rel_path).replace('\\', '/'),  # Use forward slashes
                    'content': content_b64
                })

                print(f"Added to upload: {rel_path} ({len(content)} bytes)")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        return files

    def prepare_payload(self, code: str, analyze_dependencies: bool = True) -> Dict[str, Any]:
        """
        Prepare complete payload for RunPod execution.

        Args:
            code: Python code to execute
            analyze_dependencies: Whether to analyze and upload dependencies

        Returns:
            Complete payload dictionary
        """
        files = []

        if analyze_dependencies:
            print("Analyzing code dependencies...")
            analyzer = FileDependencyAnalyzer(code, str(self.base_dir))
            dependencies = analyzer.analyze()

            print(f"\nFound {len(dependencies)} file dependencies:")
            for dep in sorted(dependencies):
                print(f"  - {dep.relative_to(self.base_dir) if dep.is_relative_to(self.base_dir) else dep}")

            if dependencies:
                files = self.build_file_tree(dependencies)

        payload = {
            'input': {
                'code': code,
                'files': files
            }
        }

        return payload

    def add_manual_files(self, payload: Dict[str, Any], file_patterns: List[str]) -> Dict[str, Any]:
        """
        Manually add files matching patterns to the payload.

        Args:
            payload: Existing payload
            file_patterns: List of glob patterns (e.g., ['data/*.csv', 'models/*.pth'])

        Returns:
            Updated payload
        """
        additional_files = set()

        for pattern in file_patterns:
            matches = self.base_dir.glob(pattern)
            for match in matches:
                if match.is_file():
                    additional_files.add(match)

        if additional_files:
            print(f"\nAdding {len(additional_files)} manual files:")
            for file_path in sorted(additional_files):
                print(f"  - {file_path.relative_to(self.base_dir)}")

            new_files = self.build_file_tree(additional_files)
            payload['input']['files'].extend(new_files)

        return payload


# Example usage
if __name__ == '__main__':
    # Example code to analyze
    example_code = """
import pandas as pd
import numpy as np
from my_utils import helper_function

# Load data
df = pd.read_csv('data/dataset.csv')
model = torch.load('models/model.pth')

# Process
result = helper_function(df)
print(result)
"""

    # Create uploader
    uploader = FileUploader(base_dir='.')

    # Prepare payload with automatic dependency detection
    payload = uploader.prepare_payload(example_code, analyze_dependencies=True)

    # Optionally add more files manually
    payload = uploader.add_manual_files(payload, ['data/*.csv', 'models/*.pth'])

    # Save payload to file
    with open('upload_payload.json', 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"\nPayload saved to upload_payload.json")
    print(f"Total files to upload: {len(payload['input']['files'])}")
