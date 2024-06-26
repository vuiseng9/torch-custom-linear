import os
import shutil
from setuptools import setup, Command
from torch.utils.cpp_extension import BuildExtension, CppExtension

MODULE_NAME="custom_linear"

class CustomCleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        # Add paths to clean
        paths_to_clean = [
            './build',
            './dist',
        ]

        # Patterns to remove
        patterns_to_remove = [
            '*.egg-info',
            '*.so',
            '*.pyd',
            '*.egg',
        ]

        # Clean specific paths
        for path in paths_to_clean:
            if os.path.isdir(path):
                print(f'Removing directory: {path}')
                shutil.rmtree(path)
            elif os.path.isfile(path):
                print(f'Removing file: {path}')
                os.remove(path)

        # Clean patterns in the root directory
        for pattern in patterns_to_remove:
            for filename in os.listdir('.'):
                if filename.endswith(pattern.split('*')[-1]):
                    file_path = os.path.join('.', filename)
                    if os.path.isdir(file_path):
                        print(f'Removing directory: {file_path}')
                        shutil.rmtree(file_path)
                    elif os.path.isfile(file_path):
                        print(f'Removing file: {file_path}')
                        os.remove(file_path)

        # Recursively clean specific patterns
        for root, dirs, files in os.walk('.'):
            for dir in dirs:
                if dir == '__pycache__' or dir.endswith('.egg-info'):
                    dir_path = os.path.join(root, dir)
                    print(f'Removing directory: {dir_path}')
                    shutil.rmtree(dir_path)

            for file in files:
                if file.endswith(('.pyc', '.pyo', '.so', '.pyd', '.egg')):
                    file_path = os.path.join(root, file)
                    print(f'Removing file: {file_path}')
                    os.remove(file_path)

setup(
    name=MODULE_NAME,
    ext_modules=[
        CppExtension(
            name=MODULE_NAME,
            sources=['csrc/dense_linear.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension,
        'clean': CustomCleanCommand,
    })
