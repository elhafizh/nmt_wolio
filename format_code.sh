#!/bin/bash

file_path="helpers/utils.py"

# Run black on the file
black "$file_path"

# Run isort on the file
isort "$file_path"

# Run mypy with --ignore-missing-imports on the file
# mypy --ignore-missing-imports "$file_path"
