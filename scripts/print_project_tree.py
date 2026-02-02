import os

EXCLUDE_DIRS = {
    "__pycache__", ".git", ".venv", "venv",
    "build", "dist", ".idea", ".vscode",
    ".mypy_cache", ".pytest_cache",
    "data", "generated_datasets",
    "logs", "output", "fact_out", "parquet_dims", 
    "PBIP Parquet", "PBIP CSV", "ui", "samples", "scripts", "docs"
}

EXCLUDE_EXT = {".pyc", ".pyo", ".pyd", ".so"}

# file types to display
INCLUDE_EXT = {".py", ".ps1", ".pbix", ".pbit", '.png', ".parquet", ".sql"}

def print_tree(root=".", prefix=""):
    try:
        entries = sorted(os.listdir(root))
    except PermissionError:
        return

    for i, entry in enumerate(entries):
        path = os.path.join(root, entry)

        # skip excluded directories
        if os.path.isdir(path) and entry in EXCLUDE_DIRS:
            continue

        # skip excluded file types
        if os.path.isfile(path) and os.path.splitext(entry)[1] in EXCLUDE_EXT:
            continue

        connector = "└── " if i == len(entries) - 1 else "├── "

        if os.path.isdir(path):
            print(prefix + connector + entry + "/")
            new_prefix = prefix + ("    " if i == len(entries) - 1 else "│   ")
            print_tree(path, new_prefix)
        else:
            ext = os.path.splitext(entry)[1]
            if ext in INCLUDE_EXT:
                print(prefix + connector + entry)


if __name__ == "__main__":
    import sys

    root_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else r"." # generated_datasets\2026-01-31 02_02_58 PM Customers 53K Sales 50K CSV
    )

    print(f"Project Structure ({os.path.abspath(root_path)}):\n")
    print_tree(root_path)
