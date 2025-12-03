import os

EXCLUDE_DIRS = {
    "__pycache__", ".git", ".venv", "venv",
    "build", "dist", ".idea", ".vscode",
    ".mypy_cache", ".pytest_cache",
    "data", "generated_datasets",   # ← your added exclusions
    "logs", "output", "fact_out", "parquet_dims"
}

EXCLUDE_EXT = {".pyc", ".pyo", ".pyd", ".so"}

def print_tree(root=".", prefix=""):
    try:
        entries = sorted(os.listdir(root))
    except PermissionError:
        return

    for i, entry in enumerate(entries):
        path = os.path.join(root, entry)

        # skip excluded directories entirely
        if os.path.isdir(path) and entry in EXCLUDE_DIRS:
            continue

        # skip compiled/unwanted file types
        if os.path.isfile(path) and os.path.splitext(entry)[1] in EXCLUDE_EXT:
            continue

        connector = "└── " if i == len(entries) - 1 else "├── "

        # only show directories and .py files
        if os.path.isdir(path):
            print(prefix + connector + entry + "/")
            new_prefix = prefix + ("    " if i == len(entries) - 1 else "│   ")
            print_tree(path, new_prefix)
        elif entry.endswith(".py"):
            print(prefix + connector + entry)


if __name__ == "__main__":
    print("Project Structure:\n")
    print_tree(".")
