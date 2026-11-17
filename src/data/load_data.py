from pathlib import Path
import pandas as pd


def load_data(
    file_path: str | Path,
    file_type: str | None = None,
    **kwargs
) -> pd.DataFrame:

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_type is None:
        file_type = file_path.suffix.lower().lstrip(".")

    loaders = {
        "csv": pd.read_csv,
        "xlsx": pd.read_excel,
        "xls": pd.read_excel,
        "parquet": pd.read_parquet,
    }

    if file_type not in loaders:
        raise ValueError(f"Unsupported file format: {file_type}")

    return loaders[file_type](file_path, **kwargs)
