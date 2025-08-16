from pathlib import Path

def read_any(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore')
