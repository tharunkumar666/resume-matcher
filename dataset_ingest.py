from pathlib import Path
from typing import Dict
from .parse_docs import read_any

def load_texts_from_dir(d: Path) -> Dict[str, str]:
    out={}
    if not d.exists(): return out
    for p in sorted(d.glob('**/*')):
        if p.is_file() and p.suffix.lower() in {'.txt','.pdf','.docx'}:
            out[p.stem]=read_any(p)
    return out
