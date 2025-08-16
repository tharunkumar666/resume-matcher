import re

def clean_text(s: str) -> str:
    if not s: return ''
    s = s.replace('\x00',' ')
    s = re.sub(r'\s+',' ', s)
    return s.strip()

def normalize_skill(skill: str) -> str:
    return skill.strip().lower()
