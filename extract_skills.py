from pathlib import Path
import spacy
from spacy.matcher import PhraseMatcher
from typing import List
from .preprocess import normalize_skill

def load_skills(skills_path: Path) -> List[str]:
    return [s.strip() for s in skills_path.read_text(encoding='utf-8').splitlines() if s.strip()]

def build_nlp_and_matcher(skills: List[str]):
    nlp = spacy.blank('en')
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    patterns = [nlp.make_doc(s) for s in skills]
    matcher.add('SKILL', patterns)
    return nlp, matcher

def extract_skill_list(nlp, matcher, text: str) -> List[str]:
    if not text: return []
    doc = nlp(text)
    spans = [doc[start:end] for _, start, end in matcher(doc)]
    spans = spacy.util.filter_spans(spans)
    out=[]
    seen=set()
    for sp in spans:
        v = normalize_skill(sp.text)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out
