import streamlit as st
from pathlib import Path
from src.parse_docs import read_any
from src.preprocess import clean_text
from src.extract_skills import load_skills, build_nlp_and_matcher, extract_skill_list
from src.vectorize_match import SkillVectorizer
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
skills_path = BASE / 'data' / 'curated_skills.txt'
skills = load_skills(skills_path)
nlp, matcher = build_nlp_and_matcher(skills)

jobs_dir = BASE / 'data' / 'jobs_raw'
jd_files = {p.stem:p for p in sorted(jobs_dir.glob('**/*')) if p.is_file()}
jd_texts = {jid: read_any(p) for jid,p in jd_files.items()}
jd_skills = {jid: extract_skill_list(nlp, matcher, clean_text(txt)) for jid,txt in jd_texts.items()}
vec = SkillVectorizer()
vec.fit_jobs(list(jd_skills.keys()), [jd_skills[j] for j in jd_skills])

st.title('Resume Matcher')
up = st.file_uploader('Upload resume', type=['txt'])
if up:
    text = read_any(Path(up.name)) if False else up.read().decode('utf-8')
    rskills = extract_skill_list(nlp, matcher, clean_text(text))
    res = vec.rank_for_resume(rskills, top_k=5)
    df = pd.DataFrame([{'job_id':r[0],'score':r[1],'matched_skills':', '.join(r[2])} for r in res])
    st.dataframe(df)
