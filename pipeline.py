from pathlib import Path
from .preprocess import clean_text
from .extract_skills import load_skills, build_nlp_and_matcher, extract_skill_list
from .vectorize_match import SkillVectorizer
from .dataset_ingest import load_texts_from_dir

def build_pipeline(base: Path):
    skills_path = base / 'data' / 'curated_skills.txt'
    resumes_dir = base / 'data' / 'resumes_raw'
    jobs_dir = base / 'data' / 'jobs_raw'
    skills = load_skills(skills_path)
    nlp, matcher = build_nlp_and_matcher(skills)
    resumes = load_texts_from_dir(resumes_dir)
    jobs = load_texts_from_dir(jobs_dir)
    resume_skills = {rid: extract_skill_list(nlp, matcher, clean_text(txt)) for rid, txt in resumes.items()}
    job_skills = {jid: extract_skill_list(nlp, matcher, clean_text(txt)) for jid, txt in jobs.items()}
    vec = SkillVectorizer()
    job_ids = list(job_skills.keys())
    vec.fit_jobs(job_ids, [job_skills[j] for j in job_ids])
    results={}
    for rid, rskills in resume_skills.items():
        results[rid]=vec.rank_for_resume(rskills, top_k=10)
    return results

if __name__=='__main__':
    import pathlib
    res = build_pipeline(pathlib.Path('.').resolve())
    print(res)
