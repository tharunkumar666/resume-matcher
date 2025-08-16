from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def matched_skills(resume, jd):
    return sorted(set([s.lower() for s in resume]).intersection(set([s.lower() for s in jd])))

class SkillVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2))
        self.job_ids=[]
        self.job_matrix=None
        self.job_skills=[]
    def fit_jobs(self, job_ids, job_skills):
        texts=[" ".join(s) for s in job_skills]
        self.job_ids=job_ids
        self.job_skills=job_skills
        self.job_matrix = self.vectorizer.fit_transform(texts)
    def rank_for_resume(self, resume_skills, top_k=10):
        if self.job_matrix is None: raise RuntimeError('Call fit_jobs')
        res_vec = self.vectorizer.transform([" ".join(resume_skills)])
        sims = cosine_similarity(res_vec, self.job_matrix).ravel()
        order = np.argsort(-sims)[:top_k]
        out=[]
        for i in order:
            out.append((self.job_ids[i], float(sims[i]), matched_skills(resume_skills, self.job_skills[i])))
        return out
