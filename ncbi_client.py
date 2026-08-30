"""
ncbi_client.py
==============
Real NCBI E-utilities client optimized for hybrid semantic retrieval (Bio-Lens).
"""

import re
import time
import requests
from xml.etree import ElementTree as ET

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
TOOL_NAME = "pubmed-semantic-search-demo"

STUDY_TYPE_MAP = {
    "randomized controlled trials": "Randomized Controlled Trial[pt]",
    "randomized controlled trial": "Randomized Controlled Trial[pt]",
    "clinical trials": "Clinical Trial[pt]",
    "clinical trial": "Clinical Trial[pt]",
    "rct": "Randomized Controlled Trial[pt]",
    "systematic reviews": "Systematic Review[pt]",
    "systematic review": "Systematic Review[pt]",
    "meta-analyses": "Meta-Analysis[pt]",
    "meta-analysis": "Meta-Analysis[pt]",
    "case reports": "Case Reports[pt]",
    "case report": "Case Reports[pt]",
    "observational studies": "Observational Study[pt]",
    "observational study": "Observational Study[pt]",
    "reviews": "Review[pt]",
    "review": "Review[pt]",
}

AGE_GROUP_MAP = {
    "elderly": "Aged[mesh]",
    "older adult": "Aged[mesh]",
    "older adults": "Aged[mesh]",
    "geriatric": "Aged[mesh]",
    "pediatric": "Child[mesh]",
    "paediatric": "Child[mesh]",
    "children": "Child[mesh]",
    "infant": "Infant[mesh]",
    "neonate": "Infant, Newborn[mesh]",
    "adolescent": "Adolescent[mesh]",
    "adolescents": "Adolescent[mesh]",
    "adult": "Adult[mesh]",
    "adults": "Adult[mesh]",
}


def deterministic_parse(query: str):
    """
    Parses natural language input into clean search terms without over-constraining.
    Removes conversational fluff and avoids forcing [tiab] AND clauses that break recall.
    """
    q_lower = query.lower()
    detected = {"study_type": None, "age_group": None}

    for phrase in STUDY_TYPE_MAP:
        if phrase in q_lower:
            detected["study_type"] = phrase
            break

    for phrase in AGE_GROUP_MAP:
        if phrase in q_lower:
            detected["age_group"] = phrase
            break

    # Strip special characters
    cleaned = re.sub(r"[^a-z0-9\s\-]", " ", q_lower)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Extended stopwords to remove natural language query fluff
    stopwords = {
        "find", "for", "in", "the", "a", "an", "of", "with", "and", "on", 
        "newly", "published", "evaluating", "evaluates", "study", "studies", 
        "trial", "trials", "paper", "papers", "article", "articles", "show", "showing"
    }

    words = [w for w in cleaned.split() if w not in stopwords and len(w) > 1]
    
    # Construct a space-separated unconstrained query (lets NCBI ATM handle field matching)
    # This prevents missing papers when abstracts use synonyms like "cognitive impairment" instead of "decline"
    topic_clause = " ".join(words) if words else query

    return topic_clause, detected, topic_clause


_last_request_time = {"t": 0.0}


def _throttle(has_api_key: bool):
    min_interval = 1.0 / 10 if has_api_key else 1.0 / 3
    elapsed = time.time() - _last_request_time["t"]
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time["t"] = time.time()


def _ncbi_params(email: str, api_key: str, extra: dict):
    params = dict(extra)
    params["tool"] = TOOL_NAME
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key
    return params


def esearch(term: str, retmax: int = 1000, sort: str = "relevance", email: str = "", api_key: str = ""):
    params = _ncbi_params(
        email, api_key,
        {"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax, "sort": sort},
    )
    _throttle(has_api_key=bool(api_key))
    resp = requests.get(ESEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def efetch_records(pmids: list, email: str = "", api_key: str = ""):
    if not pmids:
        return []
    params = _ncbi_params(
        email, api_key,
        {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"},
    )
    _throttle(has_api_key=bool(api_key))
    resp = requests.get(EFETCH_URL, params=params, timeout=20)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    records = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "(no title)"

        abstract_parts = article.findall(".//Abstract/AbstractText")
        abstract = " ".join("".join(p.itertext()).strip() for p in abstract_parts) if abstract_parts else ""

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else ""

        year_el = article.find(".//JournalIssue/PubDate/Year")
        if year_el is None:
            year_el = article.find(".//JournalIssue/PubDate/MedlineDate")
        year = year_el.text if year_el is not None else ""

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last = author.find("LastName")
            fore = author.find("ForeName")
            if last is not None:
                name = last.text
                if fore is not None:
                    name = f"{fore.text} {name}"
                authors.append(name)

        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "authors": ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else ""),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return records