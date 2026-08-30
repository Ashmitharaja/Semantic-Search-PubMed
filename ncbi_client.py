"""
ncbi_client.py — Enhanced NCBI Entrez E-utilities Client
"""

import re
import xml.etree.ElementTree as ET
import requests

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

AGE_MESH_MAP = {
    "Child: birth-18 years": '"Child"[Mesh] OR "Infant"[Mesh] OR "Adolescent"[Mesh]',
    "Newborn: birth-1 month": '"Infant, Newborn"[Mesh]',
    "Infant: birth-23 months": '"Infant"[Mesh]',
    "Infant: 1-23 months": '"Infant"[Mesh]',
    "Preschool Child: 2-5 years": '"Child, Preschool"[Mesh]',
    "Child: 6-12 years": '"Child"[Mesh]',
    "Adolescent: 13-18 years": '"Adolescent"[Mesh]',
    "Adult: 19+ years": '"Adult"[Mesh]',
    "Young Adult: 19-24 years": '"Young Adult"[Mesh]',
    "Adult: 19-44 years": '"Adult"[Mesh]',
    "Middle Aged + Aged: 45+ years": '"Middle Aged"[Mesh] OR "Aged"[Mesh]',
    "Middle Aged: 45-64 years": '"Middle Aged"[Mesh]',
    "Aged: 65+ years": '"Aged"[Mesh]',
    "80 and over: 80+ years": '"Aged, 80 and over"[Mesh]',
}

STOP_WORDS = {
    "a", "an", "the", "newly", "published", "evaluating", "evaluation", 
    "study", "trial", "trials", "for", "in", "with", "and", "or", "of", 
    "to", "on", "at", "by", "from", "about", "is", "are", "was", "were"
}

def deterministic_parse(query: str) -> tuple[str, str, str]:
    """
    Extracts significant keywords for PubMed keyword searching when given
    a long, natural language query prompt.
    """
    clean_text = re.sub(r'[^\w\s-]', ' ', query)
    tokens = clean_text.split()
    
    # Filter out conversational stop words while preserving medical terms & numbers
    keywords = [t for t in tokens if t.lower() not in STOP_WORDS]
    
    # Fallback to original text if filtering yields empty list
    parsed_query = " ".join(keywords) if keywords else query
    return parsed_query, "", ""

def build_pubmed_filter_query(
    base_query: str, 
    ages: list, 
    species: list, 
    sex: list, 
    languages: list,
    text_avail: list
) -> str:
    """Appends PubMed tags to enforce filter constraints."""
    query_parts = [f"({base_query})"]

    if ages:
        age_terms = [AGE_MESH_MAP[a] for a in ages if a in AGE_MESH_MAP]
        if age_terms:
            query_parts.append(f"({' OR '.join(age_terms)})")

    if species:
        spec_terms = []
        if "Humans" in species:
            spec_terms.append('"Humans"[Mesh]')
        if "Other Animals" in species:
            spec_terms.append('"Animals"[Mesh] NOT "Humans"[Mesh]')
        if spec_terms:
            query_parts.append(f"({' OR '.join(spec_terms)})")

    if sex:
        sex_terms = [f'"{s}"[Mesh]' for s in sex]
        query_parts.append(f"({' OR '.join(sex_terms)})")

    if languages:
        lang_terms = [f'"{l}"[Language]' for l in languages]
        query_parts.append(f"({' OR '.join(lang_terms)})")

    if "Free full text" in text_avail:
        query_parts.append("ffrft[Filter]")
    elif "Full text" in text_avail:
        query_parts.append("full text[Filter]")

    return " AND ".join(query_parts)

def esearch(query: str, retmax: int = 30, sort: str = "relevance", email: str = "", api_key: str = "") -> list:
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "sort": sort,
        "retmode": "json",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(ESEARCH_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])

def efetch_records(pmids: list, email: str = "", api_key: str = "") -> list:
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(EFETCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    
    root = ET.fromstring(resp.content)
    records = []

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        title_el = article.find(".//ArticleTitle")
        title = title_el.text if title_el is not None else "No title"

        abstract_texts = article.findall(".//AbstractText")
        abstract = " ".join([a.text for a in abstract_texts if a.text])

        journal_el = article.find(".//Journal/Title")
        journal = journal_el.text if journal_el is not None else "Unknown Journal"

        year_el = article.find(".//JournalIssue/PubDate/Year")
        year = year_el.text if year_el is not None else "2026"

        authors = []
        for author in article.findall(".//Author"):
            last = author.findtext("LastName", "")
            fore = author.findtext("ForeName", "")
            if last:
                authors.append(f"{last} {fore}".strip())

        pub_types = [pt.text for pt in article.findall(".//PublicationType") if pt.text]
        mesh_terms = [m.findtext("DescriptorName") for m in article.findall(".//MeshHeading") if m.findtext("DescriptorName")]
        langs = [l.text for l in article.findall(".//Language") if l.text]

        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "journal": journal,
            "year": year,
            "authors": ", ".join(authors),
            "publication_types": pub_types,
            "mesh_terms": mesh_terms,
            "languages": langs,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        })

    return records