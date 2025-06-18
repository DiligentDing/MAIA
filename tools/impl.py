# tools/impl.py
from __future__ import annotations
import os 
import json, requests, pymysql
from typing import List, Dict, Any, Tuple
import re  
# ───────────────────────────────────────────
# 1) PubMed
# ───────────────────────────────────────────
PUBMED = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def pubmed_search(term: str, retmax: int = 10) -> List[str]:
    params = {"db": "pubmed", "term": term, "retmode": "json",
              "sort": "pub+date", "retmax": retmax}
    r = requests.get(f"{PUBMED}/esearch.fcgi", params=params, timeout=20)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


# ───────────────────────────────────────────
# 2) ClinicalTrials.gov v2
# ───────────────────────────────────────────

import requests, urllib.parse
from typing import Dict, List, Optional

CTGOV = "https://clinicaltrials.gov/api/v2/studies"

# --- utils ------------------------------------------------------------
ALLOWED_STATUS = {
    "NOT_YET_RECRUITING", "RECRUITING", "ACTIVE",
    "COMPLETED", "SUSPENDED", "TERMINATED", "WITHDRAWN"
}

def _build_params(
    *,
    conditions: Optional[str] = None,
    startDateFrom: Optional[str] = None,
    overallStatus: Optional[str] = None,
    interventions_name: Optional[str] = None,
    locations_country: Optional[str] = None,
    page_size: int = 100,
    page_token: Optional[str] = None
) -> dict[str, str]:

    if not any([conditions, interventions_name, locations_country, overallStatus, startDateFrom]):
        raise ValueError("At least one filter criterion must be supplied")

    params: dict[str, str] = {
        "pageSize": str(page_size),
        "countTotal": "false",
        "markupFormat": "markdown",
    }

    if conditions:
        params["query.cond"] = conditions

    if interventions_name:
        intr = " AND ".join(
            seg.strip() for seg in re.split(r"[;,]", interventions_name) if seg.strip()
        )
        params["query.intr"] = intr

    if locations_country:
        params["query.locn"] = f'"{locations_country}"'

    if overallStatus:
        params["filter.overallStatus"] = overallStatus.upper()

    if startDateFrom:
        params["filter.advanced"] = f"AREA[StartDate]RANGE[{startDateFrom},MAX]"

    if page_token:
        params["pageToken"] = page_token

    return params


def ctgov_search(
    conditions: str | None = None,
    startDateFrom: str | None = None,
    overallStatus: str | None = None,
    interventions_name: str | None = None,
    locations_country: str | None = None,
    page_size: int = 100
) -> list[str]:
    """Return a list of NCT IDs that meet the given criteria."""
    studies, token = [], None
    while True:
        params = _build_params(
            conditions=conditions,
            startDateFrom=startDateFrom,
            overallStatus=overallStatus,
            interventions_name=interventions_name,
            locations_country=locations_country,
            page_size=page_size,
            page_token=token
        )
        r = requests.get(CTGOV, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        studies.extend(data.get("studies", []))
        token = data.get("nextPageToken")
        if not token:
            break

    return [
        s["protocolSection"]["identificationModule"]["nctId"]
        for s in studies
    ]

# ───────────────────────────────────────────
# 3) OpenTargets
# ───────────────────────────────────────────
OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"

def _ot_query(query: str) -> Dict:
    r = requests.post(OT_URL, json={"query": query}, timeout=20)
    r.raise_for_status()
    return r.json()["data"]

def ot_associated_diseases(target_id: str, min_score: float = 0.5) -> List[Dict]:
    q = f"""
    {{ target(ensemblId: "{target_id}") {{
        associatedDiseases {{ rows {{ disease {{id name}} score }} }} }} }}
    """
    rows = _ot_query(q)["target"]["associatedDiseases"]["rows"]
    return [r for r in rows if r["score"] >= min_score]

def ot_tractability(target_id: str, value: bool = True) -> List[Dict]:
    q = f"""
    {{ target(ensemblId: "{target_id}") {{
        tractability {{ modality label value }} }} }}
    """
    rows = _ot_query(q)["target"]["tractability"]
    return [r for r in rows if r["value"] is value]

def ot_safety(symbol: str, event: str) -> Dict:
    search_q = f'''
    {{
      search(queryString: "{symbol}") {{
        hits {{
          id
          entity
          description
        }}
      }}
    }}
    '''
    search_data = _ot_query(search_q)["search"]["hits"]
    target_id = None
    for hit in search_data:
        if hit["entity"] == "target":
            target_id = hit["id"]
            break
    if not target_id:
        return {}
    
    safety_q = f'''
    {{
      target(ensemblId: "{target_id}") {{
        safetyLiabilities {{
          event
          biosamples {{ tissueLabel tissueId }}
          effects {{ dosing direction }}
        }}
      }}
    }}
    '''
    rows = _ot_query(safety_q)["target"]["safetyLiabilities"]
    for r in rows:
        if r["event"].lower() == event.lower():
            return {"biosamples": r["biosamples"], "effects": r["effects"]}
    return {}


# ───────────────────────────────────────────
# 4) UMLS MySQL
# ───────────────────────────────────────────
DB_CFG = dict(
    host=os.getenv("UMLS_DB_HOST", "localhost"),
    port=int(os.getenv("UMLS_DB_PORT", "3306")),
    user=os.getenv("UMLS_DB_USER"),
    password=os.getenv("UMLS_DB_PASSWORD"),
    database=os.getenv("UMLS_DB_NAME", "umls"),
    cursorclass=pymysql.cursors.DictCursor,
    autocommit=True,
)
_conn = pymysql.connect(**DB_CFG)

def umls_concept_lookup(name: str) -> str:
    with _conn.cursor() as cur:
        cur.execute("SELECT cui FROM concepts WHERE STR = %s LIMIT 1", (name,))
        row = cur.fetchone()
        return row["cui"] if row else ""

def umls_get_related(from_cui: str, rela: str) -> List[str]:
    with _conn.cursor() as cur:
        cur.execute(
            "SELECT cui1 FROM MRREL WHERE cui2=%s AND rela=%s",
            (from_cui, rela))
        return [row["cui1"] for row in cur.fetchall()]

def umls_cui_to_name(cui: str) -> str:
    """
    输入CUI，返回英文名，优先PF/PT类型，否则任选一个。
    """
    sql = """
        SELECT STR, TTY
        FROM   MRCONSO
        WHERE  LAT='ENG' AND CUI=%s
    """
    with _conn.cursor() as cur:
        cur.execute(sql, (cui,))
        names = []
        pfpt = None
        for row in cur.fetchall():
            names.append(row["STR"])
            if row["TTY"] in ("PF", "PT"):
                pfpt = row["STR"]
        if pfpt:
            return pfpt
        if names:
            return names[0]
        return ""

# ───────────────────────────────────────────
# 5) Guideline path
# ───────────────────────────────────────────
def oncology_path_query(nodes: List[str]) -> List[str]:
    return nodes

