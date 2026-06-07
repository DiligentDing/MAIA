# tools/schema.py
# JSON schema（Python 字典形式），可直接传给 chat.completions.create tools 参数
pubmed_schema = {
    "name": "pubmed.search",
    "description": "Search PubMed and return a list of PMIDs.",
    "parameters": {
        "type": "object",
        "properties": {
            "term":    {"type": "string"},
            "retmax":  {"type": "integer", "default": 10}
        },
        "required": ["term"]
    }
}

ctgov_schema = {
    "name": "ctgov.search",
    "description": "Search ClinicalTrials.gov v2 and return matching NCT IDs.",
    "parameters": {
        "type": "object",
        "properties": {
            "conditions": {
                "type": "string",
                "description": "Condition or disease, e.g. 'Multiple Myeloma'"
            },
            "startDateFrom": {
                "type": "string",
                "format": "date",
                "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                "description": "Earliest study start date (YYYY-MM-DD)"
            },
            "overallStatus": {
                "type": "string",
                "enum": [
                    "NOT_YET_RECRUITING", "RECRUITING", "ACTIVE",
                    "COMPLETED", "SUSPENDED", "TERMINATED", "WITHDRAWN"
                ],
                "description": "Overall recruitment status"
            },
            "interventions_name": {
                "type": "string",
                "description": "Comma-separated intervention names"
            },
            "locations_country": {
                "type": "string",
                "description": "Exact country name"
            },
            "page_size": {
                "type": "integer",
                "minimum": 1, "maximum": 100,
                "default": 100,
                "description": "Page size (1–100)"
            }
        },
    "additionalProperties": False
    }
}

# Backward-compatible alias (previous name used underscore style)
ctgov_schema_legacy = {
    **ctgov_schema,
    "name": "ctgov_search",
}




ot_search_schema = {
    "name": "opentargets.associated_diseases",
    "description": "Return diseases associated with a target (with score cutoff).",
    "parameters": {
        "type": "object",
        "properties": {
            "target_id": {"type": "string"},
            "min_score": {"type": "number", "default": 0.5}
        },
        "required": ["target_id"]
    }
}

# Legacy alias to preserve older tool name
ot_search_schema_legacy = {
    **ot_search_schema,
    "name": "opentargets.search",
}

ot_tract_schema = {
    "name": "opentargets.tractability",
    "description": "Return tractability modalities with value==True/False.",
    "parameters": {
        "type": "object",
        "properties": {
            "target_id": {"type": "string"},
            "value":     {"type": "boolean", "default": True}
        },
        "required": ["target_id"]
    }
}

ot_safety_schema = {
    "name": "opentargets.safety",
    "description": "Return biosamples & effects for a given safety event.",
    "parameters": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "event":  {"type": "string"}
        },
        "required": ["symbol", "event"]
    }
}

umls_lookup_schema = {
    "name": "umls.concept_lookup",
    "description": "Return the CUI for a given concept name.",
    "parameters": {
        "type": "object",
        "properties": {
            "name": {"type": "string"}
        },
        "required": ["name"]
    }
}

umls_related_schema = {
    "name": "umls.get_related",
    "description": "Return CUIs related by a specified rela.",
    "parameters": {
        "type": "object",
        "properties": {
            "from_cui": {"type": "string"},
            "rela":     {"type": "string"}
        },
        "required": ["from_cui", "rela"]
    }
}

umls_cui_to_name_schema = {
    "name": "umls.cui_to_name",
    "description": "Return the English name (STR, PF/PT preferred) for a given CUI.",
    "parameters": {
        "type": "object",
        "properties": {
            "cui": {"type": "string"}
        },
        "required": ["cui"]
    }
}


onco_path_schema = {
    "name": "oncology.path_query",
    "description": "Return guideline nodes supplied by params.",
    "parameters": {
        "type": "object",
        "properties": {
            "nodes": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["nodes"]
    }
}

ALL_SCHEMAS = [
    pubmed_schema, ctgov_schema, ctgov_schema_legacy,
    ot_search_schema, ot_search_schema_legacy, ot_tract_schema, ot_safety_schema,
    umls_lookup_schema, umls_related_schema, umls_cui_to_name_schema,
    onco_path_schema
]
