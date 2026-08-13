"""
Quality profiles for different food ingredients.
Defines which visual quality dimensions are relevant for each class.
"""

from typing import Dict, List

# Common metrics applicable to most fruits and vegetables
COMMON_METRICS = ["ripeness", "bruising", "mold", "discoloration", "freshness"]

QUALITY_PROFILES: Dict[str, List[str]] = {
    "tomato": [
        "ripeness",
        "color_uniformity",
        "bruising",
        "mold",
        "cracking",
        "skin_damage",
        "shriveling",
        "visible_decay",
        "freshness",
    ],
    "banana": [
        "ripeness",
        "browning",
        "bruising",
        "black_spots",
        "mold",
        "peel_damage",
        "shriveling",
        "freshness",
    ],
    "apple": [
        "ripeness",
        "bruising",
        "discoloration",
        "rot",
        "mold",
        "skin_damage",
        "shriveling",
        "freshness",
    ],
    "potato": [
        "sprouting",
        "greening",
        "bruising",
        "cuts",
        "rot",
        "mold",
        "shriveling",
        "freshness",
    ],
    "strawberry": [
        "ripeness",
        "bruising",
        "mold",
        "leakage",
        "shriveling",
        "freshness",
    ],
    "onion": [
        "sprouting",
        "mold",
        "soft_spots",
        "skin_integrity",
        "freshness",
    ],
}

def get_quality_metrics(label: str) -> List[str]:
    """Get the list of relevant quality metrics for a given label."""
    # Normalize label (handle capitalization and synonyms if necessary)
    label_lower = label.lower()
    
    # Check for direct match
    if label_lower in QUALITY_PROFILES:
        return QUALITY_PROFILES[label_lower]
    
    # Handle cases like "Tomato" (capitalized) or "tomato" (lowercase) in model.names
    # and handle synonyms like "cucumber/cuke"
    parts = label_lower.replace("/", " ").split()
    for part in parts:
        if part in QUALITY_PROFILES:
            return QUALITY_PROFILES[part]
            
    return COMMON_METRICS
