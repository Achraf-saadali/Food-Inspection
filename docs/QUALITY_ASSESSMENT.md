# Structured Quality Assessment System

This document explains the structured, ingredient-aware quality assessment framework implemented in the VLM reasoning stage.

## Overview

The generic `quality` field has been replaced with a structured representation that adapts to the specific fruit or vegetable being inspected. This allows the system to evaluate relevant visual characteristics (e.g., "ripeness" for tomatoes vs. "browning" for bananas) rather than using a one-size-fits-all metric.

## The New Schema

The `QualityAssessment` object now contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `status` | `enum` | `ok`, `defect`, or `uncertain`. |
| `overall_quality_score` | `float` | A normalized score (0.0 to 1.0) where 1.0 is perfect quality. |
| `quality_metrics` | `dict` | A dictionary of ingredient-specific metrics and their scores. |
| `defects` | `list[str]` | A list of specific defects detected by the VLM. |
| `explanation` | `string` | A natural language explanation of the visual evidence from the VLM. |
| `commentary` | `string` | A short farmer-facing paragraph derived locally from score, defects, and action. |
| `required_action` | `enum` | `none`, `flag_for_review`, or `remove`. |

### Example JSON Output (Banana)

```json
{
  "status": "defect",
  "overall_quality_score": 0.45,
  "quality_metrics": {
    "ripeness": 0.9,
    "browning": 0.7,
    "bruising": 0.5,
    "mold": 0.0,
    "freshness": 0.6
  },
  "defects": ["browning", "bruising"],
  "explanation": "Significant browning and visible bruising on the mid-section of the peel.",
  "commentary": "Banana. Quality score: 0.45/1.00. Observed issues: browning, bruising. Recommended action: remove from the saleable batch.",
  "required_action": "remove"
}
```

## Quality Profiles

Relevant metrics are controlled by **Quality Profiles** defined in `backend/quality_profiles.py`.

### How it works:
1. **YOLO** detects the class (e.g., "banana").
2. The **VLM Reasoning** stage fetches the profile for "banana".
3. The **Prompt Generator** injects the relevant metrics into the VLM prompt.
4. The **VLM** performs visual assessment against those specific dimensions.
5. The **Parser** validates the response against the schema.

### Adding a New Fruit/Vegetable
To add support for a new ingredient:
1. Open `backend/quality_profiles.py`.
2. Add a new entry to the `QUALITY_PROFILES` dictionary.
3. Define the list of visual metrics you want the VLM to evaluate.

```python
QUALITY_PROFILES["strawberry"] = [
    "ripeness", "bruising", "mold", "leakage", "shriveling", "freshness"
]
```

## Handling Unknown Classes
If the YOLO model detects a class that is not configured in `backend/quality_profiles.py`, the system falls back to a set of `COMMON_METRICS`:
- Ripeness
- Bruising
- Mold
- Discoloration
- Freshness

This ensures the system remains robust and extensible as the detection model evolves.
