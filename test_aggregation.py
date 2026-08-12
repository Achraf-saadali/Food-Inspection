import sys
import types

import numpy as np

sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))
from inspection_pipeline import run_inspection
from schemas import InspectionStatus


class Value:
    def __init__(self, value):
        self.value = value

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def tolist(self):
        return self.value


class Box:
    def __init__(self, cls, conf, xyxy):
        self.cls = [Value(cls)]
        self.conf = [Value(conf)]
        self.xyxy = [Value(xyxy)]


class Result:
    def __init__(self):
        self.boxes = [
            Box(0, 0.95, [1, 1, 5, 5]),
            Box(1, 0.90, [6, 1, 10, 5]),
        ]


class Model:
    names = {0: "apple", 1: "banana"}

    def __call__(self, image):
        return [Result()]


class Backend:
    name = "test"

    def analyze(self, crop, label, confidence):
        from schemas import QualityAssessment, RequiredAction
        return QualityAssessment(
            status=InspectionStatus.OK,
            overall_quality_score=0.9,
            quality_metrics={"freshness": 0.9},
            defects=[],
            explanation=f"{label} looks acceptable.",
            required_action=RequiredAction.NONE,
            vlm_backend=self.name,
        )


result = run_inspection(np.zeros((20, 20, 3), dtype=np.uint8), Model(), 1, "test.jpg", Backend())
assert [item.detection.crop_id for item in result.items] == ["CROP_001", "CROP_002"]
assert len(result.commentary) > 0
assert result.items[0].quality.status == InspectionStatus.OK
assert result.items[1].quality.status == InspectionStatus.OK
print(result.model_dump(mode="json"))

skipped = run_inspection(np.zeros((20, 20, 3), dtype=np.uint8), Model(), 2, "test.jpg", None)
assert all(item.quality.status == InspectionStatus.SKIPPED for item in skipped.items)
assert "not fully evaluated" in skipped.commentary
print(skipped.model_dump(mode="json"))
print("aggregation regression checks passed")
