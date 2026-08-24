# Food Inspection AI — 10-Minute Manager Presentation

**Presenter:** Internship project review  
**Audience:** Manager and technical stakeholders  
**Recommended duration:** 9–10 minutes, followed by questions

## Presentation objective

The presentation should demonstrate that the project is not only an object detector. It is an inspection system: a YOLO-based visual detection stage identifies food items, a vision-language model evaluates visible quality dimensions, and a reporting layer converts the result into an understandable operational recommendation. The main message is that the architecture has now been made consistent across the web application and the live-camera runner, while the experimental analysis distinguishes measured evidence from future work.

## Cover

### Food Inspection AI

**From visual detection to farmer-readable quality inspection**  
Internship project review

**Speaking commentary:** “This project aims to support food-quality inspection by combining fast object detection with a second visual reasoning stage. Rather than returning only a class label and a confidence score, the system produces a structured inspection result, explains visible quality concerns in plain language, and preserves the result as a report.”

## Slide 1 — The operational problem

**Core message:** Detection alone is not an inspection decision.

| Challenge | Consequence |
| --- | --- |
| Manual inspection is repetitive and subjective | Results can vary between inspectors or across time |
| A detector only answers “what is present?” | It does not directly explain freshness, bruising, mold, or action |
| Real-time use has latency constraints | A slow second-stage model can interrupt the workflow |

**Speaking commentary:** “The system is designed around the difference between detection and inspection. Detection tells us that an item is present. Inspection must additionally describe its visible condition and recommend what to do next. This distinction is the reason the backend separates the YOLO result from the VLM quality assessment.”

## Slide 2 — What I built

**Core message:** One typed pipeline serves multiple interfaces.

```text
Image or camera frame
        ↓
YOLO11m object detection
        ↓
Crop each detected item
        ↓
VLM quality assessment using class-specific metrics
        ↓
Schema validation and safe fallback
        ↓
Farmer commentary + inspection report
```

**Speaking commentary:** “The important architectural decision is that the frontend and the additional live runner now call the same `run_inspection` function. This avoids having one behavior in the web application and a different behavior in the camera script.”

## Slide 3 — Evidence from the first experiment

**Core message:** The training run was a real, resumable YOLO experiment rather than an untracked one-off run.

| Item | Recorded setting or evidence |
| --- | --- |
| Detector | Ultralytics YOLO11m initialized from `yolo11m.pt` |
| Dataset | LVIS fruits-and-vegetables YOLO-format dataset, configured for 63 classes |
| Input size | 640 × 640 |
| Target | 80 epochs |
| Hardware record | Tesla T4, approximately 14.56 GiB reported GPU memory |
| Checkpointing | `best.pt`, `last.pt`, and periodic epoch checkpoints copied to Drive |
| Current supplied log | Progress recorded through epoch 62 of 80 |

**Speaking commentary:** “The notebook explicitly distinguishes a fresh training run, an exact interrupted resume, and a new fine-tuning stage. That distinction is important for a scientific comparison because a resumed run and a new fine-tuning stage are not the same experiment.”

## Slide 4 — Current training signal

**Core message:** Training loss improved, but validation performance plateaued at a modest level in the supplied snapshot.

| Checkpoint | Box loss | Class loss | DFL loss | Precision | Recall | mAP50 | mAP50–95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Epoch 50 | 0.9910 | 0.7819 | 0.9957 | 0.338 | 0.260 | 0.239 | 0.172 |
| Epoch 60 | 0.9496 | 0.7038 | 0.9805 | 0.365 | 0.265 | 0.244 | 0.177 |
| Epoch 62 | 0.9454 | 0.6993 | 0.9788 | 0.378 | 0.259 | 0.239 | 0.173 |

**Speaking commentary:** “Between epochs 50 and 62, the losses continued to decline, but mAP oscillated rather than improving monotonically. The best values in this supplied window are approximately 0.244 mAP50 and 0.177 mAP50–95 at epoch 60. I would therefore avoid claiming final convergence until the 80-epoch log and an independent test evaluation are available.”

## Slide 5 — Why YOLO11m was fine-tuned

**Core message:** This was transfer learning, not training YOLO from scratch.

| Concept | Meaning in this project |
| --- | --- |
| Training from scratch | Randomly initialized detector weights learn visual features from the project dataset alone |
| Fine-tuning / transfer learning | A pretrained detector is adapted to the food dataset |
| This experiment | YOLO11m was initialized from `yolo11m.pt`, so the correct statement is “fine-tuned a pretrained YOLO11m detector” |
| CNN/MobileNet from scratch | Correct only if those models began with random initialization and learned all relevant weights from the project data |

**Speaking commentary:** “The distinction is not about whether I ran `model.train`. It is about the initialization. Because YOLO11m started from pretrained weights, I should present this as supervised transfer learning or fine-tuning.”

## Slide 6 — The new shared backend design

**Core message:** Minimal restructuring improved consistency without rewriting the application.

| Change | Benefit |
| --- | --- |
| Added `backend/live_inspection.py` | Camera/video execution uses the same `run_inspection` path as the API |
| Added `backend/reporting.py` | Farmer metrics are computed from the typed result, not from a second inference implementation |
| Added Gemma and NVIDIA adapters | Provider experiments do not require changing the pipeline |
| Added `GET /reports/{id}/farmer-report` | A saved report can be presented in operational terms |
| Added `backend/vlm_benchmark.py` | Equal-image comparisons can be repeated and exported as CSV |

**Speaking commentary:** “I deliberately added narrow modules instead of replacing the existing backend. This keeps the project understandable and lowers the risk of breaking the current frontend.”

## Slide 7 — VLM benchmark: what must be measured

**Core message:** “Better” must be demonstrated against a fixed evaluation set.

| Dimension | Measurement |
| --- | --- |
| Structured-output reliability | Valid JSON / total responses |
| Quality agreement | Agreement on status, defect, and recommended action against a human-labeled reference |
| Latency | Median and 95th-percentile response time per crop |
| Operational usefulness | Explanation completeness and farmer readability |
| Cost and quota | Requests, tokens, provider limits, and data-use terms |

**Speaking commentary:** “NVIDIA may be better in my observations, but the defensible conclusion requires identical crops, identical prompt, deterministic settings, repeated measurements, and human labels. The benchmark script records these measurements without changing the production pipeline.”

## Slide 8 — Preliminary recommendation

**Core message:** Use provider selection as an experiment variable, not as business logic.

Google’s official hosted Gemma documentation lists Gemma 4 image-capable models, including `gemma-4-31b-it` and `gemma-4-26b-a4b-it`, and documents image inputs through the Gemini API [1]. NVIDIA’s NIM documentation describes an OpenAI-compatible VLM API and exposes health, model, and metrics endpoints [2]. Google’s free tier is suitable for prototyping but includes limited model access and states that free-tier content may be used to improve products [3].

**Speaking commentary:** “For the internship prototype I would keep both adapters. If the controlled benchmark confirms that NVIDIA gives better structured-output reliability or latency on our crops, I would use NVIDIA as the primary experimental provider and keep Gemma as a reproducible comparison baseline. I would not claim superiority from intuition alone.”

## Slide 9 — Limitations and next experiment

**Core message:** The next improvement is evaluation quality, not blindly increasing epochs.

The current evidence does not yet establish per-class AP, a clean held-out test result, VLM agreement with an inspector, or real camera throughput. The next experiment should freeze the dataset split, record class distributions, finish the 80-epoch run, evaluate `best.pt` on an independent test split, and benchmark both VLMs on the same labeled crop set.

**Speaking commentary:** “This makes the next stage scientifically useful. It will tell us whether the detector generalizes, whether the VLM recommendation is reliable, and whether the end-to-end system is fast enough for the intended workflow.”

## Slide 10 — Closing

**Core message:** The project now connects model evidence to an operational inspection workflow.

**Speaking commentary:** “The contribution is an end-to-end, explainable prototype: a trained food detector, a provider-independent quality-inspection stage, a shared live and web execution path, and reports that translate model outputs into actions a farmer or inspector can understand. The immediate next step is controlled evaluation and calibration, not a large rewrite.”

## Questions to anticipate

| Manager question | Recommended answer |
| --- | --- |
| “Did you train YOLO from scratch?” | “No. The notebook initializes YOLO11m from pretrained `yolo11m.pt`; I fine-tuned it on the food dataset.” |
| “Why use a VLM after YOLO?” | “YOLO localizes and classifies objects. The VLM evaluates visible quality dimensions and generates an explanation.” |
| “Is this really real time?” | “The live runner processes camera frames through the exact shared pipeline. End-to-end real-time throughput still needs to be measured because VLM latency depends on detections and provider response time.” |
| “Why is mAP not higher?” | “The supplied snapshot is incomplete and shows modest validation performance. I would inspect label quality, class imbalance, small-object scale, and per-class AP before changing the architecture.” |
| “Which VLM is better?” | “I have a reproducible benchmark path. The final recommendation should be based on latency, valid structured outputs, agreement with human labels, and quota/data-use constraints.” |

## References

[1]: https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api "Run Gemma with the Gemini API — Google AI for Developers"
[2]: https://docs.nvidia.com/nim/vision-language-models/latest/api-reference.html "NVIDIA NIM for Vision Language Models — API Reference"
[3]: https://ai.google.dev/gemini-api/docs/pricing "Gemini Developer API pricing — Google AI for Developers"
