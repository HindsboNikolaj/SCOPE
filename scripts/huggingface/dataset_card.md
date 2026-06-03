---
license: cc-by-nc-4.0
language:
  - en
tags:
  - robotics
  - vision-language
  - human-robot-interaction
  - ptz-camera
  - benchmark
  - blender
size_categories:
  - n<1K
task_categories:
  - visual-question-answering
  - image-classification
  - object-detection
pretty_name: SCOPE — A Real-Time Natural Language Camera Agent Benchmark
---

# SCOPE Benchmark

The SCOPE benchmark accompanies the HRI '26 paper [*SCOPE: A Real-Time Natural Language Camera Agent at the Edge*](https://doi.org/10.1145/3757279.3785641). It evaluates modular multimodal agentic systems controlling PTZ cameras in simulated and physical settings.

## Contents

- **scenes/** — 4 Blender `.blend` scenes (with packed textures where the original assets were available):
  - `whitechapel/whitechapel.blend` — French Quarter exterior, ~95% textured
  - `book-nook/book-nook.blend` — small interior, geometry-only (original SL/MySims textures lost)
  - `city-street/city-street.blend` — urban scene, geometry-only (re-download CC0 textures via `scripts/fetch_textures/`)
  - `postwar-city/postwar-city.blend` — partial textures, ~35% textured
- **scope_541.csv** — 541-question benchmark with columns: `question_id`, `file_location`, `question`, `expected_answer`, `eval_category`, `difficulty`, `multi_step_mode`, `required_tools_policy`, `expected_tool_order_json`, `evaluation_notes`.

## Task categories

8 categories (see paper §4):
1. Object identification
2. Object counting
3. Spatial reasoning
4. Multi-step planning
5. Camera control validation
6. Perception robustness
7. Error recovery
8. Tool-use correctness

## Usage

```bash
pip install huggingface_hub
huggingface-cli download HindsboNikolaj/scope-benchmark \
    --repo-type dataset \
    --local-dir benchmark/
```

Then run the benchmark from the [main repository](https://github.com/HindsboNikolaj/SCOPE):

```bash
git clone https://github.com/HindsboNikolaj/SCOPE
cd SCOPE
bash scripts/01_install.sh
bash scripts/run_eval_pipeline.sh
```

## Texture state

Two of the four scenes (`book-nook`, `city-street`) shipped without textures from their original authors — they referenced absolute Windows paths (`D:/SL/...`, `E:/New folder/...`) that were never bundled. The packed `.blend` files in this dataset are honest about what's available:

| Scene | Texture refs | Packed | Missing |
|---|---|---|---|
| whitechapel | 193 | 188 (97%) | 5 (paid HDR addon) |
| book-nook | 385 | 0 | 296 (SL/MySims rips) |
| city-street | 127 | 0 | 126 (CC0 — re-downloadable) |
| postwar-city | 71 | 25 (35%) | 46 (mixed sources) |

To restore textures for the partially-bundled scenes, see [`docs/MISSING_TEXTURES.md`](https://github.com/HindsboNikolaj/SCOPE/blob/main/docs/MISSING_TEXTURES.md) in the main repo — there's an automated AmbientCG downloader and a manifest of where to find the rest.

The benchmark questions are designed to be answerable from geometry alone for most rows, so even untextured scenes produce meaningful results.

## License

The dataset is released under CC-BY-NC-4.0 for research use. Individual scene assets retain their original licenses — see the per-scene README in each subfolder for source attribution.

## Citation

```bibtex
@inproceedings{hindsbo2026scope,
  title={SCOPE: A Real-Time Natural Language Camera Agent at the Edge},
  author={Hindsbo, Nikolaj and Ehsani, Sina and Mishra, Pragyana},
  booktitle={Proceedings of the ACM/IEEE International Conference on Human-Robot Interaction (HRI '26)},
  year={2026},
  publisher={ACM},
  doi={10.1145/3757279.3785641}
}
```
