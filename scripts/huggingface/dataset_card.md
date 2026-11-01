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

- **scenes/**: the four Blender `.blend` worlds, with textures packed into the files:
  - `book-nook/book-nook.blend`: a modern city corner with a bookshop; textures packed, none missing
  - `city-street/city-street.blend`: a one-way street with hotel frontage; textures packed, none missing
  - `postwar-city/postwar-city.blend`: a bombed-out European street; 46 of 71 textures absent
  - `whitechapel/whitechapel.blend`: a rain-soaked Victorian yard; 5 of 193 textures absent, among them its sky
- **scope_v2.csv**: dataset v2, 536 tasks, the dataset SCOPE 0.2.0 ships and scores, with `CHANGES_v2.md` (every change from v1 and why) and **archive/** (dataset v1, the paper's 536 tasks, and the lineage from the 541-row file SCOPE 0.1.0 shipped). These are copies for convenience. The authoritative CSVs are in the [GitHub repository](https://github.com/HindsboNikolaj/SCOPE/tree/main/benchmark), and SCOPE's scripts read those, not these.

This dataset is mainly the download for the four 3D worlds; `scripts/03_download_scenes.sh` fetches only `scenes/`.

## Task categories

Eight evaluation categories (`eval_category`): `single_call`, `multi_step_command`,
`multi_step_reasoning`, `counting`, `descriptor`, `ocr_identification`, `location_spatial`,
`comparative_relational`.

## Usage

Run the benchmark from the [main repository](https://github.com/HindsboNikolaj/SCOPE); its
download script fetches the scenes from here:

```bash
git clone https://github.com/HindsboNikolaj/SCOPE
cd SCOPE
bash scripts/01_install.sh
bash scripts/03_download_scenes.sh
bash scripts/run_eval_pipeline.sh
```

## Texture state

The original authors' files referenced many textures by absolute paths on their own machines
(`D:/SL/...`, `E:/New folder/...`). The `.blend` files here were re-packed for release, so most
textures travel inside them. Counted by opening each downloaded scene (image datablocks neither
packed nor found on disk):

| Scene | Texture refs | Packed | Missing |
|---|---|---|---|
| book-nook | 387 | 385 | 0 |
| city-street | 127 | 126 | 0 |
| postwar-city | 71 | 21 | 46 (surface textures from mixed sources) |
| whitechapel | 193 | 187 | 5 (including the sky map, its only light) |

[`docs/MISSING_TEXTURES.md`](https://github.com/HindsboNikolaj/SCOPE/blob/main/docs/MISSING_TEXTURES.md)
in the main repo lists the texture references by category.

The missing textures do not change benchmark results: SCOPE captures the viewport with studio
lighting, which ignores the scene world, so they only show (as magenta) when a scene is rendered
as authored.

## License

The dataset is released under CC-BY-NC-4.0 for research use. Scene assets keep their original licences.

## Credits

- city-street: "City Street" by amaryase, BlendSwap #30474, CC-BY.

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
