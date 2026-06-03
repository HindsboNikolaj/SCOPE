"""
SCOPE — Hugging Face Space landing page.

A static, paper-style landing for the SCOPE project. Drop into a Streamlit
or Gradio Space — Gradio is lighter for static content.

Deploy:
  1. Create Space at huggingface.co/new-space (SDK: gradio)
  2. Copy this file as app.py
  3. Push.
"""
import gradio as gr

ABSTRACT = """
**SCOPE** is a modular multimodal agentic system for natural-language PTZ camera control.
A Small Language Model planner orchestrates a fixed action space — a set of *skills*
(camera-control and perception workflows) exposed through an OpenAI-compatible JSON tool
schema identical on Blender simulation and a physical AXIS PTZ. A Vision-Language Model
handles perception as a callable skill.

This Space hosts the project landing page. The benchmark dataset lives at
[HindsboNikolaj/scope-benchmark](https://huggingface.co/datasets/HindsboNikolaj/scope-benchmark)
and the source code at [github.com/HindsboNikolaj/SCOPE](https://github.com/HindsboNikolaj/SCOPE).
"""

LINKS = """
### Resources
- 📄 [Paper (HRI '26, DOI 10.1145/3757279.3785641)](https://doi.org/10.1145/3757279.3785641)
- 💻 [Source code (GitHub)](https://github.com/HindsboNikolaj/SCOPE)
- 📊 [Benchmark dataset (HF Datasets)](https://huggingface.co/datasets/HindsboNikolaj/scope-benchmark)
- 🎥 [Physical demo video](https://github.com/HindsboNikolaj/SCOPE/raw/main/assets/scope-demo.mp4)

### Authors
- **Nikolaj Hindsbo** — Armada AI
- **Sina Ehsani** — Armada AI
- **Pragyana Mishra** — Armada AI
"""

CITATION = """
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
"""

with gr.Blocks(title="SCOPE — Camera Agent at the Edge") as demo:
    gr.Markdown("# SCOPE: A Real-Time Natural Language Camera Agent at the Edge")
    gr.Markdown("*HRI '26 · Armada AI*")
    gr.Markdown(ABSTRACT)
    gr.Markdown(LINKS)
    gr.Markdown("---")
    gr.Markdown("## Citation")
    gr.Markdown(CITATION)

if __name__ == "__main__":
    demo.launch()
