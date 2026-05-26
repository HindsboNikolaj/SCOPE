# Publishing checklist for HindsboNikolaj/SCOPE

What's automated vs. manual after the professionalization pass.

## Automated (already done — `refactor/professionalize-2026`)

- [x] Repo restructure (src/, paper/, prompts/, docs/, scripts/)
- [x] README slim to ~140 lines + reword as modular multimodal agentic system
- [x] AGENT_INSTRUCTIONS.md for Claude Code / Codex setup
- [x] `.env.example` + `LICENSE` + `.gitignore`
- [x] Scripts split into numbered pipeline stages (01_install → 12_metrics)
- [x] System prompts surfaced to `prompts/`
- [x] PEP 621 `pyproject.toml`
- [x] Scene paths renamed: `whitechapel/`, `book-nook/`, `city-street/`, `postwar-city/`
- [x] Benchmark CSV `file_location` column updated for all 541 rows
- [x] `03_download_scenes.sh` switched to Hugging Face Hub (Google Drive as fallback)
- [x] AmbientCG auto-downloader: `scripts/fetch_textures/download_ambientcg.sh`
- [x] Texture manifest with source attribution: `docs/MISSING_TEXTURES.md`
- [x] HF dataset card + Space landing page drafts: `scripts/huggingface/`
- [x] Install-verified end-to-end on real Blender + Ollama (5/5 smoke passed)

## Done (automated, after token handoff)

### Hugging Face — published

- **Dataset**: https://huggingface.co/datasets/HindsboNikolaj/scope-benchmark
- **Space**: https://huggingface.co/spaces/HindsboNikolaj/scope
- **Collection**: https://huggingface.co/collections/HindsboNikolaj/scope-hri-26-6a1626e0b8e9b9205c09fffc

## Still manual (your turn)

### 0. ⏳ arXiv preprint (pending)

The paper isn't on arXiv yet, which means there's no `huggingface.co/papers/<id>` page and the HF Collection can't include the paper as a first-class item. To fix:

1. Sign in at https://arxiv.org/login with `NikolajHindsbo@gmail.com` (Google SSO).
2. Go to https://arxiv.org/submit. License: arXiv non-exclusive (default).
3. Upload `paper/SCOPE_HRI26.pdf`.
4. Metadata (copy verbatim):
   - **Title**: `SCOPE: A Real-Time Natural Language Camera Agent at the Edge -- A Sim-to-Real Benchmark and Analysis of Open-Source Vision and Language Agents for PTZ Camera Tasks`
   - **Authors**: `Nikolaj Hindsbo and Sina Ehsani and Pragyana Mishra`
   - **Abstract**: page-1 abstract from the PDF
   - **Comments**: `Accepted at ACM/IEEE International Conference on Human-Robot Interaction (HRI '26), Edinburgh, Scotland, UK, March 16-19, 2026. 9 pages.`
   - **Journal-ref**: `Proceedings of the ACM/IEEE International Conference on Human-Robot Interaction (HRI 2026)`
   - **DOI**: `10.1145/3757279.3785641`
   - **ACM classes**: `I.2.9; I.2.10; I.4.8`
5. Categories: primary `cs.RO`, cross-list `cs.AI`, `cs.CV`.
6. Submit. Expected processing time: 1-2 business days.

**Endorsement note**: if first-time submitter, may hit "endorsement required for cs.RO" — fastest fix is co-author endorsement (Sina or Pragyana, if either has prior cs.* arXiv papers).

Once an arXiv ID is assigned (e.g. `2603.NNNNN`), the follow-up tasks (Collection paper item, README badge, Space link, citation update) are mechanical — just paste the ID into a Claude/Codex session.

### 1. ~~Hugging Face setup~~ — done above

1. **Create HF account / token** if you don't have one:
   - https://huggingface.co/join
   - Settings → Access Tokens → New (Write scope)
   - `pip install huggingface_hub && huggingface-cli login`

2. **Create the dataset repo**:
   - https://huggingface.co/new-dataset
   - Name: `scope-benchmark`
   - Owner: HindsboNikolaj
   - Visibility: Public
   - License: cc-by-nc-4.0

3. **Run the upload**:
   ```bash
   cd /tmp/scope-refactor
   bash scripts/huggingface/upload_to_hf.sh
   ```

4. **Create the Space (landing page)**:
   - https://huggingface.co/new-space
   - Name: `scope`
   - SDK: Gradio
   - Hardware: CPU basic (free tier is fine)
   - Visibility: Public
   - After created, `git clone` it locally and copy `scripts/huggingface/space_app.py` → `app.py` and `space_readme.md` → `README.md`. Push.

5. **Create the Collection** (groups dataset + Space + paper):
   - On your profile page, click Collections → New
   - Title: "SCOPE — HRI '26"
   - Add: dataset + Space + a link to the arXiv/DOI of the paper

### 2. Texture re-acquisition — variable time

The two unbundled scenes (`book-nook`, `city-street`) and partial scenes need work. See `docs/MISSING_TEXTURES.md` for the full breakdown.

**Easy wins (~10 min):**
```bash
# AmbientCG CC0 textures (17 + 6 = 23 files for city-street and postwar-city)
bash scripts/fetch_textures/download_ambientcg.sh
```

**Medium effort (~1 hour):**
- textures.com / poliigon.com — 61 files, need free account. Names in `scripts/fetch_textures/textures_com_names.txt`. Drop downloaded files into the appropriate scene's `textures/` folder.
- 3dtextures community — 14 files. Search each name on 3dtextures.me, cgbookcase.com, sharetextures.com.

**Hard (~1-2 hours):**
- **Book Nook** (296 SL/MySims UUIDs) — search Sketchfab for "The Book Nook" or "MySims Book Nook" model listings. The textures are bundled in the `.blend` export. Re-download and re-extract.
- **Pro Lighting Skies HDRs** (4 files) — paid Blender Market addon ($29). If you don't have it, substitute with free Poly Haven HDRs of similar tone (cloudy, morning).

After downloading any new textures, re-pack:
```bash
for s in whitechapel book-nook city-street postwar-city; do
    blender --background --python scripts/repack_scene.py -- benchmark/scenes/$s/$s.blend
done
```

### 3. Open the PR

When ready (independent of texture work — you can ship without):
```bash
gh auth switch -u HindsboNikolaj
cd /tmp/scope-refactor
gh pr create --title "Professionalize SCOPE repo for 2026" \
  --body "$(cat docs/PR_BODY.md)" \
  --base main --head refactor/professionalize-2026
```

