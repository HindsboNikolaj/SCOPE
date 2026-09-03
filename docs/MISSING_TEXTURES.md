# Missing Textures Manifest

After publishing the SCOPE benchmark scenes, **408 unique textures** referenced inside the .blend files are not bundled in the Google Drive / Hugging Face download. This is because the original scene authors left absolute paths pointing at their own machines (`D:/`, `E:/`, etc.) and the textures were never packed into the .blend or shipped alongside.

This manifest categorizes them by source so they can be reacquired.

## Quick stats

| Scene | Total refs | Already packed | Still missing |
|---|---|---|---|
| whitechapel | 193 | 187 | 5 |
| book-nook | 387 | 385 | 0 |
| city-street | 127 | 126 | 0 |
| postwar-city | 71 | 21 | 46 |

**Updated after checking the published files.** The book-nook and city-street rows previously
read 0 packed with 296 and 61 missing. Those numbers described the scenes before they were
re-packed for release. The files on Hugging Face have everything embedded, so nothing needs
recovering for either one, and in particular the 296 Second Life texture rips listed below
for book-nook are already in the download. Measured by opening each scene from
`scripts/03_download_scenes.sh` and counting image datablocks that are neither packed nor
resolvable on disk.

**Which asset is missing matters more than how many.** postwar-city is missing 46 files and
renders perfectly usably: they are ordinary surface textures, so the damage is a magenta door
and a few flat panels in an otherwise complete scene. whitechapel is missing 5 and every
Cycles frame is unusable, because one of the five is `207-free-hdri-skies-com.hdr`, the world
environment map, and it is the scene's only light source. A missing world costs more than
forty missing surfaces.

This does not affect benchmark results. SCOPE captures the viewport with studio lighting,
which ignores the scene world, so the magenta only appears when somebody renders the scene as
authored.

## By source

| Source | Count | Recoverable? | Action |
|---|---|---|---|
| AmbientCG (CC0) | 17 | Yes — bulk-downloadable | `bash scripts/fetch_textures/download_ambientcg.sh` |
| AmbientCG (legacy naming) | 6 | Yes — search by name | Same script, falls back to website search |
| textures.com / poliigon.com | 61 | Mostly yes (free + paid mix) | Login required; see `scripts/fetch_textures/textures_com_names.txt` |
| 3dtextures.me / cgbookcase.com | 14 | Yes — manual download | Search by filename on each site |
| Stock photo (Freepik etc.) | 1 | Maybe (paywalled) | Search by filename ID |
| Pro Lighting Skies addon | 4 | Paid — $29 on Blender Market | Buy at blendermarket.com/products/pro-lighting-skies |
| free-hdri-skies.com (defunct) | 1 | Substitute on polyhaven.com | Pick equivalent cloudy/morning HDR |
| Scene-author custom textures | 7 | Maybe — check Sketchfab listing | Search Sketchfab / BlendSwap for the scene name |
| Second Life / MySims asset rips | 292 | Maybe — check original Sketchfab page | Re-download `.blend` export from Sketchfab; textures may be bundled there |
| Unidentified | 5 | Manual | Eyeball each name |

## Per-scene detail

### `book_nook`

- **sl_or_sketchfab_uuid**: 292 files
- **unknown**: 4 files

### `city_street`

- **textures_com**: 28 files
- **ambientcg**: 17 files
- **3dtextures_community**: 14 files
- **freepik_or_stock**: 1 files
- **unknown**: 1 files

### `postwar_city`

- **textures_com**: 33 files
- **scene_author_custom**: 6 files
- **ambientcg_old**: 5 files
- **blender_market_pro_lighting**: 2 files

### `whitechapel`

- **blender_market_pro_lighting**: 2 files
- **hdri_skies_dot_com**: 1 files
- **ambientcg_old**: 1 files
- **scene_author_custom**: 1 files

## How to use this manifest

1. **Auto-download CC0 textures from AmbientCG**:
   ```bash
   bash scripts/fetch_textures/download_ambientcg.sh
   ```
   This pulls every AmbientCG asset by code into a staging directory, then copies the right resolution files into each scene's `textures/` folder.

2. **textures.com / poliigon.com** — needs an account. The list of names is in `scripts/fetch_textures/textures_com_names.txt`. After downloading, drop the files into the appropriate scene's `textures/` folder and re-pack with `blender --background --python scripts/repack_scene.py -- <scene.blend>`.

3. **Pro Lighting Skies addon** — if you own the addon, the HDRs are in `~/Library/Application Support/Blender/<version>/scripts/addons/pro_lighting_skies_demo/hdris/`. Copy the referenced ones into the scene folder. If you don't own it, substitute with a free Poly Haven HDR of similar tone.

4. **Second Life / MySims rips (Book Nook)** — the cleanest path is to find the original Sketchfab listing for 'The Book Nook' and re-download the `.blend` export which typically bundles the textures. The 296 UUID filenames in `sl_rip_uuids.txt` are useless without the original source listing.

5. **Re-pack any updated scene**:
   ```bash
   blender --background --python scripts/repack_scene.py -- <path/to/scene.blend>
   ```
   This relinks any newly-present textures and writes `<scene>.packed.blend` with everything embedded.