# Creating New Blender Scenes for SCOPE

This guide explains how to create new Blender scenes for use with the SCOPE
benchmark. Scenes provide the 3D environments that the PTZ camera agent
observes and interacts with.

---

## Requirements

- **Blender 4.0+** (SCOPE uses `bpy` APIs available in Blender 4.x)
- A scene with at least one camera object set as the active scene camera
- One or more camera presets (recommended)

---

## Scene Setup

### 1. Create the Blender File

Open Blender and build or import your 3D environment. The scene can contain
any combination of meshes, materials, lighting, and textures. SCOPE captures
frames by rendering the active camera's viewport, so ensure the scene looks
correct from the camera's perspective.

### 2. Set the Active Camera

SCOPE reads the active scene camera via `bpy.context.scene.camera`. Ensure
exactly one camera is set as active:

1. Select your camera object.
2. In the Properties panel, go to **Scene Properties**.
3. Under **Scene > Camera**, assign your camera object.

Alternatively, select the camera and press `Ctrl+Numpad 0` to make it the
active camera.

### 3. Configure Camera Properties

The camera's focal length (`lens` in mm) controls the initial zoom level.
SCOPE's `ptz_adjust` tool modifies this value to simulate zoom. A starting
focal length between 35mm and 50mm works well for most scenes.

Set the render resolution in **Output Properties > Format**:
- The default SCOPE configuration uses 1920 x 1080.
- Match this to the `blender.render_resolution` value in your YAML config.

### 4. Create Camera Presets

Camera presets allow the agent to navigate to named viewpoints. SCOPE reads
presets from Blender's user preset system.

**To create a preset programmatically** (recommended for reproducibility):

```python
import bpy
from scope.blender.preset_helpers import create_preset

# Position camera at the desired viewpoint first
cam = bpy.context.scene.camera
cam.location = (5.0, -3.0, 2.5)
cam.rotation_euler = (1.2, 0.0, 0.8)
cam.data.lens = 50.0

# Save as a named preset
create_preset("Entrance")
```

**To create a preset manually in Blender:**

1. Position the camera at the desired viewpoint.
2. Open the Python console in Blender.
3. Run:
   ```python
   from scope.blender.preset_helpers import create_preset
   create_preset("MyPresetName")
   ```

Preset files are saved as `.py` scripts in Blender's user presets directory
under `presets/camera/`. Each preset stores the camera's location, rotation,
and focal length.

**Recommended presets:**
- Always include a preset named `"Home"` that represents the default overview
  position. The `home_action` tool will use this preset if it exists.
- Create presets for key viewpoints referenced in your benchmark tasks
  (e.g., `"Entrance"`, `"Parking Lot"`, `"Stage"`).

### 5. Verify Preset Round-Trip

After creating presets, verify that they apply correctly:

```python
from scope.blender.preset_helpers import list_presets, apply_preset

print(list_presets())       # Should include your preset names
apply_preset("Entrance")    # Camera should move to the saved position
```

---

## Scene Directory Structure

Place your `.blend` files under `benchmark/scenes/`. The directory structure
is flexible, but each scene file should be reachable from the `file_location`
column in the benchmark CSV.

```
benchmark/
  scenes/
    my-environment/
      MyScene.blend
    another-scene/
      AnotherScene.blend
  scope_536.csv
```

The `file_location` value in the CSV would be:
```
scenes/my-environment/MyScene.blend
```

---

## Adding Benchmark Tasks for Your Scene

Once your scene is ready, add evaluation tasks to the benchmark CSV. Each row
requires:

1. **`question_id`** -- A unique identifier (e.g., `Q_537`).
2. **`file_location`** -- Relative path to the `.blend` file from `benchmark/`.
3. **`preset_start`** -- The preset to apply before asking the question.
4. **`presets_available`** -- JSON array of all presets in the scene.
5. **`question`** -- The natural-language question for the agent.
6. **`expected_answer`** -- The ground-truth answer.
7. **`eval_category`** -- One of: `counting`, `descriptor`, `location_spatial`,
   `ocr_identification`, `single_call`, `multi_step_command`, `multi_step_reasoning`,
   `comparative_relational`. (These are the values the shipped CSV actually uses; five of them
   were previously listed here in an abbreviated form that no row uses.)
8. **`expected_tool_order_json`** -- The expected sequence of tool calls.

See [`benchmark/README.md`](../benchmark/README.md) for the complete column
schema.

### Example Row

```csv
Q_537,scenes/my-environment/MyScene.blend,Home,"[""Home"", ""Entrance""]",How many chairs are visible?,There are 4 chairs.,QA,counting,none,,"[{""name"": ""count_pointing"", ""args"": {""instruction"": ""chairs"", ""view_type"": ""current""}}]",TRUE,"{""count_pointing"": {""instruction"": ""chairs"", ""view_type"": ""current""}}",current,,Easy,chairs,v1.0,,,,,,,
```

---

## Scene Design Guidelines

### Object Placement

- Place objects at varying distances and angles from the camera to test the
  agent's ability to navigate and perceive.
- Include objects that overlap or occlude each other to test spatial reasoning.
- For counting tasks, use exact known quantities of target objects.

### Lighting

- Use consistent, well-distributed lighting. Avoid extreme darkness or
  blown-out highlights that might confuse the VLM.
- Scene lighting affects VLM accuracy significantly. Test your scene with
  the VLM before finalizing.

### Text and Signs (for OCR tasks)

- Place readable text on flat surfaces facing the camera.
- Use clear fonts at sufficient size. Text should be legible at the scene's
  render resolution.
- Avoid text at extreme angles where perspective distortion makes it
  unreadable.

### Scale and Units

- Use consistent real-world scale (1 Blender unit = 1 meter is standard).
- Camera movements via `ptz_adjust` use degrees for pan/tilt, so real-world
  scale affects how far the camera "sees" at a given focal length.

---

## Testing Your Scene

Before adding tasks to the benchmark, verify the scene works end-to-end:

```python
import bpy
from scope.agent import AgentClient
from scope.tools.blender_tools import take_image, get_presets

# Check presets
result = get_presets()
print(result["presets"])

# Capture a test frame
shot = take_image()
print(f"Screenshot saved: {shot['path']}")

# Run an interactive query
agent = AgentClient()
text, _, timings, _ = agent.ask("Describe what you see in the current view.")
print(text)
print(timings)
```

If the agent produces reasonable answers and the timing breakdown shows
non-zero VLM time, the scene is ready for benchmark integration.

---

## Full views, and why to capture them once

A question whose `answer_view` is `full` needs the agent to look around before answering. About
one row in six is like that, and every viewpoint in the shipped scenes carries some.

### The shape of a full view is not always a 360 sweep

A sweep is the obvious implementation and it is the wrong shape at some viewpoints. Which ones
is not guessable from the camera height, so measure two things for each candidate:

- **content** -- how much of the output is scene rather than flat viewport background
- **scene coverage** -- how many of the scene's mesh objects fall inside the captured frames

Coverage decides first, because a full-view question has to see the thing at all. Among the
candidates that see everything, keep the sweep if it is also a good picture; otherwise take the
cheapest candidate that sees comparably much.

Across the four shipped scenes this came out roughly half and half. From a camera above a
bounded scene, a single 90-degree frame contained every object a nine-frame sweep did, in a
tenth of the time. And levelling a pitched camera before sweeping, which is the right correction
at street level, was the *worst* option for an elevated one, because it aims the camera at empty
sky. `docs/FULL_VIEW.md` has the measurements.

Note also that scenes are often modelled to be seen from one side. Turning right round may show
nothing. That is a property of the asset, and it is a good reason to prefer a wide frame at that
viewpoint.

### Capture them once, not per question

Stitched panoramas are sensitive to frame overlap, step angle, camera pitch and capture
resolution, and those interact: seam quality stops improving somewhere around forty percent
overlap while the frame count keeps climbing, and a low-resolution capture stitches into
something blurry however good the geometry is.

Nothing in a scene moves between rows, so the same sweep is otherwise recomputed for an answer
that cannot differ. Capture once per viewpoint, look at the result, and store it:

```bash
SCOPE_PANO_CACHE=benchmark/panoramas SCOPE_PANO_CACHE_MODE=write \
  blender benchmark/scenes/<scene>/<scene>.blend \
    --python scripts/precapture_panoramas.py
python3 scripts/index_panorama_cache.py    # writes the metadata a lookup validates
```

`docs/PANORAMA_CACHE.md` explains how an entry is validated. The short version: entries are
named after the scene and preset, but served only when the live camera pose matches, so a stale
or mislabelled entry cannot answer the wrong question.

## After adding a scene

1. **Pack the textures** (`File > External Data > Pack Resources`). Unpacked textures use paths
   relative to wherever the author had them, and those do not survive being shared. Every
   missing texture in the shipped scenes is this.
2. **Record what is still missing** in `benchmark/expected_assets.json`, so absent files are
   reported as expected rather than as a broken download.
3. **Re-run `scripts/build_preset_index.py`**, which writes
   `benchmark/presets/presets_by_scene.json`. Presets are exported keyed by each scene's
   original asset filename, which does not match the name the CSV uses and cannot be derived
   from it; the index records the correspondence so tools do not have to guess.
4. **Add the scene's pictures to `docs/VISUAL_SMOKE_TEST.md`**, so the next person can check
   their setup against them by eye.

### Watch out for expensive materials

One of the shipped scenes costs hundreds of times more per frame than the others under software
rendering, and the cost does not fall with resolution, because it is per-material shader
compilation rather than per pixel. If a new scene is unexpectedly slow, time a capture at two
resolutions before trying to optimise. If the time barely changes, resolution is not the lever.
