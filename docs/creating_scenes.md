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
7. **`eval_category`** -- One of: `counting`, `descriptor`, `location`, `ocr`,
   `single_call`, `multi_cmd`, `multi_reason`, `comparative`.
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
