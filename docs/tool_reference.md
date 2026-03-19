# SCOPE Tool Reference

This document provides the complete API reference for all tools exposed by
SCOPE. These tools are defined in `scope/tools/schema.json` (OpenAI
function-calling format) and implemented in `scope/tools/blender_tools.py`.

The tool schema is identical between the Blender simulation and real PTZ
camera deployments. Only the underlying implementations differ.

---

## PTZ Tools

These tools manipulate the camera and do not require a VLM backend.

### ptz_adjust

Adjust pan, tilt, and zoom by numeric amounts (relative moves). Use this for
explicit numeric camera adjustments -- not for object-targeted framing (use
`zoom_bounding` for that).

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `pan_deg` | number | No | Pan adjustment in degrees. Positive = right, negative = left. |
| `tilt_deg` | number | No | Tilt adjustment in degrees. Positive = up, negative = down. |
| `zoom_percent` | number | No | Relative zoom as a percentage. `+50` zooms in 50%, `-50` zooms out 50%. |
| `zoom_factor` | number | No | Multiplicative zoom factor. `2.0` doubles the zoom, `0.5` halves it. |
| `zoom_full` | string | No | Set to `"in"` for maximum zoom or `"out"` for maximum wide angle. Enum: `in`, `out`. |

All parameters are optional. Omit any parameter to leave that axis unchanged.
Multiple parameters can be combined in a single call.

**Returns:**

```json
{
  "result": "PTZ adjusted",
  "path": "/path/to/screenshot_after_adjustment.png",
  "timings": { "vlm": 0.0, "script": 0.012 }
}
```

---

### go_to_preset

Move the camera to a named preset position and zoom level. The name must
match one of the presets returned by `get_presets`. If the preset is not
found, the camera remains unchanged.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | Yes | The preset name to navigate to. |

**Returns:**

```json
{
  "result": "Moved to preset 'Entrance'",
  "timings": { "vlm": 0.0, "script": 0.005 }
}
```

---

### home_action

Return the camera to its home position. If a preset named `"Home"` exists, it
is applied. Otherwise, the camera reverts to the position captured at the start
of the session.

**Parameters:** None.

**Returns:**

```json
{
  "result": "Returned to Home position (preset)",
  "timings": { "vlm": 0.0, "script": 0.003 }
}
```

---

### get_presets

Retrieve a list of all available camera preset names.

**Parameters:** None.

**Returns:**

```json
{
  "result": "Available presets: Home, Entrance, Parking Lot",
  "presets": ["Home", "Entrance", "Parking Lot"],
  "timings": { "vlm": 0.0, "script": 0.001 }
}
```

---

### take_image

Capture the current camera frame and save it to disk. Returns the file path
of the saved screenshot.

**Parameters:** None.

**Returns:**

```json
{
  "result": "/absolute/path/to/screenshot.png",
  "path": "/absolute/path/to/screenshot.png",
  "timings": { "vlm": 0.0, "script": 0.0 }
}
```

---

### track_object

Continuously track a specified object in the camera's view for a set duration.
In the Blender simulation this is a stub that returns immediately; on real
hardware it engages the PTZ tracking system.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `object_of_interest` | string | Yes | Description of the target to track (e.g., `"person wearing blue"`). |
| `duration` | integer | Yes | Length of time to track the object. |
| `unit` | string | Yes | Time unit for the duration. Enum: `seconds`, `minutes`, `hours`. |

**Returns:**

```json
{
  "result": "Tracked 'person wearing blue' for 30 seconds.",
  "timings": { "vlm": 0.0, "script": 0.001 }
}
```

---

## Perception Tools

These tools require a bound VLM backend. They capture a frame from the
Blender scene, send it to the VLM for inference, and return structured
results.

### count_pointing

Count the number of objects matching a descriptive query. The VLM's
`point` capability is used to locate instances, and the count is derived
from the number of returned points.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `instruction` | string | Yes | Description of items to count (e.g., `"red cars"`, `"people wearing hats"`). |
| `view_type` | string | No | `"current"` for the live frame (default), `"full"` for a 360-degree panorama sweep. |

**Returns:**

```json
{
  "result": "Counted 3 'red cars' in current view.",
  "count": 3,
  "timings": { "vlm": 0.45, "script": 0.08 }
}
```

When `view_type="full"`, the tool is a generator that yields intermediate
panorama capture progress before returning the final count.

---

### query_answer

Answer a natural-language question about the current scene using the VLM's
visual question answering (VQA) capability.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `instruction` | string | Yes | The question to ask about the scene (e.g., `"What color is the truck?"`). |
| `view_type` | string | No | `"current"` for the live frame (default), `"full"` for a 360-degree panorama. |

**Returns:**

```json
{
  "result": "The truck is red.",
  "answer": "The truck is red.",
  "timings": { "vlm": 0.32, "script": 0.06 }
}
```

---

### zoom_bounding

Zoom the camera to fill the frame with a specific described object. The VLM's
`detect` capability (falling back to `point` if no bounding boxes are returned)
is used to locate the target, and the Blender camera is adjusted to center and
zoom on the detected region.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `instruction` | string | Yes | What to zoom in on (e.g., `"the stop sign"`, `"person on the left"`). |

**Returns:**

```json
{
  "result": "Zoomed to target: the stop sign",
  "bbox": [0.35, 0.22, 0.68, 0.71],
  "path": "/path/to/postzoom_screenshot.png",
  "timings": { "vlm": 0.51, "script": 0.09 }
}
```

The `bbox` field contains normalized coordinates `[x1, y1, x2, y2]` in the
range `[0, 1]`.

---

## Return Value Convention

All tools return a dictionary with at least these keys:

| Key | Type | Description |
|-----|------|-------------|
| `result` | string | Human-readable result passed back to the SLM as the tool response. |
| `timings` | dict | Timing breakdown with `vlm` (VLM inference seconds) and `script` (Blender scripting seconds). |

Additional keys vary by tool (e.g., `count`, `answer`, `bbox`, `path`,
`presets`). The `AgentClient` extracts the `result` value and sends it to the
SLM as the tool response content; the other keys are available for evaluation
and logging.

---

## VLM Capabilities Required per Tool

| Tool | `caption` | `vqa` | `detect` | `point` |
|------|-----------|-------|----------|---------|
| `zoom_bounding` | | | Yes | Yes (fallback) |
| `count_pointing` | | | | Yes |
| `query_answer` | | Yes | | |
| `ptz_adjust` | | | | |
| `go_to_preset` | | | | |
| `home_action` | | | | |
| `get_presets` | | | | |
| `take_image` | | | | |
| `track_object` | | | | |
