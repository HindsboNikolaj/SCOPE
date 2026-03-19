#!/usr/bin/env python3
"""
Zoom-In + Agent Dashboard Add-on for SCOPE
"""

bl_info = {
    "name":        "Zoom-In + Agent Dashboard",
    "author":      "Armada AI",
    "version":     (2, 0),
    "blender":     (3, 0, 0),
    "location":    "View3D > Sidebar > Zoom",
    "description": "SCOPE: Zoom into detections, take camera screenshots & chat with agent",
    "category":    "3D View",
}

import bpy, os, sys, math, time, textwrap
from datetime import datetime

try:
    from dotenv import load_dotenv, find_dotenv
except ImportError:
    load_dotenv = None
    find_dotenv = None

try:
    from PIL import Image
except ImportError:
    Image = None

# load .env
if find_dotenv is not None:
    dotenv = find_dotenv()
    if dotenv:
        load_dotenv(dotenv)

from scope.blender import helper_funcs as bhf
from scope.tools import blender_tools
from scope.agent.client import AgentClient

# ─── Global agent client ─────────────────────────────────────────────────────

_agent_client = None


def _ensure_agent(context):
    """Lazily create or recreate the AgentClient from current scene settings."""
    global _agent_client
    if _agent_client is not None:
        return _agent_client
    model_id = (context.scene.agent_model_id or "").strip() or None
    try:
        _agent_client = AgentClient(model_id=model_id)
        print(f"[SCOPE] Agent client loaded (model={_agent_client.model_id})")
    except Exception as e:
        _agent_client = None
        print(f"[SCOPE] Failed to create AgentClient: {e}")
    return _agent_client


def update_agent_model(self, context):
    """Called when the user changes the model id string property."""
    global _agent_client
    _agent_client = None          # force re-creation on next ask
    _ensure_agent(context)


# ─── Operators ────────────────────────────────────────────────────────────────

class ZOOMIN_OT_by_class(bpy.types.Operator):
    bl_idname = "zoom.zoom_in_by_class"
    bl_label  = "Zoom In"
    bl_description = "Detect your class and zoom camera to it"

    def execute(self, context):
        if Image is None:
            self.report({'ERROR'}, "Pillow (PIL) is not installed")
            return {'CANCELLED'}

        sc  = context.scene
        cls = sc.zoom_class_name.strip()
        out = bpy.path.abspath(sc.zoom_output_dir)
        os.makedirs(out, exist_ok=True)
        start = time.time()

        # Snapshot
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out, f"{ts}_raw.png")
        bhf.screenshot_camera_view(path)
        print(f"[SCOPE] Screenshot: {time.time()-start:.3f}s")

        # Detect via VLM
        det_start = time.time()
        vlm = blender_tools._VLM
        if vlm is None:
            self.report({'WARNING'}, "VLM not available — set VLM_* env vars")
            return {'CANCELLED'}
        objs = vlm.detect(Image.open(path).convert('RGB'), cls).get('objects', [])
        print(f"[SCOPE] Detection: {time.time()-det_start:.3f}s, found {len(objs)}")
        if not objs:
            self.report({'WARNING'}, f"No '{cls}' found")
            return {'CANCELLED'}

        # Zoom
        zoom_start = time.time()
        w, h = Image.open(path).size
        x0, y0 = objs[0]['x_min'] * w, objs[0]['y_min'] * h
        x1, y1 = objs[0]['x_max'] * w, objs[0]['y_max'] * h
        u0, v0 = x0 / w, y0 / h
        u1, v1 = x1 / w, y1 / h
        applied, (dp, dt) = bhf.corrected_persp_area_zoom_fov(
            context.scene.camera, u0, v0, u1, v1, margin=sc.zoom_margin)
        print(f"[SCOPE] Zoom: {time.time()-zoom_start:.3f}s")

        total = time.time() - start
        self.report({'INFO'}, f"Zoom x{applied:.2f} (total {total:.2f}s)")
        return {'FINISHED'}


class ZOOMIN_OT_take_screenshot(bpy.types.Operator):
    bl_idname = "zoom.take_screenshot"
    bl_label  = "Take Screenshot"
    bl_description = "Capture a fast camera-view screenshot"

    def execute(self, context):
        out = bpy.path.abspath(context.scene.zoom_output_dir)
        os.makedirs(out, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out, f"{ts}_snapshot.png")
        start = time.time()
        try:
            bhf.screenshot_camera_view(path)
            self.report({'INFO'}, f"Saved {path} ({time.time()-start:.2f}s)")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Screenshot failed: {e}")
            return {'CANCELLED'}


class ZOOMIN_OT_ask_agent(bpy.types.Operator):
    bl_idname = "zoom.ask_agent"
    bl_label  = "Ask Agent"
    bl_description = "Send prompt to agent and display reply with timings"

    def execute(self, context):
        sc = context.scene
        client = _ensure_agent(context)
        if client is None:
            self.report({'WARNING'}, "Agent client not loaded — check env vars")
            return {'CANCELLED'}
        prompt = sc.agent_prompt.strip()
        if not prompt:
            self.report({'WARNING'}, "Prompt empty")
            return {'CANCELLED'}
        try:
            reply, _msgs, times, _tree = client.ask(prompt, reset_history=True)
        except Exception as e:
            self.report({'ERROR'}, f"Agent error: {e}")
            return {'CANCELLED'}

        sc.agent_response      = reply
        sc.agent_timing_total  = times.get('total', 0.0)
        sc.agent_timing_llm    = times.get('llm', 0.0)
        sc.agent_timing_vlm    = times.get('vlm', 0.0)
        sc.agent_timing_script = times.get('script', 0.0)
        return {'FINISHED'}


# ─── Panel ────────────────────────────────────────────────────────────────────

class ZOOMIN_PT_panel(bpy.types.Panel):
    bl_label       = "Zoom & Agent"
    bl_idname      = "ZOOMIN_PT_panel"
    bl_space_type  = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category    = 'Zoom'

    def draw(self, context):
        l  = self.layout
        sc = context.scene

        # Zoom Controls
        l.label(text="Zoom Settings:")
        l.prop(sc, 'zoom_class_name', text="Class")
        l.prop(sc, 'zoom_output_dir', text="Out Dir")
        l.prop(sc, 'zoom_margin',     text="Margin")
        l.prop(sc, 'zoom_max_zoom',   text="Max Zoom")
        row = l.row(align=True)
        row.operator('zoom.zoom_in_by_class', icon='VIEWZOOM')
        row.operator('zoom.take_screenshot',  icon='RENDER_STILL')

        # Camera Info
        l.separator()
        l.label(text="Camera:")
        cam = sc.camera
        if cam:
            cd = cam.data
            l.prop(cd, 'type', text="Type")
            if cd.type == 'PERSP':
                l.prop(cd, 'angle', text="FOV")
            else:
                l.prop(cd, 'ortho_scale', text="Ortho Scale")
        else:
            l.label(text="No camera set", icon='ERROR')

        # Agent Chat
        l.separator()
        l.label(text="Agent Chat:")
        l.prop(sc, 'agent_model_id', text="Model")

        # Prompt Input
        l.label(text="Prompt:")
        row = l.row()
        row.scale_y = 1.2
        row.prop(sc, 'agent_prompt', text="", emboss=True)
        # Wrapped preview of full prompt
        if sc.agent_prompt:
            for ln in textwrap.wrap(sc.agent_prompt, width=50):
                l.label(text=ln, icon='BLANK1')

        l.operator('zoom.ask_agent', icon='QUESTION')

        # Full Reply Display
        if sc.agent_response:
            l.separator()
            l.label(text="Reply:")
            for ln in textwrap.wrap(sc.agent_response, width=50):
                l.label(text=ln)

        # Timings
        if sc.agent_timing_total > 0:
            l.separator()
            l.label(text="Timing Breakdown:")
            l.label(text=f"Total:  {sc.agent_timing_total:.2f}s")
            l.label(text=f"LLM:    {sc.agent_timing_llm:.2f}s")
            l.label(text=f"VLM:    {sc.agent_timing_vlm:.2f}s")
            l.label(text=f"Script: {sc.agent_timing_script:.2f}s")


# ─── Registration ─────────────────────────────────────────────────────────────

_CLASSES = (
    ZOOMIN_OT_by_class,
    ZOOMIN_OT_take_screenshot,
    ZOOMIN_OT_ask_agent,
    ZOOMIN_PT_panel,
)

_PROPS = (
    'zoom_class_name', 'zoom_output_dir', 'zoom_margin', 'zoom_max_zoom',
    'agent_model_id', 'agent_prompt', 'agent_response',
    'agent_timing_total', 'agent_timing_llm',
    'agent_timing_vlm', 'agent_timing_script',
)


def register():
    sc = bpy.types.Scene

    # Zoom Props
    sc.zoom_class_name = bpy.props.StringProperty(
        name="Class", default="fire hydrant")
    sc.zoom_output_dir = bpy.props.StringProperty(
        name="Out Dir", subtype='DIR_PATH', default="//images/")
    sc.zoom_margin = bpy.props.FloatProperty(
        name="Margin", default=1.02, min=1.0)
    sc.zoom_max_zoom = bpy.props.FloatProperty(
        name="Max Zoom", default=6.0, min=1.0)

    # Agent Props
    sc.agent_model_id = bpy.props.StringProperty(
        name="Model ID",
        default=os.getenv("AGENT_MODEL_ID", ""),
        description="Model identifier (env: AGENT_MODEL_ID)",
        update=update_agent_model,
    )
    sc.agent_prompt = bpy.props.StringProperty(
        name="Prompt", default="Ask me...")
    sc.agent_response = bpy.props.StringProperty(
        name="Reply", default="", options={'SKIP_SAVE'})
    sc.agent_timing_total  = bpy.props.FloatProperty(name="Total Time",  default=0.0)
    sc.agent_timing_llm    = bpy.props.FloatProperty(name="LLM Time",    default=0.0)
    sc.agent_timing_vlm    = bpy.props.FloatProperty(name="VLM Time",    default=0.0)
    sc.agent_timing_script = bpy.props.FloatProperty(name="Script Time", default=0.0)

    for cls in _CLASSES:
        bpy.utils.register_class(cls)

    # Preload default client
    _ensure_agent(bpy.context)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    for p in _PROPS:
        if hasattr(bpy.types.Scene, p):
            delattr(bpy.types.Scene, p)


register()
