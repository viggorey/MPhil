import bpy
import bl_ui
import os
import pyperclip  # Library to handle clipboard operations

from bpy.props import (StringProperty,
                       PointerProperty,
                       )

from bpy.types import PropertyGroup

bl_info = {
    "name": "Export Track Coordinates to Clipboard",
    "description": "Copies tracked XY coordinates to clipboard",
    "author": "Modified by AI Assistant",
    "blender": (2, 80, 0),
    "doc_url": "",
    "location": "Clip Editor > Side Panel: Track",
    "category": "Tracking",
    "support": "COMMUNITY"
}


def copy_all_tracks_from_clip(clip):
    x = clip.size[0]
    y = clip.size[1]
    
    clipboard_content = []
    for t in clip.tracking.tracks:
        clipboard_content.append(copy_track_to_clipboard(t, clip.name, x, y))
    
    # Combine all track data into one clipboard string
    combined_data = "\n\n".join(clipboard_content)
    pyperclip.copy(combined_data)
    print("All track data copied to clipboard.")


def copy_track_to_clipboard(track, clip_name, clip_x_size, clip_y_size):
    # Prepare data for clipboard (coordinates only)
    clipboard_lines = []
    for m in track.markers:
        x_coord = m.co[0] * clip_x_size
        y_coord = m.co[1] * clip_y_size
        clipboard_lines.append(f"{x_coord:.2f}, {y_coord:.2f}")
    
    # Combine the coordinates into a single string
    clipboard_data = "\n".join(clipboard_lines)
    
    # Copy to clipboard
    pyperclip.copy(clipboard_data)
    print(f"Track data for {track.name} copied.")
    return clipboard_data


class SaveDirectory(PropertyGroup):
    path: StringProperty(
        name="",
        description="Path to Directory",
        default="",
        maxlen=1024,
        subtype='DIR_PATH')


class CLIP_PT_export_track(bl_ui.space_clip.CLIP_PT_tracking_panel, bpy.types.Panel):
    bl_space_type = 'CLIP_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Track"
    bl_label = "Copy Track to Clipboard"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        clip = context.space_data.clip
        active_track = clip.tracking.tracks.active

        layout.label(text="Copy Coordinates")
        
        layout.operator("wm.copy_all_tracks")

        if not active_track:
            layout.active = False
            layout.label(text="No active track")
            return

        layout.operator("wm.copy_active_track")
        

class CopyAllTracks(bpy.types.Operator):
    bl_idname = "wm.copy_all_tracks"
    bl_label = "Copy All Tracks"

    def execute(self, context):
        clip = context.space_data.clip
        copy_all_tracks_from_clip(clip)
        self.report({'INFO'}, f"Copied all tracks from clip: {clip.name}")
        return {'FINISHED'}


class CopyCurrentTrack(bpy.types.Operator):
    bl_idname = "wm.copy_active_track"
    bl_label = "Copy Selected Track Only"

    def execute(self, context):
        clip = context.space_data.clip
        active_track = clip.tracking.tracks.active
        x, y = clip.size

        clipboard_data = copy_track_to_clipboard(active_track, clip.name, x, y)
        pyperclip.copy(clipboard_data)
        self.report({'INFO'}, f"Copied track {active_track.name} to clipboard.")
        return {'FINISHED'}


classes = (
    CLIP_PT_export_track,
    CopyAllTracks,
    CopyCurrentTrack,
    SaveDirectory
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
        
    bpy.types.Scene.my_tool = PointerProperty(type=SaveDirectory)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
    
    del bpy.types.Scene.my_tool


if __name__ == "__main__":
    register()