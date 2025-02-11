# /Applications/Blender.app/Contents/Resources/4.2/python/bin/python3.11 -m pip install --upgrade pip
# /Applications/Blender.app/Contents/Resources/4.2/python/bin/python3.11 -m pip install openpyxl

import bpy
import bl_ui
import csv
import os.path
import openpyxl
from openpyxl import Workbook

from bpy.props import (StringProperty,
                       PointerProperty,
                       )

from bpy.types import PropertyGroup

"""
Blender Add-on by Lorenz Mammen, GitHub @LorMam

exports all tracks from all movieclips
API reference to the track coordinates: bpy.data.movieclips[0].tracking.tracks[0].markers[0].co

Help on developing a UI for a Blender Add-on
    https://blender.stackexchange.com/questions/57306/how-to-create-a-custom-ui
    https://blender.stackexchange.com/questions/26898/how-to-create-a-folder-file-dialog
"""

bl_info = {
    "name": "Export Video Track as CSV",
    "description": "Exports tracked xy coordinates to a .csv file",
    "author": "Lorenz Mammen",
    "blender": (2, 80, 0),
    "doc_url": "",
    "location": "Clip Editor > Side Panel: Track",
    "category": "Tracking",
    "support": "COMMUNITY"
}


def export_all_tracks_from_clip(clip, folder):
    x = clip.size[0]
    y = clip.size[1]
    
    for t in clip.tracking.tracks:
        export_track(t, clip.name, x, y, folder)


def export_track(track, clip_name, clip_x_size, clip_y_size, folder):
    path = os.path.join(folder, clip_name.split(".")[0] + "_" + track.name.replace(".", "_") + ".xlsx")
    
    # Create a new Excel workbook and select the active worksheet
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Track Data"

    # Write headers
    sheet.append(["Frame", "X", "Y"])

    # Write data
    for m in track.markers:
        x_coord = m.co[0] * clip_x_size
        y_coord = m.co[1] * clip_y_size
        sheet.append([m.frame, x_coord, y_coord])

    # Save the workbook
    workbook.save(path)
    print(f"Excel file saved to: {path}")
            

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
    bl_label = "Export Track"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        self.layout.label(text="Select Folder")

        scn = context.scene
        col = layout.column(align=True)
        col.prop(scn.my_tool, "path", text="")
        clip = context.space_data.clip
        active_track = clip.tracking.tracks.active
        
        self.layout.label(text="")

        layout.operator("wm.export_all_tracks")

        if not active_track:
            layout.active = False
            layout.label(text="No active track")
            return
        
        layout.operator("wm.export_active_track")
        

class ExportAllTracks(bpy.types.Operator):
    bl_idname = "wm.export_all_tracks"
    bl_label = "Export all Tracks"

    def execute(self, context):

        if not os.path.isdir(context.scene.my_tool.path):
            self.report({"ERROR"}, "No proper folder selected.")
            return {'CANCELLED'}
        
        clip = context.space_data.clip
        export_all_tracks_from_clip(clip, context.scene.my_tool.path)
        self.report({'INFO'}, "Exported all tracks from current clip: " + str(clip.name))

        return {'FINISHED'}


class ExportCurrentTrack(bpy.types.Operator):
    bl_idname = "wm.export_active_track"
    bl_label = "Export selected Track only"

    def execute(self, context):

        if not os.path.isdir(context.scene.my_tool.path):
            self.report({"ERROR"}, "No proper folder selected.")
            return {'CANCELLED'}

        clip = context.space_data.clip
        active_track = clip.tracking.tracks.active
           
        x, y = clip.size

        export_track(active_track, clip.name, x, y, context.scene.my_tool.path)
        self.report({'INFO'}, "Exported track " + str(active_track.name))

        return {'FINISHED'}


classes = (
    CLIP_PT_export_track,
    ExportAllTracks,
    ExportCurrentTrack,
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