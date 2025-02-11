import bpy
import bl_ui
import os.path
import openpyxl
from openpyxl import Workbook
from bpy.props import StringProperty, PointerProperty
from bpy.types import PropertyGroup

bl_info = {
    "name": "Export Video Track as CSV (pictures)",
    "description": "Exports tracked xy coordinates to a .csv file",
    "author": "Lorenz Mammen",
    "blender": (2, 80, 0),
    "category": "Tracking",
    "support": "COMMUNITY"
}

def export_all_tracks_from_clip(clip, folder):
    x = clip.size[0]
    y = clip.size[1]
    
    # Create a new Excel workbook and select the active worksheet
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Track Data"
    
    # Initialize a list to store track data for each frame
    all_tracks_data = []
    track_names = []
    
    for t in clip.tracking.tracks:
        track_data_x = []  # X coordinates for this track
        track_data_y = []  # Y coordinates for this track
        track_names.append(f"Track {len(track_names) + 1}")  # Track name as Track 1, Track 2, etc.
        
        # Process all markers, even if there is only one
        for m in t.markers:
            x_coord = m.co[0] * x
            y_coord = m.co[1] * y
            track_data_x.append(x_coord)  # Append X-coordinate for this frame
            track_data_y.append(y_coord)  # Append Y-coordinate for this frame
        
        # Ensure the length of the track matches the total number of frames
        while len(track_data_x) < clip.frame_duration:
            track_data_x.append(None)
            track_data_y.append(None)
        
        all_tracks_data.append(track_data_x)
        all_tracks_data.append(track_data_y)
    
    # Now transpose the data so each column represents a frame
    transposed_data = list(zip(*all_tracks_data))  # Transpose the data
    
    # Create headers for each frame
    headers = []
    num_frames = clip.frame_duration
    for frame in range(1, num_frames + 1):
        headers.append(f"Frame {frame}")
    
    # Write headers (without the rank column)
    sheet.append(["Track Name", "Coordinate Type"] + headers)  # Add Track Name header
    
    # Write each track's X and Y data as two rows (one for X, one for Y)
    for track_name in track_names:
        track_index = track_names.index(track_name)
        # Label and write X coordinates for this track
        sheet.append([track_name, "X"] + all_tracks_data[track_index * 2])  # X coordinates (row for X)
        # Label and write Y coordinates for this track
        sheet.append([track_name, "Y"] + all_tracks_data[track_index * 2 + 1])  # Y coordinates (row for Y)

    # Save the workbook using the video file name without .mp4
    video_name_without_extension = os.path.splitext(clip.name)[0]  # Remove .mp4
    path = os.path.join(folder, f"{video_name_without_extension}.xlsx")  # Save using the video name
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

        # Export just the active track
        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Track Data"
        sheet.append(["Track Name", "Coordinate Type", "Frame", "X", "Y"])
        
        export_track(active_track, clip.name, x, y, context.scene.my_tool.path, sheet)
        workbook.save(os.path.join(context.scene.my_tool.path, f"{clip.name}_active_track.xlsx"))
        
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