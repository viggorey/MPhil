import bpy
import os
from openpyxl import Workbook
from bpy.props import StringProperty, PointerProperty
from bpy.types import PropertyGroup

bl_info = {
    "name": "Export Video Track as CSV",
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
    
    # Initialize a list to store track data and frame ranges
    all_tracks_data = []
    track_names = []
    frame_ranges = []

    for t in clip.tracking.tracks:
        track_data_x = []  # X coordinates for this track
        track_data_y = []  # Y coordinates for this track

        track_names.append(f"Track {len(track_names) + 1}")  # Track name as Track 1, Track 2, etc.
        frame_start = t.markers[0].frame
        frame_end = t.markers[-1].frame
        frame_ranges.append((frame_start, frame_end))

        for m in t.markers:
            x_coord = m.co[0] * x
            y_coord = m.co[1] * y
            track_data_x.append(x_coord)
            track_data_y.append(y_coord)

        all_tracks_data.append((track_data_x, track_data_y, frame_start, len(t.markers)))

    # Determine the total frame range
    if frame_ranges:
        min_frame = min(fr[0] for fr in frame_ranges)
        max_frame = max(fr[1] for fr in frame_ranges)
    else:
        min_frame = 1
        max_frame = 1

    # Create headers for the range of frames (starting at Frame 1)
    headers = [f"Frame {i}" for i in range(min_frame + 1, max_frame)]  # Exclude Frame 1 and the last column
    sheet.append(["Track Name", "Coordinate Type"] + headers)

    # Write each track's X and Y data
    for track_index, (track_data_x, track_data_y, start_frame, num_markers) in enumerate(all_tracks_data):
        track_name = track_names[track_index]
        num_frames = max_frame - min_frame + 1
        
        # Initialize empty rows for the entire range
        full_x = [None] * num_frames
        full_y = [None] * num_frames

        # If fewer than 4 points, leave the row blank
        if num_markers < 4:
            sheet.append([track_name, "X"] + [None] * (num_frames - 2))
            sheet.append([track_name, "Y"] + [None] * (num_frames - 2))
            continue
        
        # Fill the rows starting at the correct index
        start_idx = start_frame - min_frame
        for i, (x, y) in enumerate(zip(track_data_x, track_data_y)):
            full_x[start_idx + i] = x
            full_y[start_idx + i] = y

        # Append rows for this track
        sheet.append([track_name, "X"] + full_x[1:-1])  # Skip Frame 1 and the last column
        sheet.append([track_name, "Y"] + full_y[1:-1])  # Skip Frame 1 and the last column

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

class CLIP_PT_export_track(bpy.types.Panel):
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

class ExportAllTracks(bpy.types.Operator):
    bl_idname = "wm.export_all_tracks"
    bl_label = "Export all Tracks"

    def execute(self, context):
        if not os.path.isdir(context.scene.my_tool.path):
            self.report({"ERROR"}, "No proper folder selected.")
            return {'CANCELLED'}
        
        clip = context.space_data.clip
        export_all_tracks_from_clip(clip, context.scene.my_tool.path)
        self.report({'INFO'}, f"Exported all tracks from current clip: {clip.name}")

        return {'FINISHED'}

classes = (
    CLIP_PT_export_track,
    ExportAllTracks,
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