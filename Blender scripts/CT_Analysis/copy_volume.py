"""
Volume Calculator Add-on for Blender
====================================

INSTALLATION:
1. Save this script as a .py file (e.g., 'volume_calculator.py')
2. In Blender: Edit > Preferences > Add-ons
3. Click "Install..." and select your .py file
4. Find "Volume Calculator" in the add-ons list (you can search for it)
5. Check the checkbox to enable it

USAGE:
1. Select a mesh object in the 3D viewport
2. Press N to open the sidebar if not already visible
3. Go to the "Tool" tab in the sidebar
4. Find the "Volume Calculator" panel
5. Click "Calculate Volume" button
6. Volume will be calculated and automatically copied to clipboard
7. Check the panel for the result and info area for confirmation

REQUIREMENTS:
- Blender 2.80 or newer
- Linux users may need to install 'xclip' or 'xsel' for clipboard functionality:
  Ubuntu/Debian: sudo apt install xclip
  Fedora: sudo dnf install xclip

FEATURES:
- Only works with mesh objects (button disabled for other object types)
- Accounts for object transforms (scale, rotation, position)
- Shows last calculated result in the panel
- Automatic clipboard copying with system notifications
- Volume calculated in Blender units³ (based on scene unit settings)

NOTES:
- Object must be a manifold (watertight) mesh for accurate volume calculation
- Volume is formatted to 6 decimal places
- Works in Object mode (automatically switches if needed)
"""

bl_info = {
    "name": "Volume Calculator",
    "author": "Your Name",
    "version": (1, 0, 0),
    "blender": (2, 80, 0),
    "location": "3D Viewport > Sidebar (N) > Tool Tab",
    "description": "Calculate mesh volume and copy to clipboard",
    "category": "Mesh",
}

import bpy
import bmesh
import subprocess
import platform
from bpy.types import Panel, Operator
from bpy.props import StringProperty

def copy_to_clipboard(text):
    """Copy text to system clipboard"""
    system = platform.system()
    
    if system == "Windows":
        # Windows
        subprocess.run(['clip'], input=text, text=True, shell=True)
    elif system == "Darwin":
        # macOS
        subprocess.run(['pbcopy'], input=text, text=True)
    else:
        # Linux
        try:
            subprocess.run(['xclip', '-selection', 'clipboard'], input=text, text=True)
        except FileNotFoundError:
            try:
                subprocess.run(['xsel', '--clipboard', '--input'], input=text, text=True)
            except FileNotFoundError:
                print("Clipboard access requires 'xclip' or 'xsel' to be installed on Linux")
                return False
    return True

class MESH_OT_calculate_volume(Operator):
    """Calculate the volume of selected mesh and copy to clipboard"""
    bl_idname = "mesh.calculate_volume"
    bl_label = "Calculate Volume"
    bl_options = {'REGISTER', 'UNDO'}
    
    # Property to store the last calculated volume
    last_volume: StringProperty(
        name="Last Volume",
        default="No calculation yet"
    )
    
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH')
    
    def execute(self, context):
        obj = context.active_object
        
        # Make sure we're in object mode
        if context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        
        # Create bmesh representation
        bm = bmesh.new()
        
        try:
            # Load mesh data into bmesh
            bm.from_mesh(obj.data)
            
            # Apply object transforms to get world-space volume
            bm.transform(obj.matrix_world)
            
            # Calculate volume
            volume = bm.calc_volume()
            
            # Format volume string
            volume_text = f"{volume:.6f}"
            
            # Store in the operator property for display
            self.last_volume = volume_text
            
            # Update the scene property for the panel
            context.scene.volume_calc_result = volume_text
            
            # Copy to clipboard
            if copy_to_clipboard(volume_text):
                self.report({'INFO'}, f"Volume: {volume_text} (copied to clipboard)")
            else:
                self.report({'WARNING'}, f"Volume: {volume_text} (clipboard copy failed)")
                
        except Exception as e:
            self.report({'ERROR'}, f"Error calculating volume: {e}")
            return {'CANCELLED'}
        
        finally:
            # Clean up bmesh
            bm.free()
        
        return {'FINISHED'}

class VIEW3D_PT_volume_calculator(Panel):
    """Panel in the 3D Viewport sidebar"""
    bl_label = "Volume Calculator"
    bl_idname = "VIEW3D_PT_volume_calculator"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Tool"
    
    def draw(self, context):
        layout = self.layout
        
        # Check if there's an active mesh object
        if context.active_object and context.active_object.type == 'MESH':
            layout.label(text=f"Object: {context.active_object.name}")
            layout.operator("mesh.calculate_volume", text="Calculate Volume")
            
            # Display last result if available
            if hasattr(context.scene, 'volume_calc_result'):
                layout.separator()
                layout.label(text="Last Result:")
                layout.label(text=context.scene.volume_calc_result)
        else:
            layout.label(text="Select a mesh object", icon='INFO')

def register():
    bpy.utils.register_class(MESH_OT_calculate_volume)
    bpy.utils.register_class(VIEW3D_PT_volume_calculator)
    
    # Add property to store results
    bpy.types.Scene.volume_calc_result = StringProperty(
        name="Volume Result",
        default="No calculation yet"
    )

def unregister():
    bpy.utils.unregister_class(MESH_OT_calculate_volume)
    bpy.utils.unregister_class(VIEW3D_PT_volume_calculator)
    
    # Remove property
    del bpy.types.Scene.volume_calc_result

if __name__ == "__main__":
    register()