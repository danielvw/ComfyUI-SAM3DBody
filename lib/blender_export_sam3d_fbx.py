"""
Blender script to export SAM3D Body mesh and skeleton to FBX file.

Usage: blender --background --python blender_export_sam3d_fbx.py -- <input_obj> <output_fbx> [skeleton_json]
"""

import bpy
import sys
import os
import json
import numpy as np
from mathutils import Vector

# Get arguments after '--'
argv = sys.argv
argv = argv[argv.index("--") + 1:] if "--" in argv else []

if len(argv) < 2:
    print("Usage: blender --background --python blender_export_sam3d_fbx.py -- <input_obj> <output_fbx> [skeleton_json]")
    sys.exit(1)

input_obj = argv[0]
output_fbx = argv[1]
skeleton_json = argv[2] if len(argv) > 2 and argv[2] else None

# Parse export flags
export_mesh = "--no-mesh" not in argv
export_skeleton = "--no-skeleton" not in argv

print(f"[SAM3D] ========================================")
print(f"[SAM3D] Export flags: mesh={export_mesh}, skeleton={export_skeleton}")
print(f"[SAM3D] Input OBJ: {input_obj}")
print(f"[SAM3D] Output FBX: {output_fbx}")
print(f"[SAM3D] Skeleton JSON: {skeleton_json}")
print(f"[SAM3D] ========================================")




def get_joint_name(index, joint_names_list):
    """Get joint name from list or fallback to numbered."""
    if joint_names_list and index < len(joint_names_list):
        name = joint_names_list[index]
        # Sanitize name for Blender (replace spaces, special chars)
        name = name.replace(' ', '_').replace('-', '_')
        return name
    return f'Joint_{index:03d}'


# Load skeleton data from JSON if provided
joints = None
num_joints = 0
joint_parents_list = None
skinning_weights = None
joint_names_list = None

print(f"[SAM3D] Checking skeleton JSON...")
if skeleton_json and os.path.exists(skeleton_json):
    print(f"[SAM3D] Loading skeleton data from: {skeleton_json}")
    try:
        with open(skeleton_json, 'r') as f:
            skeleton_data = json.load(f)

        joint_positions = skeleton_data.get('joint_positions', [])
        num_joints = skeleton_data.get('num_joints', len(joint_positions))
        joint_parents_list = skeleton_data.get('joint_parents')
        skinning_weights = skeleton_data.get('skinning_weights')

        print(f"[SAM3D] Skeleton data loaded: num_joints={num_joints}, has_parents={joint_parents_list is not None}, has_weights={skinning_weights is not None}")

        if joint_positions:
            joints = np.array(joint_positions, dtype=np.float32)
            print(f"[SAM3D] Joint positions array shape: {joints.shape}")

        # Load joint names from skeleton data (extracted by ComfyUI from MHR model)
        joint_names_list = skeleton_data.get('joint_names')
        if joint_names_list:
            print(f"[SAM3D] Joint names loaded from skeleton data: {len(joint_names_list)} names")
        else:
            print(f"[SAM3D] No joint names in skeleton data, will use numbered names")
    except Exception as e:
        print(f"[SAM3D] ERROR loading skeleton data: {e}")
        import traceback
        traceback.print_exc()
        joints = None
else:
    print(f"[SAM3D] No skeleton JSON provided or file does not exist")


# Clean default scene
def clean_bpy():
    """Remove all default Blender objects"""
    for c in bpy.data.actions:
        bpy.data.actions.remove(c)
    for c in bpy.data.armatures:
        bpy.data.armatures.remove(c)
    for c in bpy.data.cameras:
        bpy.data.cameras.remove(c)
    for c in bpy.data.collections:
        bpy.data.collections.remove(c)
    for c in bpy.data.images:
        bpy.data.images.remove(c)
    for c in bpy.data.materials:
        bpy.data.materials.remove(c)
    for c in bpy.data.meshes:
        bpy.data.meshes.remove(c)
    for c in bpy.data.objects:
        bpy.data.objects.remove(c)
    for c in bpy.data.textures:
        bpy.data.textures.remove(c)

clean_bpy()


def make_mesh_double_sided(mesh_obj):
    """Make mesh double-sided by duplicating faces with flipped normals."""
    # Select the mesh and enter edit mode
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Select all geometry
    bpy.ops.mesh.select_all(action='SELECT')

    # Duplicate faces
    bpy.ops.mesh.duplicate()

    # Flip normals on duplicated faces
    bpy.ops.mesh.flip_normals()

    # Return to object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    print(f"[SAM3D] Made mesh double-sided: {mesh_obj.name}")


# Create collection
collection = bpy.data.collections.new('SAM3D_Export')
bpy.context.scene.collection.children.link(collection)

# Import OBJ mesh (only if export_mesh is enabled)
mesh_obj = None
if export_mesh:
    print(f"[SAM3D] Importing mesh from OBJ...")
    try:
        bpy.ops.wm.obj_import(filepath=input_obj)

        imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
        if not imported_objects:
            raise RuntimeError("No mesh found after OBJ import")

        mesh_obj = imported_objects[0]
        mesh_obj.name = 'SAM3D_Character'
        print(f"[SAM3D] Mesh imported: {mesh_obj.name}, vertices={len(mesh_obj.data.vertices)}")

        # Move to our collection
        if mesh_obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(mesh_obj)
        collection.objects.link(mesh_obj)
        print(f"[SAM3D] Mesh added to collection")

    except Exception as e:
        print(f"[SAM3D] Failed to import OBJ: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print(f"[SAM3D] Skipping mesh import (export_mesh=False)")

# Create armature from skeleton if provided (only if export_skeleton is enabled)
print(f"[SAM3D] Checking armature creation condition:")
print(f"[SAM3D]   export_skeleton = {export_skeleton}")
print(f"[SAM3D]   joints is not None = {joints is not None}")
print(f"[SAM3D]   num_joints = {num_joints}")

if export_skeleton and joints is not None and num_joints > 0:
    print(f"[SAM3D] Creating armature with {num_joints} joints...")
    try:
        # Create armature in edit mode
        bpy.ops.object.armature_add(enter_editmode=True)
        armature = bpy.data.armatures.get('Armature')
        armature.name = 'SAM3D_Skeleton'
        armature_obj = bpy.context.active_object
        armature_obj.name = 'SAM3D_Skeleton'
        print(f"[SAM3D] Armature object created: {armature_obj.name}")

        # Move to our collection
        if armature_obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(armature_obj)
        collection.objects.link(armature_obj)
        print(f"[SAM3D] Armature added to collection")

        edit_bones = armature.edit_bones
        extrude_size = 0.05

        # Remove default bone
        default_bone = edit_bones.get('Bone')
        if default_bone:
            edit_bones.remove(default_bone)

        # Calculate skeleton center for root bone placement
        skeleton_center = joints.mean(axis=0)
        print(f"[SAM3D] Skeleton center: {skeleton_center}")

        # Make positions relative to skeleton center
        rel_joints = joints - skeleton_center

        # Apply coordinate system correction to match mesh orientation
        rel_joints_corrected = np.zeros_like(rel_joints)
        rel_joints_corrected[:, 0] = rel_joints[:, 0]
        rel_joints_corrected[:, 1] = -rel_joints[:, 2]
        rel_joints_corrected[:, 2] = rel_joints[:, 1]

        # Create all bones with proper names
        print(f"[SAM3D] Creating {num_joints} bones...")
        bones_dict = {}
        for i in range(num_joints):
            bone_name = get_joint_name(i, joint_names_list)
            bone = edit_bones.new(bone_name)
            bone.head = Vector((rel_joints_corrected[i, 0], rel_joints_corrected[i, 1], rel_joints_corrected[i, 2]))
            bone.tail = Vector((rel_joints_corrected[i, 0], rel_joints_corrected[i, 1], rel_joints_corrected[i, 2] + extrude_size))
            bones_dict[bone_name] = bone
        print(f"[SAM3D] Created {len(bones_dict)} bones")

        # Build hierarchical structure using joint parents if available
        if joint_parents_list and len(joint_parents_list) == num_joints:
            print(f"[SAM3D] Building hierarchical bone structure...")
            for i in range(num_joints):
                parent_idx = joint_parents_list[i]
                if parent_idx >= 0 and parent_idx < num_joints and parent_idx != i:
                    bone_name = get_joint_name(i, joint_names_list)
                    parent_bone_name = get_joint_name(parent_idx, joint_names_list)
                    bones_dict[bone_name].parent = bones_dict[parent_bone_name]
                    bones_dict[bone_name].use_connect = False
            print(f"[SAM3D] Hierarchical structure built")
        else:
            print(f"[SAM3D] Building flat hierarchy (fallback)...")
            # Fallback: create flat hierarchy with first joint as root
            root_bone_name = get_joint_name(0, joint_names_list)
            for i in range(1, num_joints):
                bone_name = get_joint_name(i, joint_names_list)
                bones_dict[bone_name].parent = bones_dict[root_bone_name]
                bones_dict[bone_name].use_connect = False
            print(f"[SAM3D] Flat hierarchy built")

        # Switch to object mode
        bpy.ops.object.mode_set(mode='OBJECT')
        print(f"[SAM3D] Switched to object mode")

        # Position armature at skeleton center
        skeleton_center_corrected = np.zeros(3)
        skeleton_center_corrected[0] = skeleton_center[0]
        skeleton_center_corrected[1] = -skeleton_center[2]
        skeleton_center_corrected[2] = skeleton_center[1]
        armature_obj.location = Vector((skeleton_center_corrected[0], skeleton_center_corrected[1], skeleton_center_corrected[2]))
        print(f"[SAM3D] Armature positioned at: {armature_obj.location}")

        # Apply skinning weights if available
        if skinning_weights:
            print(f"[SAM3D] Applying skinning weights...")
            if mesh_obj is None:
                print(f"[SAM3D] WARNING: Cannot apply skinning weights - mesh_obj is None!")
            else:
                # Create vertex groups for each bone with proper names
                for i in range(num_joints):
                    bone_name = get_joint_name(i, joint_names_list)
                    mesh_obj.vertex_groups.new(name=bone_name)

                # Assign weights to vertices
                num_vertices = len(mesh_obj.data.vertices)
                for vert_idx in range(min(num_vertices, len(skinning_weights))):
                    influences = skinning_weights[vert_idx]
                    if influences and len(influences) > 0:
                        for bone_idx, weight in influences:
                            if 0 <= bone_idx < num_joints and weight > 0.0001:
                                bone_name = get_joint_name(bone_idx, joint_names_list)
                                vertex_group = mesh_obj.vertex_groups.get(bone_name)
                                if vertex_group:
                                    vertex_group.add([vert_idx], weight, 'REPLACE')
                print(f"[SAM3D] Skinning weights applied to {num_vertices} vertices")
        else:
            print(f"[SAM3D] No skinning weights available")

        # Deselect all
        for obj in bpy.context.selected_objects:
            obj.select_set(False)

        # Parent mesh to armature (only if mesh exists)
        if mesh_obj:
            print(f"[SAM3D] Parenting mesh to armature...")
            mesh_obj.select_set(True)
            armature_obj.select_set(True)
            bpy.context.view_layer.objects.active = armature_obj

            if skinning_weights:
                bpy.ops.object.parent_set(type='ARMATURE')
                print(f"[SAM3D] Mesh parented with ARMATURE (with weights)")
            else:
                bpy.ops.object.parent_set(type='ARMATURE_NAME')
                print(f"[SAM3D] Mesh parented with ARMATURE_NAME (no weights)")
        else:
            print(f"[SAM3D] No mesh to parent to armature")

        print(f"[SAM3D] Armature creation completed successfully!")

    except Exception as e:
        print(f"[SAM3D] Armature creation failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"[SAM3D] Skipping armature creation (condition not met)")

# Make mesh double-sided AFTER skinning (so duplicated vertices inherit weights)
if mesh_obj:
    print(f"[SAM3D] Making mesh double-sided...")
    make_mesh_double_sided(mesh_obj)
else:
    print(f"[SAM3D] No mesh to make double-sided")

# Export to FBX
print(f"[SAM3D] Preparing FBX export...")
print(f"[SAM3D] Collection objects before export: {[obj.name for obj in collection.objects]}")
os.makedirs(os.path.dirname(output_fbx) if os.path.dirname(output_fbx) else '.', exist_ok=True)

try:
    # Select all objects in our collection
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in collection.objects:
        obj.select_set(True)
        print(f"[SAM3D] Selected for export: {obj.name} (type={obj.type})")

    # Export FBX
    print(f"[SAM3D] Exporting to: {output_fbx}")

    # Check if we have armature in selection
    has_armature = any(obj.type == 'ARMATURE' for obj in bpy.context.selected_objects)
    has_mesh = any(obj.type == 'MESH' for obj in bpy.context.selected_objects)
    print(f"[SAM3D] Export contains: armature={has_armature}, mesh={has_mesh}")

    # Use original export settings that worked before
    bpy.ops.export_scene.fbx(
        filepath=output_fbx,
        check_existing=False,
        use_selection=True,
        add_leaf_bones=False,
        path_mode='COPY',
        embed_textures=True,
    )

    print(f"[SAM3D] Export completed successfully!")
    print(f"[SAM3D] ========================================")

except Exception as e:
    print(f"[SAM3D] Export failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
