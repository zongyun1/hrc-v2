#!/usr/bin/env python3
"""
Import an FBX file into Blender and export motion to motion.pkl.
Run headless: blender -b -P import_and_export_fbx.py -- --fbx Talking.fbx --name Talking --output motion.pkl --merge motion.pkl
"""
import bpy
import sys
import os

def main():
    # Parse args after "--"
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender -b -P import_and_export_fbx.py -- --fbx FILE --name NAME [--output PKL] [--merge PKL]")
        return

    idx = argv.index("--")
    args = argv[idx + 1:]

    fbx_path = None
    motion_name = None
    output_path = None
    merge_path = None

    i = 0
    while i < len(args):
        if args[i] == "--fbx" and i + 1 < len(args):
            fbx_path = args[i + 1]
            i += 2
        elif args[i] == "--name" and i + 1 < len(args):
            motion_name = args[i + 1]
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_path = args[i + 1]
            i += 2
        elif args[i] == "--merge" and i + 1 < len(args):
            merge_path = args[i + 1]
            i += 2
        else:
            i += 1

    if fbx_path is None:
        print("Error: --fbx required")
        return

    if not os.path.isabs(fbx_path):
        fbx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fbx_path)

    if not os.path.isfile(fbx_path):
        print(f"Error: FBX not found: {fbx_path}")
        return

    # Clear default scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import FBX
    print(f"Importing FBX: {fbx_path}")
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # Find the armature
    armature = None
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            armature = obj
            break

    if armature is None:
        print("Error: No armature found in FBX")
        return

    print(f"Found armature: {armature.name}")
    bpy.context.view_layer.objects.active = armature

    # Now run the export
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if output_path is None:
        output_path = os.path.join(script_dir, "output_motion", "motion.pkl")
    elif not os.path.isabs(output_path):
        output_path = os.path.join(script_dir, "output_motion", output_path)

    if merge_path is not None and not os.path.isabs(merge_path):
        merge_path = os.path.join(script_dir, "output_motion", merge_path)

    # Import and run the export function
    sys.path.insert(0, script_dir)
    from export_motion_from_blender import export_motion_from_blender

    export_motion_from_blender(
        armature_name=armature.name,
        motion_name=motion_name,
        output_path=output_path,
        merge_existing=merge_path,
    )
    print(f"Done! Motion '{motion_name}' exported to {output_path}")


if __name__ == "__main__":
    main()
