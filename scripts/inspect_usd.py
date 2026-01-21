#!/usr/bin/env python3
"""Inspect USD file structure to understand the hierarchy."""

import sys
sys.path.insert(0, "/home/qyy/IsaacLab")

from isaaclab.app import AppLauncher

# Create app launcher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app


def print_prim_tree(prim, indent=0):
    """Recursively print prim hierarchy."""
    indent_str = "  " * indent
    
    # Get prim info
    prim_type = prim.GetTypeName()
    
    # Check physics APIs
    from pxr import UsdPhysics
    is_articulation = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    is_rigid_body = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    is_joint = prim.IsA(UsdPhysics.Joint)
    
    # Build info string
    info_parts = []
    if prim_type:
        info_parts.append(prim_type)
    if is_articulation:
        info_parts.append("ARTICULATION_ROOT")
    if is_rigid_body:
        info_parts.append("RIGID_BODY")
    if is_joint:
        info_parts.append("JOINT")
    
    info = f" ({', '.join(info_parts)})" if info_parts else ""
    
    print(f"{indent_str}{prim.GetName()}{info}")
    
    # Print children
    for child in prim.GetChildren():
        print_prim_tree(child, indent + 1)


def inspect_joints(stage):
    """Print all joints and their properties."""
    from pxr import UsdPhysics
    
    print("\n" + "="*80)
    print("JOINTS:")
    print("="*80)
    
    joint_names = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            joint_name = prim.GetName()
            joint_names.append(joint_name)
            
            print(f"\nJoint: {prim.GetPath()}")
            print(f"  Name: {joint_name}")
            print(f"  Type: {prim.GetTypeName()}")
            
            # Get joint limits if it's a revolute joint
            if prim.IsA(UsdPhysics.RevoluteJoint):
                rev_joint = UsdPhysics.RevoluteJoint(prim)
                lower_limit = rev_joint.GetLowerLimitAttr().Get()
                upper_limit = rev_joint.GetUpperLimitAttr().Get()
                if lower_limit is not None and upper_limit is not None:
                    print(f"  Limits: [{lower_limit}, {upper_limit}]")
    
    print(f"\n\nAll joint names: {joint_names}")
    return joint_names


def main():
    from pxr import Usd, UsdPhysics
    
    usd_path = "/home/qyy/hdd/work/alice_isaac/alice_isaac/assets/humanoid_quadrotor.usd"
    
    print(f"Loading USD file: {usd_path}")
    print("="*80)
    
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"ERROR: Could not open USD file: {usd_path}")
        return
    
    print("\nPRIM HIERARCHY:")
    print("="*80)
    root = stage.GetPseudoRoot()
    for child in root.GetChildren():
        print_prim_tree(child)
    
    joint_names = inspect_joints(stage)
    
    print("\n" + "="*80)
    print("ARTICULATION ROOTS:")
    print("="*80)
    articulation_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_count += 1
            print(f"\n{articulation_count}. {prim.GetPath()}")
            print(f"   Name: {prim.GetName()}")
            print(f"   Type: {prim.GetTypeName()}")
            
            # Count joints under this articulation
            joint_count = 0
            from pxr import Usd as UsdCore
            for child in UsdCore.PrimRange(prim):
                if child.IsA(UsdPhysics.Joint):
                    joint_count += 1
            print(f"   Joints: {joint_count}")
    
    print(f"\nTotal articulation roots found: {articulation_count}")
    print("="*80)


if __name__ == "__main__":
    main()
