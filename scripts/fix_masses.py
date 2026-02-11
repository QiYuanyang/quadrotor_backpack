# fix_masses.py
from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open("assets/humanoid_quadrotor/humanoid_quadrotor.usd")

# Define mass for each body (kg)
masses = {
    "torso": 8.0,         # Main body
    "quadrotor": 5.0,     # Backpack frame
    "pelvis": 1.5,
    "lwaist": 1.0,
    "right_thigh": 1.2,
    "left_thigh": 1.2,
    "right_shin": 0.8,
    "left_shin": 0.8,
    "right_foot": 0.5,
    "left_foot": 0.5,
    "right_upper_arm": 0.6,
    "left_upper_arm": 0.6,
    "right_lower_arm": 0.4,
    "left_lower_arm": 0.4,
}

for prim in stage.Traverse():
    prim_name = prim.GetName()
    if prim_name in masses:
        if not prim.HasAPI(UsdPhysics.MassAPI):
            UsdPhysics.MassAPI.Apply(prim)
        mass_api = UsdPhysics.MassAPI(prim)
        mass_api.GetMassAttr().Set(masses[prim_name])
        print(f"Set {prim_name}: {masses[prim_name]} kg")

stage.Save()
print(f"\nTotal mass: {sum(masses.values()):.1f} kg")