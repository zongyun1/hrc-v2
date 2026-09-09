"""Phase 0.4 milestone B (step 1) - voxelize a RoboCasa kitchen into TRUMANS occupancy.

TRUMANS grid: bounds x[-3,3] y[0,2] z[-4,4] (Y-UP), res (300,100,400), voxel 0.02m,
occ[x,y,z] bool, True=occupied, floor (y=0 layer) fully occupied by convention.

RoboCasa/Genesis is Z-UP with the kitchen offset from origin. Remap (right-handed,
cyclic so no mirroring):
    T_x = RC_y - ycen      (room depth  -> lateral)
    T_y = RC_z             (height      -> up)
    T_z = RC_x - xcen      (galley len  -> forward/walk axis)
Fixtures are axis-aligned boxes, so filling each geom's world AABB gives solid,
gap-free occupancy. Robot (parked ~(10,10)) and RoboCasa marker geoms (z~10.46)
are filtered out by keeping only geoms inside the kitchen region.

Writes kitchen.npy into the TRUMANS Scene/ folder and saves the transform.

Usage (Genesis venv):
  cd genesis && uv run python ../tools/voxelize_kitchen_for_trumans.py --mjcf ../exports/robocasa_layout1_style1.xml
"""

import argparse
import json

import numpy as np
import genesis as gs

GRID_MIN = np.array([-3.0, 0.0, -4.0])
GRID_MAX = np.array([3.0, 2.0, 4.0])
RES = np.array([300, 100, 400])
VOX = (GRID_MAX - GRID_MIN) / RES  # 0.02 each


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=str, required=True)
    parser.add_argument("--out", type=str,
                        default="../third_party/trumans_utils/Data_blocks_motion_all/Scene/kitchen.npy")
    parser.add_argument("--transform_out", type=str, default="../exports/kitchen_transform.json")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False)
    kitchen = scene.add_entity(gs.morphs.MJCF(file=args.mjcf))
    scene.build()

    # Collect world-frame AABBs of kitchen geoms only (exclude robot + far markers).
    boxes = []
    for link in kitchen.links:
        name = link.name
        if any(k in name for k in ("robot0", "mobilebase", "gripper", "manipulator",
                                   "eef_target", "base")):
            continue
        for geom in link.geoms:
            aabb = geom.get_AABB().cpu().numpy()  # (2,3) RC z-up
            c = (aabb[0] + aabb[1]) / 2.0
            # keep only geoms inside the kitchen region (RC coords)
            if not (-1.0 < c[0] < 6.0 and -4.5 < c[1] < 1.5 and -0.1 < c[2] < 3.0):
                continue
            boxes.append(aabb)
    boxes = np.array(boxes)
    print(f"kept {len(boxes)} geom AABBs inside kitchen region")

    rc_lo = boxes[:, 0, :].min(axis=0)
    rc_hi = boxes[:, 1, :].max(axis=0)
    print(f"RC kitchen bounds lo={np.round(rc_lo,2)} hi={np.round(rc_hi,2)}")
    xcen = (rc_lo[0] + rc_hi[0]) / 2.0  # galley length center (RC x)
    ycen = (rc_lo[1] + rc_hi[1]) / 2.0  # room depth center   (RC y)
    print(f"remap centers: xcen(RCx->Tz)={xcen:.3f} ycen(RCy->Tx)={ycen:.3f}")

    def rc_to_t(p):  # p: (...,3) RC z-up -> TRUMANS y-up
        out = np.empty_like(p)
        out[..., 0] = p[..., 1] - ycen
        out[..., 1] = p[..., 2]
        out[..., 2] = p[..., 0] - xcen
        return out

    occ = np.zeros(tuple(RES), dtype=bool)
    clipped = 0
    for aabb in boxes:
        # 8 corners -> T coords -> min/max box in T
        corners = np.array([[aabb[i, 0], aabb[j, 1], aabb[k, 2]]
                            for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        t = rc_to_t(corners)
        tlo, thi = t.min(axis=0), t.max(axis=0)
        ilo = np.floor((tlo - GRID_MIN) / VOX).astype(int)
        ihi = np.ceil((thi - GRID_MIN) / VOX).astype(int)
        if np.any(ihi < 0) or np.any(ilo >= RES):
            clipped += 1
            continue
        ilo = np.clip(ilo, 0, RES - 1)
        ihi = np.clip(ihi, 0, RES)
        occ[ilo[0]:ihi[0], ilo[1]:ihi[1], ilo[2]:ihi[2]] = True

    occ[:, 0, :] = True  # floor layer, TRUMANS convention
    print(f"occupied frac {occ.mean():.4f}  ({clipped} boxes out of grid)")

    np.save(args.out, occ)
    print(f"saved {args.out}")

    transform = {"xcen": float(xcen), "ycen": float(ycen),
                 "note": "RC->T: Tx=RCy-ycen, Ty=RCz, Tz=RCx-xcen; inverse RCx=Tz+xcen, RCy=Tx+ycen, RCz=Ty"}
    with open(args.transform_out, "w") as f:
        json.dump(transform, f, indent=2)
    print(f"saved {args.transform_out}: {transform}")


if __name__ == "__main__":
    main()
