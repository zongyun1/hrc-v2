"""Procedural warehouse packing station with collidable furniture (meters, Z up)."""


def build_warehouse(box, x, y):
    box("WarehouseBackWall", (8.0, 0.12, 3.2), (x, y + 3.2, 1.6), (0.72, 0.77, 0.80))
    box("WarehouseSideWall", (0.12, 6.4, 3.2), (x - 3.5, y, 1.6), (0.64, 0.70, 0.75))
    box("PackingTable", (0.65, 0.85, 0.06), (x + 0.80, y, 0.77), (0.60, 0.42, 0.24))
    for i, (dx, dy) in enumerate(((0.53, -0.34), (0.53, 0.34), (1.07, -0.34), (1.07, 0.34))):
        box(f"PackingLeg{i}", (0.05, 0.05, 0.74), (x + dx, y + dy, 0.37), (0.18, 0.22, 0.25))
    for bay in range(3):
        cx = x - 2.5 + bay * 1.65
        for side_id, side in enumerate((-0.7, 0.7)):
            for depth_id, depth in enumerate((-0.30, 0.30)):
                box(f"RackPost{bay}_{side_id}_{depth_id}", (0.06, 0.06, 2.5),
                    (cx + side, y + 2.7 + depth, 1.25), (0.11, 0.24, 0.36))
        for level in range(3):
            z = 0.20 + level * 0.85
            box(f"RackShelf{bay}_{level}", (1.5, 0.72, 0.05), (cx, y + 2.7, z), (0.88, 0.40, 0.09))
            for col in range(3):
                box(f"StockBox{bay}_{level}_{col}", (0.34, 0.40, 0.38),
                    (cx - 0.47 + col * 0.47, y + 2.7, z + 0.215), (0.56 + 0.06 * col, 0.39, 0.23))
    # Thin painted route markings stay out of the collision space.
    for side_id, side in enumerate((-0.65, 0.65)):
        box(f"LaneMark{side_id}", (0.025, 2.5, 0.002), (x + side, y + 0.55, -0.004), (0.94, 0.69, 0.12))

    for i, dy in enumerate((-0.28, 0.28)):
        box(f"HumanStopMark{i}", (0.55, 0.025, 0.002), (x + 0.9, y + 1.65 + dy, -0.004), (0.18, 0.68, 0.80))
