"""Robot ↔ avatar collision detection via analytic point-vs-capsule SDF.

Why analytic (not Genesis's `detect_collision()`): Genesis does not expose
a "query-only / phantom collider" mode — any geom with `collision=True`
is fully in the physics world and will push the robot.  To detect
collisions *without* perturbing the physics we stay out of the collision
world entirely and do the distance test in Python.

Algorithm per call (`check()`):
  1. For every robot collision geom, read its current world-space mesh
     vertices via `geom.get_verts()`.  A pre-chosen subsample (~30 pts
     per geom by default) keeps the work bounded.
  2. For every avatar capsule `(A, B, radius)` and oriented box
     `(center, axes, half_extents)`, compute each point's signed
     distance.
  3. A pair `(robot_link, capsule)` is in collision iff its minimum
     signed distance is ≤ `margin` (default 0).

Vectorized via NumPy, ~O(N_robot_pts × N_capsules).  For a Franka with
≈11 links and 43 avatar capsules, each check is <1 ms.

Typical usage:

    from envs.avatar import AvatarCollider, AvatarCollisionChecker
    self.collider = AvatarCollider(scene, self.avatar.robot)
    # ... scene.build(), avatar.reset() ...
    self.collision_checker = AvatarCollisionChecker(
        self.collider, self.robot.get_arm("right").entity
    )

    # Each sim step:
    self.collider.update()
    result = self.collision_checker.check()
    if result["collided"]:
        print("hit", result["pairs"])  # [(link, capsule, depth_m), ...]
"""
import numpy as np


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x)


class AvatarCollisionChecker:
    """Analytic point-vs-capsule collision checker.

    Parameters
    ----------
    avatar_collider : AvatarCollider
        Source of capsules (`current_capsules()` returns (name, A, B, r)).
    robot_entity : genesis RigidEntity OR iterable of RigidEntity
        Robot(s) to check.  Each entity's collision geoms contribute
        sample points.  Pass a list to cover multiple arms.
    n_points_per_geom : int
        Target subsample count per collision geom.  The module takes a
        stride over `init_verts` so small geoms keep more coverage and
        large geoms don't explode the point count.
    margin : float
        Distance threshold for "in collision" (metres).  Set `0.0` for
        strict overlap, positive to flag near-misses (e.g. 0.01 → within
        1 cm).  Negative to allow small interpenetration without
        tripping (unusual).
    """

    def __init__(
        self,
        avatar_collider,
        robot_entity,
        n_points_per_geom: int = 30,
        margin: float = 0.0,
        ignored_capsules=(),
        ignored_capsule_prefixes=(),
    ):
        self.collider = avatar_collider
        self.margin = float(margin)
        self.ignored_capsules = {str(x) for x in (ignored_capsules or ())}
        self.ignored_capsule_prefixes = tuple(
            str(x) for x in (ignored_capsule_prefixes or ())
        )

        entities = robot_entity if isinstance(robot_entity, (list, tuple)) \
            else [robot_entity]

        # [(link_name, geom, numpy-int32 sub-indices)] — indices applied
        # to get_verts() output each step.
        self._geom_refs = []
        for ent in entities:
            for link in ent.links:
                link_geoms = getattr(link, "_geoms", None) or []
                for g in link_geoms:
                    try:
                        n_v = int(g.init_verts.shape[0])
                    except Exception:
                        continue
                    if n_v == 0:
                        continue
                    step = max(1, n_v // n_points_per_geom)
                    idxs = np.arange(0, n_v, step, dtype=np.int64)
                    # Always include the last vertex (covers extremes).
                    if idxs[-1] != n_v - 1:
                        idxs = np.append(idxs, n_v - 1)
                    self._geom_refs.append((link.name, g, idxs))

    def _is_ignored_capsule(self, name: str) -> bool:
        if name in self.ignored_capsules:
            return True
        return any(name.startswith(prefix) for prefix in self.ignored_capsule_prefixes)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _collect_points(self):
        """Stack world-space sample points across all robot geoms.

        Returns (M, 3) points and a length-M array of originating link
        names (used to build the per-link collision report).
        """
        pts_parts = []
        link_parts = []
        for link_name, geom, idxs in self._geom_refs:
            verts = _to_numpy(geom.get_verts())[idxs]
            pts_parts.append(verts)
            link_parts.extend([link_name] * len(verts))
        if not pts_parts:
            return np.zeros((0, 3)), []
        return np.concatenate(pts_parts, axis=0), link_parts

    def check(self) -> dict:
        """Run one collision query.

        Returns
        -------
        dict with keys
            collided      : bool — any signed-distance ≤ margin
            min_distance  : float — minimum signed distance over all
                            (point, capsule) pairs.  Negative means
                            interpenetration depth; positive means
                            closest approach.
            pairs         : list of (link_name, capsule_name, depth) for
                            every (link, capsule) pair where at least
                            one sample point is within `margin`.  Depth
                            is the most-negative signed distance found
                            for that pair.
            closest_pair  : (link, capsule, depth) for the single
                            deepest / closest point (always populated
                            when there are points and capsules).
        """
        capsules = self.collider.current_capsules() if self.collider else []
        boxes = []
        if self.collider is not None and hasattr(self.collider, "current_boxes"):
            boxes = self.collider.current_boxes()
        pts, link_names = self._collect_points()
        if (not capsules and not boxes) or len(pts) == 0:
            return {"collided": False, "min_distance": float("inf"),
                    "pairs": [], "closest_pair": None}

        # For each point, track (min_signed_distance, capsule_name).
        min_d = np.full(len(pts), np.inf)
        which_cap = np.array(["" for _ in range(len(pts))], dtype=object)

        for cname, A, B, r in capsules:
            A = np.asarray(A, dtype=np.float64)
            B = np.asarray(B, dtype=np.float64)
            AB = B - A
            ab_sq = float(np.dot(AB, AB))
            if ab_sq < 1e-12:
                continue
            # Closest point on segment to each sample, clamped to [0,1].
            t = np.clip((pts - A) @ AB / ab_sq, 0.0, 1.0)
            closest = A[None, :] + np.outer(t, AB)
            dist = np.linalg.norm(pts - closest, axis=1) - float(r)
            mask = dist < min_d
            if mask.any():
                min_d = np.where(mask, dist, min_d)
                which_cap[mask] = cname

        for bname, center, axes, half_extents in boxes:
            center = np.asarray(center, dtype=np.float64)
            axes = np.asarray(axes, dtype=np.float64)
            half_extents = np.asarray(half_extents, dtype=np.float64)
            local = (pts - center) @ axes
            q = np.abs(local) - half_extents[None, :]
            outside = np.maximum(q, 0.0)
            outside_dist = np.linalg.norm(outside, axis=1)
            inside_dist = np.minimum(np.max(q, axis=1), 0.0)
            dist = outside_dist + inside_dist
            mask = dist < min_d
            if mask.any():
                min_d = np.where(mask, dist, min_d)
                which_cap[mask] = bname

        # Build per-pair (link, capsule) min-depth report for pts that
        # breach `margin`.
        breaches = np.where(min_d <= self.margin)[0]
        pair_depth = {}
        for i in breaches:
            key = (link_names[i], str(which_cap[i]))
            d = float(min_d[i])
            if key not in pair_depth or d < pair_depth[key]:
                pair_depth[key] = d
        ignored_pairs = []
        pairs = []
        for (ln, cn), d in pair_depth.items():
            item = (ln, cn, d)
            if self._is_ignored_capsule(cn):
                ignored_pairs.append(item)
            else:
                pairs.append(item)
        pairs.sort(key=lambda x: x[2])  # deepest first
        ignored_pairs.sort(key=lambda x: x[2])

        arg_closest = int(np.argmin(min_d))
        closest_pair = pairs[0] if pairs else (
            link_names[arg_closest],
            str(which_cap[arg_closest]),
            float(min_d[arg_closest]),
        )

        return {
            "collided": bool(pairs),
            "min_distance": float(pairs[0][2] if pairs else min_d.min()),
            "pairs": pairs,
            "ignored_pairs": ignored_pairs,
            "closest_pair": closest_pair,
        }
