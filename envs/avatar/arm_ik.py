"""Two-bone IK for the avatar's right arm.

The avatar's skin is driven by per-joint 4×4 transforms stored in
`robot.node_trans`.  We override three rows (55=RightHand, 56=RightForeArm,
57=RightArm) with IK-computed transforms so the hand lands at a desired
world-frame target.  The rest of the body follows the active motion.

The math is classic analytic 2-bone IK:
    - Given shoulder S, wrist target H, upper-arm length Lu, forearm length Lf.
    - Distance d = |H - S|, clamp to (0, Lu + Lf - 1e-4].
    - Elbow interior angle: cos θ_elbow = (Lu² + Lf² - d²) / (2·Lu·Lf).
    - Shoulder aim angle:   cos φ       = (d² + Lu² - Lf²) / (2·d·Lu).
    - Aim direction a = (H - S) / d.
    - Pole direction p: built from a preferred "elbow-down-and-back" hint,
      orthogonalised against a (keeps elbow in a natural pose rather than
      flipping out to the side).
    - Elbow position E = S + Lu · (cos φ · a + sin φ · p_ortho).

Then we emit 4×4 matrices for the upper arm (rotates rest upper-arm direction
to E-S) and forearm (rotates rest forearm direction to H-E), leaving the
translation part in the row[3, :3] slot which is the convention used by
`forward_kinematics` in `envs/avatar/utils.py` (row-major, "position last row"
Genesis-style).
"""
import numpy as np


def _minimum_rotation(u, v):
    """Return a 3×3 rotation matrix that rotates unit vector u into v.

    Uses Rodrigues' formula around axis (u × v); falls back to identity
    for near-parallel and to a 180° rotation around any perpendicular for
    near-antiparallel.  No twist — just the shortest rotation.
    """
    u = np.asarray(u, dtype=np.float64) / (np.linalg.norm(u) + 1e-12)
    v = np.asarray(v, dtype=np.float64) / (np.linalg.norm(v) + 1e-12)
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if dot > 1.0 - 1e-8:
        return np.eye(3)
    if dot < -1.0 + 1e-8:
        # Antiparallel — rotate 180° around any axis perpendicular to u.
        axis = np.cross(u, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(u, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=np.float64)
        return np.eye(3) + 2.0 * K @ K
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    axis = axis / (s + 1e-12)
    c = dot
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=np.float64)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


class ArmIK:
    """Two-bone IK for a fixed shoulder + elbow + wrist chain."""

    # node_trans rows for the avatar's right arm (Adrian_Keller rig).
    ROW_HAND = 55
    ROW_FOREARM = 56
    ROW_ARM = 57

    def __init__(self, upper_len, forearm_len, rest_upper_dir, rest_forearm_dir):
        self.Lu = float(upper_len)
        self.Lf = float(forearm_len)
        self.rest_u = np.asarray(rest_upper_dir, dtype=np.float64)
        self.rest_u /= np.linalg.norm(self.rest_u) + 1e-12
        self.rest_f = np.asarray(rest_forearm_dir, dtype=np.float64)
        self.rest_f /= np.linalg.norm(self.rest_f) + 1e-12

    def solve_elbow(self, shoulder, target_hand, pole_hint=np.array([0.0, -1.0, -0.5])):
        """Return (elbow_pos, upper_dir, forearm_dir) all in the same frame
        as shoulder and target_hand."""
        S = np.asarray(shoulder, dtype=np.float64)
        H = np.asarray(target_hand, dtype=np.float64)
        v = H - S
        d = float(np.linalg.norm(v))
        max_reach = self.Lu + self.Lf - 1e-4
        if d > max_reach:
            # Clamp along aim direction — arm fully stretched.
            v = v * (max_reach / d)
            d = max_reach
        if d < 1e-4:
            # Target is at shoulder — degenerate; collapse arm.
            return S + self.rest_u * 1e-3, self.rest_u, self.rest_f
        a = v / d
        # Shoulder-to-elbow angle from law of cosines.
        cos_phi = np.clip((d * d + self.Lu * self.Lu - self.Lf * self.Lf)
                          / (2.0 * d * self.Lu), -1.0, 1.0)
        sin_phi = float(np.sqrt(max(0.0, 1.0 - cos_phi * cos_phi)))
        # Pole direction: orthogonalise hint against aim.
        p = np.asarray(pole_hint, dtype=np.float64)
        p_ortho = p - np.dot(p, a) * a
        n = float(np.linalg.norm(p_ortho))
        if n < 1e-4:
            # Pole hint parallel to aim — fall back to a world-down vector.
            p = np.array([0.0, 0.0, -1.0])
            p_ortho = p - np.dot(p, a) * a
            n = float(np.linalg.norm(p_ortho))
            if n < 1e-4:
                p_ortho = np.cross(a, np.array([1.0, 0.0, 0.0]))
                n = float(np.linalg.norm(p_ortho)) + 1e-6
        p_ortho = p_ortho / n
        elbow = S + self.Lu * (cos_phi * a + sin_phi * p_ortho)
        upper_dir = (elbow - S)
        upper_dir = upper_dir / (np.linalg.norm(upper_dir) + 1e-12)
        forearm_dir = (H - elbow)
        forearm_dir = forearm_dir / (np.linalg.norm(forearm_dir) + 1e-12)
        return elbow, upper_dir, forearm_dir

    def arm_transforms(self, shoulder, elbow, hand, upper_dir, forearm_dir,
                       rest_shoulder, rest_elbow, rest_hand):
        """Build 4×4 node_trans rows for RightArm, RightForeArm, RightHand.

        Convention (envs/avatar/utils.py forward_kinematics): the matrix
        stores rotation in [:3, :3] and position in [3, :3] (row-major,
        'translation is last row').  We match that.
        """
        # Rotations: align rest upper-arm direction with new upper_dir, and
        # rest forearm direction with new forearm_dir.  These are world
        # rotations applied to the mesh.
        R_upper = _minimum_rotation(self.rest_u, upper_dir)
        R_fore = _minimum_rotation(self.rest_f, forearm_dir)

        def make_mat(R, p):
            m = np.eye(4)
            m[:3, :3] = R
            m[3, :3] = p
            return m

        return {
            self.ROW_ARM: make_mat(R_upper, shoulder),
            self.ROW_FOREARM: make_mat(R_fore, elbow),
            self.ROW_HAND: make_mat(R_fore, hand),
        }
