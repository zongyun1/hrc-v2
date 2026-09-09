"""One place to build Genesis morphs, so import-time defaults stay honest.

Genesis's ``MJCF`` morph exposes an ``align`` option documented as "Default to False",
but the field's actual default is ``None``, and ``None`` behaves like ``True``: root
(floating-base) links get re-framed onto their centre of mass. For a **single-body free
object** that silently changes what a free joint's ``qpos`` position *means* — Genesis
places the COM where MuJoCo would place the body frame origin, so the geometry lands
displaced by the body's ``ipos``.

Measured on a one-body test model whose box sits 0.1 m off the body origin, given the
identical ``qpos``:

===============  ==========================  ==============
                 box centre                  error vs MuJoCo
===============  ==========================  ==============
MuJoCo           (1.1, 2, 3)                 --
``align=None``   (1.0, 2, 3)                 100 mm
``align=True``   (1.0, 2, 3)                 100 mm
``align=False``  (1.1, 2, 3)                 0 mm
===============  ==========================  ==============

On a real RoboCasa episode this showed up as objects landing 0.01-12.45 mm from where the
placement sampler put them, each off by exactly its own ``body_ipos`` — small enough to
look like numerical noise, large enough to make an IK grasp target wrong.

Poly-articulated entities are unaffected (the SMPL-X human's pelvis carries a 20 mm
``ipos`` and transfers identically either way), so this costs nothing where it does not
matter. Load every MJCF through here rather than reaching for ``gs.morphs.MJCF`` directly.
"""

from __future__ import annotations

#: Defaults applied to every MJCF morph. See the module docstring for why ``align``
#: cannot be left at its own default.
MJCF_DEFAULTS = {"align": False}


def mjcf(file, **kwargs):
    """A ``gs.morphs.MJCF`` with this project's defaults applied."""
    import genesis as gs

    return gs.morphs.MJCF(file=str(file), **{**MJCF_DEFAULTS, **kwargs})


__all__ = ["mjcf", "MJCF_DEFAULTS"]
