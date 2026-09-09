"""Eval-mode avatar driver mixin.

When ``config["eval_mode"]=True``, the policy (VLA / RL / diffusion / random)
drives the robot via ``BaseTask.take_action`` and the scripted ``play_once``
choreography is bypassed.  Tasks with avatars need a way to advance the
avatar's animation state in that mode — otherwise the avatar sits idle
forever and the test scenario differs from the demonstration.

This mixin centralises the two hooks subclasses need:

  ``_eval_at_reset()``
    Fired exactly once after ``BaseTask.reset`` returns.  Use it to queue
    the avatar's opening animation and attach any objects the avatar
    starts the episode holding.  Most "single-shot" tasks (avatar plays
    one motion at episode start) need only this hook.

  ``_eval_at_step(step_idx)``
    Fired at the START of every ``take_action`` call before the physics
    substep loop.  ``step_idx`` is a 1-based policy-step counter.  Use
    this for: (a) chaining the next motion when ``avatar.spare()`` flips
    True, (b) randomly-triggered interrupts at step T, (c) cooperative
    loops that must keep firing motions across the rollout.

The mixin must precede the task class (and any other mixins that override
``reset`` / ``take_action``) in the MRO so its ``super()`` calls fall
through correctly:

    class PourWater(EvalModeAvatarMixin, BaseTask):
        ...

When ``eval_mode`` is False (the default) the mixin is a no-op — scripted
``play_once`` flows are unaffected.
"""

from __future__ import annotations


class EvalModeAvatarMixin:
    """See module docstring."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, seed: int = 0):
        obs = super().reset(seed=seed)
        if self._eval_active():
            self._eval_policy_step_count = 0
            self._eval_at_reset()
        return obs

    def take_action(self, action, action_type: str = "qpos"):
        if self._eval_active():
            self._eval_policy_step_count = (
                getattr(self, "_eval_policy_step_count", 0) + 1
            )
            self._eval_at_step(self._eval_policy_step_count)
        return super().take_action(action, action_type=action_type)

    # ------------------------------------------------------------------
    # Predicates / hooks
    # ------------------------------------------------------------------

    def _eval_active(self) -> bool:
        """Mixin is a no-op unless an avatar exists AND eval_mode is on."""
        return (
            getattr(self, "avatar", None) is not None
            and bool(self.config.get("eval_mode", False))
        )

    def _eval_at_reset(self) -> None:
        """Override to queue the avatar's opening behaviour."""
        return None

    def _eval_at_step(self, step_idx: int) -> None:
        """Override to drive avatar state across the rollout."""
        return None
