"""Benchmark-approved avatar model inventory.

Celebrity likenesses are intentionally excluded from runtime sampling because
their licensing/status is not suitable for benchmark distribution.  The Mixamo
sentinel model is also excluded because it is a game character rather than a
human-realistic avatar.  Models that do not satisfy the benchmark avatar
skeleton mapping are excluded too; random sampling must never pick a model that
crashes AvatarController initialization.
"""

from __future__ import annotations

from pathlib import Path


AVATAR_MODELS = (
    "avatars/models/custom_Adrian_Keller.glb",
    "avatars/models/custom_Clara_Voss.glb",
    "avatars/models/custom_Dylan_Mercer.glb",
    "avatars/models/custom_Elena_Markovic.glb",
    "avatars/models/custom_Elena_Petrovna.glb",
    "avatars/models/custom_Elliot_Marsh.glb",
    "avatars/models/custom_Ethan_Miyamoto.glb",
    "avatars/models/custom_Felix_Braun.glb",
    "avatars/models/custom_Freya_Novak.glb",
    "avatars/models/custom_Ivy_Holloway.glb",
    "avatars/models/custom_Julian_Cross.glb",
    "avatars/models/custom_Kenji_Watanabe.glb",
    "avatars/models/custom_Layla_Haddad.glb",
    "avatars/models/custom_Liam_Novak.glb",
    "avatars/models/custom_Malik_Okoro.glb",
    "avatars/models/custom_Marco_Ruiz.glb",
    "avatars/models/custom_Marcus_Yuen.glb",
    "avatars/models/custom_Mira_Johansson.glb",
    "avatars/models/custom_Morten_Lindqvist.glb",
    "avatars/models/custom_Naomi_Zhang.glb",
    "avatars/models/custom_Nico_Vega.glb",
    "avatars/models/custom_Rafael_Costa.glb",
    "avatars/models/custom_Sophia_Lindberg.glb",
    "avatars/models/custom_Tariq_Johnson.glb",
    "avatars/models/custom_Theo_Caldwell.glb",
    "avatars/models/custom_Victor_Han.glb",
    "avatars/models/custom_Walter_Greene.glb",
    "avatars/models/custom_Yara_Mbatha.glb",
    "avatars/models/mixamo_Alex_Jefferson.glb",
    "avatars/models/mixamo_Brian_Carter.glb",
    "avatars/models/mixamo_Brycer_Rodriguez.glb",
    "avatars/models/mixamo_Chad_Thompson.glb",
    "avatars/models/mixamo_Elizabeth_Mensah.glb",
    "avatars/models/mixamo_James_Thompson.glb",
    "avatars/models/mixamo_Jody_Petersen.glb",
    "avatars/models/mixamo_Joe_Anderson.glb",
    "avatars/models/mixamo_Kate_Novak.glb",
    "avatars/models/mixamo_Leonard_Park.glb",
    "avatars/models/mixamo_Megan_Carter.glb",
    "avatars/models/mixamo_Olivia_Adams.glb",
    "avatars/models/mixamo_Pika_Hayes.glb",
    "avatars/models/mixamo_Shannon_Steves.glb",
    "avatars/models/mixamo_Sophie_Carlsen.glb",
)


EXCLUDED_AVATAR_MODEL_PATTERNS = (
    "avatars/models/celebrity_*.glb",
)


EXCLUDED_AVATAR_MODELS = (
    "avatars/models/custom_Scarlett_Byrne.glb",
    "avatars/models/custom_Zane_Matthews.glb",
    "avatars/models/mixamo_Malik_Okoro.glb",
    "avatars/models/mixamo_Sophia_Lindberg.glb",
    "avatars/models/mixamo_sentinel.glb",
    "avatars/models/mixamo_Eve_By_J_Gonzales.glb",
    "avatars/models/mixamo_Erika_Archer.glb",
    "avatars/models/mixamo_Emily_Johnson.glb",
    "avatars/models/mixamo_Steve_Johnson.glb",
    "avatars/models/mixamo_Zara_Williams.glb",
    # Excluded: GLB baseColorFactor alpha > 1.0, which Genesis 1.2.0's
    # ImageTexture validation rejects ("image_color[3] must be <= 1"), crashing
    # AvatarController init. Adam_Pierce (1.8) observed; Louise_Roberts (1.3) and
    # Roth_Miller (5.4) share the same defect.
    "avatars/models/mixamo_Adam_Pierce.glb",
    "avatars/models/mixamo_Louise_Roberts.glb",
    "avatars/models/mixamo_Roth_Miller.glb",
)


DEFAULT_AVATAR_POOL = AVATAR_MODELS


def avatar_model_name(model_path: str) -> str:
    return Path(model_path).stem


def iter_avatar_models() -> tuple[str, ...]:
    return AVATAR_MODELS
