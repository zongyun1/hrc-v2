from envs.exr_envmap import normalize_env_offset, offset_env_texture


def test_normalize_env_offset_sequence_and_mapping():
    assert normalize_env_offset([12, -3.5]) == (12.0, -3.5)
    assert normalize_env_offset(
        {"horizontal_deg": -7, "vertical_deg": 4}
    ) == (-7.0, 4.0)


def test_zero_offset_does_not_touch_source():
    source = "does/not/need/to/exist.exr"
    assert offset_env_texture(source, [0, 0]) == source
