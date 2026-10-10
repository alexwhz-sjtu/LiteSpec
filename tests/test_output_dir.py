from litespec.training.checkpoint import reserve_output_dir


def test_reserve_output_dir_adds_incrementing_suffix(tmp_path):
    requested = tmp_path / "run"

    assert reserve_output_dir(str(requested)) == str(requested)
    assert reserve_output_dir(f"{requested}/") == f"{requested}_1"
    assert reserve_output_dir(str(requested)) == f"{requested}_2"
