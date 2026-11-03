import json

from reflexone.cli import main


def test_cli_catalog_describes_real_profiles(capsys):
    assert main(["profiles"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert set(result) == {"support", "commands", "spam"}
    assert result["support"]["domains"] == ["banking", "hotels"]


def test_cli_missing_artifact_is_clear_error(tmp_path, capsys):
    result = main(
        ["predict", "--artifacts", str(tmp_path), "--domain", "banking", "--text", "lost my card"]
    )
    assert result == 2
    assert "error" in capsys.readouterr().err.lower()
