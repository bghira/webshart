import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("public_text", Path(__file__).parents[1] / "scripts" / "check_public_text.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_generic_payload():
    assert not guard.contains_private_text("Run pytest tests/ and inspect src/lib.rs", {"private-value"})


def test_local_policy():
    assert guard.contains_private_text("PRIVATE-VALUE", {"private-value"})


def test_failure_does_not_echo_input(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(guard, "private_terms", lambda: {"private-value"})
    payload = tmp_path / "payload.txt"
    payload.write_text("private-value")
    assert guard.main(["--text-file", str(payload)]) == 1
    output = capsys.readouterr()
    assert output.err.strip() == guard.BLOCKED
    assert "private-value" not in output.err + output.out


def test_unreadable_payload_is_blocked(tmp_path, capsys):
    assert guard.main(["--text-file", str(tmp_path / "missing")]) == 2
    assert capsys.readouterr().err.strip() == "Blocked: public text could not be checked."


def test_range_checks_intermediate_changes(tmp_path, monkeypatch, capsys):
    import subprocess

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(guard, "private_terms", lambda: {"private-value"})
    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    payload = tmp_path / "payload.txt"
    payload.write_text("generic")
    git("add", ".")
    git("commit", "-m", "Initial")
    payload.write_text("private-value")
    git("commit", "-am", "Intermediate")
    payload.write_text("generic")
    git("commit", "-am", "Final")
    assert guard.main(["--git-range", "HEAD~2..HEAD"]) == 1
    assert capsys.readouterr().err.strip() == guard.BLOCKED


def test_staged_changes_are_checked(tmp_path, monkeypatch, capsys):
    import subprocess

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(guard, "private_terms", lambda: {"private-value"})
    (tmp_path / "payload.txt").write_text("private-value")
    subprocess.run(["git", "add", "."], check=True, capture_output=True)
    assert guard.main(["--staged"]) == 1
    assert capsys.readouterr().err.strip() == guard.BLOCKED
