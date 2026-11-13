import subprocess
import sys


def test_strider_imports_and_lists_models_without_torch():
    """``import strider`` and ``strider models`` must work where torch cannot be imported (the PIT containers)."""
    code = "import sys; sys.modules['torch'] = None; import strider.cli; strider.cli.main(['models'])"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "Model cache" in run.stdout


def test_without_a_version_file_the_version_is_unknown_and_the_command_works():
    code = (
        "import sys; sys.modules['strider._version'] = None; import strider, strider.cli; "
        "print(strider.__version__); strider.cli.main(['models'])"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert run.stdout.splitlines()[0] == "unknown" and "Model cache" in run.stdout
