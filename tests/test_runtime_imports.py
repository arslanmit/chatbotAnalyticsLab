import subprocess
import sys
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def test_main_module_import_does_not_require_ml_stack():
    result = _run_python(
        """
        import src.main
        print("ok")
        """
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_api_app_import_does_not_require_ml_stack():
    result = _run_python(
        """
        from src.api.app import create_app
        app = create_app()
        print(app.title)
        """
    )

    assert result.returncode == 0, result.stderr
    assert "Chatbot Analytics API" in result.stdout


def test_dashboard_import_does_not_require_ml_stack():
    result = _run_python(
        """
        from src.dashboard import compute_overview_metrics
        print(callable(compute_overview_metrics))
        """
    )

    assert result.returncode == 0, result.stderr
    assert "True" in result.stdout
