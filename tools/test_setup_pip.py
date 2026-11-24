"""setup.py's package installs (#753): uv only on Linux ppc64le (IBM POWER9); every other OS / CPU installs with pip
exactly as before, even when uv is on PATH."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import setup as S  # noqa: E402
import maya  # noqa: E402

PKGS = ["numpy", "jinja2"]
PIP = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *PKGS]   # (v1.1.52's)


def uv_on_path(name):
    return "/usr/local/bin/uv" if name == "uv" else None


def missing_distribution(name):
    raise S.importlib.metadata.PackageNotFoundError(name)


class UvOnlyOnPpc64le(unittest.TestCase):
    def test_pip_everywhere_else_even_with_uv_on_path(self):
        for machine, system in [("x86_64", "linux"), ("AMD64", "win32"), ("x86_64", "win32"), ("aarch64", "linux"),
                                ("arm64", "darwin"), ("ARM64", "win32"), ("ppc64", "linux"), ("ppc64le", "win32"),
                                ("ppc64le", "darwin"), ("riscv64", "linux"), ("", "linux")]:
            with self.subTest(machine=machine, system=system):
                self.assertEqual(S.pip_command(PKGS, machine, system, which=uv_on_path), PIP)

    def test_uv_on_linux_ppc64le(self):
        self.assertEqual(S.pip_command(PKGS, "ppc64le", "linux", which=uv_on_path),
                         ["/usr/local/bin/uv", "pip", "install", "--python", sys.executable, "--quiet", *PKGS])

    def test_pip_on_ppc64le_without_uv(self):
        self.assertEqual(S.pip_command(PKGS, "ppc64le", "linux", which=lambda name: None), PIP)

    def test_this_pc_installs_with_pip_with_uv_on_path(self):
        """pip_install itself, on the machine the tests run on (not ppc64le): uv on PATH changes nothing."""
        if S.platform.machine().lower() == "ppc64le" and sys.platform.startswith("linux"):
            self.skipTest("this PC is Linux ppc64le: uv is meant to be used here")
        ran = []
        with tempfile.TemporaryDirectory() as d, patch.object(S.sys, "prefix", d), \
                patch.object(S.importlib.metadata, "version", side_effect=missing_distribution), \
                patch.object(S.shutil, "which", uv_on_path), patch.object(S, "run", lambda cmd, **kw: ran.append(cmd)), \
                patch.object(S, "say", lambda *a, **k: None), patch.object(S, "ok", lambda *a, **k: None):
            S.pip_install(PKGS, "test packages")
        self.assertEqual(ran, [PIP])


class InstalledPackages(unittest.TestCase):
    def test_tools_step_reuses_tokenizers_without_an_installer_record(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "ggml").mkdir()
            (root / "ggml" / "CMakeLists.txt").touch()
            (root / "gguf-py").mkdir()
            args = SimpleNamespace(llama_dir=d, yes=False)
            with patch.object(S.sys, "prefix", d), patch.object(S.importlib.metadata, "version", return_value="0.23.3"), \
                    patch.object(maya, "ask", side_effect=AssertionError("unexpected download prompt")), \
                    patch.object(S, "run", side_effect=AssertionError("unexpected pip install")), \
                    patch.object(maya, "step"), patch.object(maya, "ok"), patch.object(S, "ok"):
                self.assertEqual(maya.tools_step(args, ["tokenizers"]), root.resolve())

    def test_stale_installer_record_does_not_hide_a_missing_package(self):
        with tempfile.TemporaryDirectory() as d, patch.object(S.sys, "prefix", d), \
                patch.object(S.importlib.metadata, "version", side_effect=missing_distribution):
            (Path(d) / ".strata-pip.json").write_text('["tokenizers", "jsonschema>=4.18,<5"]')
            _, _, need = S.pip_package_status(["tokenizers", "jsonschema>=4.18,<5"])
            self.assertEqual(need, ["tokenizers"])


if __name__ == "__main__":
    unittest.main()
