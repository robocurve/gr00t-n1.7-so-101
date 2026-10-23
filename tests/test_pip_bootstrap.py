"""Exercise old/new pip bootstrap selection without downloads or installs."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pip_bootstrap", ROOT / "src/pip_bootstrap.py"
)
bootstrap = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap)


class PipBootstrapTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.bundle = Path(self.directory.name) / "ca.crt"
        self.bundle.write_text("bundle fixture; no network call is made")

    def run_recipe(self, initial):
        runner = Mock()
        environment = {
            "PIP_USE_FEATURE": "truststore",
            "PIP_TRUSTED_HOST": "bad.invalid",
            "PIP_USE_DEPRECATED": "legacy-certs",
            "REQUESTS_CA_BUNDLE": "/old/roots",
            "CURL_CA_BUNDLE": "/old/roots",
        }
        bootstrap.bootstrap(initial, environment, runner, str(self.bundle))
        return runner.call_args

    def test_old_pip_avoids_unsupported_option_and_opt_in_truststore(self):
        args, kwargs = self.run_recipe("22.0.2")
        self.assertNotIn("PIP_USE_DEPRECATED", kwargs["env"])
        self.assertNotIn("PIP_USE_FEATURE", kwargs["env"])
        self.assertEqual(kwargs["env"]["PIP_CONFIG_FILE"], "/dev/null")
        self.assertEqual(kwargs["env"]["PIP_CERT"], str(self.bundle))
        self.assertIn("pip==26.2.2", args[0])

    def test_modern_pip_uses_exclusive_bundle_and_wheel_only_bootstrap(self):
        args, kwargs = self.run_recipe("25.1.1")
        self.assertEqual(kwargs["env"]["PIP_USE_DEPRECATED"], "legacy-certs")
        for key in ("PIP_TRUSTED_HOST", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            self.assertNotIn(key, kwargs["env"])
        self.assertIn("--only-binary=:all:", args[0])
        self.assertIn("--no-deps", args[0])
        self.assertTrue(kwargs["check"])

    def test_unknown_version_fails_before_download(self):
        runner = Mock()
        with self.assertRaises(RuntimeError):
            bootstrap.bootstrap("unknown", {}, runner, str(self.bundle))
        runner.assert_not_called()

    def test_missing_bundle_fails_before_download(self):
        runner = Mock()
        with self.assertRaises(RuntimeError):
            bootstrap.bootstrap("25.1.1", {}, runner, str(self.bundle) + ".missing")
        runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
