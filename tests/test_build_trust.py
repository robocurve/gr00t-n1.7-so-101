"""Check image bootstrap ordering without importing Modal or running workloads."""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src/modal_app.py"


def image_steps(node):
    """Read a fluent image recipe as ordered method calls."""
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return []
    return image_steps(node.func.value) + [(node.func.attr, node.args)]


class BootstrapTrustTest(unittest.TestCase):
    def test_every_image_refreshes_exclusive_pip_trust_before_installing(self):
        tree = ast.parse(APP.read_text())
        recipes = [
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in {"image", "probe_image"}
                for target in node.targets
            )
        ]
        self.assertEqual(len(recipes), 1)
        for recipe in recipes:
            roots_refreshed = False
            environment = {}
            pip_seen = False
            bootstrap_seen = False
            for method, args in image_steps(recipe):
                if method == "apt_install":
                    roots_refreshed |= any(
                        isinstance(arg, ast.Constant) and arg.value == "ca-certificates"
                        for arg in args
                    )
                if method == "env":
                    for key, value in zip(args[0].keys, args[0].values):
                        if isinstance(key, ast.Constant) and key.value in {
                            "PIP_CERT",
                            "PIP_USE_DEPRECATED",
                        }:
                            environment[key.value] = ast.literal_eval(value)
                if method == "run_commands" and any(
                    isinstance(arg, ast.Constant)
                    and arg.value == "python /opt/robocurve-bootstrap-pip.py"
                    for arg in args
                ):
                    self.assertTrue(roots_refreshed)
                    self.assertNotIn("PIP_USE_DEPRECATED", environment)
                    bootstrap_seen = True
                if method == "pip_install":
                    pip_seen = True
                    self.assertTrue(roots_refreshed)
                    self.assertTrue(bootstrap_seen)
                    self.assertEqual(
                        environment["PIP_CERT"], "/etc/ssl/certs/ca-certificates.crt"
                    )
                    self.assertEqual(environment["PIP_USE_DEPRECATED"], "legacy-certs")
            self.assertTrue(pip_seen)

    def test_bootstrap_never_disables_certificate_verification(self):
        source = APP.read_text()
        for insecure in ("--trusted-host", "PIP_TRUSTED_HOST", "verify=False"):
            self.assertNotIn(insecure, source)


if __name__ == "__main__":
    unittest.main()
