from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from game_connections.services.beat_backend_spec import BACKEND_BEAT_THIS, build_beat_ctx
from game_connections.services.beat_install import build_beat_uninstall_plan


class BeatInstallRegressionTests(unittest.TestCase):
    def test_build_beat_ctx_preserves_managed_runtime_paths(self):
        managed_ctx = {
            "gpu_vendor": "NVIDIA",
            "libs_dir": r"C:\\managed\\overlay\\site-packages",
            "target_dir": r"C:\\managed\\overlay\\site-packages",
            "python_paths": [
                r"C:\\managed\\core\\site-packages",
                r"C:\\managed\\overlay\\site-packages",
            ],
            "strict_target": True,
        }

        with patch.dict(
            os.environ,
            {"NEUROMITA_LIB_DIR": r"C:\\NeuroMita\\Lib"},
            clear=False,
        ):
            result = build_beat_ctx(managed_ctx)

        self.assertEqual(result["libs_dir"], managed_ctx["libs_dir"])
        self.assertEqual(result["target_dir"], managed_ctx["target_dir"])
        self.assertEqual(result["python_paths"], managed_ctx["python_paths"])
        self.assertTrue(result["strict_target"])

    def test_beat_this_uninstall_plan_contains_only_environment_mutations(self):
        plan = build_beat_uninstall_plan(
            BACKEND_BEAT_THIS,
            ctx={
                "gpu_vendor": "NVIDIA",
                "libs_dir": r"C:\\managed\\overlay\\site-packages",
            },
        )

        self.assertGreater(len(plan.actions), 0)
        self.assertTrue(
            all(action.environment_mutation for action in plan.actions),
            [action.description for action in plan.actions],
        )


if __name__ == "__main__":
    unittest.main()
