import ast
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class SFTTrainingSurfaceTest(unittest.TestCase):
    def test_sft_has_independent_entrypoints_and_unprefixed_losses(self):
        recipe = (ROOT / "litespec/training/recipes/dlite.py").read_text(
            encoding="utf-8"
        )
        parser = next(
            node
            for node in ast.parse(recipe).body
            if isinstance(node, ast.FunctionDef) and node.name == "parse_sft_args"
        )
        trainer = ast.get_source_segment(recipe, parser)
        launcher = (ROOT / "scripts/dlite/run_training_dlite_sft.sh").read_text(
            encoding="utf-8"
        )
        yaml_launcher = (ROOT / "scripts/config/launch_dlite.py").read_text(
            encoding="utf-8"
        )
        config = (ROOT / "scripts/config/dlite_sft.yaml").read_text(encoding="utf-8")

        self.assertIn('model_role="student"', recipe)
        self.assertIn('"sft": "scripts.dlite.train_sft"', yaml_launcher)
        self.assertIn("dlite_sft.yaml", launcher)
        self.assertNotIn("train_dlite_two_stage", trainer + launcher + config)
        self.assertNotIn("stage1_", trainer + launcher + config)
        self.assertNotIn("stage2_", trainer + launcher + config)
        self.assertNotIn("teacher_draft", trainer + launcher + config)

        for option in (
            "--num-epochs",
            "--loss-decay-gamma",
            "--final-ce-weight",
            "--l1-loss-weight",
            "--base-lm-ce-weight",
            "--base-lm-ce-decay-gamma",
            "--local-position",
        ):
            self.assertIn(option, trainer)
            self.assertIn(option[2:].replace("-", "_") + ":", config)
        self.assertIn("flex_attention_backend:", config)
        self.assertNotIn("dlite_version", trainer + launcher + config)


if __name__ == "__main__":
    unittest.main()
