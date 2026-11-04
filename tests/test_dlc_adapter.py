import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from futsal_markerless_analysis.adapters.dlc import (
    _snapshot_index_for_checkpoint,
    model_registry,
)


class DeepLabCutAdapterTest(unittest.TestCase):
    def test_snapshot_index_resolves_the_registry_checkpoint_by_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = [
                SimpleNamespace(path=root / "snapshot-005.pt"),
                SimpleNamespace(path=root / "snapshot-025.pt"),
                SimpleNamespace(path=root / "snapshot-best-035.pt"),
            ]

            self.assertEqual(_snapshot_index_for_checkpoint(snapshots, root / "snapshot-025.pt"), 1)

    def test_snapshot_index_rejects_missing_or_ambiguous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = SimpleNamespace(path=root / "snapshot-025.pt")

            with self.assertRaisesRegex(ValueError, "found 0"):
                _snapshot_index_for_checkpoint([snapshot], root / "snapshot-030.pt")
            with self.assertRaisesRegex(ValueError, "found 2"):
                _snapshot_index_for_checkpoint([snapshot, snapshot], root / "snapshot-025.pt")

    def test_registry_lists_all_five_configurations_without_distributing_weights(self):
        registry = model_registry()

        self.assertEqual(
            set(registry["models"]),
            {"ResNet-101", "HRNet-W48", "RTMPose-S", "RTMPose-M", "RTMPose-X"},
        )
        self.assertFalse(registry["weights_distributed"])


if __name__ == "__main__":
    unittest.main()
