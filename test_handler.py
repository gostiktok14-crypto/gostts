import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


runpod = types.ModuleType("runpod")
runpod.serverless = types.SimpleNamespace(
    progress_update=lambda *_args, **_kwargs: None,
    start=lambda *_args, **_kwargs: None,
)
sys.modules.setdefault("runpod", runpod)

MODULE_PATH = Path(__file__).with_name("handler.py")
SPEC = importlib.util.spec_from_file_location("worudub_export_handler", MODULE_PATH)
handler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(handler)


class DemuxFallbackTests(unittest.TestCase):
    def test_demucs_failure_is_fatal_by_default(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"DEMUCS_DEVICE": "cuda", "DEMUCS_ALLOW_CENTER_CANCEL": "false"},
            clear=False,
        ), patch.object(handler, "run", side_effect=RuntimeError("demucs unavailable")):
            with self.assertRaisesRegex(RuntimeError, "Demucs separation failed"):
                handler.demux_background(Path(tmp) / "source.wav", Path(tmp), {})

    def test_center_cancel_requires_explicit_opt_in(self):
        calls = []

        def fake_run(command, cwd=None, timeout=None):
            calls.append(command)
            if "demucs" in command:
                raise RuntimeError("demucs unavailable")
            Path(command[-1]).write_bytes(b"RIFF-test")
            return ""

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ,
            {"DEMUCS_DEVICE": "cuda", "DEMUCS_ALLOW_CENTER_CANCEL": "true"},
            clear=False,
        ), patch.object(handler, "run", side_effect=fake_run):
            output, vocal_output, mode = handler.demux_background(Path(tmp) / "source.wav", Path(tmp), {})
            self.assertEqual(mode, "center_cancel")
            self.assertTrue(output.is_file())
            self.assertEqual(vocal_output, Path(tmp) / "source.wav")
            self.assertEqual(len(calls), 2)


class InputValidationTests(unittest.TestCase):
    def test_download_rejects_non_http_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "absolute HTTP"):
                handler.download_file("file:///etc/passwd", Path(tmp) / "input.bin")


class MixFilterTests(unittest.TestCase):
    def test_each_delayed_input_is_separated_in_filter_graph(self):
        captured = []

        def fake_input_file(segment, _base_name, workdir):
            return workdir / f"source_{segment['id']}.wav"

        def fake_run(command, cwd=None, timeout=None):
            captured.append(command)
            return ""

        segments = [
            {"id": 1, "start_seconds": 0, "target_duration": 1, "volume": 1},
            {"id": 2, "start_seconds": 1, "target_duration": 1, "volume": 1},
        ]
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            handler, "input_file", side_effect=fake_input_file
        ), patch.object(handler, "prepare_dub_audio"), patch.object(
            handler, "run", side_effect=fake_run
        ):
            handler.mix_audio(Path(tmp) / "background.wav", segments, Path(tmp), {})

        filter_graph = captured[0][captured[0].index("-filter_complex") + 1]
        self.assertEqual(filter_graph.count(";"), 2)
        self.assertIn("[1:a]adelay=0:all=1,volume=1.0[d1];[2:a]", filter_graph)
        self.assertIn("[0:a][d1][d2]amix=inputs=3", filter_graph)


if __name__ == "__main__":
    unittest.main()
