from __future__ import annotations

import unittest

from backend.media import build_video_filters, keep_intervals, merge_ranges
from backend.models import RenderRequest, VideoSettings
from backend.speech import detect_fillers, detect_repeated_starts


class RangeTests(unittest.TestCase):
    def test_ranges_merge_and_are_clipped(self):
        self.assertEqual(merge_ranges([(2, 3), (2.95, 4), (8, 9)]), [(2.0, 4.0), (8.0, 9.0)])
        self.assertEqual(
            keep_intervals(1, 10, [(0, 2), (4, 5), (9, 12)]),
            [(2.0, 4.0), (5.0, 9.0)],
        )

    def test_cannot_remove_entire_clip(self):
        with self.assertRaises(ValueError):
            keep_intervals(0, 3, [(0, 3)])


class SpeechTests(unittest.TestCase):
    @staticmethod
    def words(values):
        result = []
        cursor = 0.0
        for value in values:
            result.append({"word": value, "token": value.lower().strip(".,-"), "start": cursor, "end": cursor + 0.22})
            cursor += 0.3
        return result

    def test_filler_detection(self):
        found = detect_fillers(self.words(["Today", "um", "we", "begin"]))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["text"], "um")

    def test_repeated_phrase_detection(self):
        found = detect_repeated_starts(self.words(["I", "wanted", "to", "I", "wanted", "to", "explain"]))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["reason"], "repeated false start")
        self.assertIn("wanted", found[0]["text"])

    def test_intentional_short_repeat_is_not_cut(self):
        found = detect_repeated_starts(self.words(["very", "very", "good"]))
        self.assertEqual(found, [])


class ModelTests(unittest.TestCase):
    def test_request_defaults_and_clamps(self):
        request = RenderRequest.from_dict(
            {
                "clips": [{"upload_id": "0123456789abcdef", "trim_start": -8}],
                "video": {"exposure": 90},
                "audio": {"target_lufs": -99},
                "transition": {"style": "unknown", "duration": 99},
            }
        )
        self.assertEqual(request.clips[0].trim_start, 0)
        self.assertEqual(request.video.exposure, 2)
        self.assertEqual(request.audio.target_lufs, -24)
        self.assertEqual(request.transition.style, "fade")
        self.assertEqual(request.transition.duration, 2)

    def test_video_filter_is_deterministic(self):
        filters = build_video_filters(VideoSettings(), 1920, 1080, 30)
        self.assertTrue(any(value.startswith("normalize=") for value in filters))
        self.assertTrue(any(value.startswith("eq=") for value in filters))
        self.assertEqual(filters[-1], "format=yuv420p")


if __name__ == "__main__":
    unittest.main()

