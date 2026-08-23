"""
tests/test_tiled_detect.py
───────────────────────────
Sliced-inference geometry and merging.

The load-bearing case is a person straddling a tile seam: it must come out of
the merge as ONE box. Pure IoU NMS cannot do this — one tile sees the whole
person, its neighbour sees a narrow sliver, and if the sliver is under half the
parent box their IoU is below 0.5 so both survive. That case is tested
explicitly, including a regression guard proving IoU alone fails it.
"""
from __future__ import annotations

import numpy as np
import pytest

from cameras.tiled_detect import (_axis_starts, containment, detect_with_tiling,
                                  iou, merge_detections, should_tile,
                                  slice_frame, translate)
from config.identity_config import IdentityConfig


def det(x, y, w, h, conf=0.9, label="person"):
    return {"label": label, "confidence": conf, "bbox": [x, y, w, h]}


# ═══════════════════════════════════════════════════════════════════════════
# 1. Slicing geometry
# ═══════════════════════════════════════════════════════════════════════════

class TestAxisStarts:

    def test_dimension_smaller_than_tile_gives_one_tile(self):
        assert _axis_starts(600, 640, 512) == [0]

    def test_exact_multiple(self):
        assert _axis_starts(1280, 640, 640) == [0, 640]

    def test_last_start_is_shifted_back_not_padded(self):
        """800 wide, 640 tile: the second tile starts at 160 so it ENDS at 800."""
        starts = _axis_starts(800, 640, 512)
        assert starts == [0, 160]
        assert starts[-1] + 640 == 800

    def test_no_start_ever_overruns(self):
        for dim in (200, 640, 641, 800, 1080, 1920, 3840):
            for tile in (320, 640, 960):
                for stride in (tile // 2, int(tile * 0.8), tile):
                    for s in _axis_starts(dim, tile, stride):
                        assert s >= 0
                        assert s + tile <= dim or dim <= tile


class TestSliceFrame:

    def test_tiles_cover_the_whole_frame(self):
        f = np.zeros((600, 800, 3), np.uint8)
        cover = np.zeros((600, 800), bool)
        for tile, (x, y) in slice_frame(f, 640, 0.2):
            cover[y:y + tile.shape[0], x:x + tile.shape[1]] = True
        assert cover.all(), "every pixel must appear in at least one tile"

    def test_no_tile_is_padded(self):
        """A padded tile would be exactly tile_size with synthetic border; the
        real tile must match the frame region it names."""
        f = np.zeros((600, 800, 3), np.uint8)
        for tile, (x, y) in slice_frame(f, 640, 0.2):
            th, tw = tile.shape[:2]
            assert y + th <= 600 and x + tw <= 800
            assert np.shares_memory(tile, f), "tiles should be views, not copies"

    def test_800x600_yields_two_tiles(self):
        tiles = slice_frame(np.zeros((600, 800, 3), np.uint8), 640, 0.2)
        assert len(tiles) == 2
        assert [o for _t, o in tiles] == [(0, 0), (160, 0)]
        assert all(t.shape[:2] == (600, 640) for t, _o in tiles)

    def test_4k_frame_tiles(self):
        tiles = slice_frame(np.zeros((2160, 3840, 3), np.uint8), 640, 0.2)
        assert len(tiles) > 20
        assert all(t.shape[0] <= 640 and t.shape[1] <= 640 for t, _o in tiles)

    def test_frame_smaller_than_tile_is_one_untouched_tile(self):
        f = np.zeros((480, 640, 3), np.uint8)
        tiles = slice_frame(f, 640, 0.2)
        assert len(tiles) == 1
        assert tiles[0][1] == (0, 0)
        assert tiles[0][0].shape[:2] == (480, 640)

    def test_empty_frame(self):
        assert slice_frame(np.zeros((0, 0, 3), np.uint8)) == []
        assert slice_frame(None) == []

    @pytest.mark.parametrize("bad", [0, -1])
    def test_bad_tile_size(self, bad):
        with pytest.raises(ValueError, match="tile_size"):
            slice_frame(np.zeros((10, 10, 3), np.uint8), bad)

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_bad_overlap(self, bad):
        with pytest.raises(ValueError, match="overlap_ratio"):
            slice_frame(np.zeros((10, 10, 3), np.uint8), 640, bad)


class TestTranslate:

    def test_offsets_applied_to_position_only(self):
        out = translate([det(10, 20, 30, 40)], 160, 0)
        assert out[0]["bbox"] == [170, 20, 30, 40]

    def test_does_not_mutate_input(self):
        src = [det(10, 20, 30, 40)]
        translate(src, 100, 100)
        assert src[0]["bbox"] == [10, 20, 30, 40]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Merging — the seam case
# ═══════════════════════════════════════════════════════════════════════════

class TestGeometryHelpers:

    def test_iou_identical(self):
        assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)

    def test_iou_disjoint(self):
        assert iou([0, 0, 10, 10], [50, 50, 10, 10]) == 0.0

    def test_containment_of_a_fragment_is_one(self):
        """A sliver wholly inside the parent is fully contained regardless of
        how small it is — this is what IoU misses."""
        assert containment([0, 0, 10, 100], [0, 0, 100, 100]) == pytest.approx(1.0)
        assert iou([0, 0, 10, 100], [0, 0, 100, 100]) == pytest.approx(0.1)


class TestSeamMerging:
    """A synthetic person placed on a tile seam."""

    # 800x600 frame tiles to [0,640] and [160,800] (seam region 160-640).
    # Person occupies x=600..700, so:
    #   tile 0 sees x 600..640  -> a 40 px sliver
    #   tile 1 sees x 600..700  -> the whole 100 px person
    SLIVER = det(600, 200, 40, 300, conf=0.55)
    WHOLE = det(600, 200, 100, 300, conf=0.90)

    def test_seam_person_merges_to_one_box(self):
        merged = merge_detections([self.SLIVER, self.WHOLE])
        assert len(merged) == 1, "one person on a seam must yield ONE box"
        assert merged[0]["bbox"] == [600, 200, 100, 300], "the WHOLE box must win"

    def test_iou_alone_would_fail_this_case(self):
        """Regression guard for the design decision: disabling containment
        leaves the fragment behind as a second box."""
        assert iou(self.SLIVER["bbox"], self.WHOLE["bbox"]) < 0.5
        iou_only = merge_detections([self.SLIVER, self.WHOLE],
                                    containment_threshold=2.0)
        assert len(iou_only) == 2, (
            "pure IoU NMS keeps the fragment — which is why merging uses "
            "IoU OR containment")

    def test_order_does_not_matter(self):
        a = merge_detections([self.SLIVER, self.WHOLE])
        b = merge_detections([self.WHOLE, self.SLIVER])
        assert a[0]["bbox"] == b[0]["bbox"] == [600, 200, 100, 300]

    def test_fragments_from_both_tiles_still_merge(self):
        """Worst case: neither tile sees the whole person, both see halves that
        overlap inside the seam."""
        left = det(600, 200, 60, 300, conf=0.7)
        right = det(640, 200, 60, 300, conf=0.6)
        merged = merge_detections([left, right])
        assert len(merged) <= 2
        assert merged[0]["bbox"] == [600, 200, 60, 300]

    def test_two_genuinely_distinct_people_are_not_merged(self):
        """The merge must not become a people-fuser: separated subjects stay
        separate. This is the failure that would silently reduce coverage."""
        p1 = det(100, 200, 80, 300, conf=0.9)
        p2 = det(400, 200, 80, 300, conf=0.9)
        assert len(merge_detections([p1, p2])) == 2

    def test_adjacent_touching_people_are_not_merged(self):
        p1 = det(100, 200, 80, 300, conf=0.9)
        p2 = det(182, 200, 80, 300, conf=0.9)      # 2 px gap
        assert len(merge_detections([p1, p2])) == 2

    def test_different_labels_never_merge(self):
        person = det(100, 100, 50, 50, label="person")
        bag = det(100, 100, 50, 50, label="backpack")
        assert len(merge_detections([person, bag])) == 2

    def test_duplicate_across_overlap_region_merges(self):
        """Same person seen twice in the shared strip — the ordinary case."""
        a = det(300, 100, 90, 280, conf=0.88)
        b = det(303, 102, 92, 276, conf=0.81)
        assert len(merge_detections([a, b])) == 1

    def test_empty_input(self):
        assert merge_detections([]) == []

    def test_highest_confidence_survives(self):
        lo = det(100, 100, 80, 200, conf=0.4)
        hi = det(101, 101, 80, 200, conf=0.95)
        out = merge_detections([lo, hi])
        assert len(out) == 1 and out[0]["confidence"] == 0.95


# ═══════════════════════════════════════════════════════════════════════════
# 3. Gating and orchestration
# ═══════════════════════════════════════════════════════════════════════════

class _FakeDetector:
    """Records what it was asked to detect; returns one box per call."""

    def __init__(self, per_call=None):
        self.calls = []
        self._per_call = per_call or []

    def detect(self, frame):
        self.calls.append(frame.shape[:2])
        i = len(self.calls) - 1
        if i < len(self._per_call):
            return self._per_call[i]
        return []


class TestShouldTile:

    def test_off_by_default(self):
        assert IdentityConfig().enable_tiled_detection is False
        assert should_tile(3840, 2160, IdentityConfig()) is False

    def test_enabled_above_the_floor(self):
        cfg = IdentityConfig(enable_tiled_detection=True)
        assert should_tile(1920, 1080, cfg) is True

    def test_skipped_below_the_floor(self):
        """Tiling a 640 px webcam frame is pure cost."""
        cfg = IdentityConfig(enable_tiled_detection=True)
        assert should_tile(640, 480, cfg) is False

    def test_floor_uses_the_longer_edge(self):
        cfg = IdentityConfig(enable_tiled_detection=True,
                             tiled_min_source_resolution=1280)
        assert should_tile(1280, 720, cfg) is True
        assert should_tile(720, 1280, cfg) is True
        assert should_tile(1279, 719, cfg) is False


class TestDetectWithTiling:

    def test_disabled_is_exactly_plain_detect(self):
        d = _FakeDetector([[det(1, 2, 3, 4)]])
        out = detect_with_tiling(d, np.zeros((2160, 3840, 3), np.uint8),
                                 IdentityConfig())
        assert out == [det(1, 2, 3, 4)]
        assert len(d.calls) == 1, "must not run tiles when disabled"

    def test_below_floor_is_exactly_plain_detect(self):
        cfg = IdentityConfig(enable_tiled_detection=True)
        d = _FakeDetector([[det(1, 2, 3, 4)]])
        detect_with_tiling(d, np.zeros((480, 640, 3), np.uint8), cfg)
        assert len(d.calls) == 1

    def test_runs_full_frame_plus_every_tile(self):
        cfg = IdentityConfig(enable_tiled_detection=True,
                             tiled_min_source_resolution=640)
        d = _FakeDetector()
        frame = np.zeros((600, 800, 3), np.uint8)
        detect_with_tiling(d, frame, cfg)
        assert d.calls[0] == (600, 800), "first call is the full frame"
        assert len(d.calls) == 1 + len(slice_frame(frame, 640, 0.2))

    def test_full_frame_box_survives_for_a_large_person(self):
        """A person taller than a tile cannot fit in any tile; the full-frame
        pass is what keeps them from being fragmented."""
        cfg = IdentityConfig(enable_tiled_detection=True,
                             tiled_min_source_resolution=640)
        big = det(100, 0, 300, 600, conf=0.95)
        frag = det(100, 0, 300, 600, conf=0.4)
        d = _FakeDetector([[big], [frag], []])
        out = detect_with_tiling(d, np.zeros((600, 800, 3), np.uint8), cfg)
        assert len(out) == 1 and out[0]["bbox"] == [100, 0, 300, 600]

    def test_tile_only_detection_is_recovered(self):
        """The point of the feature: a box the full-frame pass missed."""
        cfg = IdentityConfig(enable_tiled_detection=True,
                             tiled_min_source_resolution=640)
        small = det(20, 300, 18, 44, conf=0.42)
        d = _FakeDetector([[], [small], []])
        out = detect_with_tiling(d, np.zeros((600, 800, 3), np.uint8), cfg)
        assert len(out) == 1 and out[0]["bbox"] == [20, 300, 18, 44]

    def test_tile_boxes_land_in_frame_coordinates(self):
        cfg = IdentityConfig(enable_tiled_detection=True,
                             tiled_min_source_resolution=640)
        # second tile starts at x=160; a box at tile-local x=10 is frame x=170
        d = _FakeDetector([[], [], [det(10, 5, 20, 50)]])
        out = detect_with_tiling(d, np.zeros((600, 800, 3), np.uint8), cfg)
        assert out[0]["bbox"] == [170, 5, 20, 50]

    def test_a_failing_tile_does_not_lose_the_frame(self):
        class Flaky(_FakeDetector):
            def detect(self, frame):
                self.calls.append(frame.shape[:2])
                if len(self.calls) == 2:
                    raise RuntimeError("tile blew up")
                return [det(1, 1, 5, 5)] if len(self.calls) == 1 else []
        cfg = IdentityConfig(enable_tiled_detection=True,
                             tiled_min_source_resolution=640)
        out = detect_with_tiling(Flaky(), np.zeros((600, 800, 3), np.uint8), cfg)
        assert out and out[0]["bbox"] == [1, 1, 5, 5]

    def test_empty_frame(self):
        assert detect_with_tiling(_FakeDetector(), None, IdentityConfig()) == []
