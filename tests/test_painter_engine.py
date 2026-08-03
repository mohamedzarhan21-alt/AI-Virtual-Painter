"""Tests for the AI Virtual Painter engine.

Everything except the Tk window is exercised headlessly: the raster primitives,
the brush engine, the document/undo model, shape recognition, export, the
gesture state machine and a full HUD render pass.
"""

from __future__ import annotations

import math
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ai_virtual_painter as aivp

# --------------------------------------------------------------------------- #
# Colour, maths and easing
# --------------------------------------------------------------------------- #


def test_hex_to_bgr_reverses_channel_order() -> None:
    assert aivp.hex_to_bgr("#112233") == (0x33, 0x22, 0x11)
    assert aivp.hex_to_bgr("FF8000") == (0x00, 0x80, 0xFF)


@pytest.mark.parametrize("value", ["#12345", "", "#GGGGGG12"])
def test_hex_to_bgr_rejects_malformed_input(value: str) -> None:
    with pytest.raises(ValueError):
        aivp.hex_to_bgr(value)


def test_clamp_and_lerp_boundaries() -> None:
    assert aivp.clamp(-5.0, 0.0, 1.0) == 0.0
    assert aivp.clamp(5.0, 0.0, 1.0) == 1.0
    assert aivp.lerp(10.0, 20.0, 0.25) == pytest.approx(12.5)


@pytest.mark.parametrize("ease", [aivp.ease_out_cubic, aivp.ease_in_out_sine])
def test_easings_are_normalised(ease) -> None:
    assert ease(0.0) == pytest.approx(0.0, abs=1e-6)
    assert ease(1.0) == pytest.approx(1.0, abs=1e-6)


def test_ease_out_back_overshoots_before_settling() -> None:
    assert max(aivp.ease_out_back(t / 50) for t in range(51)) > 1.0
    assert aivp.ease_out_back(1.0) == pytest.approx(1.0, abs=1e-6)


def test_hue_to_bgr_cycles_through_primaries() -> None:
    assert aivp.hue_to_bgr(0.0) == (0, 0, 255)
    assert aivp.hue_to_bgr(1.0 / 3.0) == (0, 255, 0)
    assert aivp.hue_to_bgr(2.0 / 3.0) == (255, 0, 0)


def test_mix_bgr_is_clamped_and_symmetric() -> None:
    assert aivp.mix_bgr((0, 0, 0), (255, 255, 255), 0.5) == (127, 127, 127)
    assert aivp.mix_bgr((0, 0, 0), (255, 255, 255), -3.0) == (0, 0, 0)
    assert aivp.mix_bgr((0, 0, 0), (255, 255, 255), 9.0) == (255, 255, 255)


def test_animated_eases_towards_target_and_settles() -> None:
    animation = aivp.Animated(0.0, speed=20.0)
    animation.set(1.0)
    for _ in range(200):
        animation.update(1 / 60)
    assert animation.value == pytest.approx(1.0, abs=1e-3)


def test_animated_snap_is_instantaneous() -> None:
    animation = aivp.Animated(0.0)
    animation.snap(0.5)
    assert animation.value == 0.5 and animation.target == 0.5


def test_animated_color_cross_fades() -> None:
    fade = aivp.AnimatedColor((0, 0, 0), speed=30.0)
    fade.set((255, 255, 255))
    for _ in range(200):
        fade.update(1 / 60)
    assert fade.value == (255, 255, 255)


# --------------------------------------------------------------------------- #
# Filtering and path sampling
# --------------------------------------------------------------------------- #


def test_one_euro_filter_suppresses_jitter_on_a_still_signal() -> None:
    rng = np.random.default_rng(3)
    noisy = 100.0 + rng.normal(0.0, 3.0, size=300)
    filtered = []
    smoother = aivp.OneEuroFilter()
    for index, sample in enumerate(noisy):
        filtered.append(smoother.filter(float(sample), index / 60.0))
    assert float(np.std(filtered[50:])) < float(np.std(noisy[50:])) * 0.45


def test_one_euro_filter_tracks_a_fast_ramp() -> None:
    smoother = aivp.OneEuroFilter()
    for index in range(240):
        result = smoother.filter(index * 12.0, index / 60.0)
    assert result == pytest.approx(239 * 12.0, rel=0.06)


def test_point_filter_reset_forgets_history() -> None:
    point_filter = aivp.PointFilter()
    point_filter.filter((0.0, 0.0), 0.0)
    point_filter.reset()
    assert point_filter.filter((900.0, 400.0), 1.0) == (900.0, 400.0)


def test_resample_polyline_produces_uniform_spacing() -> None:
    points = aivp.resample_polyline([(0.0, 0.0), (100.0, 0.0)], 10.0)
    gaps = [math.dist(a, b) for a, b in pairwise(points)]
    assert all(gap == pytest.approx(10.0, abs=1e-6) for gap in gaps)


def test_catmull_rom_interpolates_its_control_points() -> None:
    p0, p1, p2, p3 = (0.0, 0.0), (10.0, 0.0), (20.0, 10.0), (30.0, 10.0)
    assert aivp.catmull_rom(p0, p1, p2, p3, 0.0) == pytest.approx(p1)
    assert aivp.catmull_rom(p0, p1, p2, p3, 1.0) == pytest.approx(p2)


def test_path_sampler_emits_uniformly_spaced_samples() -> None:
    sampler = aivp.PathSampler(spacing=5.0)
    produced: list[aivp.PathSample] = []
    for step in range(20):
        produced.extend(sampler.push(step * 20.0, 100.0, 0.0))
    produced.extend(sampler.finish())
    assert len(produced) > 20
    gaps = [math.dist((a.x, a.y), (b.x, b.y)) for a, b in pairwise(produced[1:])]
    assert gaps and max(gaps) < 5.6


def test_path_sampler_indices_are_contiguous_and_deterministic() -> None:
    def run() -> list[aivp.PathSample]:
        sampler = aivp.PathSampler(spacing=4.0)
        out: list[aivp.PathSample] = []
        for step in range(12):
            out.extend(sampler.push(step * 13.0, 90.0 + step * 7.0, 100.0))
        out.extend(sampler.finish())
        return out

    first, second = run(), run()
    assert [s.index for s in first] == list(range(len(first)))
    assert [(s.x, s.y, s.distance) for s in first] == [(s.x, s.y, s.distance) for s in second]


# --------------------------------------------------------------------------- #
# Raster primitives
# --------------------------------------------------------------------------- #


def test_clip_rect_returns_none_when_fully_outside() -> None:
    assert aivp.clip_rect((-40, -40, -10, -10), 100, 100) is None
    assert aivp.clip_rect((-5, -5, 20, 20), 100, 100) == (0, 0, 20, 20)


def test_union_rect_handles_missing_operands() -> None:
    assert aivp.union_rect(None, (1, 2, 3, 4)) == (1, 2, 3, 4)
    assert aivp.union_rect((1, 2, 3, 4), None) == (1, 2, 3, 4)
    assert aivp.union_rect((0, 0, 5, 5), (3, 3, 9, 9)) == (0, 0, 9, 9)


def test_composite_over_matches_the_porter_duff_result() -> None:
    destination = aivp.new_layer(1, 1)
    destination[0, 0] = (0, 0, 255, 255)
    source = aivp.new_layer(1, 1)
    source[0, 0] = (255, 0, 0, 128)
    aivp.composite_over(destination, source)
    blue, green, red, alpha = destination[0, 0]
    assert alpha == 255
    assert blue == pytest.approx(128, abs=2)
    assert red == pytest.approx(127, abs=2)
    assert green == 0


def test_erase_with_subtracts_alpha_only() -> None:
    destination = aivp.new_layer(1, 1)
    destination[0, 0] = (10, 20, 30, 200)
    cutter = aivp.new_layer(1, 1)
    cutter[0, 0, 3] = 80
    aivp.erase_with(destination, cutter)
    assert tuple(destination[0, 0]) == (10, 20, 30, 120)


def test_flatten_onto_blends_by_alpha() -> None:
    background = np.zeros((1, 1, 3), dtype=np.uint8)
    layer = aivp.new_layer(1, 1)
    layer[0, 0] = (200, 100, 50, 128)
    aivp.flatten_onto(background, layer)
    assert tuple(background[0, 0]) == pytest.approx((100, 50, 25), abs=2)


def test_stamp_sprite_max_mode_does_not_compound_opacity() -> None:
    layer = aivp.new_layer(40, 40)
    sprite = aivp.SPRITES.disc(6.0, 0.2)
    for _ in range(8):
        aivp.stamp_sprite(layer, sprite, 20, 20, (0, 0, 255), 0.4, aivp.BlendMode.MAX)
    assert layer[20, 20, 3] == pytest.approx(102, abs=3)


def test_stamp_sprite_over_mode_accumulates_density() -> None:
    layer = aivp.new_layer(40, 40)
    sprite = aivp.SPRITES.disc(6.0, 0.2)
    for _ in range(8):
        aivp.stamp_sprite(layer, sprite, 20, 20, (0, 0, 255), 0.2, aivp.BlendMode.OVER)
    assert layer[20, 20, 3] > 200


def test_stamp_sprite_off_canvas_is_a_no_op() -> None:
    layer = aivp.new_layer(20, 20)
    assert aivp.stamp_sprite(layer, aivp.SPRITES.disc(3.0, 0.5), -50, -50, (1, 2, 3), 1.0, aivp.BlendMode.MAX) is None
    assert not layer.any()


def test_sprite_cache_returns_identical_objects_for_quantised_radii() -> None:
    first = aivp.SPRITES.disc(8.0, 0.4)
    second = aivp.SPRITES.disc(8.04, 0.401)
    assert first is second


def test_glass_panel_alters_only_its_own_rectangle() -> None:
    frame = np.full((80, 120, 3), 40, dtype=np.uint8)
    aivp.draw_glass_panel(frame, (10, 10, 60, 50), 8, border=aivp.THEME.primary)
    assert frame[5, 5].tolist() == [40, 40, 40]
    assert frame[30, 30].tolist() != [40, 40, 40]


def test_text_renderer_measures_and_draws() -> None:
    width, height = aivp.TEXT.measure("AIVP", aivp.FontRole.DISPLAY, 24, 2)
    assert width > 0 and height > 0
    frame = np.zeros((60, 200, 3), dtype=np.uint8)
    assert aivp.TEXT.draw(frame, "AIVP", (10, 10), size=24, color=(255, 255, 255)) is not None
    assert frame.any()


def test_text_renderer_ignores_empty_strings() -> None:
    frame = np.zeros((30, 60, 3), dtype=np.uint8)
    assert aivp.TEXT.draw(frame, "", (5, 5)) is None
    assert not frame.any()


# --------------------------------------------------------------------------- #
# Palette and brushes
# --------------------------------------------------------------------------- #


def test_palette_covers_every_requested_colour() -> None:
    names = {slot.name for slot in aivp.PALETTE}
    required = {
        "RED",
        "BLUE",
        "GREEN",
        "YELLOW",
        "ORANGE",
        "PURPLE",
        "PINK",
        "WHITE",
        "BLACK",
        "BROWN",
        "CYAN",
        "MAGENTA",
        "GRADIENT",
        "RAINBOW",
    }
    assert required <= names


def test_solid_slot_ignores_distance() -> None:
    red = next(slot for slot in aivp.PALETTE if slot.name == "RED")
    assert red.sample(0.0) == red.sample(1234.5)


def test_gradient_and_rainbow_slots_vary_along_the_stroke() -> None:
    for name in ("GRADIENT", "RAINBOW"):
        slot = next(item for item in aivp.PALETTE if item.name == name)
        assert slot.sample(0.0) != slot.sample(150.0)


def test_every_brush_is_specified_and_renderable() -> None:
    assert set(aivp.BRUSH_SPECS) == set(aivp.BrushId)
    assert set(aivp.BRUSH_RENDERERS) == set(aivp.BrushId)


@pytest.mark.parametrize("brush", list(aivp.BrushId))
def test_every_brush_paints_visible_pixels(brush: aivp.BrushId) -> None:
    layer = aivp.new_layer(240, 160)
    stroke = aivp.Stroke(
        brush=brush,
        color_index=0,
        thickness=20,
        opacity=1.0,
        seed=11,
        points=[(30.0 + index * 18.0, 80.0, 220.0) for index in range(11)],
    )
    dirty = aivp.StrokeRasteriser.render_complete(stroke, layer)
    assert dirty is not None
    assert int(layer[:, :, 3].sum()) > 0


@pytest.mark.parametrize("brush", list(aivp.BrushId))
def test_brush_rendering_is_deterministic(brush: aivp.BrushId) -> None:
    stroke = aivp.Stroke(
        brush=brush,
        color_index=12,
        thickness=15,
        opacity=0.8,
        seed=99,
        points=[(20.0 + index * 15.0, 60.0 + index * 4.0, 150.0) for index in range(10)],
    )
    first, second = aivp.new_layer(220, 140), aivp.new_layer(220, 140)
    aivp.StrokeRasteriser.render_complete(stroke, first)
    aivp.StrokeRasteriser.render_complete(stroke, second)
    assert np.array_equal(first, second)


def test_incremental_rendering_matches_a_full_rebuild() -> None:
    """The live drawing path and the undo rebuild path must agree exactly."""
    points = [(25.0 + index * 12.0, 70.0 + math.sin(index * 0.6) * 25.0, 180.0) for index in range(16)]
    stroke = aivp.Stroke(brush=aivp.BrushId.GALAXY, color_index=6, thickness=20, opacity=0.9, seed=5, points=points)
    incremental = aivp.new_layer(260, 160)
    rasteriser = aivp.StrokeRasteriser(stroke)
    for x, y, velocity in points:
        rasteriser.feed(incremental, x, y, velocity)
    rasteriser.flush(incremental)
    rebuilt = aivp.new_layer(260, 160)
    aivp.StrokeRasteriser.render_complete(stroke, rebuilt)
    assert np.array_equal(incremental, rebuilt)


def test_thickness_scale_changes_stroke_footprint() -> None:
    def footprint(thickness: int) -> int:
        layer = aivp.new_layer(200, 200)
        stroke = aivp.Stroke(
            brush=aivp.BrushId.MARKER,
            color_index=10,
            thickness=thickness,
            opacity=1.0,
            seed=1,
            points=[(40.0, 100.0, 0.0), (160.0, 100.0, 0.0)],
        )
        aivp.StrokeRasteriser.render_complete(stroke, layer)
        return int((layer[:, :, 3] > 0).sum())

    assert footprint(40) > footprint(5) * 3


def test_brush_swatch_renders_within_its_bounds() -> None:
    swatch = aivp.render_brush_swatch(aivp.BrushId.NEON, 4, 10, 1.0, (80, 40))
    assert swatch.shape == (40, 80, 4)
    assert swatch[:, :, 3].any()


# --------------------------------------------------------------------------- #
# Shape recognition
# --------------------------------------------------------------------------- #


def _traced(points: list[tuple[float, float]]) -> list[tuple[float, float, float]]:
    return [(x, y, 120.0) for x, y in points]


def _circle_trace(radius: float = 90.0, wobble: float = 2.0) -> list[tuple[float, float, float]]:
    rng = np.random.default_rng(1)
    return _traced(
        [
            (
                300.0 + math.cos(step / 44 * math.tau) * (radius + rng.normal(0.0, wobble)),
                300.0 + math.sin(step / 44 * math.tau) * (radius + rng.normal(0.0, wobble)),
            )
            for step in range(45)
        ]
    )


def test_recognises_a_circle() -> None:
    fit = aivp.ShapeRecogniser.identify(_circle_trace())
    assert fit is not None and fit.kind is aivp.ShapeKind.CIRCLE
    centre = np.mean(np.array(fit.points), axis=0)
    assert centre == pytest.approx((300.0, 300.0), abs=6.0)


def test_recognises_a_rectangle() -> None:
    corners = [(120.0, 120.0), (420.0, 118.0), (422.0, 300.0), (118.0, 302.0), (120.0, 120.0)]
    fit = aivp.ShapeRecogniser.identify(_traced(aivp.resample_polyline(corners, 9.0)))
    assert fit is not None and fit.kind is aivp.ShapeKind.RECTANGLE


def test_recognises_a_triangle() -> None:
    corners = [(200.0, 400.0), (400.0, 400.0), (300.0, 200.0), (200.0, 400.0)]
    fit = aivp.ShapeRecogniser.identify(_traced(aivp.resample_polyline(corners, 8.0)))
    assert fit is not None and fit.kind is aivp.ShapeKind.TRIANGLE


def test_recognises_a_line_and_returns_its_endpoints() -> None:
    fit = aivp.ShapeRecogniser.identify(_traced([(100.0 + index * 12.0, 200.0 + index * 0.4) for index in range(30)]))
    assert fit is not None and fit.kind is aivp.ShapeKind.LINE
    assert math.dist(fit.points[0], (100.0, 200.0)) < 8.0


def test_recognises_a_single_stroke_arrow() -> None:
    shaft = aivp.resample_polyline([(100.0, 300.0), (400.0, 300.0)], 8.0)
    barb = aivp.resample_polyline([(400.0, 300.0), (340.0, 250.0)], 8.0)
    fit = aivp.ShapeRecogniser.identify(_traced([*shaft, *barb]))
    assert fit is not None and fit.kind is aivp.ShapeKind.ARROW


def test_rejects_short_strokes_and_scribbles() -> None:
    assert aivp.ShapeRecogniser.identify(_traced([(0.0, 0.0), (3.0, 3.0)])) is None
    rng = np.random.default_rng(12)
    scribble = _traced([(200.0 + rng.uniform(-90, 90), 200.0 + rng.uniform(-90, 90)) for _ in range(40)])
    assert aivp.ShapeRecogniser.identify(scribble) is None


def test_arrow_is_rejected_when_the_barb_is_as_long_as_the_shaft() -> None:
    shaft = aivp.resample_polyline([(100.0, 300.0), (400.0, 300.0)], 8.0)
    barb = aivp.resample_polyline([(400.0, 300.0), (200.0, 150.0)], 8.0)  # barb/shaft = 0.83
    assert aivp.ShapeRecogniser.identify(_traced([*shaft, *barb])) is None


def test_a_stroke_doubling_back_on_itself_reads_as_a_line() -> None:
    shaft = aivp.resample_polyline([(100.0, 300.0), (400.0, 300.0)], 8.0)
    back = aivp.resample_polyline([(400.0, 300.0), (250.0, 300.0)], 8.0)
    fit = aivp.ShapeRecogniser.identify(_traced([*shaft, *back]))
    assert fit is not None and fit.kind is aivp.ShapeKind.LINE


# --------------------------------------------------------------------------- #
# Document, history and export
# --------------------------------------------------------------------------- #


def _paint(document: aivp.PaintDocument, settings: aivp.BrushSettings, y: float, eraser: bool = False) -> None:
    document.begin_stroke(settings, eraser=eraser)
    for step in range(12):
        document.extend_stroke(60.0 + step * 20.0, y, 140.0)
    document.end_stroke(snap_shapes=False)


def test_document_starts_empty() -> None:
    document = aivp.PaintDocument(320, 240)
    assert document.is_empty and not document.can_undo and not document.can_redo
    assert document.content_rect is None


def test_painting_marks_the_canvas_and_records_a_stroke() -> None:
    document = aivp.PaintDocument(400, 300)
    _paint(document, aivp.BrushSettings(brush=aivp.BrushId.MARKER), 150.0)
    assert document.stroke_count == 1
    assert document.canvas[:, :, 3].any()
    assert document.content_rect is not None


def test_undo_and_redo_restore_identical_rasters() -> None:
    document = aivp.PaintDocument(400, 300)
    settings = aivp.BrushSettings(brush=aivp.BrushId.BRUSH)
    _paint(document, settings, 100.0)
    after_first = document.canvas.copy()
    _paint(document, settings, 180.0)
    after_second = document.canvas.copy()

    assert document.undo()
    assert np.array_equal(document.canvas, after_first)
    assert document.redo()
    assert np.array_equal(document.canvas, after_second)


def test_undo_walks_all_the_way_back_to_a_blank_canvas() -> None:
    document = aivp.PaintDocument(300, 200)
    settings = aivp.BrushSettings(brush=aivp.BrushId.MARKER)
    for index in range(4):
        _paint(document, settings, 40.0 + index * 30.0)
    while document.undo():
        pass
    assert document.stroke_count == 0
    assert not document.canvas[:, :, 3].any()


def test_new_stroke_discards_the_redo_branch() -> None:
    document = aivp.PaintDocument(300, 200)
    settings = aivp.BrushSettings(brush=aivp.BrushId.MARKER)
    _paint(document, settings, 60.0)
    _paint(document, settings, 120.0)
    document.undo()
    assert document.can_redo
    _paint(document, settings, 160.0)
    assert not document.can_redo


def test_clear_is_undoable() -> None:
    document = aivp.PaintDocument(300, 200)
    _paint(document, aivp.BrushSettings(brush=aivp.BrushId.MARKER), 100.0)
    populated = document.canvas.copy()
    assert document.clear()
    assert not document.canvas[:, :, 3].any()
    assert document.undo()
    assert np.array_equal(document.canvas, populated)


def test_clear_on_an_empty_document_reports_no_change() -> None:
    assert aivp.PaintDocument(120, 120).clear() is False


def test_eraser_removes_previously_painted_alpha() -> None:
    document = aivp.PaintDocument(400, 300)
    settings = aivp.BrushSettings(brush=aivp.BrushId.MARKER, thickness_index=5)
    _paint(document, settings, 150.0)
    before = int(document.canvas[:, :, 3].sum())
    _paint(document, settings, 150.0, eraser=True)
    assert int(document.canvas[:, :, 3].sum()) < before * 0.25


def test_cancel_stroke_leaves_no_trace() -> None:
    document = aivp.PaintDocument(300, 220)
    document.begin_stroke(aivp.BrushSettings(brush=aivp.BrushId.MARKER), eraser=False)
    for step in range(8):
        document.extend_stroke(50.0 + step * 20.0, 110.0, 100.0)
    document.cancel_stroke()
    assert not document.canvas[:, :, 3].any()
    assert document.stroke_count == 0


def test_shape_snapping_replaces_the_freehand_path() -> None:
    document = aivp.PaintDocument(600, 600)
    document.begin_stroke(aivp.BrushSettings(brush=aivp.BrushId.MARKER), eraser=False)
    for x, y, velocity in _circle_trace():
        document.extend_stroke(x, y, velocity)
    assert document.end_stroke(snap_shapes=True) == "CIRCLE"


def test_shape_snapping_can_be_disabled() -> None:
    document = aivp.PaintDocument(600, 600)
    document.begin_stroke(aivp.BrushSettings(brush=aivp.BrushId.MARKER), eraser=False)
    for x, y, velocity in _circle_trace():
        document.extend_stroke(x, y, velocity)
    assert document.end_stroke(snap_shapes=False) is None


def test_render_artwork_scales_vectors_rather_than_pixels() -> None:
    document = aivp.PaintDocument(200, 150)
    _paint(document, aivp.BrushSettings(brush=aivp.BrushId.MARKER), 75.0)
    scaled = document.render_artwork(3.0)
    assert scaled.shape == (450, 600, 4)
    assert int(scaled[:, :, 3].sum()) > int(document.canvas[:, :, 3].sum()) * 5


@pytest.mark.parametrize("fmt", list(aivp.ExportFormat))
def test_every_export_format_writes_a_readable_file(tmp_path: Path, fmt: aivp.ExportFormat) -> None:
    config = aivp.AppConfig(canvas_width=200, canvas_height=150, output_dir=tmp_path, export_scale=2)
    document = aivp.PaintDocument(200, 150)
    _paint(document, aivp.BrushSettings(brush=aivp.BrushId.NEON), 75.0)
    backdrop = np.full((150, 200, 3), 60, dtype=np.uint8)
    written = aivp.ArtworkExporter(config).save(document, fmt, backdrop)
    assert written.exists() and written.stat().st_size > 0
    assert written.suffix == fmt.suffix


def test_transparent_export_keeps_an_alpha_channel(tmp_path: Path) -> None:
    config = aivp.AppConfig(canvas_width=160, canvas_height=120, output_dir=tmp_path)
    document = aivp.PaintDocument(160, 120)
    _paint(document, aivp.BrushSettings(brush=aivp.BrushId.MARKER), 60.0)
    image = aivp.ArtworkExporter(config).build_image(document, aivp.ExportFormat.PNG_TRANSPARENT, None)
    assert image.mode == "RGBA"
    assert min(image.getchannel("A").getextrema()) == 0


def test_high_resolution_export_uses_the_configured_scale(tmp_path: Path) -> None:
    config = aivp.AppConfig(canvas_width=160, canvas_height=120, output_dir=tmp_path, export_scale=4)
    document = aivp.PaintDocument(160, 120)
    _paint(document, aivp.BrushSettings(brush=aivp.BrushId.MARKER), 60.0)
    image = aivp.ArtworkExporter(config).build_image(document, aivp.ExportFormat.HIGH_RES, None)
    assert image.size == (640, 480)


# --------------------------------------------------------------------------- #
# Gesture engine
# --------------------------------------------------------------------------- #

_WRIST_POSITION = (400.0, 520.0)
_FINGER_ANGLES = (-2.45, -1.83, -1.57, -1.31, -0.87)  # thumb → pinky, radians


def make_hand(
    extended: tuple[bool, bool, bool, bool, bool],
    pinch_with: int | None = None,
) -> aivp.HandObservation:
    """Synthesise a landmark set with the requested fingers extended.

    ``pinch_with`` places the thumb tip on top of landmark 8 (index) or 12
    (middle) so the pinch detector can be exercised deterministically.
    """
    landmarks = np.tile(np.array(_WRIST_POSITION, dtype=np.float32), (21, 1))
    for finger, angle in enumerate(_FINGER_ANGLES):
        joint_radius, tip_radius = 90.0, (150.0 if extended[finger] else 70.0)
        for index, radius in ((aivp.FINGER_JOINTS[finger], joint_radius), (aivp.FINGER_TIPS[finger], tip_radius)):
            landmarks[index] = (
                _WRIST_POSITION[0] + math.cos(angle) * radius,
                _WRIST_POSITION[1] + math.sin(angle) * radius,
            )
    landmarks[aivp.MIDDLE_MCP] = (_WRIST_POSITION[0], _WRIST_POSITION[1] - 60.0)
    if pinch_with is not None:
        landmarks[aivp.THUMB_TIP] = landmarks[pinch_with] + np.float32([4.0, 4.0])
    palm = float(np.linalg.norm(landmarks[aivp.MIDDLE_MCP] - landmarks[aivp.WRIST]))
    return aivp.HandObservation(landmarks, "RIGHT", 0.0, palm)


def _settle(engine: aivp.GestureEngine, observation: aivp.HandObservation, start: float = 0.0) -> aivp.GestureState:
    """Feed the same pose repeatedly so debounce and smoothing converge."""
    state = aivp.GestureState()
    for step in range(8):
        state = engine.update(
            aivp.HandObservation(
                observation.landmarks, observation.handedness, start + step * 0.05, observation.palm_size
            ),
            start + step * 0.05,
        )
    return state


@pytest.mark.parametrize(
    ("extended", "expected"),
    [
        ((False, True, False, False, False), aivp.HandPose.DRAW),
        ((False, True, True, False, False), aivp.HandPose.SELECT),
        ((False, True, True, True, False), aivp.HandPose.ERASE),
        ((True, True, True, True, True), aivp.HandPose.PALM),
        ((False, False, False, False, False), aivp.HandPose.UNKNOWN),
    ],
)
def test_pose_classification(extended, expected) -> None:
    state = _settle(aivp.GestureEngine(), make_hand(extended))
    assert state.pose is expected


def test_tools_follow_poses() -> None:
    engine = aivp.GestureEngine()
    assert _settle(engine, make_hand((False, True, False, False, False))).tool is aivp.Tool.DRAW
    engine.reset()
    assert _settle(engine, make_hand((False, True, True, True, False))).tool is aivp.Tool.ERASE


def test_missing_observation_produces_an_absent_state() -> None:
    state = aivp.GestureEngine().update(None, 1.0)
    assert not state.present and state.tool is aivp.Tool.IDLE and state.pointer is None


def test_stale_observation_is_treated_as_lost_tracking() -> None:
    engine = aivp.GestureEngine()
    observation = make_hand((False, True, False, False, False))
    assert not engine.update(observation, observation.timestamp + 5.0).present


def test_pointer_tracks_the_index_fingertip() -> None:
    observation = make_hand((False, True, False, False, False))
    state = _settle(aivp.GestureEngine(), observation)
    assert state.pointer is not None
    assert math.dist(state.pointer, tuple(observation.landmarks[aivp.INDEX_TIP])) < 12.0


def _hold_pose(
    engine: aivp.GestureEngine, observation: aivp.HandObservation, frames: int, start: float
) -> list[aivp.GestureEvent]:
    """Feed one pose for ``frames`` frames and collect every event emitted."""
    collected: list[aivp.GestureEvent] = []
    for step in range(frames):
        now = start + step * 0.05
        state = engine.update(aivp.HandObservation(observation.landmarks, "RIGHT", now, observation.palm_size), now)
        collected.extend(state.events)
    return collected


def test_thumb_index_pinch_emits_undo_exactly_once() -> None:
    engine = aivp.GestureEngine()
    _settle(engine, make_hand((False, True, False, False, False)))
    pinched = make_hand((False, True, False, False, False), pinch_with=aivp.INDEX_TIP)
    events = _hold_pose(engine, pinched, frames=8, start=1.0)
    assert events.count(aivp.GestureEvent.UNDO) == 1


def test_thumb_middle_pinch_emits_redo() -> None:
    engine = aivp.GestureEngine()
    _settle(engine, make_hand((False, True, True, False, False)))
    pinched = make_hand((False, True, True, False, False), pinch_with=aivp.MIDDLE_TIP)
    assert aivp.GestureEvent.REDO in _hold_pose(engine, pinched, frames=8, start=1.0)


def test_open_palm_held_for_two_seconds_clears_the_canvas() -> None:
    engine = aivp.GestureEngine()
    palm = make_hand((True, True, True, True, True))
    fired = False
    progress = 0.0
    for step in range(70):
        now = step * 0.05
        state = engine.update(aivp.HandObservation(palm.landmarks, "RIGHT", now, palm.palm_size), now)
        progress = max(progress, state.hold_progress)
        fired = fired or aivp.GestureEvent.CLEAR in state.events
    assert fired and progress == pytest.approx(1.0)


def test_hold_gesture_does_not_fire_early() -> None:
    engine = aivp.GestureEngine()
    palm = make_hand((True, True, True, True, True))
    for step in range(20):  # one second only
        now = step * 0.05
        state = engine.update(aivp.HandObservation(palm.landmarks, "RIGHT", now, palm.palm_size), now)
        assert aivp.GestureEvent.CLEAR not in state.events
    assert 0.0 < state.hold_progress < 1.0


def test_stationary_victory_pose_saves_the_artwork() -> None:
    engine = aivp.GestureEngine()
    victory = make_hand((False, True, True, False, False))
    fired = False
    for step in range(70):
        now = step * 0.05
        state = engine.update(aivp.HandObservation(victory.landmarks, "RIGHT", now, victory.palm_size), now)
        fired = fired or aivp.GestureEvent.SAVE in state.events
    assert fired


def test_moving_hand_is_not_reported_as_stationary() -> None:
    engine = aivp.GestureEngine()
    state = aivp.GestureState()
    for step in range(24):
        now = step * 0.04
        landmarks = make_hand((False, True, True, False, False)).landmarks.copy()
        landmarks[:, 0] += step * 22.0
        state = engine.update(aivp.HandObservation(landmarks, "RIGHT", now, 60.0), now)
    assert not state.stationary
    assert state.velocity > 0.0


# --------------------------------------------------------------------------- #
# Interface model
# --------------------------------------------------------------------------- #


def test_brush_settings_expose_the_selected_thickness_and_slot() -> None:
    settings = aivp.BrushSettings(thickness_index=4, color_index=2)
    assert settings.thickness == aivp.THICKNESSES[4]
    assert settings.slot is aivp.PALETTE[2]


def test_hit_region_supports_rectangular_and_circular_shapes() -> None:
    rectangle = aivp.HitRegion(aivp.Command.SELECT_COLOR, 1, (10, 10, 50, 30))
    assert rectangle.contains(20, 20) and not rectangle.contains(60, 20)
    circle = aivp.HitRegion(aivp.Command.SELECT_BRUSH, 2, (0, 0, 40, 40), radius=20.0)
    assert circle.contains(20, 20) and not circle.contains(38, 38)
    assert circle.key == (aivp.Command.SELECT_BRUSH, 2)


def test_toast_envelope_rises_and_falls() -> None:
    toast = aivp.Toast("SAVED", "", aivp.ToastKind.SUCCESS, created=0.0, duration=2.0)
    assert toast.envelope(0.0) == pytest.approx(0.0, abs=1e-6)
    assert toast.envelope(1.0) > 0.9
    assert toast.envelope(3.0) == 0.0
    assert toast.expired(3.0)


def test_animation_bank_reuses_one_animation_per_key() -> None:
    bank = aivp.AnimationBank(speed=25.0)
    for _ in range(120):
        value = bank.drive("dock", 1.0, 1 / 60)
    assert value == pytest.approx(1.0, abs=1e-3)


def _hud_model(**overrides) -> aivp.HudModel:
    defaults = {
        "settings": aivp.BrushSettings(),
        "gesture": aivp.GestureState(),
        "tool": aivp.Tool.IDLE,
        "fps": 60.0,
        "inference_ms": 8.0,
        "stroke_count": 3,
        "can_undo": True,
        "can_redo": False,
        "shape_ai": True,
        "show_skeleton": True,
        "art_only": False,
        "export_format": aivp.ExportFormat.PNG,
        "output_dir": Path("artwork"),
        "hover_key": None,
        "dwell_progress": 0.0,
        "menu_open": False,
        "menu_origin": (640, 360),
        "mouse": None,
        "boot_progress": 1.0,
        "boot_status": "TRACKING ONLINE",
        "elapsed": 4.0,
    }
    defaults.update(overrides)
    return aivp.HudModel(**defaults)  # type: ignore[arg-type]


def test_hud_renders_and_publishes_hit_regions() -> None:
    renderer = aivp.HudRenderer(1280, 720)
    frame = np.full((720, 1280, 3), 70, dtype=np.uint8)
    regions = renderer.render(frame, _hud_model(), 1 / 60)
    commands = {region.command for region in regions}
    assert {
        aivp.Command.SELECT_BRUSH,
        aivp.Command.SELECT_COLOR,
        aivp.Command.SELECT_THICKNESS,
        aivp.Command.SET_OPACITY,
        aivp.Command.CLOSE,
    } <= commands
    assert frame.any()


def test_hud_regions_stay_inside_the_canvas() -> None:
    renderer = aivp.HudRenderer(1280, 720)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    for region in renderer.render(frame, _hud_model(), 1 / 60):
        assert region.rect[0] >= 0 and region.rect[1] >= 0
        assert region.rect[2] <= 1280 and region.rect[3] <= 720


def test_hud_dock_regions_do_not_overlap_the_telemetry_column() -> None:
    renderer = aivp.HudRenderer(1280, 720)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dock = [r for r in renderer.render(frame, _hud_model(), 1 / 60) if r.command is aivp.Command.SELECT_BRUSH]
    assert dock and max(region.rect[2] for region in dock) < aivp.HudRenderer.TELEMETRY_X


def test_hud_renders_the_radial_palette_when_selecting() -> None:
    renderer = aivp.HudRenderer(1280, 720)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    gesture = aivp.GestureState(present=True, pose=aivp.HandPose.SELECT, tool=aivp.Tool.SELECT, pointer=(640.0, 360.0))
    model = _hud_model(tool=aivp.Tool.SELECT, gesture=gesture, menu_open=True)
    for _ in range(30):  # let the bloom animation finish
        regions = renderer.render(frame, model, 1 / 60)
    circular = [region for region in regions if region.radius > 0.0]
    assert len(circular) >= len(aivp.BrushId) + len(aivp.PALETTE) + len(aivp.THICKNESSES)


def test_hud_renders_every_tool_state_and_the_boot_sequence() -> None:
    renderer = aivp.HudRenderer(960, 540)
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    for tool in aivp.Tool:
        gesture = aivp.GestureState(
            present=True,
            tool=tool,
            pointer=(480.0, 270.0),
            landmarks=make_hand((True,) * 5).landmarks,
            hold_label="CLEAR CANVAS",
            hold_progress=0.5,
        )
        renderer.render(frame, _hud_model(tool=tool, gesture=gesture, boot_progress=0.4, mouse=(480, 270)), 1 / 60)
    renderer.notify("SAVED", "AIVP.PNG", aivp.ToastKind.SUCCESS)
    renderer.render(frame, _hud_model(), 1 / 60)
    assert frame.any()


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_config_reads_environment_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIVP_CAMERA_INDEX", "2")
    monkeypatch.setenv("AIVP_CAMERA_WIDTH", "640")
    monkeypatch.setenv("AIVP_OUTPUT_DIR", str(tmp_path / "art"))
    monkeypatch.setenv("AIVP_HAND_MODEL", str(tmp_path / "model.task"))
    config = aivp.AppConfig.from_environment()
    assert config.camera_index == 2
    assert config.camera_width == 640
    assert config.output_dir == tmp_path / "art"
    assert config.model_path == tmp_path / "model.task"


def test_config_ignores_invalid_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIVP_CAMERA_WIDTH", "not-a-number")
    monkeypatch.setenv("AIVP_CAMERA_INDEX", "-4")
    config = aivp.AppConfig.from_environment()
    assert config.camera_width == 1280
    assert config.camera_index == 0


def test_export_format_metadata_is_consistent() -> None:
    assert aivp.ExportFormat.JPG.suffix == ".jpg"
    assert all(fmt.suffix in {".png", ".jpg"} for fmt in aivp.ExportFormat)
    assert all(fmt.label for fmt in aivp.ExportFormat)


def test_bgr_to_hex_round_trips() -> None:
    assert aivp.bgr_to_hex(aivp.hex_to_bgr("#22e6ff")) == "#22e6ff"
