from pathlib import Path
import re


ROOT = Path(__file__).parents[1]
STATIC = ROOT / "static"


def read(name):
    return (STATIC / name).read_text()


def test_index_keeps_png_fallback_and_loads_local_viewer_module():
    html = read("index.html")
    assert 'id="current-image"' in html
    assert 'id="vtk-viewer"' in html
    assert 'id="viewer-status"' in html
    assert 'id="viewer-error"' in html
    assert '<script src="/static/viewer.js"></script>' in html


def test_index_pins_vtk_js_without_package_manifest():
    html = read("index.html")
    assert re.search(r"vtk\.js@\d+\.\d+\.\d+", html)


def test_viewer_fetches_typed_chunks_and_reconstructs_fortran_volume():
    viewer = read("viewer.js")
    assert "/api/datasets/" in viewer
    assert "/metadata" in viewer
    assert "/chunks/" in viewer
    assert "Float32Array" in viewer
    assert "byte_offset" in viewer
    assert "dimensions" in viewer
    assert "spacing" in viewer
    assert re.search(r"order\s*===?\s*[\"']F[\"']|order.*F", viewer)
    assert "validateMetadata" in viewer
    assert "byte_length" in viewer
    assert "chunk.byte_offset" in viewer


def test_viewer_adapts_exactly_24_values_and_exposes_camera_state():
    viewer = read("viewer.js")
    assert re.search(r"length\s*!==\s*24|length\s*===\s*24", viewer)
    assert "setTransferFunction" in viewer
    assert "getCamera" in viewer
    assert "setCamera" in viewer
    assert "render" in viewer
    assert "resetCamera" in viewer
    assert "value.zoom" in viewer
    assert "camera.dolly" not in viewer
    assert "setParallelScale(baseScale / value.zoom)" in viewer
    assert "camera.zoom(" not in viewer


def test_viewer_cancels_and_ignores_stale_dataset_loads():
    viewer = read("viewer.js")
    assert "AbortController" in viewer
    assert "loadGeneration" in viewer
    assert "signal" in viewer
    assert "isCurrentLoad" in viewer


def test_viewer_renders_locally_and_does_not_request_png_frames():
    viewer = read("viewer.js")
    assert "/api/render" not in viewer
    assert "image_b64" not in viewer
    assert "vtk" in viewer
    assert "vtk-viewer" in viewer


def test_app_integrates_viewer_with_state_and_dataset_changes():
    app = read("app.js")
    assert "window.volumeViewer" in app or "volumeViewer" in app
    assert "load" in app
    assert "chooseDataset" in app


def test_styles_cover_viewer_loading_error_and_fallback_states():
    css = read("style.css")
    for selector in ("#vtk-viewer", "#viewer-status", "#viewer-error", "#current-image"):
        assert selector in css
    assert "@media (max-width: 444px)" in css
    assert "min-width: 0" in css
