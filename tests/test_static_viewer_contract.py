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
    assert re.search(r"https://unpkg\.com/vtk\.js@\d+\.\d+\.\d+(?:/vtk\.js)?", html)


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
    assert "metadata.name !== name" in viewer
    assert "metadata.version" in viewer
    assert "X-Dataset-Version" in viewer
    assert "little-endian" in viewer


def test_viewer_contract_exposes_label_transport_and_unlabeled_fallback():
    viewer = read("viewer.js")
    assert "/labels/metadata" in viewer
    assert "/labels/chunks/" in viewer
    assert "Uint8Array" in viewer
    assert "reconstructLabels" in viewer
    assert "fetchLabelMetadata" in viewer
    assert "no anatomical labels" in viewer
    assert "label-unavailable" in viewer
    assert "label_layout_version" in viewer
    assert "anatomy-v2" in viewer
    assert "labelStatus" in viewer
    assert "metadata.chunks.length !== metadata.chunk_count" in viewer
    assert "chunk.byte_offset + chunk.byte_length > metadata.total_bytes" in viewer
    assert "chunks.length !== metadata.chunk_count" in viewer
    assert "bytes.byteLength !== length" in viewer
    assert "offset + length > raw.byteLength" in viewer
    assert "bytes.byteLength !== chunk.byte_length" in viewer
    assert "chunks.length !== metadata.chunk_count" in viewer
    assert "labelStatus" in viewer[viewer.index("setStatus(`Local ${datasetName} volume${labelStatus}`)"):]


def test_viewer_adapts_exactly_48_values_and_exposes_camera_state():
    viewer = read("viewer.js")
    assert "const TOTAL_PARAMS = 48" in viewer
    assert "params.length !== TOTAL_PARAMS" in viewer
    assert "setTransferFunction" in viewer
    assert "getCamera" in viewer
    assert "setCamera" in viewer
    assert "render" in viewer
    assert "resetCamera" in viewer
    assert "value.zoom" in viewer
    assert "camera.dolly" not in viewer
    assert "setParallelScale(cameraBaseScale / appliedCameraState.zoom)" in viewer
    assert "camera.zoom(" not in viewer
    assert "toRendererCamera" in viewer
    assert "fromRendererCamera" in viewer
    assert "position" in viewer and "focal_point" in viewer and "view_up" in viewer
    assert "Number.isFinite(value.azimuth)" in viewer
    assert "Number.isFinite(value.elevation)" in viewer
    assert "Number.isFinite(value.zoom)" in viewer
    assert "value.azimuth !== 0" in viewer
    assert "value.elevation !== 0" in viewer


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


def test_viewer_wires_and_initializes_render_window_interactor():
    viewer = read("viewer.js")
    assert "vtkRenderWindowInteractor.newInstance()" in viewer
    assert "interactor.setView(openGLRenderWindow)" in viewer
    assert "renderWindow.setInteractor(interactor)" in viewer
    assert "interactor.initialize()" in viewer
    assert "interactor.bindEvents(viewerEl)" in viewer


def test_viewer_tears_down_partial_vtk_setup_on_initialization_failure():
    viewer = read("viewer.js")
    assert "interactor.unbindEvents()" in viewer
    assert "openGLRenderWindow.delete()" in viewer
    assert "renderWindow.delete()" in viewer
    assert "renderer.delete()" in viewer
    assert "fallbackEl.hidden = false" in viewer


def test_app_integrates_viewer_with_state_and_dataset_changes():
    app = read("app.js")
    assert "window.volumeViewer" in app or "volumeViewer" in app
    assert "load" in app
    assert "chooseDataset" in app
    assert "/api/scenes/transition" in app
    assert "parent_scene_id" in app
    assert "goal" in app
    assert "dataset_version" in app
    assert "postSceneTransition" in app
    assert "client_metadata" in app
    assert "session_id" in app
    assert "captureSceneRoot" in app
    assert "before.scene_id" in app
    assert 'attribute: "neutral"' in app
    assert "sceneSequence += 1" in app
    assert "sessionStorage" in app
    assert "sceneNonce" in app
    assert "randomUUID" in app
    assert "setItem(\"localViewerSceneNonce\"" not in app
    assert "sceneSequence" in app
    assert 'safeStorageGet("session", sceneSequenceKey)' in app
    assert "sceneSequenceKey" in app
    assert 'safeStorageSet("session", sceneSequenceKey' in app
    assert "transitionIdentity" in app
    assert "sceneNonce" in app and "before.scene_id" in app
    assert "event_id: eventId" in app
    assert "sceneSnapshot = null" in app
    assert "showToast(`Scene transition failed" in app


def test_app_does_not_fabricate_opacity_goals_for_non_opacity_commands():
    app = read("app.js")
    assert "non_extractable" in app
    assert "original_command" in app
    assert "cmd_dict?.target || \"soft\"" not in app


def test_app_defines_grouped_eight_class_anatomy_metadata():
    app = read("app.js")
    html = read("index.html")
    for name in ("skeleton", "lungs", "heart", "vessels", "liver", "kidneys", "spleen", "soft"):
        assert name in app
        assert f"telem-{name}" in html
    for group in ("Thoracic", "Abdominal", "Structural"):
        assert group in html
    assert "ANATOMY_CLASSES" in app
    assert "updateTelemetry(state.current.class_visibility, state.current.class_brightness)" in app


def test_app_compare_and_curve_use_shared_anatomy_metadata():
    app = read("app.js")
    assert "ANATOMY_CLASSES.map" in app
    assert "collision" in app.lower()
    assert "CLASS_ORDER = ANATOMY_CLASSES.map" in app


def test_dataset_boundary_identity_includes_document_and_scene_identity():
    app = read("app.js")
    assert "boundaryEventId" in app
    assert "sceneNonce" in app
    assert "data.session_id" in app
    assert "sceneSnapshot.scene_id" in app
    assert "boundaryEventId" in app and "dedupe_key: boundaryEventId" in app


def test_styles_cover_viewer_loading_error_and_fallback_states():
    css = read("style.css")
    for selector in ("#vtk-viewer", "#viewer-status", "#viewer-error", "#current-image"):
        assert selector in css
    assert "@media (max-width: 444px)" in css
    assert "min-width: 0" in css


def test_compare_layout_fits_inside_viewer_region_without_card_overflow():
    css = read("style.css")
    assert "#image-panel" in css and "overflow: hidden" in css
    assert re.search(r"#viewport\s*\{[^}]*width:\s*100%[^}]*min-width:\s*0[^}]*max-width:\s*100%", css, re.S)
    assert re.search(r"#compare-view\s*\{[^}]*width:\s*100%[^}]*min-width:\s*0[^}]*max-width:\s*100%", css, re.S)
    assert "grid-template-columns: repeat(4, minmax(0, 1fr))" in css
    assert re.search(r"\.compare-arm\s*\{[^}]*min-width:\s*0", css, re.S)
    assert re.search(r"\.compare-arm-image\s*\{[^}]*aspect-ratio:\s*1[^}]*max-height:", css, re.S)
    assert "@media (max-width: 1100px)" in css
    assert "@media (max-width: 640px)" in css


def test_telemetry_reads_out_the_four_goal_classes_not_retired_bands():
    html = read("index.html")
    for goal_class in ("skeleton", "lungs", "soft", "vessels"):
        assert f'id="telem-{goal_class}"' in html
    for retired in ("air", "fat", "spongy", "bone"):
        assert f'id="telem-{retired}"' not in html


def test_app_reads_class_visibility_for_the_telemetry():
    app = read("app.js")
    assert "class_visibility" in app
    assert '"air", "fat"' not in app


def test_app_explains_unavailable_policy_sweep_arm_without_nan_or_blank_values():
    app = read("app.js")
    assert "Policy unavailable" in app
    assert "policy arm has no checkpoint-backed samples" in app
    assert "Number.isFinite(arm.median)" in app


def test_viewer_has_global_async_loading_state_and_chat_overflow_contract():
    html = read("index.html")
    app = read("app.js")
    css = read("style.css")
    assert 'id="loading-overlay"' in html
    assert 'aria-live="polite"' in html
    assert "withLoading" in app
    for function_name in ("sendCommand", "navigate", "chooseDataset", "runCompare", "runSweep"):
        assert f"withLoading(() => {function_name}Impl" in app
    assert "await withLoading(async () =>" in app
    assert re.search(r"#messages\s*\{[^}]*overflow-y:\s*auto", css, re.S)


def test_viewer_contains_containment_contract_for_chat_history_and_panels():
    css = read("style.css")
    assert re.search(r"#app\s*\{[^}]*overflow:\s*hidden", css, re.S)
    assert re.search(r"#main\s*\{[^}]*min-height:\s*0", css, re.S)
    assert re.search(r"#image-panel\s*\{[^}]*min-height:\s*0[^}]*overflow:\s*hidden", css, re.S)
    assert re.search(r"#chat-panel\s*\{[^}]*min-height:\s*0", css, re.S)
    assert re.search(r"#messages\s*\{[^}]*min-height:\s*0", css, re.S)
    assert re.search(r"#composer\s*\{[^}]*flex-shrink:\s*0", css, re.S)


def test_mobile_main_keeps_intentional_stack_scrolling():
    css = read("style.css")
    mobile = css[css.index("@media (max-width: 860px)"):]
    assert re.search(r"#main\s*\{[^}]*overflow:\s*(?:auto|scroll)", mobile, re.S)


def test_viewer_parses_with_the_llm_but_applies_the_command_exactly():
    # Policy mode answers "show only bones" with soft tissue still dominant
    # (43.1% soft vs 4.9% skeleton on ct_chest); exact application isolates
    # the class the user named. The policy proposes, it does not execute.
    app = read("app.js")
    assert re.search(r"config\s*=\s*\{[^}]*parser:\s*\"llm\"", app)
    assert re.search(r"config\s*=\s*\{[^}]*mode:\s*\"exact\"", app)
    html = read("index.html")
    assert re.search(r'id="policy-toggle-btn"[^>]*data-active="false"', html)


def test_viewer_leaves_the_camera_alone_when_a_step_does_not_move_it():
    viewer = read("viewer.js")
    assert "lastRequestedCamera" in viewer
    assert "sameCameraRequest" in viewer
    # the guard must run before the camera is re-seated
    guard = viewer.index("if (sameCameraRequest(value)) return;")
    reseat = viewer.index("appliedCameraState = toRendererCamera(value);", guard)
    assert guard < reseat


def test_app_persists_theme_and_updates_accessible_toggle_state():
    app = read("app.js")
    assert 'const THEME_KEY = "tf-rl-theme"' in app
    assert "function applyTheme(theme)" in app
    assert "document.documentElement.dataset.theme" in app
    assert 'setAttribute("aria-pressed"' in app
    assert 'setAttribute("title"' in app
    assert 'safeStorageGet("local", THEME_KEY)' in app
    assert 'safeStorageSet("local", THEME_KEY, next)' in app
    assert "try" in app and "catch" in app


def test_app_handles_about_dialog_without_touching_viewer_state():
    app = read("app.js")
    assert 'el("about-modal")' in app
    assert 'el("about-btn")' in app
    assert 'el("about-modal-close")' in app
    assert "showModal()" in app
    assert "close()" in app
    assert "e.key === \"Escape\"" in app
    assert "document.activeElement" in app
    assert "focus()" in app


def test_app_visible_copy_is_english_only():
    app = read("app.js")
    for word in ("LOKALE SITZUNG", "SKELETT", "LUNGE", "WEICHGEW.", "GEFÄSSE"):
        assert word not in app


def test_viewer_has_theme_about_controls_and_english_copy():
    html = read("index.html")
    for control_id in ("theme-toggle", "about-btn", "about-modal", "about-modal-close"):
        assert f'id="{control_id}"' in html
    assert 'aria-label="Toggle color theme"' in html
    assert 'aria-label="About this project"' in html
    assert 'aria-labelledby="about-modal-title"' in html
    assert "Conversational Transfer Functions" in html
    assert "semantic Gaussian peaks for lungs, soft tissue, vessels, and skeleton" in html
    assert 'id="session-indicator"' not in html
    for word in ("LOKALE SITZUNG", "SKELETT", "LUNGE", "WEICHGEW.", "GEFÄSSE"):
        assert word not in html


def test_app_source_keeps_theme_and_about_lifecycle_contracts():
    app = read("app.js")
    assert "initTheme();" in app
    assert "initAboutDialog();" in app
    assert "modal.addEventListener(\"click\"" in app
    assert "modal.addEventListener(\"cancel\"" in app
    assert "modal.addEventListener(\"close\"" in app
    assert "modal.addEventListener(\"keydown\"" in app
    assert 'safeStorageGet("local", THEME_KEY)' in app
    assert 'safeStorageSet("local", THEME_KEY, next)' in app


def test_command_dialog_cleans_up_native_close_and_restores_opener_focus():
    app = read("app.js")
    assert "let commandsReturnFocus = el(\"commands-help-btn\");" in app
    assert "commandsReturnFocus = document.activeElement instanceof HTMLElement" in app
    assert "modal.addEventListener(\"cancel\", (event) =>" in app
    assert "modal.addEventListener(\"close\", () => {" in app
    assert "modal.classList.remove(\"dialog-open\");" in app
    assert "commandsReturnFocus?.focus();" in app


def test_command_dialog_ignores_duplicate_open_requests_while_loading_or_open():
    app = read("app.js")
    assert "let commandsOpenPromise = null;" in app
    assert "if (modal.open || commandsOpenPromise) return;" in app
    assert "commandsOpenPromise = (async () => {" in app
    assert "commandsOpenPromise = null;" in app


def test_answer_mode_toggles_expose_and_sync_pressed_state():
    html = read("index.html")
    app = read("app.js")
    assert 'id="search-toggle-btn"' in html
    assert 'id="policy-toggle-btn"' in html
    assert 'id="search-toggle-btn" class="toggle" data-active="false" aria-pressed="false"' in html
    assert 'id="policy-toggle-btn" class="toggle" data-active="false" aria-pressed="false"' in html
    assert 'searchToggleBtn.setAttribute("aria-pressed", String(config.search));' in app
    assert 'policyToggleBtn.setAttribute("aria-pressed", "false");' in app
    assert 'policyToggleBtn.setAttribute("aria-pressed", String(active));' in app
    assert 'searchToggleBtn.setAttribute("aria-pressed", "false");' in app


def test_theme_control_exposes_visible_current_mode_and_mobile_bar_wraps():
    html = read("index.html")
    app = read("app.js")
    css = read("style.css")
    assert 'id="theme-toggle"' in html
    assert 'id="theme-label"' in html
    assert 'button?.querySelector("#theme-label")' in app
    assert "#app-bar-right {" in css
    assert "flex-wrap: wrap;" in css
    assert "@media (max-width: 860px)" in css


def test_collector_has_english_blind_judging_controls_and_standalone_theme():
    html = read("collect.html")
    js = read("collect.js")
    for label in ("Preference collection", "Candidate A", "Candidate B", "Start state", "Show start", "Choose A", "Choose B", "Equal", "Skip"):
        assert label in html
    assert 'id="about-btn"' not in html
    assert 'id="theme-toggle"' in html
    assert 'id="theme-label"' in html
    assert 'aria-label="Toggle color theme"' in html
    assert 'data-choice="a"' in html and 'data-choice="b"' in html
    assert 'data-choice="equal"' in html and 'data-choice="skip"' in html
    assert 'const THEME_KEY = "tf-rl-theme"' in js
    assert "document.documentElement.dataset.theme" in js
    assert 'setAttribute("aria-pressed"' in js
    assert "storageGet(THEME_KEY)" in js
    assert "storageSet(THEME_KEY, next)" in js
    assert 'button?.querySelector("#theme-label")' in js
    assert 'const KEY_CHOICES = { a: "a", b: "b", e: "equal", s: "skip" }' in js
    assert '"/api/collect/judge"' in js


def test_collector_uses_safe_storage_and_reconciles_after_committed_judgment():
    js = read("collect.js")
    html = read("collect.html")
    assert "function storageGet(key)" in js
    assert "function storageSet(key, value)" in js
    assert "storageGet(RATER_KEY)" in js
    assert "storageSet(RATER_KEY, id)" in js
    assert "storageGet(ROLE_KEY)" in js
    assert "storageSet(ROLE_KEY, role)" in js
    assert "storageGet(EXPERIENCE_KEY)" in js
    assert "storageSet(EXPERIENCE_KEY, experience)" in js
    assert "storageGet(countKey)" in js
    assert "storageSet(countKey, String(judgedCount))" in js
    assert "judgeCommitted" in js
    assert "currentItem = null" in js
    assert "showItem(await fetchNext())" in js
    assert 'aria-live="polite"' in html
