#!/bin/zsh
# fps.sh <js-to-apply-first> [seconds]  -- applies JS in the headset tab, then measures XR frames/s in the page.
P=$(dirname $0)/page.py
PY="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/.venv/bin/python"
$PY $P "8000" "(() => { const v = volumeViewer; if (!window.__probe) { const orig = v.getPhysicalRotation; window.__frames = 0; v.getPhysicalRotation = (...a) => { window.__frames += 1; return orig(...a); }; window.__probe = true; } const out = (${1:-'() => null'})(); window.__t0 = performance.now(); window.__f0 = window.__frames; return JSON.stringify(out); })()"
sleep ${2:-6}
$PY $P "8000" 'JSON.stringify({inXR: volumeViewer.inXR(), fps: +((window.__frames - window.__f0) / ((performance.now() - window.__t0) / 1000)).toFixed(1)})'
