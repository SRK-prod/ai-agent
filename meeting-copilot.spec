# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: `pyinstaller meeting-copilot.spec` (see docs/installation.md).

Bundles the FastAPI backend + PySide6 overlay into one macOS .app via
src/meeting_copilot/packaged_app.py (backend on a thread, Qt on main thread).
configs/ ships inside the bundle read-only; data/logs/.env move to
~/Library/Application Support/meeting-copilot at runtime (see paths.py).
"""

from PyInstaller.utils.hooks import collect_all

datas = [("configs", "configs")]
binaries = []
hiddenimports = []

# These packages carry C extensions/data files PyInstaller can't infer on its own.
for pkg in (
    "torch",
    "torchaudio",
    "pyannote.audio",
    "faster_whisper",
    "ctranslate2",
    "silero_vad",
    "av",
):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    ["src/meeting_copilot/packaged_app.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="meeting-copilot",
    debug=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name="meeting-copilot",
)

app = BUNDLE(
    coll,
    name="meeting-copilot.app",
    icon=None,
    bundle_identifier="com.local.meeting-copilot",
    info_plist={
        "NSMicrophoneUsageDescription": (
            "meeting-copilot needs microphone access to listen to meeting audio."
        ),
        "LSUIElement": False,
        # Accessibility/Input Monitoring (needed by pynput's global hotkeys) is
        # granted via System Settings > Privacy & Security after first launch --
        # there's no Info.plist usage-description key for those, unlike mic access.
    },
)
