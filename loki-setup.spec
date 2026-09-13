# PyInstaller spec for the Windows setup editor (loki-setup.exe).
#
# A separate executable on purpose.  The Tk editor must not share a process with
# the credential-holding chat: keeping profile and ACL mutation in this binary
# means the runtime has no code path that can widen its own sandbox, and only
# this executable needs tcl/tk bundled -- loki.exe and loki-acp.exe stay free of
# it.
#
# "console" is deliberate: the editor is normally launched by loki.exe, which
# owns the console, and its --verify/--list/--uninstall modes print their
# results there.  A GUI subsystem build would discard those.
#
# Build with:
#   python -m PyInstaller --noconfirm --clean loki-setup.spec

a = Analysis(
    ["loki-setup"],
    pathex=[],
    binaries=[],
    datas=[],
    # The editor imports tkinter lazily (inside run_editor) so that
    # --verify/--list never load Tk; list the submodules explicitly so a future
    # refactor cannot hide them from the analysis.
    hiddenimports=["tkinter", "tkinter.filedialog", "tkinter.messagebox"],
    hookspath=[],
    hooksconfig={},
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
    name="loki-setup",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="loki-setup",
)
