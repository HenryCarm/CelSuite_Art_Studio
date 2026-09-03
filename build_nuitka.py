#!/usr/bin/env python3
"""
CelSuite Art Studio (CelAS v269.2.0) - Nuitka Packaging Script
Ultra-Slim Featherlight Edition for Linux & Windows 10/11.
Prunes multi-gigabyte PyTorch bloat and unneeded Qt modules.
"""

import sys
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

# Fix Windows cp1252 charmap encoding crash when printing Unicode/emojis
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parent

# Cross-platform python executable detection (Central Linux venv or system python / Windows runner)
CENTRAL_VENV = Path("/home/henry/Documents/Projects/Python/venv/bin/python")
if CENTRAL_VENV.exists():
    VENV_PYTHON = CENTRAL_VENV
else:
    VENV_PYTHON = Path(sys.executable)

MAIN_SCRIPT = PROJECT_DIR / "CelAS.py"
ICON_FILE   = PROJECT_DIR / "icon.png"
ICO_FILE    = PROJECT_DIR / "icon.ico"
DIST_DIR    = PROJECT_DIR / "dist"
APP_NAME    = "CelSuite Art Studio"

# Aggressively prune unused frameworks, submodules, and CUDA bloat to slash package size
EXCLUDE_MODULES = [
    # Standard lib bloat
    "tkinter", "unittest", "pydoc", "doctest", "email", "http", "xmlrpc",
    "distutils", "setuptools", "pip", "pkg_resources", "curses", "idlelib",
    "turtledemo", "sqlite3", "matplotlib", "pandas", "tensorflow", "scipy",
    "PIL.SpiderImagePlugin", "PIL.FitsImagePlugin", "PIL.MpoImagePlugin", "PIL.PdfImagePlugin",
    # Unneeded PySide6 modules (drops 250MB+ of Qt bloat)
    "PySide6.QtNetwork", "PySide6.QtSql", "PySide6.QtQml", "PySide6.QtQuick",
    "PySide6.QtOpenGL", "PySide6.QtSvg", "PySide6.QtTest", "PySide6.QtXml",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtPdf", "PySide6.QtPrintSupport",
    "PySide6.QtBluetooth", "PySide6.QtPositioning", "PySide6.QtSensors", "PySide6.QtNfc",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
    # PyTorch bloat pruning (drops 1GB+ of training/distributed/cuda overhead)
    "torch.cuda",
    "torch.distributed",
    "torch.testing",
    "torch.autograd.profiler",
    "torch.utils.tensorboard",
    "torch.utils.benchmark",
    "torchvision",
    "torchaudio",
    "caffe2"
]

def zip_directory(dir_path: Path, zip_path: Path):
    """Compresses a standalone folder into a clean .zip distribution archive."""
    print(f"📦 Zipping standalone package: {dir_path.name} -> {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(dir_path.parent)
                zipf.write(full_path, rel_path)
    print(f"✅ Standalone archive created: {zip_path} ({round(zip_path.stat().st_size / (1024*1024), 2)} MB)")

def build(target="standalone"):
    """
    target: 'standalone' or 'onefile'
    """
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    is_win = sys.platform == "win32"
    os_name = "Windows" if is_win else "Linux"
    exe_suffix = ".exe" if is_win else ""
    out_name = f"CelSuite_ArtStudio{'_onefile' if target == 'onefile' else ''}{exe_suffix}"

    cmd = [
        str(VENV_PYTHON), "-m", "nuitka",
        f"--{target}",
        "--enable-plugin=pyside6",
        "--include-qt-plugins=sensible",
        f"--output-dir={DIST_DIR}",
        f"--output-filename={out_name}",
        f"--include-data-file={ICON_FILE}=icon.png",
        "--assume-yes-for-downloads",
        "--remove-output",
        "--lto=no",
        "--jobs=3",
        "--python-flag=-OO",
        "--prefer-source-code",
        "--module-parameter=torch-disable-jit=yes",
        "--noinclude-unittest-mode=nofollow",
        "--noinclude-setuptools-mode=nofollow",
        "--windows-console-mode=disable",
    ]

    # Icon handling: .ico on Windows, .png on Linux
    if is_win and ICO_FILE.exists():
        cmd.append(f"--windows-icon-from-ico={ICO_FILE}")
    elif ICON_FILE.exists():
        cmd.append(f"--linux-icon={ICON_FILE}")

    # Add pruning exclusions
    for mod in EXCLUDE_MODULES:
        cmd.append(f"--nofollow-import-to={mod}")

    cmd.append(str(MAIN_SCRIPT))

    print(f"\n🚀 Starting Nuitka {APP_NAME} {target.upper()} build on {os_name}...")
    print(f"📦 Command: {' '.join(cmd)}\n")

    res = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    if res.returncode == 0:
        print(f"\n🎉 {APP_NAME} {target.upper()} build finished successfully in {DIST_DIR}!")
        
        # Auto-zip standalone folder distribution
        if target == "standalone":
            standalone_folder = DIST_DIR / f"CelSuite_ArtStudio.dist"
            if not standalone_folder.exists():
                candidates = list(DIST_DIR.glob("*.dist"))
                if candidates:
                    standalone_folder = candidates[0]
            if standalone_folder.exists():
                zip_target = DIST_DIR / f"CelSuite_ArtStudio_Standalone_{os_name}.zip"
                zip_directory(standalone_folder, zip_target)
    else:
        print(f"\n❌ {APP_NAME} build failed with exit code {res.returncode}")
        sys.exit(res.returncode)

def spawn_external_terminal(mode="standalone"):
    """Spawns an external terminal so Henny can watch compilation progress in real-time."""
    cmd_str = f"cd '{PROJECT_DIR}' && '{VENV_PYTHON}' build_nuitka.py --run={mode} ; exec bash"
    try:
        subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', cmd_str])
        print(f"🚀 Spawned GNOME Terminal for {APP_NAME} live compilation monitoring!")
    except FileNotFoundError:
        try:
            subprocess.Popen(['x-terminal-emulator', '-e', f'bash -c "{cmd_str}"'])
            print(f"🚀 Spawned x-terminal-emulator for {APP_NAME} live compilation monitoring!")
        except FileNotFoundError:
            try:
                subprocess.Popen(['konsole', '-e', f'bash -c "{cmd_str}"'])
                print(f"🚀 Spawned Konsole for {APP_NAME} live compilation monitoring!")
            except FileNotFoundError:
                print("⚠️ No GUI terminal emulator found. Running inline...")
                build(mode)

def interactive_menu():
    """Displays an interactive selection prompt when no CLI flags are passed."""
    print(f"\n✨ =========================================")
    print(f"📦  {APP_NAME} - Nuitka Packaging Menu (Slim Edition)")
    print(f"✨ =========================================")
    print("  [1] Standalone .zip   (Fastest, instant launch - RECOMMENDED)")
    print("  [2] OneFile portable  (Single file executable)")
    print("  [3] Spawn Terminal    (Launch build in separate GNOME Terminal window)")
    print("  [0 / q] Cancel")
    
    try:
        choice = input("\n👉 Select an option [1-3 / q] (default: 1): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Build cancelled.")
        sys.exit(0)
        
    if choice in ("", "1", "standalone"):
        build("standalone")
    elif choice in ("2", "onefile"):
        build("onefile")
    elif choice in ("3", "spawn"):
        spawn_external_terminal("standalone")
    elif choice in ("0", "q", "exit", "quit"):
        print("👋 Build cancelled.")
        sys.exit(0)
    else:
        print(f"❌ Invalid choice '{choice}'. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        interactive_menu()
    elif "--run=standalone" in sys.argv:
        build("standalone")
    elif "--run=onefile" in sys.argv:
        build("onefile")
    elif "--spawn" in sys.argv:
        spawn_external_terminal("standalone")
    else:
        print("Usage:")
        print("  python build_nuitka.py                    (Interactive selection menu)")
        print("  python build_nuitka.py --run=standalone   (Builds fast zipped folder)")
        print("  python build_nuitka.py --run=onefile      (Builds single portable binary)")
        print("  python build_nuitka.py --spawn            (Spawns live terminal window)")
