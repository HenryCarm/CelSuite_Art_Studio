#!/usr/bin/env python3
"""
CelSuite Art Studio (CelAS v269.2.0) - Multi-Target Packaging Script
Builds:
  - Universal (PyTorch CPU + PySide6)
  - Intel (stable-diffusion.cpp + PySide6 - Ultra-Light AVX2)
Exports both:
  - Standard .zip (universal 1-click)
  - Ultra .7z (Maximum LZMA2 -mx=9 compression)
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

# Cross-platform python executable detection
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

# Base exclusions for both editions
BASE_EXCLUDE = [
    "tkinter", "unittest", "pydoc", "doctest", "email", "http", "xmlrpc",
    "distutils", "setuptools", "pip", "pkg_resources", "curses", "idlelib",
    "turtledemo", "sqlite3", "matplotlib", "pandas", "scipy",
    "PIL.SpiderImagePlugin", "PIL.FitsImagePlugin", "PIL.MpoImagePlugin", "PIL.PdfImagePlugin",
    # Unneeded PySide6 modules
    "PySide6.QtNetwork", "PySide6.QtSql", "PySide6.QtQml", "PySide6.QtQuick",
    "PySide6.QtOpenGL", "PySide6.QtSvg", "PySide6.QtTest", "PySide6.QtXml",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtPdf", "PySide6.QtPrintSupport",
    "PySide6.QtBluetooth", "PySide6.QtPositioning", "PySide6.QtSensors", "PySide6.QtNfc",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore"
]

# Exclusions specific to Intel SD.cpp (drops PyTorch/Diffusers completely!)
INTEL_EXCLUDE = BASE_EXCLUDE + [
    "torch", "diffusers", "transformers", "accelerate", "safetensors.torch"
]

# Exclusions for Universal PyTorch CPU
UNIVERSAL_EXCLUDE = BASE_EXCLUDE + [
    "torch.cuda", "torch.distributed", "torch.testing", "torch.autograd.profiler",
    "torch.utils.tensorboard", "torch.utils.benchmark", "torchvision", "torchaudio", "caffe2",
    "stable_diffusion_cpp"
]

def compress_archives(folder_path: Path, base_name: str):
    """Creates both .zip (Deflate) and .7z (Ultra LZMA2) archives."""
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    
    zip_path = DIST_DIR / f"{base_name}.zip"
    seven_z_path = DIST_DIR / f"{base_name}.7z"
    
    print(f"\n📦 Creating Standard Zip archive: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(folder_path.parent)
                zipf.write(full_path, rel_path)
    print(f"✅ Zip Created: {zip_path.name} ({round(zip_path.stat().st_size / (1024*1024), 2)} MB)")

    print(f"\n🗜️ Creating Ultra 7z (LZMA2 Maximum Compression): {seven_z_path.name}...")
    # Use 7z CLI if available (pre-installed on Ubuntu & Windows runners)
    seven_z_cmd = shutil.which("7z") or shutil.which("7za")
    if seven_z_cmd:
        cmd = [
            seven_z_cmd, "a", "-t7z", "-m0=lzma2", "-mx=9", "-mfb=64", "-md=64m", "-ms=on",
            str(seven_z_path), str(folder_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL)
        if res.returncode == 0 and seven_z_path.exists():
            print(f"✅ Ultra 7z Created: {seven_z_path.name} ({round(seven_z_path.stat().st_size / (1024*1024), 2)} MB)")
        else:
            print("⚠️ 7z execution returned error. Zip remains primary.")
    else:
        print("⚠️ 7z tool not found on system PATH. Standard .zip is ready.")

def build(variant="universal"):
    """
    variant: 'universal' (PyTorch CPU) or 'intel' (stable-diffusion.cpp)
    """
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    is_win = sys.platform == "win32"
    os_name = "Windows" if is_win else "Linux"
    exe_suffix = ".exe" if is_win else ""
    out_name = f"CelSuite_ArtStudio{exe_suffix}"

    variant_title = "Universal" if variant.lower() == "universal" else "Intel"
    base_archive_name = f"CelSuite_ArtStudio-{os_name}-{variant_title}"

    cmd = [
        str(VENV_PYTHON), "-m", "nuitka",
        "--standalone",
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
        "--noinclude-unittest-mode=nofollow",
        "--noinclude-setuptools-mode=nofollow",
        "--windows-console-mode=disable",
    ]

    if variant.lower() == "universal":
        cmd.append("--module-parameter=torch-disable-jit=yes")
        exclusions = UNIVERSAL_EXCLUDE
    else:
        exclusions = INTEL_EXCLUDE

    if is_win and ICO_FILE.exists():
        cmd.append(f"--windows-icon-from-ico={ICO_FILE}")
    elif ICON_FILE.exists():
        cmd.append(f"--linux-icon={ICON_FILE}")

    for mod in exclusions:
        cmd.append(f"--nofollow-import-to={mod}")

    cmd.append(str(MAIN_SCRIPT))

    print(f"\n🚀 Starting Nuitka {APP_NAME} ({variant_title} Edition) on {os_name}...")
    res = subprocess.run(cmd, cwd=str(PROJECT_DIR))
    if res.returncode == 0:
        print(f"\n🎉 Standalone build finished successfully in {DIST_DIR}!")
        dist_folder = DIST_DIR / "CelAS.dist"
        if not dist_folder.exists():
            candidates = list(DIST_DIR.glob("*.dist"))
            if candidates:
                dist_folder = candidates[0]
        if dist_folder.exists():
            compress_archives(dist_folder, base_archive_name)
    else:
        print(f"\n❌ Build failed with exit code {res.returncode}")
        sys.exit(res.returncode)

if __name__ == "__main__":
    variant_arg = "universal"
    for arg in sys.argv:
        if arg.startswith("--variant="):
            variant_arg = arg.split("=", 1)[1]
    build(variant_arg)
