#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║             CelSuite Art Studio (CelAS v269.4.0) 🎨✨                        ║
# ║                   The Liquid Glass & Yellow Weave Edition                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

import sys
import os
import json
import datetime
import glob
import multiprocessing
import shutil
import random
import gc
import threading
import subprocess
from pathlib import Path

# ─── MAXIMUM CPU THREADS (HENNY'S FULL SPEED BEAST MODE) ──────────────────────
torch_threads = multiprocessing.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(torch_threads)
os.environ["MKL_NUM_THREADS"] = str(torch_threads)

from PIL import Image, ImageGrab, ImageOps

# ─── DUAL AI INFERENCE ENGINE: PYTORCH (UNIVERSAL) vs SD.CPP (INTEL AVX2) ────
try:
    import torch
    torch.set_num_threads(torch_threads)
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        EulerAncestralDiscreteScheduler,
        DPMSolverMultistepScheduler,
        DDIMScheduler
    )
    HAS_TORCH = True
except ImportError:
    torch = None
    StableDiffusionPipeline = None
    StableDiffusionImg2ImgPipeline = None
    EulerAncestralDiscreteScheduler = None
    DPMSolverMultistepScheduler = None
    DDIMScheduler = None
    HAS_TORCH = False

try:
    import stable_diffusion_cpp
    HAS_SDCPP = True
except ImportError:
    stable_diffusion_cpp = None
    HAS_SDCPP = False

# ─── FUTURE-PROOF MONKEY PATCH FOR TRANSFORMERS 5.6+ ──────────────────────────
if HAS_TORCH:
    try:
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse("5.6"):
            from transformers import CLIPTextModel
            CLIPTextModel.text_model = property(lambda self: self)
    except Exception:
        pass

# ─── LLAMA-CPP (DOLPHIN AI AUTO-PROMPTING) ────────────────────────────────────
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

# ─── PYSIDE6 QT STANDARD ──────────────────────────────────────────────────────
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QFileDialog, QTabWidget,
    QScrollArea, QFrame, QComboBox, QLineEdit, QTextEdit, QCheckBox,
    QMessageBox, QDialog, QSizePolicy, QProgressBar, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize, QRect
from PySide6.QtGui import (
    QPainter, QImage, QPixmap, QColor, QFont, QIcon, QKeySequence, QShortcut,
    QTextOption
)

# ─── PATHS & CELSTUDIO CONSTANTS (CROSS-PLATFORM LINUX & WINDOWS) ─────────────
SCRIPT_DIR = Path(__file__).resolve().parent
USER_DOCS = Path.home() / "Documents"

# Wallpaper search: checks Projects/Python/.png, local wallpapers, or user Pictures
DEFAULT_WP_DIR = USER_DOCS / "Projects" / "Python" / ".png"
if DEFAULT_WP_DIR.exists():
    WALLPAPER_DIR = DEFAULT_WP_DIR
elif (SCRIPT_DIR / "wallpapers").exists():
    WALLPAPER_DIR = SCRIPT_DIR / "wallpapers"
else:
    WALLPAPER_DIR = Path.home() / "Pictures"

# Safe Trash location: checks Projects/OpenCode/tmp/Trash or user .celsuite/Trash
DEFAULT_TRASH = USER_DOCS / "Projects" / "OpenCode" / "tmp" / "Trash"
if DEFAULT_TRASH.parent.exists():
    TRASH_DIR = DEFAULT_TRASH
else:
    TRASH_DIR = Path.home() / ".celsuite" / "Trash"

HISTORY_DIR = Path.home() / "Pictures" / "HenJay" / "CelAS"
SETTINGS_FILE = SCRIPT_DIR / "cel_settings.json"
CIRCLE_ICON_ORIG = WALLPAPER_DIR / "Cel Logo CelAS Circle.png"
ICON_PATH = SCRIPT_DIR / "icon.png"
ICO_PATH = SCRIPT_DIR / "icon.ico"

HISTORY_DIR.mkdir(parents=True, exist_ok=True)
TRASH_DIR.mkdir(parents=True, exist_ok=True)

# ─── COLOR PALETTE (CELSUITE GOLD & AMBER YELLOW THEME) ───────────────────────
BG_DEEP       = "#090703"
BORDER_AMBER  = "rgba(234, 179, 8, 0.45)"
BORDER_GLOW   = "rgba(250, 204, 21, 0.85)"
AMBER_GOLD    = "#eab308"
AMBER_HOVER   = "#facc15"
AMBER_PRESS   = "#a16207"
AMBER_BTN_BG  = "rgba(50, 38, 12, 0.85)"
TEXT_MAIN     = "#fefce8"
TEXT_MUTED    = "#d4b886"
TEXT_AMBER    = "#facc15"
GREEN_GLOW    = "#00ff88"


# ─── OFFICIAL CELSTUDIO WR-STANDARD ICON LOADER ──────────────────────────────
def get_app_icon() -> QIcon:
    target_path = ICON_PATH if ICON_PATH.exists() else CIRCLE_ICON_ORIG
    if not target_path.exists():
        return QIcon()
    base_pixmap = QPixmap(str(target_path))
    if base_pixmap.isNull():
        return QIcon()
    icon = QIcon()
    # High-quality bicubic downsampling directly from the 1024x1024 master circle PNG
    for size in (16, 24, 32, 48, 64, 128, 256, 512):
        scaled = base_pixmap.scaled(
            size, size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        icon.addPixmap(scaled)
    icon.addPixmap(base_pixmap)
    return icon


# ─── WORKER THREAD: STABLE DIFFUSION GENERATION ───────────────────────────────
class SDWorkerThread(QThread):
    sig_log = Signal(str)
    sig_preview = Signal(object)      # PIL Image
    sig_saved = Signal(str, object)   # saved_path, PIL Image
    sig_finished = Signal()
    sig_error = Signal(str)

    def __init__(self, pipe_holder, params):
        super().__init__()
        self.pipe_holder = pipe_holder
        self.p = params
        self.abort_requested = False

    def request_abort(self):
        self.abort_requested = True

    def run(self):
        pipe = self.pipe_holder.get("pipe")
        if pipe is None:
            self.sig_error.emit("AI Brain is not loaded! Load checkpoint first.")
            return

        is_sdcpp = (self.pipe_holder.get("engine") == "sdcpp")
        engine_label = "Intel SD.cpp AVX2 ⚡" if is_sdcpp else "PyTorch Universal 🎨"
        self.sig_log.emit(f"\n--- INITIATING BATCH OF {batch_size} ({engine_label} - FULL BEAST MODE 🚀🔥) ---")

        try:
            sampler = self.p["sampler"]
            if not is_sdcpp and HAS_TORCH:
                if sampler == "Euler A":
                    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
                elif sampler == "DPM++ 2M":
                    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                elif sampler == "DDIM":
                    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

            prompt = self.p["prompt"]
            neg_prompt = self.p["neg_prompt"]
            steps = self.p["steps"]
            w = self.p["width"]
            h = self.p["height"]
            base_img_path = self.p["base_img_path"]
            use_low_ram = self.p["use_low_ram"]

            # Dolphin Auto-Prompting
            if self.p["use_dolphin"] and self.p["dolphin_path"] and os.path.exists(self.p["dolphin_path"]):
                if Llama is not None:
                    self.sig_log.emit("🐬 Dolphin 3.0 is expanding your prompt aesthetic...")
                    try:
                        llm = Llama(model_path=self.p["dolphin_path"], n_ctx=2048, verbose=False)
                        raw_chat = (
                            f"<|im_start|>system\n"
                            f"You are a master anime/digital art prompt engineer. Expand the user's prompt "
                            f"into high-detail Danbooru visual tags, cinematic lighting, and composition tokens. "
                            f"Output only comma-separated tags with no conversational filler.<|im_end|>\n"
                            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                        )
                        res = llm(raw_chat, max_tokens=120, stop=["<|im_end|>", "\n\n"], echo=False)
                        expanded = res['choices'][0]['text'].strip()
                        if expanded:
                            prompt = f"{prompt}, {expanded}"
                            self.sig_log.emit(f"🐬 Dolphin Added Tags: {expanded[:80]}...")
                    except Exception as dolph_err:
                        self.sig_log.emit(f"Dolphin Warning: {dolph_err}")

            for img_num in range(batch_size):
                if self.abort_requested:
                    self.sig_log.emit("\n🛑 Generation aborted by user!")
                    break

                self.sig_log.emit(f"\n✨ Image {img_num+1} of {batch_size} ✨")

                c = random.uniform(5.0, 15.0) if self.p["cfg_random"] else self.p["cfg"]
                d = random.uniform(0.3, 0.9) if self.p["denoise_random"] else self.p["denoising"]
                if d <= 0.0:
                    d = 0.01

                gen = torch.Generator("cpu")
                seed_val = self.p["seed"].strip()
                if seed_val.isdigit() and batch_size == 1:
                    actual_seed = int(seed_val)
                    gen.manual_seed(actual_seed)
                else:
                    actual_seed = gen.seed()
                    self.sig_log.emit(f"🌱 Auto-Seed: {actual_seed}")

                self.sig_log.emit(f"📏 CFG: {round(c, 1)} | 🧬 Denoise: {round(d, 2)} | 🏃 Steps: {steps}")

                partial_img = None

                def step_callback(step, t, latents):
                    if self.abort_requested:
                        try:
                            with torch.no_grad():
                                lat = latents / pipe.vae.config.scaling_factor
                                img_tmp = pipe.vae.decode(lat, return_dict=False)[0]
                                nonlocal partial_img
                                partial_img = pipe.image_processor.postprocess(img_tmp, output_type="pil")[0]
                        except Exception:
                            pass
                        raise Exception("Killed! 🛑")
                    self.sig_log.emit(f"👨‍🍳 Step {step+1}/{steps}...")

                if is_sdcpp:
                    self.sig_log.emit("⚡ Processing image with pure C++ Intel AVX2 Engine...")
                    if base_img_path and os.path.exists(base_img_path):
                        raw_img = Image.open(base_img_path).convert("RGB")
                        img_input = ImageOps.fit(raw_img, (w, h), Image.Resampling.LANCZOS)
                        imgs = pipe.img_to_img(
                            image=img_input,
                            prompt=prompt,
                            negative_prompt=neg_prompt,
                            cfg_scale=c,
                            sample_steps=steps,
                            seed=actual_seed
                        )
                    else:
                        imgs = pipe.txt_to_img(
                            prompt=prompt,
                            negative_prompt=neg_prompt,
                            width=w,
                            height=h,
                            cfg_scale=c,
                            sample_steps=steps,
                            seed=actual_seed
                        )
                    result = imgs[0] if isinstance(imgs, list) else imgs
                elif base_img_path and os.path.exists(base_img_path):
                    if not isinstance(pipe, StableDiffusionImg2ImgPipeline):
                        self.sig_log.emit("🔄 Switching pipeline to Img2Img mode...")
                        pipe = StableDiffusionImg2ImgPipeline(
                            vae=pipe.vae,
                            text_encoder=pipe.text_encoder,
                            tokenizer=pipe.tokenizer,
                            unet=pipe.unet,
                            scheduler=pipe.scheduler,
                            safety_checker=None,
                            feature_extractor=None
                        )
                        if use_low_ram:
                            pipe.enable_attention_slicing()
                        self.pipe_holder["pipe"] = pipe

                    raw_img = Image.open(base_img_path).convert("RGB")
                    img_input = ImageOps.fit(raw_img, (w, h), Image.Resampling.LANCZOS)
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        image=img_input,
                        guidance_scale=c,
                        strength=d,
                        num_inference_steps=steps,
                        generator=gen,
                        callback=step_callback,
                        callback_steps=1
                    ).images[0]
                else:
                    if not isinstance(pipe, StableDiffusionPipeline):
                        self.sig_log.emit("🔄 Switching pipeline to Text2Img mode...")
                        pipe = StableDiffusionPipeline(
                            vae=pipe.vae,
                            text_encoder=pipe.text_encoder,
                            tokenizer=pipe.tokenizer,
                            unet=pipe.unet,
                            scheduler=pipe.scheduler,
                            safety_checker=None,
                            feature_extractor=None
                        )
                        if use_low_ram:
                            pipe.enable_attention_slicing()
                        self.pipe_holder["pipe"] = pipe

                    result = pipe(
                        prompt=prompt,
                        negative_prompt=neg_prompt,
                        width=w,
                        height=h,
                        guidance_scale=c,
                        num_inference_steps=steps,
                        generator=gen,
                        callback=step_callback,
                        callback_steps=1
                    ).images[0]

                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"Cel_{timestamp}_Seed-{actual_seed}_Den-{round(float(d),2)}_CFG-{round(float(c),1)}_Stp-{steps}.png"
                filepath = os.path.join(str(HISTORY_DIR), filename)
                result.save(filepath)

                self.sig_preview.emit(result)
                self.sig_saved.emit(filepath, result)

                if self.p["evolve"] and (img_num < batch_size - 1):
                    base_img_path = filepath
                    self.sig_log.emit(f"🧬 Evolution: Image fed forward as next base!")

            self.sig_log.emit("\n✨ BATCH COMPLETED! ✨")

        except Exception as e:
            if "Killed!" in str(e):
                self.sig_log.emit(f"\n🛑 Generation stopped cleanly.")
            else:
                self.sig_log.emit(f"\n🛑 Error encountered: {e}")
                self.sig_error.emit(str(e))
        finally:
            self.sig_finished.emit()


# ─── MAIN WINDOW: CELSTUDIO ART STUDIO ────────────────────────────────────────
class CelStudioWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CelSuite Art Studio v269.4.0 (Liquid Glass Yellow Edition) 🎨✨")
        self.setWindowIcon(get_app_icon())

        # Dynamic Resizing & Memory
        self.setMinimumSize(740, 420)

        # App State & Settings
        self.settings = self.load_settings()
        self.model_path = self.settings.get("model_path", "")
        self.dolphin_path = self.settings.get("dolphin_path", "")
        self.lora_path = self.settings.get("lora_path", "")
        self.base_image_path = None
        self.active_display_image = None
        self.trash_mode = False
        self.is_panicking = False
        self.current_page = 0
        self.ITEMS_PER_PAGE = 15

        # Restore saved window size smoothly
        init_w = max(740, self.settings.get("window_width", 1440))
        init_h = max(420, self.settings.get("window_height", 840))
        self.resize(init_w, init_h)

        # Brain Pipeline State
        self.pipe_holder = {"pipe": None}
        self.is_loading_brain = False
        self.worker = None

        # Wallpaper Cache
        self._cached_wp_path = None
        self._cached_wp_pixmap = None

        # Build UI
        self._build_ui()
        self._build_panic_overlay()
        self._apply_glass_styles()

        # Restore Splitter Sizes ensuring Console fits greeting without wrap
        saved_splitters = self.settings.get("splitter_sizes", None)
        if saved_splitters and len(saved_splitters) == 3:
            if saved_splitters[2] < 380:
                diff = 380 - saved_splitters[2]
                saved_splitters[1] = max(320, saved_splitters[1] - diff)
                saved_splitters[2] = 380
            self.splitter.setSizes(saved_splitters)
        else:
            self.splitter.setSizes([400, 640, 380])

        # Keyboard Shortcuts
        self.panic_shortcut = QShortcut(QKeySequence("Ctrl+Space"), self)
        self.panic_shortcut.activated.connect(self.toggle_panic)

        # Greeting on Console (Fits on single line naturally with 380px default width!)
        self.log_console("✨ CelSuite Art Studio Initialized :) ✨\nAll CPU Cores Online! 🎨")
        self.refresh_gallery()

        # Auto-load model if configured
        if self.model_path and os.path.exists(self.model_path):
            self.log_console(f"🤫 Silently loading AI Checkpoint:\n{self.model_path}")
            threading.Thread(target=self._bg_load_model, daemon=True).start()

    # ─── SETTINGS MANAGEMENT ──────────────────────────────────────────────────
    def load_settings(self):
        default_neg = (
            "worst quality, low quality, normal quality, ugly, blurry, mutated, "
            "poorly drawn, extra limbs, bad anatomy, missing fingers, jpeg artifacts, "
            "watermark, signature, cringe"
        )
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "model_path": "",
            "dolphin_path": "",
            "lora_path": "",
            "pos_prompt": "1girl, cute anime girl, wearing cozy oversized sweater, warm cafe, soft sunlight, gentle smile, detailed eyes, masterpiece",
            "sticky_prompt": "masterpiece, best quality, ultra-detailed, highres, 8k resolution, cinematic lighting, soft lighting, sharp focus",
            "neg_prompt": default_neg,
            "use_dolphin": False,
            "use_low_ram": False,
            "wallpaper_name": "CelWeave Yellow.png",
            "wallpaper_tint": 65,
            "glass_opacity": 82,
            "use_pytorch": HAS_TORCH,
            "window_width": 1440,
            "window_height": 840,
            "splitter_sizes": [400, 640, 380]
        }

    def save_settings(self):
        data = {
            "model_path": self.model_path,
            "dolphin_path": self.dolphin_path,
            "lora_path": self.lora_path,
            "pos_prompt": self.pos_prompt_edit.toPlainText(),
            "sticky_prompt": self.sticky_prompt_edit.toPlainText(),
            "neg_prompt": self.neg_prompt_edit.toPlainText(),
            "use_dolphin": self.chk_dolphin.isChecked(),
            "use_low_ram": self.chk_low_ram.isChecked(),
            "use_pytorch": self.chk_use_pytorch.isChecked() if hasattr(self, "chk_use_pytorch") else HAS_TORCH,
            "wallpaper_name": self.combo_wallpaper.currentText(),
            "wallpaper_tint": self.slider_tint.value(),
            "glass_opacity": self.slider_glass.value(),
            "window_width": self.width(),
            "window_height": self.height(),
            "splitter_sizes": self.splitter.sizes() if hasattr(self, "splitter") else [400, 640, 380]
        }
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.log_console(f"Failed to save settings: {e}")

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    # ─── WALLPAPER ENGINE & PAINT EVENT ───────────────────────────────────────
    def _get_wallpaper_pixmap(self):
        wp_name = self.settings.get("wallpaper_name", "CelWeave Yellow.png")
        target_path = WALLPAPER_DIR / wp_name

        if not target_path.exists():
            target_path = WALLPAPER_DIR / "CelWeave Yellow.png"
            if not target_path.exists():
                pngs = list(WALLPAPER_DIR.glob("*.png")) + list(WALLPAPER_DIR.glob("*.webp"))
                target_path = pngs[0] if pngs else None

        if not target_path or not target_path.exists():
            return None

        if self._cached_wp_path != str(target_path) or self._cached_wp_pixmap is None:
            self._cached_wp_path = str(target_path)
            self._cached_wp_pixmap = QPixmap(str(target_path))

        return self._cached_wp_pixmap

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()

        if self.is_panicking:
            painter.fillRect(rect, QColor("#000000"))
            super().paintEvent(event)
            return

        pm = self._get_wallpaper_pixmap()
        if pm and not pm.isNull():
            scaled = pm.scaled(
                rect.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation
            )
            sx = max(0, (scaled.width() - rect.width()) // 2)
            sy = max(0, (scaled.height() - rect.height()) // 2)
            painter.drawPixmap(0, 0, scaled, sx, sy, rect.width(), rect.height())

            tint_val = self.settings.get("wallpaper_tint", 65) / 100.0
            tint_alpha = int(255 * tint_val)
            painter.fillRect(rect, QColor(0, 0, 0, tint_alpha))
        else:
            painter.fillRect(rect, QColor(BG_DEEP))

        super().paintEvent(event)

    # ─── LIQUID GLASS DYNAMIC STYLING (YELLOW THEME) ──────────────────────────
    def _apply_glass_styles(self):
        opacity = self.settings.get("glass_opacity", 82) / 100.0
        alpha_card = round(opacity * 0.72, 2)
        alpha_panel = round(opacity * 0.86, 2)
        alpha_input = round(opacity * 0.90, 2)

        style = f"""
        * {{
            font-family: 'Inter', 'Segoe UI', 'Ubuntu', sans-serif;
            font-size: 12px;
            color: {TEXT_MAIN};
            outline: none;
        }}
        QMainWindow {{ background: transparent; }}
        QWidget#central_widget {{ background: transparent; }}

        /* Liquid Glass Panels */
        QFrame.glass_panel {{
            background: rgba(18, 14, 8, {alpha_panel});
            border: 1px solid {BORDER_AMBER};
            border-radius: 12px;
        }}
        QFrame.glass_card {{
            background: rgba(28, 22, 12, {alpha_card});
            border: 1px solid rgba(234, 179, 8, 0.35);
            border-radius: 8px;
        }}

        /* Tabview Styling */
        QTabWidget::pane {{
            border: 1px solid {BORDER_AMBER};
            background: rgba(14, 10, 6, {alpha_panel});
            border-radius: 10px;
        }}
        QTabBar::tab {{
            background: rgba(24, 18, 8, 0.65);
            color: {TEXT_MUTED};
            padding: 8px 16px;
            margin-right: 4px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            border: 1px solid rgba(234, 179, 8, 0.2);
            font-weight: bold;
        }}
        QTabBar::tab:selected {{
            background: {AMBER_GOLD};
            color: #000000;
            border: 1px solid {BORDER_GLOW};
        }}
        QTabBar::tab:hover:!selected {{
            background: rgba(234, 179, 8, 0.3);
            color: #ffffff;
        }}

        /* Buttons */
        QPushButton {{
            background: {AMBER_BTN_BG};
            border: 1px solid {BORDER_AMBER};
            border-radius: 6px;
            padding: 6px 12px;
            font-weight: 600;
            color: {TEXT_MAIN};
        }}
        QPushButton:hover {{
            background: {AMBER_HOVER};
            border-color: {BORDER_GLOW};
            color: #000000;
        }}
        QPushButton:pressed {{
            background: {AMBER_PRESS};
            color: #ffffff;
        }}
        QPushButton#amber_cta {{
            background: {AMBER_GOLD};
            border: 1px solid {BORDER_GLOW};
            color: #000000;
            font-size: 14px;
            font-weight: 800;
            border-radius: 8px;
        }}
        QPushButton#amber_cta:hover {{
            background: #fde047;
            color: #000000;
        }}
        QPushButton#amber_cta:disabled {{
            background: rgba(50, 40, 15, 0.5);
            color: rgba(255, 255, 255, 0.3);
            border-color: transparent;
        }}
        QPushButton#help_btn {{
            background: rgba(202, 138, 4, 0.7);
            border: 1px solid rgba(250, 204, 21, 0.6);
            border-radius: 10px;
            padding: 0px;
            font-size: 11px;
            font-weight: bold;
            color: #ffffff;
        }}
        QPushButton#help_btn:hover {{
            background: #facc15;
            color: #000000;
            border-color: #ffffff;
        }}
        QPushButton#icon_action_btn {{
            padding: 0px;
            font-size: 13px;
        }}

        /* Text Input & TextEdits */
        QTextEdit, QLineEdit {{
            background: rgba(10, 8, 4, {alpha_input});
            border: 1px solid {BORDER_AMBER};
            border-radius: 6px;
            padding: 6px;
            color: #ffffff;
            selection-background-color: {AMBER_GOLD};
            selection-color: #000000;
        }}
        QTextEdit:focus, QLineEdit:focus {{
            border: 1px solid {BORDER_GLOW};
        }}

        /* Sliders */
        QSlider::groove:horizontal {{
            height: 4px;
            background: rgba(55, 42, 15, 0.8);
            border-radius: 2px;
        }}
        QSlider::sub-page:horizontal {{
            background: {AMBER_GOLD};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: #facc15;
            width: 14px;
            height: 14px;
            margin: -5px 0;
            border-radius: 7px;
            border: 2px solid #ffffff;
        }}

        /* Scrollbars */
        QScrollBar:vertical, QScrollBar:horizontal {{
            background: rgba(12, 9, 4, 0.5);
            border-radius: 3px;
        }}
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
            background: rgba(202, 138, 4, 0.6);
            border-radius: 3px;
        }}
        QScrollBar::handle:hover {{
            background: #facc15;
        }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}

        /* Combobox */
        QComboBox {{
            background: rgba(24, 18, 8, {alpha_input});
            border: 1px solid {BORDER_AMBER};
            border-radius: 6px;
            padding: 4px 10px;
        }}
        QComboBox QAbstractItemView {{
            background: #181206;
            border: 1px solid {BORDER_AMBER};
            selection-background-color: {AMBER_GOLD};
            selection-color: #000000;
        }}

        /* Console */
        QTextEdit#hacker_console {{
            background: rgba(6, 4, 2, 0.92);
            border: 1px solid {BORDER_AMBER};
            font-family: 'JetBrains Mono', 'Monospace', monospace;
            font-size: 11px;
            color: {TEXT_AMBER};
        }}
        """
        self.setStyleSheet(style)

    # ─── UI CONSTRUCTION WITH DYNAMIC SPLITTER ─────────────────────────────────
    def _build_ui(self):
        self.central_widget = QWidget(self)
        self.central_widget.setObjectName("central_widget")
        self.setCentralWidget(self.central_widget)

        self.main_hlayout = QHBoxLayout(self.central_widget)
        self.main_hlayout.setContentsMargins(10, 10, 10, 10)
        self.main_hlayout.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setHandleWidth(8)
        self.splitter.setStyleSheet("""
            QSplitter::handle {
                background: transparent;
            }
            QSplitter::handle:hover {
                background: rgba(234, 179, 8, 0.4);
                border-radius: 4px;
            }
        """)

        # ── LEFT COLUMN: STUDIO CONTROLS (TABS) ───────────────────────────────
        self.left_panel = QFrame()
        self.left_panel.setProperty("class", "glass_panel")
        self.left_panel.setMinimumWidth(280)
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)

        self.tab_widget = QTabWidget()
        left_layout.addWidget(self.tab_widget)

        # Tab 1: Studio Compose & Sliders
        self.tab_studio = QWidget()
        self._build_studio_tab(self.tab_studio)
        self.tab_widget.addTab(self.tab_studio, "Studio 🎨")

        # Tab 2: Settings & Models
        self.tab_settings = QWidget()
        self._build_settings_tab(self.tab_settings)
        self.tab_widget.addTab(self.tab_settings, "Settings ⚙️")

        # Generate & Abort CTA bar
        btn_bar = QHBoxLayout()
        self.btn_generate = QPushButton("Generate! ✨")
        self.btn_generate.setObjectName("amber_cta")
        self.btn_generate.setFixedHeight(48)
        self.btn_generate.clicked.connect(self.start_generation)

        self.btn_abort = QPushButton("🛑")
        self.btn_abort.setFixedSize(48, 48)
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self.abort_generation)

        btn_bar.addWidget(self.btn_generate, stretch=1)
        btn_bar.addWidget(self.btn_abort)
        left_layout.addLayout(btn_bar)

        self.splitter.addWidget(self.left_panel)

        # ── MIDDLE COLUMN: MAGIC MIRROR & GALLERY ─────────────────────────────
        self.mid_panel = QFrame()
        self.mid_panel.setProperty("class", "glass_panel")
        self.mid_panel.setMinimumWidth(320)
        mid_layout = QVBoxLayout(self.mid_panel)
        mid_layout.setContentsMargins(10, 10, 10, 10)

        # Viewport Header
        mid_top = QHBoxLayout()
        self.lbl_viewport_status = QLabel("Magic Mirror 🔮")
        self.lbl_viewport_status.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {TEXT_AMBER};")
        mid_top.addWidget(self.lbl_viewport_status)
        mid_top.addStretch()
        mid_layout.addLayout(mid_top)

        # Viewport Image
        self.lbl_image_viewer = QLabel("Waiting for magic... 🥺\n(Hit Generate or Select a Base Image)")
        self.lbl_image_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image_viewer.setStyleSheet(
            "background: rgba(0, 0, 0, 0.45); border-radius: 8px; border: 1px dashed rgba(234, 179, 8, 0.4);"
        )
        self.lbl_image_viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        mid_layout.addWidget(self.lbl_image_viewer, stretch=1)

        # Gallery Top Action Bar
        gal_bar = QHBoxLayout()
        lbl_gal = QLabel("Gallery 🕰️")
        lbl_gal.setStyleSheet("font-weight: bold; font-size: 13px;")
        gal_bar.addWidget(lbl_gal)

        btn_refresh = QPushButton("⟳")
        btn_refresh.setObjectName("icon_action_btn")
        btn_refresh.setFixedSize(30, 30)
        btn_refresh.clicked.connect(self.refresh_gallery)
        gal_bar.addWidget(btn_refresh)

        btn_open_folder = QPushButton("📂")
        btn_open_folder.setObjectName("icon_action_btn")
        btn_open_folder.setFixedSize(30, 30)
        btn_open_folder.clicked.connect(self.open_history_folder)
        gal_bar.addWidget(btn_open_folder)

        gal_bar.addSpacing(10)
        self.btn_prev_page = QPushButton("<")
        self.btn_prev_page.setObjectName("icon_action_btn")
        self.btn_prev_page.setFixedSize(30, 30)
        self.btn_prev_page.clicked.connect(self.prev_gallery_page)
        gal_bar.addWidget(self.btn_prev_page)

        self.lbl_page = QLabel("Pg 1/1")
        gal_bar.addWidget(self.lbl_page)

        self.btn_next_page = QPushButton(">")
        self.btn_next_page.setObjectName("icon_action_btn")
        self.btn_next_page.setFixedSize(30, 30)
        self.btn_next_page.clicked.connect(self.next_gallery_page)
        gal_bar.addWidget(self.btn_next_page)

        gal_bar.addStretch()

        self.btn_trash = QPushButton("Trash: OFF 🗑️")
        self.btn_trash.setFixedWidth(105)
        self.btn_trash.clicked.connect(self.toggle_trash_mode)
        gal_bar.addWidget(self.btn_trash)

        mid_layout.addLayout(gal_bar)

        # Horizontal Gallery Thumbnails ScrollArea
        self.gallery_scroll = QScrollArea()
        self.gallery_scroll.setFixedHeight(145)
        self.gallery_scroll.setWidgetResizable(True)
        self.gallery_scroll.setStyleSheet("background: rgba(6, 4, 2, 0.6); border: 1px solid rgba(234, 179, 8, 0.3); border-radius: 6px;")

        self.gallery_content = QWidget()
        self.gallery_hlayout = QHBoxLayout(self.gallery_content)
        self.gallery_hlayout.setContentsMargins(6, 6, 6, 6)
        self.gallery_hlayout.setSpacing(8)
        self.gallery_hlayout.addStretch()
        self.gallery_scroll.setWidget(self.gallery_content)

        mid_layout.addWidget(self.gallery_scroll)

        self.splitter.addWidget(self.mid_panel)

        # ── RIGHT COLUMN: CONSOLE (SIZED FOR CLEAN ONE-LINE GREETING & WRAPPING)
        self.right_panel = QFrame()
        self.right_panel.setProperty("class", "glass_panel")
        self.right_panel.setMinimumWidth(360)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)

        right_title = QLabel("Console 💻")
        right_title.setStyleSheet(f"font-weight: bold; font-size: 14px; color: {TEXT_AMBER};")
        right_layout.addWidget(right_title)

        self.txt_console = QTextEdit()
        self.txt_console.setObjectName("hacker_console")
        self.txt_console.setReadOnly(True)
        # Word wrapping enabled per Henny's request so logs/prompts wrap naturally!
        self.txt_console.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.txt_console.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        right_layout.addWidget(self.txt_console)

        self.splitter.addWidget(self.right_panel)

        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 5)
        self.splitter.setStretchFactor(2, 3)

        self.main_hlayout.addWidget(self.splitter)

    # ─── DYNAMIC RESIZING & ASPECT RATIO ADAPTATION ────────────────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "panic_overlay"):
            self.panic_overlay.setGeometry(self.rect())

        cur_w = self.width()
        cur_h = self.height()
        aspect = cur_w / max(1, cur_h)

        if hasattr(self, "gallery_scroll"):
            if cur_h < 560:
                self.gallery_scroll.setFixedHeight(115)
            else:
                self.gallery_scroll.setFixedHeight(145)

        if hasattr(self, "splitter") and not self.is_panicking:
            total_w = self.splitter.width()
            if total_w > 100:
                if cur_w < 950:
                    self.left_panel.setMaximumWidth(360)
                    self.right_panel.setMaximumWidth(360)
                elif aspect > 2.2:
                    self.left_panel.setMaximumWidth(520)
                    self.right_panel.setMaximumWidth(450)
                else:
                    self.left_panel.setMaximumWidth(16777215)
                    self.right_panel.setMaximumWidth(16777215)

    # ─── PROMPT BOX CREATOR WITH DYNAMIC AUTO-FIT / MAXIMIZE ──────────────────
    def _create_prompt_box(self, layout, title, setting_key, default_h, help_fn):
        header_w = QWidget()
        h = QHBoxLayout(header_w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        lbl = QLabel(title)
        lbl.setStyleSheet(f"font-weight: bold; color: {TEXT_AMBER};")
        h.addWidget(lbl)

        btn_help = QPushButton("?")
        btn_help.setObjectName("help_btn")
        btn_help.setFixedSize(20, 20)
        btn_help.clicked.connect(lambda: self.show_help_dialog(f"{title} Help!", help_fn()))
        h.addWidget(btn_help)

        h.addStretch()

        btn_expand = QPushButton("↕ Auto-Fit")
        btn_expand.setFixedHeight(22)
        btn_expand.setToolTip("Auto-fit height to text length or maximize")
        btn_expand.setStyleSheet(
            "font-size: 10px; font-weight: bold; padding: 2px 8px; "
            "background: rgba(234, 179, 8, 0.2); border: 1px solid rgba(234, 179, 8, 0.45); "
            "border-radius: 4px; color: #facc15;"
        )
        h.addWidget(btn_expand)
        layout.addWidget(header_w)

        edit = QTextEdit()
        edit.setFixedHeight(default_h)
        edit.setPlainText(self.settings.get(setting_key, ""))
        layout.addWidget(edit)

        edit._default_h = default_h
        edit._is_expanded = False

        def toggle_expand():
            if not edit._is_expanded:
                doc_height = int(edit.document().size().height())
                target_height = max(edit._default_h, min(550, doc_height + 26))
                edit.setFixedHeight(target_height)
                edit._is_expanded = True
                btn_expand.setText("✕ Compact")
                btn_expand.setStyleSheet(
                    "font-size: 10px; font-weight: bold; padding: 2px 8px; "
                    "background: rgba(185, 28, 28, 0.35); border: 1px solid rgba(239, 68, 68, 0.5); "
                    "border-radius: 4px; color: #fca5a5;"
                )
            else:
                edit.setFixedHeight(edit._default_h)
                edit._is_expanded = False
                btn_expand.setText("↕ Auto-Fit")
                btn_expand.setStyleSheet(
                    "font-size: 10px; font-weight: bold; padding: 2px 8px; "
                    "background: rgba(234, 179, 8, 0.2); border: 1px solid rgba(234, 179, 8, 0.45); "
                    "border-radius: 4px; color: #facc15;"
                )

        btn_expand.clicked.connect(toggle_expand)
        return edit

    # ─── STUDIO TAB CONTENTS ──────────────────────────────────────────────────
    def _build_studio_tab(self, parent):
        outer_layout = QVBoxLayout(parent)
        outer_layout.setContentsMargins(4, 4, 4, 4)
        outer_layout.setSpacing(6)

        base_frame = QFrame()
        base_frame.setProperty("class", "glass_card")
        b_layout = QHBoxLayout(base_frame)
        b_layout.setContentsMargins(6, 6, 6, 6)

        self.btn_select_base = QPushButton("Select Base 📁")
        self.btn_select_base.clicked.connect(self.select_base_image_dialog)
        b_layout.addWidget(self.btn_select_base, stretch=1)

        self.btn_paste = QPushButton("Paste 📋")
        self.btn_paste.setFixedWidth(84)
        self.btn_paste.clicked.connect(self.paste_from_clipboard)
        b_layout.addWidget(self.btn_paste)

        self.btn_clear_base = QPushButton("Clear 🧹")
        self.btn_clear_base.setFixedWidth(84)
        self.btn_clear_base.clicked.connect(self.clear_base_image)
        b_layout.addWidget(self.btn_clear_base)

        outer_layout.addWidget(base_frame)

        studio_scroll = QScrollArea()
        studio_scroll.setWidgetResizable(True)
        studio_scroll.setStyleSheet("background: transparent; border: none;")

        scroll_w = QWidget()
        s_layout = QVBoxLayout(scroll_w)
        s_layout.setContentsMargins(2, 4, 6, 4)
        s_layout.setSpacing(6)

        self.pos_prompt_edit = self._create_prompt_box(
            s_layout, "Main Subject 🗣️", "pos_prompt", 75, self.help_prompt
        )
        self.sticky_prompt_edit = self._create_prompt_box(
            s_layout, "Sticky Prompt 🍯", "sticky_prompt", 50, self.help_prompt
        )
        self.neg_prompt_edit = self._create_prompt_box(
            s_layout, "Negative Prompt 🛑", "neg_prompt", 65, self.help_prompt
        )

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet("background: rgba(234, 179, 8, 0.25); max-height: 1px; margin: 4px 0;")
        s_layout.addWidget(div)

        samp_h = QHBoxLayout()
        samp_h.addWidget(QLabel("Sampler 🧪"))
        btn_samp_help = QPushButton("?")
        btn_samp_help.setObjectName("help_btn")
        btn_samp_help.setFixedSize(20, 20)
        btn_samp_help.clicked.connect(lambda: self.show_help_dialog("Sampler Help! 🧪", self.help_samplers()))
        samp_h.addWidget(btn_samp_help)
        samp_h.addStretch()
        self.combo_sampler = QComboBox()
        self.combo_sampler.addItems(["Euler A", "DPM++ 2M", "DDIM"])
        self.combo_sampler.setFixedWidth(130)
        samp_h.addWidget(self.combo_sampler)
        s_layout.addLayout(samp_h)

        self.slider_steps, self.edit_steps = self._create_slider_row(
            s_layout, "Steps", 1, 100, 20, is_int=True, help_fn=self.help_steps
        )

        self.slider_cfg, self.edit_cfg, self.chk_cfg_rand = self._create_slider_row(
            s_layout, "CFG Scale", 10, 200, 70, is_int=False, scale=10.0, dice=True, help_fn=self.help_cfg
        )

        self.slider_denoise, self.edit_denoise, self.chk_denoise_rand = self._create_slider_row(
            s_layout, "Denoising", 1, 100, 60, is_int=False, scale=100.0, dice=True, help_fn=self.help_denoising
        )

        ratio_h = QHBoxLayout()
        ratio_h.addWidget(QLabel("Aspect Ratio 📐"))
        self.combo_ratio = QComboBox()
        self.combo_ratio.addItems(["Custom", "1:1 (Square)", "16:9 (Cinematic)", "9:16 (Portrait)", "4:3", "3:4"])
        self.combo_ratio.currentTextChanged.connect(self.apply_aspect_ratio)
        ratio_h.addWidget(self.combo_ratio)
        s_layout.addLayout(ratio_h)

        self.slider_width, self.edit_width = self._create_slider_row(
            s_layout, "Width", 128, 1024, 512, is_int=True, snap_8=True, help_fn=self.help_res
        )
        self.slider_height, self.edit_height = self._create_slider_row(
            s_layout, "Height", 128, 1024, 512, is_int=True, snap_8=True, help_fn=self.help_res
        )

        self.slider_batch, self.edit_batch = self._create_slider_row(
            s_layout, "Batch Queue 👯‍♀️", 1, 50, 1, is_int=True
        )
        self.chk_evolve = QCheckBox("Evolution Mode 🧬")
        s_layout.addWidget(self.chk_evolve)

        seed_h = QHBoxLayout()
        btn_seed_help = QPushButton("?")
        btn_seed_help.setObjectName("help_btn")
        btn_seed_help.setFixedSize(20, 20)
        btn_seed_help.clicked.connect(lambda: self.show_help_dialog("Seed Help! 🌰", self.help_seed()))
        seed_h.addWidget(btn_seed_help)
        self.edit_seed = QLineEdit()
        self.edit_seed.setPlaceholderText("Seed (Blank = Random) 🌰")
        seed_h.addWidget(self.edit_seed)
        s_layout.addLayout(seed_h)

        s_layout.addStretch()
        studio_scroll.setWidget(scroll_w)
        outer_layout.addWidget(studio_scroll, stretch=1)

    # ─── SETTINGS TAB CONTENTS ────────────────────────────────────────────────
    def _build_settings_tab(self, parent):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(10)

        lbl_models = QLabel("AI Models & Weights 🧠✨")
        lbl_models.setStyleSheet(f"font-weight: bold; font-size: 13px; color: {TEXT_AMBER};")
        layout.addWidget(lbl_models)

        m_frame = QHBoxLayout()
        self.btn_load_model = QPushButton("Load SD Checkpoint 🔮")
        self.btn_load_model.clicked.connect(self.choose_sd_model_dialog)
        m_frame.addWidget(self.btn_load_model, stretch=1)

        self.btn_unlink_model = QPushButton("Unlink 🚫")
        self.btn_unlink_model.setFixedWidth(84)
        self.btn_unlink_model.clicked.connect(self.unlink_model)
        m_frame.addWidget(self.btn_unlink_model)
        layout.addLayout(m_frame)

        lora_frame = QHBoxLayout()
        self.btn_load_lora = QPushButton("Load LoRA (Style) 🎀")
        self.btn_load_lora.clicked.connect(self.choose_lora_dialog)
        lora_frame.addWidget(self.btn_load_lora, stretch=1)

        self.btn_unlink_lora = QPushButton("Unlink 🚫")
        self.btn_unlink_lora.setFixedWidth(84)
        self.btn_unlink_lora.clicked.connect(self.unlink_lora)
        lora_frame.addWidget(self.btn_unlink_lora)
        layout.addLayout(lora_frame)

        dolph_frame = QHBoxLayout()
        self.btn_load_dolph = QPushButton("Load Dolphin GGUF 🐬")
        self.btn_load_dolph.clicked.connect(self.choose_dolphin_dialog)
        dolph_frame.addWidget(self.btn_load_dolph, stretch=1)

        self.btn_unlink_dolph = QPushButton("Unlink 🚫")
        self.btn_unlink_dolph.setFixedWidth(84)
        self.btn_unlink_dolph.clicked.connect(self.unlink_dolphin)
        dolph_frame.addWidget(self.btn_unlink_dolph)
        layout.addLayout(dolph_frame)

        self.chk_dolphin = QCheckBox("Enable Dolphin Auto-Prompting 🐬")
        self.chk_dolphin.setChecked(self.settings.get("use_dolphin", False))
        self.chk_dolphin.stateChanged.connect(self.save_settings)
        layout.addWidget(self.chk_dolphin)

        engine_h = QHBoxLayout()
        self.chk_use_pytorch = QCheckBox("Use PyTorch Engine (Diffusers) 🔮")
        default_pytorch = self.settings.get("use_pytorch", HAS_TORCH)
        self.chk_use_pytorch.setChecked(default_pytorch)
        self.chk_use_pytorch.stateChanged.connect(self.toggle_use_pytorch)
        engine_h.addWidget(self.chk_use_pytorch)

        btn_engine_help = QPushButton("?")
        btn_engine_help.setObjectName("help_btn")
        btn_engine_help.setFixedSize(20, 20)
        btn_engine_help.clicked.connect(lambda: self.show_help_dialog("Inference Engine Help ⚙️", self.help_inference_engine()))
        engine_h.addWidget(btn_engine_help)
        layout.addLayout(engine_h)

        ram_h = QHBoxLayout()
        self.chk_low_ram = QCheckBox("Low RAM Mode (Attention Slicing) 🫧")
        self.chk_low_ram.setChecked(self.settings.get("use_low_ram", False))
        self.chk_low_ram.stateChanged.connect(self.toggle_low_ram)
        ram_h.addWidget(self.chk_low_ram)

        btn_ram_help = QPushButton("?")
        btn_ram_help.setObjectName("help_btn")
        btn_ram_help.setFixedSize(20, 20)
        btn_ram_help.clicked.connect(lambda: self.show_help_dialog("Low RAM Help! 🧠", self.help_low_ram()))
        ram_h.addWidget(btn_ram_help)
        layout.addLayout(ram_h)

        btn_dump = QPushButton("Unload AI Brain (Free RAM) 🗑️💤")
        btn_dump.clicked.connect(self.unload_ai_brain)
        layout.addWidget(btn_dump)

        lbl_theme = QLabel("Wallpaper & Liquid Glass 🎨✨")
        lbl_theme.setStyleSheet(f"font-weight: bold; font-size: 13px; color: {TEXT_AMBER}; margin-top: 10px;")
        layout.addWidget(lbl_theme)

        wp_h = QHBoxLayout()
        wp_h.addWidget(QLabel("Wallpaper:"))
        self.combo_wallpaper = QComboBox()
        if WALLPAPER_DIR.exists():
            wallpapers = sorted([f.name for f in WALLPAPER_DIR.glob("*.png")] + [f.name for f in WALLPAPER_DIR.glob("*.webp")])
            self.combo_wallpaper.addItems(wallpapers)
            current_wp = self.settings.get("wallpaper_name", "CelWeave Yellow.png")
            idx = self.combo_wallpaper.findText(current_wp)
            if idx >= 0:
                self.combo_wallpaper.setCurrentIndex(idx)
        self.combo_wallpaper.currentTextChanged.connect(self._on_wallpaper_changed)
        wp_h.addWidget(self.combo_wallpaper)
        layout.addLayout(wp_h)

        tint_h = QHBoxLayout()
        tint_h.addWidget(QLabel("Dark Tint:"))
        self.slider_tint = QSlider(Qt.Orientation.Horizontal)
        self.slider_tint.setRange(20, 95)
        self.slider_tint.setValue(self.settings.get("wallpaper_tint", 65))
        self.slider_tint.valueChanged.connect(self._on_tint_changed)
        tint_h.addWidget(self.slider_tint)
        layout.addLayout(tint_h)

        glass_h = QHBoxLayout()
        glass_h.addWidget(QLabel("Glass Opacity:"))
        self.slider_glass = QSlider(Qt.Orientation.Horizontal)
        self.slider_glass.setRange(40, 100)
        self.slider_glass.setValue(self.settings.get("glass_opacity", 82))
        self.slider_glass.valueChanged.connect(self._on_glass_changed)
        glass_h.addWidget(self.slider_glass)
        layout.addLayout(glass_h)

        layout.addStretch()

        lbl_danger = QLabel("--- Danger Zone ---")
        lbl_danger.setStyleSheet("color: #ff5555; font-weight: bold;")
        layout.addWidget(lbl_danger)

        btn_reset_sliders = QPushButton("Reset Sliders Only 🧹")
        btn_reset_sliders.clicked.connect(self.reset_partial)
        layout.addWidget(btn_reset_sliders)

        nuke_h = QHBoxLayout()
        btn_nuke = QPushButton("NUKE EVERYTHING! ☢️")
        btn_nuke.setStyleSheet("background: #854d0e; color: #ffffff; font-weight: bold;")
        btn_nuke.clicked.connect(self.reset_all)
        nuke_h.addWidget(btn_nuke, stretch=1)

        btn_nuke_help = QPushButton("?")
        btn_nuke_help.setObjectName("help_btn")
        btn_nuke_help.setFixedSize(24, 24)
        btn_nuke_help.clicked.connect(lambda: self.show_help_dialog("Nuke Help! ☢️", self.help_nuke()))
        nuke_h.addWidget(btn_nuke_help)
        layout.addLayout(nuke_h)

    # ─── SLIDER HELPER CREATOR ────────────────────────────────────────────────
    def _create_slider_row(self, layout, title, min_val, max_val, default, is_int=True, scale=1.0, snap_8=False, dice=False, help_fn=None):
        frame = QVBoxLayout()
        top_h = QHBoxLayout()

        chk_dice = None
        if dice:
            chk_dice = QCheckBox(title)
            top_h.addWidget(chk_dice)
            lbl_dice = QLabel("🎲")
            lbl_dice.setStyleSheet(f"color: {TEXT_AMBER};")
            top_h.addWidget(lbl_dice)
        else:
            top_h.addWidget(QLabel(title))

        if help_fn:
            btn_h = QPushButton("?")
            btn_h.setObjectName("help_btn")
            btn_h.setFixedSize(20, 20)
            btn_h.clicked.connect(lambda: self.show_help_dialog(f"{title} Help!", help_fn()))
            top_h.addWidget(btn_h)

        top_h.addStretch()

        edit = QLineEdit()
        edit.setFixedWidth(54)
        val_str = str(int(default)) if is_int else str(round(default / scale, 2))
        edit.setText(val_str)
        top_h.addWidget(edit)
        frame.addLayout(top_h)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        frame.addWidget(slider)

        def on_slider_moved(val):
            edit.setText(str(val) if is_int else str(round(val / scale, 2)))

        def on_edit_finished():
            try:
                v = float(edit.text())
                if snap_8:
                    v = round(v / 8.0) * 8
                if is_int:
                    int_v = max(min_val, min(max_val, int(v)))
                    slider.setValue(int_v)
                    edit.setText(str(int_v))
                else:
                    scaled_v = max(min_val, min(max_val, int(v * scale)))
                    slider.setValue(scaled_v)
                    edit.setText(str(round(scaled_v / scale, 2)))
            except ValueError:
                pass

        slider.valueChanged.connect(on_slider_moved)
        edit.returnPressed.connect(on_edit_finished)

        layout.addLayout(frame)

        if dice:
            return slider, edit, chk_dice
        return slider, edit

    # ─── PANIC MODE OVERLAY (CROSS-PLATFORM DISGUISE) ──────────────────────────
    def _build_panic_overlay(self):
        self.panic_overlay = QFrame(self)
        self.panic_overlay.setStyleSheet("background: #000000; border: none;")
        self.panic_overlay.setGeometry(self.rect())
        self.panic_overlay.hide()

        p_layout = QVBoxLayout(self.panic_overlay)
        p_layout.setContentsMargins(16, 16, 16, 16)

        self.txt_panic_term = QTextEdit()
        term_color = "#eeffff" if sys.platform == "win32" else "#00ff00"
        self.txt_panic_term.setStyleSheet(
            f"background: #000000; color: {term_color}; font-family: 'Consolas', 'Monospace', monospace; font-size: 13px; border: none;"
        )
        self.txt_panic_term.setReadOnly(True)

        if sys.platform == "win32":
            fake_text = (
                "Windows PowerShell\n"
                "Copyright (C) Microsoft Corporation. All rights reserved.\n\n"
                "PS C:\\Users\\Henry> Get-Process | Sort-Object CPU -Descending | Select-Object -First 5\n\n"
                "Handles  NPM(K)    PM(K)      WS(K)     CPU(s)     Id ProcessName\n"
                "-------  ------    -----      -----     ------     -- -----------\n"
                "   1240      82   184512     195420      14.28   1420 explorer\n"
                "    840      45    95420     112400       5.12   2150 dwm\n"
                "    420      24    45100      52100       1.45   3892 svchost\n"
                "    180      12    18420      24100       0.08   4120 powershell\n\n"
                "[System Diagnostics] All background tasks healthy.\n"
                "PS C:\\Users\\Henry> _"
            )
        else:
            fake_text = (
                "user@linux:~$ top\n"
                "Tasks: 168 total,   1 running, 167 sleeping,   0 stopped,   0 zombie\n"
                "%Cpu(s):  2.3 us,  0.8 sy,  0.0 ni, 96.8 id,  0.1 wa,  0.0 hi,  0.0 si\n"
                "MiB Mem :   7963.2 total,   2154.6 free,   3812.4 used,   1996.2 buff/cache\n"
                "MiB Swap:   4096.0 total,   4096.0 free,      0.0 used.   3894.1 avail Mem\n\n"
                "    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND\n"
                "   1420 henry     20   0 1245084 184512  85412 S   2.1   2.3   0:14.28 cinnamon\n"
                "   2150 henry     20   0  842104  95420  42100 S   0.8   1.2   0:05.12 Xorg\n"
                "   3892 henry     20   0  421800  45100  21400 S   0.3   0.6   0:01.45 bash\n"
                "   4120 henry     20   0   18420   3412   2100 R   0.1   0.0   0:00.08 top\n\n"
                "[systemd] Cleaned up temporary files.\n"
                "[systemd] Started Daily apt download activities.\n"
                "user@linux:~$ _"
            )
        self.txt_panic_term.setPlainText(fake_text)
        p_layout.addWidget(self.txt_panic_term)

    def toggle_panic(self):
        self.is_panicking = not self.is_panicking
        if self.is_panicking:
            self._pre_panic_title = self.windowTitle()
            fake_title = "Windows PowerShell" if sys.platform == "win32" else "bash - user@linux:~"
            self.setWindowTitle(fake_title)
            self.central_widget.hide()
            self.panic_overlay.setGeometry(self.rect())
            self.panic_overlay.show()
            self.panic_overlay.raise_()
        else:
            self.panic_overlay.hide()
            self.central_widget.show()
            self.setWindowTitle(getattr(self, "_pre_panic_title", "CelSuite Art Studio 🎨✨"))
            self.setWindowIcon(get_app_icon())
            self.update()

    # ─── GALLERY LOGIC & SAFE TRASH ───────────────────────────────────────────
    def refresh_gallery(self):
        while self.gallery_hlayout.count() > 1:
            item = self.gallery_hlayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        imgs = sorted(glob.glob(os.path.join(str(HISTORY_DIR), "*.png")), key=os.path.getmtime, reverse=True)
        total_pages = max(1, (len(imgs) + self.ITEMS_PER_PAGE - 1) // self.ITEMS_PER_PAGE)
        if self.current_page >= total_pages:
            self.current_page = max(0, total_pages - 1)
        self.lbl_page.setText(f"Pg {self.current_page + 1}/{total_pages}")

        start = self.current_page * self.ITEMS_PER_PAGE
        page_imgs = imgs[start:start + self.ITEMS_PER_PAGE]

        for path in page_imgs:
            try:
                card = QFrame()
                card.setProperty("class", "glass_card")
                card.setFixedSize(110, 130)
                c_lay = QVBoxLayout(card)
                c_lay.setContentsMargins(4, 4, 4, 4)
                c_lay.setSpacing(2)

                btn_img = QPushButton()
                btn_img.setFixedSize(100, 100)
                btn_img.setStyleSheet("border: none; background: transparent; padding: 0;")
                pm = QPixmap(path).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                btn_img.setIcon(QIcon(pm))
                btn_img.setIconSize(QSize(100, 100))
                btn_img.clicked.connect(lambda _, p=path: self.handle_gallery_click(p))
                c_lay.addWidget(btn_img)

                display_name = os.path.basename(path).replace("Cel_", "").replace(".png", "")[:14]
                lbl_name = QLabel(display_name)
                lbl_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl_name.setStyleSheet("font-size: 9px; color: #d4b886;")
                c_lay.addWidget(lbl_name)

                self.gallery_hlayout.insertWidget(self.gallery_hlayout.count() - 1, card)
            except Exception:
                pass

    def handle_gallery_click(self, path):
        if self.trash_mode:
            try:
                dest = TRASH_DIR / os.path.basename(path)
                shutil.move(path, str(dest))
                self.log_console(f"🗑️ Moved image to Safe Trash: {dest.name}")
                if self.base_image_path == path:
                    self.clear_base_image()
                self.refresh_gallery()
            except Exception as e:
                self.log_console(f"Trash failed: {e}")
        else:
            self.display_image_path(path)
            self.base_image_path = path
            self.btn_select_base.setText(f"BASE: {os.path.basename(path)[:12]}...")
            self.btn_select_base.setStyleSheet(f"background: {AMBER_GOLD}; color: #000000; font-weight: bold;")
            self.log_console(f"🕰️ Continuity Image loaded as Base: {os.path.basename(path)}")

    def toggle_trash_mode(self):
        self.trash_mode = not self.trash_mode
        if self.trash_mode:
            self.btn_trash.setText("Trash: ON 🩸")
            self.btn_trash.setStyleSheet("background: #b91c1c; color: #ffffff; font-weight: bold;")
        else:
            self.btn_trash.setText("Trash: OFF 🗑️")
            self.btn_trash.setStyleSheet("")

    def prev_gallery_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.refresh_gallery()

    def next_gallery_page(self):
        self.current_page += 1
        self.refresh_gallery()

    def open_history_folder(self):
        try:
            if sys.platform == "win32":
                os.startfile(str(HISTORY_DIR))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(HISTORY_DIR)])
            else:
                subprocess.Popen(["xdg-open", str(HISTORY_DIR)])
        except Exception as e:
            self.log_console(f"Could not open gallery folder: {e}")

    # ─── IMAGE DISPLAY HELPER ─────────────────────────────────────────────────
    def display_image_path(self, path):
        pm = QPixmap(path)
        if not pm.isNull():
            self._set_viewport_pixmap(pm)

    def display_pil_image(self, pil_img):
        qim = ImageOps.contain(pil_img, (1024, 1024), Image.Resampling.LANCZOS)
        data = qim.convert("RGBA").tobytes("raw", "RGBA")
        qimage = QImage(data, qim.width, qim.height, QImage.Format.Format_RGBA8888)
        pm = QPixmap.fromImage(qimage)
        self._set_viewport_pixmap(pm)

    def _set_viewport_pixmap(self, pm):
        scaled = pm.scaled(
            self.lbl_image_viewer.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.lbl_image_viewer.setPixmap(scaled)
        self.lbl_image_viewer.setText("")

    # ─── BASE IMAGE ACTIONS ───────────────────────────────────────────────────
    def select_base_image_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Base Image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if p:
            self.base_image_path = p
            self.btn_select_base.setText(f"BASE: {os.path.basename(p)[:12]}...")
            self.btn_select_base.setStyleSheet(f"background: {AMBER_GOLD}; color: #000000; font-weight: bold;")
            self.display_image_path(p)
            self.log_console(f"Base Image Set: {os.path.basename(p)}")

    def paste_from_clipboard(self):
        try:
            img = ImageGrab.grabclipboard()
            if img:
                temp_p = HISTORY_DIR / "temp_clip.png"
                img.save(str(temp_p))
                self.base_image_path = str(temp_p)
                self.btn_select_base.setText("BASE: Clipboard 📋")
                self.btn_select_base.setStyleSheet(f"background: {AMBER_GOLD}; color: #000000; font-weight: bold;")
                self.display_image_path(str(temp_p))
                self.log_console("📋 Pasted image from clipboard as Base Image!")
            else:
                self.log_console("Clipboard does not contain an image! 😭")
        except Exception as e:
            self.log_console(f"Clipboard paste error: {e}")

    def clear_base_image(self):
        self.base_image_path = None
        self.btn_select_base.setText("Select Base 📁")
        self.btn_select_base.setStyleSheet("")
        self.log_console("🧹 Base image cleared. Back to Text2Img mode!")

    # ─── MODEL & LORA LOADERS ─────────────────────────────────────────────────
    def choose_sd_model_dialog(self):
        if self.is_loading_brain:
            self.log_console("🛑 Brain is already loading! Please wait.")
            return
        p, _ = QFileDialog.getOpenFileName(self, "Select SafeTensors Checkpoint", "", "SafeTensors (*.safetensors)")
        if p:
            self.unload_ai_brain()
            self.model_path = p
            self.save_settings()
            self.btn_load_model.setText("Loading AI Brain... 🧠")
            threading.Thread(target=self._bg_load_model, daemon=True).start()

    def _bg_load_model(self):
        self.is_loading_brain = True
        try:
            if HAS_TORCH:
                self.log_console(f"🧠 Loading SD checkpoint into CPU memory (Universal PyTorch Engine)...")
                pipe = StableDiffusionPipeline.from_single_file(
                    self.model_path,
                    safety_checker=None,
                    requires_safety_checker=False,
                    torch_dtype=torch.float32,
                    local_files_only=True
                ).to("cpu")

                if self.chk_low_ram.isChecked():
                    pipe.enable_attention_slicing()
                    self.log_console("🫧 Low RAM attention-slicing engaged!")

                if self.lora_path and os.path.exists(self.lora_path):
                    try:
                        pipe.load_lora_weights(self.lora_path)
                        self.log_console("🎀 Active LoRA weights loaded into brain!")
                    except Exception as le:
                        self.log_console(f"LoRA Load error: {le}")

                self.pipe_holder["pipe"] = pipe
                self.pipe_holder["engine"] = "diffusers"
                self.btn_load_model.setText("Brain Ready! 🧠✨")
                self.btn_load_model.setStyleSheet(f"background: {AMBER_GOLD}; color: #000000; font-weight: bold;")
                self.log_console("✅ Universal AI Brain loaded and ready to cook! 🚀🔥")
            elif HAS_SDCPP:
                self.log_console(f"⚡ Loading SD checkpoint into Intel SD.cpp AVX2 Engine...")
                pipe = stable_diffusion_cpp.StableDiffusion(
                    model_path=self.model_path,
                    n_threads=torch_threads,
                    verbose=False
                )
                self.pipe_holder["pipe"] = pipe
                self.pipe_holder["engine"] = "sdcpp"
                self.btn_load_model.setText("Brain Ready (SD.cpp)! ⚡")
                self.btn_load_model.setStyleSheet(f"background: {AMBER_GOLD}; color: #000000; font-weight: bold;")
                self.log_console("✅ Intel SD.cpp AVX2 Brain loaded and ready to cook! ⚡🔥")
            else:
                raise RuntimeError("Neither PyTorch nor stable-diffusion-cpp is available!")
        except Exception as e:
            self.log_console(f"🛑 Brain Load Failed: {e}")
            self.btn_load_model.setText("Load SD Checkpoint 🔮")
            self.btn_load_model.setStyleSheet("")
        finally:
            self.is_loading_brain = False

    def unlink_model(self):
        self.model_path = ""
        self.save_settings()
        self.unload_ai_brain()
        self.btn_load_model.setText("Load SD Checkpoint 🔮")
        self.btn_load_model.setStyleSheet("")
        self.log_console("🚫 Checkpoint unlinked and RAM dumped.")

    def choose_lora_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select LoRA Weight", "", "SafeTensors (*.safetensors)")
        if p:
            self.lora_path = p
            self.save_settings()
            self.btn_load_lora.setText("LoRA Linked! 🎀")
            self.btn_load_lora.setStyleSheet(f"background: {AMBER_GOLD}; color: #000000; font-weight: bold;")
            pipe = self.pipe_holder.get("pipe")
            if pipe:
                try:
                    pipe.load_lora_weights(self.lora_path)
                    self.log_console("🎀 Injected LoRA weights into running model!")
                except Exception as e:
                    self.log_console(f"LoRA Injection Failed: {e}")

    def unlink_lora(self):
        self.lora_path = ""
        self.save_settings()
        self.btn_load_lora.setText("Load LoRA (Style) 🎀")
        self.btn_load_lora.setStyleSheet("")
        pipe = self.pipe_holder.get("pipe")
        if pipe:
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass
        self.log_console("🚫 LoRA unlinked. Model back to default style.")

    def choose_dolphin_dialog(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Dolphin GGUF", "", "GGUF (*.gguf)")
        if p:
            self.dolphin_path = p
            self.save_settings()
            self.btn_load_dolph.setText("Dolphin Ready! 🐬")
            self.btn_load_dolph.setStyleSheet(f"background: {AMBER_GOLD}; color: #000000; font-weight: bold;")
            self.log_console("🐬 Dolphin GGUF registered for auto-prompting!")

    def unlink_dolphin(self):
        self.dolphin_path = ""
        self.save_settings()
        self.btn_load_dolph.setText("Load Dolphin GGUF 🐬")
        self.btn_load_dolph.setStyleSheet("")
        self.log_console("🚫 Dolphin model unlinked.")

    def toggle_use_pytorch(self, state):
        want_pytorch = (state == 2 or state == Qt.CheckState.Checked.value or bool(state))
        if want_pytorch:
            if HAS_TORCH:
                self.pipe_holder["engine"] = "diffusers"
                self.settings["use_pytorch"] = True
                self.save_settings()
                self.log_console("🔄 Switched to Universal PyTorch Engine! 🎨")
            else:
                venv_python = self.find_pytorch_venv()
                self.chk_use_pytorch.blockSignals(True)
                self.chk_use_pytorch.setChecked(False)
                self.chk_use_pytorch.blockSignals(False)
                if venv_python:
                    self.log_console(f"💡 Detected external environment with PyTorch at: {venv_python}")
                    self.show_help_dialog(
                        "PyTorch Detected Outside Binary! 🧠",
                        f"Found an external Python environment with PyTorch at:\n\n{venv_python}\n\n"
                        "To use the Universal PyTorch Engine, launch CelSuite Art Studio using that environment:\n\n"
                        f"{venv_python} \"{SCRIPT_DIR / 'CelAS.py'}\""
                    )
                else:
                    self.show_pytorch_setup_dialog()
        else:
            if HAS_SDCPP:
                self.pipe_holder["engine"] = "sdcpp"
                self.settings["use_pytorch"] = False
                self.save_settings()
                self.log_console("⚡ Switched to Intel SD.cpp AVX2 Engine!")
            else:
                self.log_console("⚠️ Intel SD.cpp Engine is not available in this bundle.")

    def find_pytorch_venv(self):
        """Scans common venv locations to see if PyTorch is installed in any external environment."""
        home = Path.home()
        candidates = [
            home / "venv",
            home / ".venv",
            home / "env",
            home / ".env",
            home / "Documents/Projects/Python/venv",
            home / "Documents/venv",
            home / "miniconda3",
            home / "anaconda3",
            SCRIPT_DIR / "venv",
            SCRIPT_DIR / ".venv",
            SCRIPT_DIR.parent / "venv"
        ]

        for cand in candidates:
            if not cand.exists():
                continue
            py_bin = cand / "bin" / "python"
            if not py_bin.exists():
                py_bin = cand / "Scripts" / "python.exe"
            if py_bin.exists():
                sp = list(cand.glob("lib/python*/site-packages/torch")) or list(cand.glob("Lib/site-packages/torch"))
                if sp:
                    return str(py_bin)
        return None

    def show_pytorch_setup_dialog(self):
        """Displays a clean, copyable dialog with step-by-step setup commands for users."""
        is_win = sys.platform == "win32"
        if is_win:
            commands = (
                "# 1. Create a Python virtual environment in your user folder\n"
                "python -m venv %USERPROFILE%\\venv\n\n"
                "# 2. Activate the virtual environment\n"
                "%USERPROFILE%\\venv\\Scripts\\activate\n\n"
                "# 3. Install lightweight CPU-optimized PyTorch & Diffusers\n"
                "pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "pip install diffusers transformers accelerate safetensors Pillow PySide6\n"
            )
        else:
            commands = (
                "# 1. Create a Python virtual environment in ~/venv\n"
                "python3 -m venv ~/venv\n\n"
                "# 2. Activate the virtual environment\n"
                "source ~/venv/bin/activate\n\n"
                "# 3. Install lightweight CPU-optimized PyTorch & Diffusers\n"
                "pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                "pip install diffusers transformers accelerate safetensors Pillow PySide6\n"
            )

        dlg = QDialog(self)
        dlg.setWindowTitle("PyTorch & Diffusers Setup Guide 🧠")
        dlg.resize(580, 420)
        dlg.setStyleSheet(f"""
            QDialog {{
                background-color: rgba(18, 14, 8, 0.96);
                border: 2px solid {AMBER_GOLD};
                border-radius: 12px;
            }}
            QLabel {{
                color: #ffffff;
                font-size: 13px;
            }}
            QTextEdit {{
                background-color: rgba(0, 0, 0, 0.7);
                color: {TEXT_AMBER};
                font-family: monospace;
                font-size: 12px;
                border: 1px solid rgba(243, 156, 18, 0.4);
                border-radius: 8px;
                padding: 8px;
            }}
            QPushButton {{
                background-color: {AMBER_GOLD};
                color: #000000;
                font-weight: bold;
                border-radius: 6px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: #f1c40f;
            }}
        """)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        title = QLabel("PyTorch & Diffusers Not Detected ⚠️")
        title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {AMBER_GOLD};")
        layout.addWidget(title)

        desc = QLabel(
            "This portable binary is running in Intel SD.cpp mode. To unlock the full Universal PyTorch "
            "engine with all custom schedulers and checkpoints, run these commands in your terminal to "
            "set up a lightweight CPU environment:"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        txt = QTextEdit()
        txt.setPlainText(commands)
        txt.setReadOnly(True)
        layout.addWidget(txt)

        btn_row = QHBoxLayout()
        btn_copy = QPushButton("Copy Commands 📋")
        btn_copy.clicked.connect(lambda: QApplication.clipboard().setText(commands))
        btn_row.addWidget(btn_copy)

        btn_close = QPushButton("Got it! 👍")
        btn_close.clicked.connect(dlg.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        dlg.exec()

    def help_inference_engine(self):
        return (
            "CelSuite Art Studio Dual-Engine System:\n\n"
            "• Universal PyTorch Engine (Diffusers):\n"
            "  Supports all custom Stable Diffusion checkpoints, LoRAs, and schedulers "
            "(Euler A, DPM++ 2M, DDIM) across all AMD and Intel processors.\n\n"
            "• Intel AVX2 Engine (stable-diffusion.cpp):\n"
            "  Pure C++ ultra-lightweight inference. Uses ~50% less RAM and runs 2x faster "
            "on Intel CPUs with AVX2 vector extensions without needing PyTorch installed."
        )

    def toggle_low_ram(self):
        self.save_settings()
        pipe = self.pipe_holder.get("pipe")
        if pipe:
            if self.chk_low_ram.isChecked():
                pipe.enable_attention_slicing()
                self.log_console("🫧 Low RAM Attention Slicing enabled on active pipeline!")
            else:
                pipe.disable_attention_slicing()
                self.log_console("🔥 Attention slicing disabled (full speed mode).")

    def unload_ai_brain(self):
        self.pipe_holder["pipe"] = None
        gc.collect()
        self.btn_load_model.setText("Load SD Checkpoint 🔮")
        self.btn_load_model.setStyleSheet("")
        self.log_console("🗑️💤 AI Brain purged from RAM. System memory freed!")

    # ─── GENERATION CONTROL ───────────────────────────────────────────────────
    def start_generation(self):
        if not self.pipe_holder.get("pipe"):
            self.log_console("🛑 No model loaded! Load an SD Checkpoint in Settings first.")
            return

        self.save_settings()

        p_main = self.pos_prompt_edit.toPlainText().strip()
        p_sticky = self.sticky_prompt_edit.toPlainText().strip()
        prompt = f"{p_main}, {p_sticky}" if p_sticky else p_main

        neg_prompt = self.neg_prompt_edit.toPlainText()
        sampler = self.combo_sampler.currentText()
        steps = self.slider_steps.value()
        cfg = self.slider_cfg.value() / 10.0
        cfg_random = self.chk_cfg_rand.isChecked()
        denoising = self.slider_denoise.value() / 100.0
        denoise_random = self.chk_denoise_rand.isChecked()

        w = self.slider_width.value()
        h = self.slider_height.value()
        w = w - (w % 8)
        h = h - (h % 8)

        batch_size = self.slider_batch.value()
        evolve = self.chk_evolve.isChecked()
        seed = self.edit_seed.text().strip()

        params = {
            "prompt": prompt,
            "neg_prompt": neg_prompt,
            "sampler": sampler,
            "steps": steps,
            "cfg": cfg,
            "cfg_random": cfg_random,
            "denoising": denoising,
            "denoise_random": denoise_random,
            "width": w,
            "height": h,
            "batch_size": batch_size,
            "evolve": evolve,
            "seed": seed,
            "base_img_path": self.base_image_path,
            "use_dolphin": self.chk_dolphin.isChecked(),
            "dolphin_path": self.dolphin_path,
            "use_low_ram": self.chk_low_ram.isChecked()
        }

        self.btn_generate.setEnabled(False)
        self.btn_generate.setText("Cooking Queue... 🍳")
        self.btn_abort.setEnabled(True)

        self.worker = SDWorkerThread(self.pipe_holder, params)
        self.worker.sig_log.connect(self.log_console)
        self.worker.sig_preview.connect(self.display_pil_image)
        self.worker.sig_saved.connect(self._on_image_saved)
        self.worker.sig_finished.connect(self._on_generation_finished)
        self.worker.sig_error.connect(lambda err: self.log_console(f"🛑 Generation Error: {err}"))
        self.worker.start()

    def abort_generation(self):
        if self.worker and self.worker.isRunning():
            self.log_console("\n🛑🔪 ABORT SIGNAL SENT!")
            self.worker.request_abort()

    def _on_image_saved(self, filepath, pil_img):
        self.refresh_gallery()

    def _on_generation_finished(self):
        self.btn_generate.setEnabled(True)
        self.btn_generate.setText("Generate! ✨")
        self.btn_abort.setEnabled(False)

    # ─── WALLPAPER & THEME HANDLERS ───────────────────────────────────────────
    def _on_wallpaper_changed(self, name):
        self.settings["wallpaper_name"] = name
        self.save_settings()
        self.update()

    def _on_tint_changed(self, val):
        self.settings["wallpaper_tint"] = val
        self.save_settings()
        self.update()

    def _on_glass_changed(self, val):
        self.settings["glass_opacity"] = val
        self.save_settings()
        self._apply_glass_styles()

    def apply_aspect_ratio(self, choice):
        ratios = {
            "1:1 (Square)": (512, 512),
            "16:9 (Cinematic)": (768, 432),
            "9:16 (Portrait)": (432, 768),
            "4:3": (640, 480),
            "3:4": (480, 640)
        }
        if choice in ratios:
            w, h = ratios[choice]
            self.slider_width.setValue(w)
            self.slider_height.setValue(h)

    def reset_partial(self):
        self.slider_steps.setValue(20)
        self.slider_cfg.setValue(70)
        self.slider_denoise.setValue(60)
        self.slider_width.setValue(512)
        self.slider_height.setValue(512)
        self.combo_sampler.setCurrentText("Euler A")
        self.combo_ratio.setCurrentText("Custom")
        self.edit_seed.clear()
        self.log_console("🧹 Sliders reset to defaults.")

    def reset_all(self):
        self.reset_partial()
        self.pos_prompt_edit.clear()
        self.sticky_prompt_edit.clear()
        self.neg_prompt_edit.clear()
        self.log_console("☢️ NUKED EVERYTHING back to factory clean state!")

    def log_console(self, text):
        self.txt_console.append(text)
        sb = self.txt_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ─── HELP MASTERCLASSES (HENNY'S ICONIC ESSAYS - DYNAMICALLY SIZED) ────────
    def show_help_dialog(self, title, text):
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setStyleSheet(f"""
            QDialog {{
                background: rgba(16, 12, 6, 0.96);
                border: 1px solid {BORDER_AMBER};
                border-radius: 10px;
            }}
            QPushButton#dialog_got_it {{
                background: {AMBER_GOLD};
                color: #000000;
                font-weight: 800;
                font-size: 13px;
                border: 1px solid {BORDER_GLOW};
                border-radius: 6px;
                padding: 8px 16px;
            }}
            QPushButton#dialog_got_it:hover {{
                background: #fde047;
                border: 1px solid #ffffff;
                color: #000000;
            }}
            QPushButton#dialog_got_it:pressed {{
                background: {AMBER_PRESS};
                color: #ffffff;
            }}
        """)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        txt = QTextEdit()
        txt.setReadOnly(True)
        clean_text = text.strip() + "\n\n"
        txt.setPlainText(clean_text)
        txt.setStyleSheet("font-size: 13px; line-height: 1.5; color: #fefce8; background: transparent; border: none;")

        dlg.setFixedWidth(520)
        doc = txt.document()
        doc.setTextWidth(480)
        doc_h = int(doc.size().height())
        target_h = max(180, min(560, doc_h + 30))
        txt.setFixedHeight(target_h)

        lay.addWidget(txt)

        btn = QPushButton("Got it! 💖")
        btn.setObjectName("dialog_got_it")
        btn.clicked.connect(dlg.accept)
        lay.addWidget(btn)

        dlg.adjustSize()
        dlg.exec()

    def help_prompt(self):
        return (
            "🔥 HOW TO TALK TO THE AI (THE MASTERCLASS) 🔥\n\n"
            "The AI does NOT speak English sentences. It does not understand verbs or prepositions.\n"
            "If you say 'Make a cat sitting on a bed and looking cute,' it will literally panic.\n"
            "It translates everything into isolated math tokens based on a system called Danbooru tags.\n\n"
            "You absolutely MUST speak to it using isolated keywords separated by commas.\n\n"
            "✅ THE PERFECT PROMPT FORMULA:\n"
            "1. The Subject (e.g., 1girl, cat, car)\n"
            "2. The Core Description (e.g., red hair, blue eyes, oversized off-shoulder sweater)\n"
            "3. The Pose/Action (e.g., sitting, blushing, pulling sweater, looking at viewer)\n"
            "4. The Background (e.g., cyberpunk city, cozy bedroom, outdoors)\n"
            "5. The Quality Modifiers (Crucial for fixing mush!)\n\n"
            "🔥 TRICKY TAGS THAT FIX MUSH:\n"
            "masterpiece, best quality, ultra-detailed, highres, 8k resolution, cinematic lighting\n\n"
            "🛑 THE NEGATIVE PROMPT (MANDATORY):\n"
            "worst quality, low quality, normal quality, ugly, blurry, mutated, poorly drawn, "
            "extra limbs, bad anatomy, missing fingers, jpeg artifacts, watermark, signature"
        )

    def help_samplers(self):
        return (
            "🧪 WHAT ARE SAMPLERS? (THE DEEP MATH) 🧪\n\n"
            "Samplers are the secret mathematical recipes the AI uses to clear away the static! ✨\n\n"
            "🎨 Euler A (Euler Ancestral):\n"
            "The most creative, chaotic, and artistic sampler! Adds a tiny bit of new random noise on every step.\n"
            "Best used for anime and soft painting styles at 20-25 steps.\n\n"
            "📸 DPM++ 2M (Stable & Photorealistic):\n"
            "Heavyweight champion of precision! Uses differential math to solve static without adding chaos.\n"
            "Creates ultra-crisp, realistic details and coherent backgrounds.\n\n"
            "🕰️ DDIM:\n"
            "Deterministic and reliable. Best when using a Base Image (Img2Img) because it respects the original structure."
        )

    def help_res(self):
        return (
            "📐 RESOLUTION RULES (THE 8-BIT LAW) 📐\n\n"
            "Stable Diffusion 1.5 was trained inside a strict 512x512 lattice.\n"
            "Its tensors scale strictly in multiples of 8. If you try non-multiples of 8, tensors misalign and crash!\n"
            "CelStudio snaps all resolutions to the nearest 8 automatically."
        )

    def help_steps(self):
        return (
            "🏃‍♂️ REFINEMENT STEPS (THE COOKING TIME) 🏃‍♂️\n\n"
            "Steps control how many denoising iterations the AI runs.\n"
            "• 1 to 15: Fast smudge.\n"
            "• 20 to 30: The Golden Zone! Fully resolved details.\n"
            "• 40+: Diminishing returns on CPU."
        )

    def help_cfg(self):
        return (
            "📏 CFG SCALE (CLASSIFIER FREE GUIDANCE) 📏\n\n"
            "The AI's obedience meter.\n"
            "• 1.0 - 3.0: High artistic freedom, dreamy.\n"
            "• 7.0 - 8.0: The industry standard sweet spot.\n"
            "• 15.0+: Deep-fried, hyper-saturated glitch art."
        )

    def help_denoising(self):
        return (
            "🧬 DENOISING STRENGTH (THE MUTATION METER) 🧬\n\n"
            "Active during Img2Img.\n"
            "• 0.01 - 0.30: Light Instagram filter.\n"
            "• 0.40 - 0.65: Magic restyle zone! Perfect for transforming photos to anime.\n"
            "• 0.80 - 1.0: Total nuclear destruction into new artwork."
        )

    def help_seed(self):
        return (
            "🌱 THE SEED (THE DNA OF THE UNIVERSE) 🌱\n\n"
            "Every image starts with random static keyed to an integer seed.\n"
            "Copying the seed from an image allows you to reproduce the exact composition and tweak words!"
        )

    def help_nuke(self):
        return (
            "☢️ NUKE EVERYTHING (THE PANIC RESET) ☢️\n\n"
            "Resets prompts, sliders, and dimensions back to factory defaults for a fresh canvas."
        )

    def help_low_ram(self):
        return (
            "🧠 LOW RAM MODE (THE SURVIVAL SWITCH) 🧠\n\n"
            "Cross-attention matrices during generation can spike memory by 4GB+.\n"
            "Attention Slicing divides the computation into smaller strips, slashing spikes to ~1GB so your 8GB Linux system never runs out of memory!"
        )


# ─── ENTRY POINT & AUTO-SCREENSHOT CLI ────────────────────────────────────────
def main():
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    app = QApplication(sys.argv)
    app.setApplicationName("CelSuite Art Studio")
    app.setApplicationVersion("269.4.0")
    app.setWindowIcon(get_app_icon())

    win = CelStudioWindow()

    # Offscreen visual test argument handling
    auto_shot = False
    shot_path = "celas_preview.png"
    test_panic = False
    test_expand_prompts = False

    for i, arg in enumerate(sys.argv):
        if arg == "--auto-screenshot":
            auto_shot = True
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                shot_path = sys.argv[i + 1]
        elif arg.startswith("--auto-screenshot="):
            auto_shot = True
            shot_path = arg.split("=", 1)[1]
        elif arg == "--test-panic":
            test_panic = True
        elif arg == "--test-expand-prompts":
            test_expand_prompts = True

    if test_panic:
        win.toggle_panic()

    if test_expand_prompts:
        for edit in (win.pos_prompt_edit, win.sticky_prompt_edit, win.neg_prompt_edit):
            doc_height = int(edit.document().size().height())
            edit.setFixedHeight(max(edit._default_h, min(550, doc_height + 26)))

    win.show()

    if auto_shot:
        def capture_and_exit():
            try:
                p = Path(shot_path).resolve()
                p.parent.mkdir(parents=True, exist_ok=True)
                pix = win.grab()
                pix.save(str(p))
                print(f"[AUTO_SCREENSHOT] Saved CelSuite snapshot to: {p}")
            except Exception as ex:
                print(f"[AUTO_SCREENSHOT] Error: {ex}")
            finally:
                app.quit()

        app.processEvents()
        QTimer.singleShot(500, capture_and_exit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
