"""Streamlit front end for the fixed Ag+ DNG prediction workflow.

The numerical steps intentionally mirror manual_ROI_prediction.py. This file
does not import that script because importing it would launch its Tk GUI.
"""

from __future__ import annotations

import base64
import hashlib
import csv
import io
import os
import tempfile
from typing import Any

import cv2
import numpy as np
import rawpy
import streamlit as st


MODEL_A = -19.301667
MODEL_B = 21.280790
VALID_MIN = 2.0
VALID_MAX = 12.0
EPS = 1e-6


@st.cache_data(show_spinner=False)
def read_dng_bytes(data: bytes) -> np.ndarray:
    """Decode an uploaded DNG exactly as the original desktop script does."""
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".DNG", delete=False) as tmp:
            tmp.write(data)
            temp_path = tmp.name

        with rawpy.imread(temp_path) as raw:
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
            )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@st.cache_data(show_spinner=False)
def read_camera_image_bytes(data: bytes) -> np.ndarray:
    """Decode a browser camera JPEG/PNG into the BGR array used downstream."""
    encoded = np.frombuffer(data, dtype=np.uint8)
    image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("浏览器返回的图片无法解码，请重新拍摄。")
    return image_bgr


def mean_rgb(
    image_bgr: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[float, float, float]:
    """Equivalent to manual_ROI_prediction.mean_rgb."""
    x, y, w, h = roi
    crop = image_bgr[y : y + h, x : x + w]
    if crop.size == 0:
        raise ValueError("分析区域为空")
    b, g, r = cv2.mean(crop)[:3]
    return float(r), float(g), float(b)


def rgb_to_y(r: float, g: float, b: float) -> float:
    return 0.299 * r + 0.587 * g + 0.114 * b


def predict_pair(
    img_left: np.ndarray,
    img_right: np.ndarray,
    roi_left: tuple[int, int, int, int],
    roi_right: tuple[int, int, int, int],
) -> dict[str, Any]:
    """Run the fixed calculation from manual_ROI_prediction.py."""
    r0_l, g0_l, b0_l = mean_rgb(img_left, roi_left)
    rs_l, gs_l, bs_l = mean_rgb(img_left, roi_right)
    y0_left = rgb_to_y(r0_l, g0_l, b0_l)
    ys_left = rgb_to_y(rs_l, gs_l, bs_l)
    y_ratio_left = ys_left / max(y0_left, EPS)

    rs_r, gs_r, bs_r = mean_rgb(img_right, roi_left)
    r0_r, g0_r, b0_r = mean_rgb(img_right, roi_right)
    ys_right = rgb_to_y(rs_r, gs_r, bs_r)
    y0_right = rgb_to_y(r0_r, g0_r, b0_r)
    y_ratio_right = ys_right / max(y0_right, EPS)

    if (
        y_ratio_left <= 0
        or y_ratio_right <= 0
        or not np.isfinite(y_ratio_left)
        or not np.isfinite(y_ratio_right)
    ):
        y_ratio_corr = np.nan
        predicted_um = np.nan
        range_status = "INVALID_RATIO"
    else:
        y_ratio_corr = float(np.sqrt(y_ratio_left * y_ratio_right))
        predicted_um = MODEL_A * y_ratio_corr + MODEL_B
        range_status = (
            "IN_RANGE"
            if VALID_MIN <= predicted_um <= VALID_MAX
            else "OUT_OF_MODEL_RANGE"
        )

    swap_difference = (
        abs(y_ratio_left - y_ratio_right)
        / max((y_ratio_left + y_ratio_right) / 2.0, EPS)
        * 100
    )

    return {
        "predicted_Ag_uM": predicted_um,
        "range_status": range_status,
        "Y_ratio_left": y_ratio_left,
        "Y_ratio_right": y_ratio_right,
        "Y_ratio_corr": y_ratio_corr,
        "position_swap_difference_percent": swap_difference,
        "left_blank_R": r0_l,
        "left_blank_G": g0_l,
        "left_blank_B": b0_l,
        "left_blank_Y": y0_left,
        "left_sample_R": rs_l,
        "left_sample_G": gs_l,
        "left_sample_B": bs_l,
        "left_sample_Y": ys_left,
        "right_sample_R": rs_r,
        "right_sample_G": gs_r,
        "right_sample_B": bs_r,
        "right_sample_Y": ys_right,
        "right_blank_R": r0_r,
        "right_blank_G": g0_r,
        "right_blank_B": b0_r,
        "right_blank_Y": y0_right,
    }


def roi_display_image_b64(image_bgr: np.ndarray) -> str:
    """Encode a display-only JPEG; ROI calculations still use the source array."""
    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    if not ok:
        raise ValueError("分析区域预览图像编码失败")
    return base64.b64encode(encoded).decode("ascii")


ROI_VIEWER_HTML = """
<div class="roi-toolbar">
  <button type="button" id="pan-mode">移动/缩放</button>
  <button type="button" id="draw-mode">框选区域</button>
  <button type="button" id="reset-view">重置视图</button>
  <span id="zoom-level">1.00×</span>
</div>
<div class="roi-help">
  手机：在移动/缩放模式下单指拖动、双指捏合 · 框选区域 模式下单指拖出矩形<br>
  桌面：滚轮缩放 · 移动/缩放模式或 Shift + 左键拖动可平移
</div>
<canvas id="roi-canvas"></canvas>
"""


ROI_VIEWER_CSS = """
:host {
  display: block;
  font-family: var(--st-font);
  color: var(--st-text-color);
}
.roi-toolbar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
}
.roi-toolbar button {
  border: 1px solid rgba(128, 128, 128, 0.45);
  border-radius: 0.35rem;
  background: var(--st-secondary-background-color);
  color: var(--st-text-color);
  padding: 0.35rem 0.75rem;
  cursor: pointer;
}
.roi-toolbar button.active {
  border-color: var(--st-primary-color);
  color: var(--st-primary-color);
  font-weight: 600;
}
#zoom-level {
  margin-left: auto;
  font-variant-numeric: tabular-nums;
}
.roi-help {
  color: rgba(128, 128, 128, 0.95);
  font-size: 0.85rem;
  margin-bottom: 0.35rem;
}
#roi-canvas {
  display: block;
  width: 100%;
  height: 640px;
  border: 1px solid rgba(128, 128, 128, 0.45);
  border-radius: 0.35rem;
  background: #111;
  touch-action: none;
  cursor: crosshair;
}
"""


ROI_VIEWER_JS = """
export default function(component) {
  const { parentElement, data, setStateValue } = component;
  const canvas = parentElement.querySelector("#roi-canvas");
  const context = canvas.getContext("2d");
  const drawButton = parentElement.querySelector("#draw-mode");
  const panButton = parentElement.querySelector("#pan-mode");
  const resetButton = parentElement.querySelector("#reset-view");
  const zoomLabel = parentElement.querySelector("#zoom-level");
  const image = new Image();

  let mode = "draw";
  let fitScale = 1;
  let scale = 1;
  let offsetX = 0;
  let offsetY = 0;
  let dragging = false;
  let panning = false;
  let interaction = null;
  let startScreen = null;
  let startImage = null;
  let startOffset = null;
  let activePointerId = null;
  const activePointers = new Map();
  let pinchStartDistance = 1;
  let pinchStartScale = 1;
  let pinchImagePoint = null;
  let roi = data.initialRoi || null;

  function setMode(nextMode) {
    mode = nextMode;
    drawButton.classList.toggle("active", mode === "draw");
    panButton.classList.toggle("active", mode === "pan");
    canvas.style.cursor = mode === "pan" ? "grab" : "crosshair";
  }

  function resizeCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      if (image.complete && image.naturalWidth) {
        resetView();
      }
    }
  }

  function viewportSize() {
    const rect = canvas.getBoundingClientRect();
    return { width: rect.width, height: rect.height };
  }

  function resetView() {
    const view = viewportSize();
    fitScale = Math.min(
      view.width / data.imageWidth,
      view.height / data.imageHeight
    );
    scale = fitScale;
    offsetX = (view.width - data.imageWidth * scale) / 2;
    offsetY = (view.height - data.imageHeight * scale) / 2;
    draw();
  }

  function screenPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  function screenToImage(point) {
    return {
      x: Math.max(0, Math.min(data.imageWidth, (point.x - offsetX) / scale)),
      y: Math.max(0, Math.min(data.imageHeight, (point.y - offsetY) / scale))
    };
  }

  function pointerPairGeometry() {
    const points = Array.from(activePointers.values()).slice(0, 2);
    const dx = points[1].x - points[0].x;
    const dy = points[1].y - points[0].y;
    return {
      midpoint: {
        x: (points[0].x + points[1].x) / 2,
        y: (points[0].y + points[1].y) / 2
      },
      distance: Math.max(1, Math.hypot(dx, dy))
    };
  }

  function beginPinch() {
    const geometry = pointerPairGeometry();
    pinchStartDistance = geometry.distance;
    pinchStartScale = scale;
    pinchImagePoint = {
      x: (geometry.midpoint.x - offsetX) / scale,
      y: (geometry.midpoint.y - offsetY) / scale
    };
    interaction = "pinch";
    dragging = true;
    panning = true;
    activePointerId = null;
  }

  function normalizeRoi(a, b) {
    const x1 = Math.max(0, Math.min(data.imageWidth, Math.min(a.x, b.x)));
    const y1 = Math.max(0, Math.min(data.imageHeight, Math.min(a.y, b.y)));
    const x2 = Math.max(0, Math.min(data.imageWidth, Math.max(a.x, b.x)));
    const y2 = Math.max(0, Math.min(data.imageHeight, Math.max(a.y, b.y)));
    const x = Math.round(x1);
    const y = Math.round(y1);
    return {
      x: x,
      y: y,
      w: Math.max(1, Math.round(x2) - x),
      h: Math.max(1, Math.round(y2) - y)
    };
  }

  function draw() {
    const view = viewportSize();
    context.save();
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.restore();

    context.save();
    context.translate(offsetX, offsetY);
    context.scale(scale, scale);
    context.drawImage(image, 0, 0, data.imageWidth, data.imageHeight);
    if (roi) {
      context.lineWidth = Math.max(2 / scale, 1);
      context.strokeStyle = data.boxColor || "#00e5ff";
      context.fillStyle = "rgba(0, 229, 255, 0.10)";
      context.fillRect(roi.x, roi.y, roi.w, roi.h);
      context.strokeRect(roi.x, roi.y, roi.w, roi.h);
    }
    context.restore();
    zoomLabel.textContent = (scale / fitScale).toFixed(2) + "×";
  }

  canvas.onwheel = (event) => {
    event.preventDefault();
    const point = screenPoint(event);
    const imagePoint = screenToImage(point);
    const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
    const nextScale = Math.max(
      fitScale * 0.35,
      Math.min(fitScale * 30, scale * factor)
    );
    offsetX = point.x - imagePoint.x * nextScale;
    offsetY = point.y - imagePoint.y * nextScale;
    scale = nextScale;
    draw();
  };

  canvas.onpointerdown = (event) => {
    if (event.pointerType === "mouse" && event.button !== 0 && event.button !== 1) {
      return;
    }
    if (activePointers.size >= 2) return;
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    const point = screenPoint(event);
    activePointers.set(event.pointerId, point);

    const wantsPan = mode === "pan" || event.shiftKey || event.button === 1;
    if (!wantsPan && (!event.isPrimary || activePointers.size > 1)) {
      activePointers.delete(event.pointerId);
      canvas.releasePointerCapture(event.pointerId);
      return;
    }

    if (wantsPan && activePointers.size === 2) {
      beginPinch();
      draw();
      return;
    }
    if (activePointers.size > 1) return;

    activePointerId = event.pointerId;
    panning = wantsPan;
    interaction = panning ? "pan" : "draw";
    dragging = true;
    startScreen = point;
    if (panning) {
      startOffset = { x: offsetX, y: offsetY };
      canvas.style.cursor = "grabbing";
    } else {
      startImage = screenToImage(point);
      roi = normalizeRoi(startImage, startImage);
    }
    draw();
  };

  canvas.onpointermove = (event) => {
    if (!activePointers.has(event.pointerId)) return;
    event.preventDefault();
    const point = screenPoint(event);
    activePointers.set(event.pointerId, point);

    if (interaction === "pinch" && activePointers.size >= 2) {
      const geometry = pointerPairGeometry();
      const nextScale = Math.max(
        fitScale * 0.35,
        Math.min(fitScale * 30, pinchStartScale * geometry.distance / pinchStartDistance)
      );
      offsetX = geometry.midpoint.x - pinchImagePoint.x * nextScale;
      offsetY = geometry.midpoint.y - pinchImagePoint.y * nextScale;
      scale = nextScale;
    } else if (interaction === "pan" && event.pointerId === activePointerId) {
      offsetX = startOffset.x + point.x - startScreen.x;
      offsetY = startOffset.y + point.y - startScreen.y;
    } else if (interaction === "draw" && event.pointerId === activePointerId) {
      roi = normalizeRoi(startImage, screenToImage(point));
    }
    draw();
  };

  function finishInteraction(event, commitRoi = true) {
    if (!activePointers.has(event.pointerId)) return;
    event.preventDefault();
    const finishingDraw = interaction === "draw" && event.pointerId === activePointerId;
    if (commitRoi && finishingDraw && roi && roi.w > 1 && roi.h > 1) {
      setStateValue("roi", roi);
    }
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    activePointers.delete(event.pointerId);

    if (interaction === "pinch" && activePointers.size === 1) {
      const remaining = activePointers.entries().next().value;
      activePointerId = remaining[0];
      startScreen = remaining[1];
      startOffset = { x: offsetX, y: offsetY };
      interaction = "pan";
      dragging = true;
      panning = true;
      canvas.style.cursor = "grabbing";
      draw();
      return;
    }

    if (event.pointerId === activePointerId || activePointers.size === 0) {
      activePointerId = null;
      interaction = null;
      dragging = false;
      panning = false;
    }
    canvas.style.cursor = mode === "pan" ? "grab" : "crosshair";
    draw();
  }

  canvas.onpointerup = (event) => finishInteraction(event, true);
  canvas.onpointercancel = (event) => finishInteraction(event, false);
  canvas.oncontextmenu = (event) => event.preventDefault();
  drawButton.onclick = () => setMode("draw");
  panButton.onclick = () => setMode("pan");
  resetButton.onclick = resetView;

  image.onload = () => {
    resizeCanvas();
    resetView();
    setMode("draw");
  };
  image.src = "data:image/jpeg;base64," + data.imageBase64;

  const observer = new ResizeObserver(() => resizeCanvas());
  observer.observe(canvas);

  return () => {
    observer.disconnect();
    canvas.onwheel = null;
    canvas.onpointerdown = null;
    canvas.onpointermove = null;
    canvas.onpointerup = null;
    canvas.onpointercancel = null;
  };
}
"""


LOCKED_ROI_VIEWER_HTML = """
<div class="roi-toolbar">
  <button type="button" id="draw-mode">开始框选区域</button>
  <button type="button" id="pan-mode">移动/缩放</button>
  <button type="button" id="lock-mode">锁定/滚动</button>
  <button type="button" id="reset-view">重置视图</button>
  <span id="zoom-level">1.00×</span>
</div>
<div class="roi-status" id="roi-status">请点击“开始框选区域”</div>
<div class="roi-help">
  框选时单指或鼠标拖出矩形，松手后自动保存并锁定；移动/缩放模式支持单指平移、双指缩放和鼠标滚轮缩放。
</div>
<canvas id="roi-canvas" aria-label="分析区域选择图像"></canvas>
"""


LOCKED_ROI_VIEWER_CSS = """
:host { display: block; font-family: var(--st-font); color: var(--st-text-color); }
.roi-toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; margin-bottom: .4rem; }
.roi-toolbar button {
  min-height: 2.45rem; border: 1px solid rgba(128,128,128,.45); border-radius: .4rem;
  background: var(--st-secondary-background-color); color: var(--st-text-color);
  padding: .4rem .72rem; cursor: pointer;
}
.roi-toolbar button.active { border-color: var(--st-primary-color); color: var(--st-primary-color); font-weight: 650; }
#zoom-level { margin-left: auto; font-variant-numeric: tabular-nums; }
.roi-status { font-weight: 650; margin-bottom: .25rem; }
.roi-status.confirmed { color: #18a558; }
.roi-help { color: rgba(128,128,128,.95); font-size: .88rem; line-height: 1.4; margin-bottom: .45rem; }
#roi-canvas {
  display: block; width: min(88vw, 100%); height: min(62vh, 640px); min-height: 360px;
  margin: 0 auto 1rem; border: 1px solid rgba(128,128,128,.45); border-radius: .45rem;
  background: #111; touch-action: auto; cursor: default;
}
@media (max-width: 640px) {
  #roi-canvas { width: 88vw; height: 52vh; min-height: 330px; }
  #zoom-level { width: 100%; margin-left: 0; }
}
"""


LOCKED_ROI_VIEWER_JS = """
export default function(component) {
  const { parentElement, data, setStateValue } = component;
  const canvas = parentElement.querySelector("#roi-canvas");
  const ctx = canvas.getContext("2d");
  const drawButton = parentElement.querySelector("#draw-mode");
  const panButton = parentElement.querySelector("#pan-mode");
  const lockButton = parentElement.querySelector("#lock-mode");
  const resetButton = parentElement.querySelector("#reset-view");
  const zoomLabel = parentElement.querySelector("#zoom-level");
  const statusLabel = parentElement.querySelector("#roi-status");
  const image = new Image();

  let mode = "locked";
  let fitScale = 1, scale = 1, offsetX = 0, offsetY = 0;
  let interaction = null, startScreen = null, startImage = null, startOffset = null;
  let activePointerId = null;
  const activePointers = new Map();
  let pinchStartDistance = 1, pinchStartScale = 1, pinchImagePoint = null;
  let roi = data.initialRoi || null;

  function updateControls() {
    drawButton.textContent = roi ? "重新框选区域" : "开始框选区域";
    drawButton.classList.toggle("active", mode === "draw");
    panButton.classList.toggle("active", mode === "pan");
    lockButton.classList.toggle("active", mode === "locked");
    canvas.style.touchAction = mode === "locked" ? "auto" : "none";
    canvas.style.cursor = mode === "pan" ? "grab" : mode === "draw" ? "crosshair" : "default";
    if (roi && mode !== "draw") {
      statusLabel.textContent = "分析区域已确认 ✓";
      statusLabel.classList.add("confirmed");
    } else if (mode === "draw") {
      statusLabel.textContent = "框选中：请拖出一个矩形";
      statusLabel.classList.remove("confirmed");
    } else {
      statusLabel.textContent = "请点击“开始框选区域”";
      statusLabel.classList.remove("confirmed");
    }
  }

  function setMode(nextMode) {
    mode = nextMode;
    activePointers.clear();
    interaction = null;
    activePointerId = null;
    updateControls();
  }

  function viewSize() {
    const rect = canvas.getBoundingClientRect();
    return { width: rect.width, height: rect.height };
  }

  function draw() {
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.restore();
    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);
    ctx.drawImage(image, 0, 0, data.imageWidth, data.imageHeight);
    if (roi) {
      ctx.lineWidth = Math.max(2 / scale, 1);
      ctx.strokeStyle = data.boxColor || "#00e5ff";
      ctx.fillStyle = "rgba(0,229,255,.10)";
      ctx.fillRect(roi.x, roi.y, roi.w, roi.h);
      ctx.strokeRect(roi.x, roi.y, roi.w, roi.h);
    }
    ctx.restore();
    zoomLabel.textContent = (scale / fitScale).toFixed(2) + "×";
  }

  function resetView() {
    const view = viewSize();
    fitScale = Math.min(view.width / data.imageWidth, view.height / data.imageHeight);
    scale = fitScale;
    offsetX = (view.width - data.imageWidth * scale) / 2;
    offsetY = (view.height - data.imageHeight * scale) / 2;
    draw();
  }

  function resizeCanvas() {
    const ratio = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width * ratio));
    const height = Math.max(1, Math.round(rect.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      if (image.complete && image.naturalWidth) resetView();
    }
  }

  function screenPoint(event) {
    const rect = canvas.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  function screenToImage(point) {
    return {
      x: Math.max(0, Math.min(data.imageWidth, (point.x - offsetX) / scale)),
      y: Math.max(0, Math.min(data.imageHeight, (point.y - offsetY) / scale))
    };
  }

  function normalizeRoi(a, b) {
    const x1 = Math.max(0, Math.min(data.imageWidth, Math.min(a.x, b.x)));
    const y1 = Math.max(0, Math.min(data.imageHeight, Math.min(a.y, b.y)));
    const x2 = Math.max(0, Math.min(data.imageWidth, Math.max(a.x, b.x)));
    const y2 = Math.max(0, Math.min(data.imageHeight, Math.max(a.y, b.y)));
    const x = Math.round(x1), y = Math.round(y1);
    return { x, y, w: Math.max(1, Math.round(x2) - x), h: Math.max(1, Math.round(y2) - y) };
  }

  function pointerPairGeometry() {
    const points = Array.from(activePointers.values()).slice(0, 2);
    const dx = points[1].x - points[0].x, dy = points[1].y - points[0].y;
    return {
      midpoint: { x: (points[0].x + points[1].x) / 2, y: (points[0].y + points[1].y) / 2 },
      distance: Math.max(1, Math.hypot(dx, dy))
    };
  }

  function beginPinch() {
    const geometry = pointerPairGeometry();
    pinchStartDistance = geometry.distance;
    pinchStartScale = scale;
    pinchImagePoint = {
      x: (geometry.midpoint.x - offsetX) / scale,
      y: (geometry.midpoint.y - offsetY) / scale
    };
    interaction = "pinch";
    activePointerId = null;
  }

  canvas.onwheel = (event) => {
    if (mode !== "pan") return;
    event.preventDefault();
    const point = screenPoint(event), imagePoint = screenToImage(point);
    const factor = event.deltaY < 0 ? 1.15 : 1 / 1.15;
    const nextScale = Math.max(fitScale * .35, Math.min(fitScale * 30, scale * factor));
    offsetX = point.x - imagePoint.x * nextScale;
    offsetY = point.y - imagePoint.y * nextScale;
    scale = nextScale;
    draw();
  };

  canvas.onpointerdown = (event) => {
    if (mode === "locked") return;
    if (event.pointerType === "mouse" && event.button !== 0 && event.button !== 1) return;
    if (activePointers.size >= 2) return;
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    const point = screenPoint(event);
    activePointers.set(event.pointerId, point);
    if (mode === "pan" && activePointers.size === 2) {
      const geometry = pointerPairGeometry();
      pinchStartDistance = geometry.distance;
      pinchStartScale = scale;
      pinchImagePoint = {
        x: (geometry.midpoint.x - offsetX) / scale,
        y: (geometry.midpoint.y - offsetY) / scale
      };
      interaction = "pinch";
      activePointerId = null;
      return;
    }
    if (activePointers.size > 1) return;
    activePointerId = event.pointerId;
    startScreen = point;
    if (mode === "pan") {
      interaction = "pan";
      startOffset = { x: offsetX, y: offsetY };
      canvas.style.cursor = "grabbing";
    } else {
      interaction = "draw";
      startImage = screenToImage(point);
      roi = normalizeRoi(startImage, startImage);
    }
    draw();
  };

  canvas.onpointermove = (event) => {
    if (!activePointers.has(event.pointerId)) return;
    event.preventDefault();
    const point = screenPoint(event);
    activePointers.set(event.pointerId, point);
    if (interaction === "pinch" && activePointers.size >= 2) {
      const geometry = pointerPairGeometry();
      const nextScale = Math.max(
        fitScale * .35,
        Math.min(fitScale * 30, pinchStartScale * geometry.distance / pinchStartDistance)
      );
      offsetX = geometry.midpoint.x - pinchImagePoint.x * nextScale;
      offsetY = geometry.midpoint.y - pinchImagePoint.y * nextScale;
      scale = nextScale;
    } else if (interaction === "pan" && event.pointerId === activePointerId) {
      offsetX = startOffset.x + point.x - startScreen.x;
      offsetY = startOffset.y + point.y - startScreen.y;
    } else if (interaction === "draw" && event.pointerId === activePointerId) {
      roi = normalizeRoi(startImage, screenToImage(point));
    }
    draw();
  };

  function finishInteraction(event, commitRoi) {
    if (!activePointers.has(event.pointerId)) return;
    event.preventDefault();
    const finishingDraw = interaction === "draw" && event.pointerId === activePointerId;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    activePointers.delete(event.pointerId);
    if (commitRoi && finishingDraw && roi && roi.w > 1 && roi.h > 1) {
      setStateValue("roi", roi);
      setMode("locked");
      draw();
      return;
    }
    if (interaction === "pinch" && activePointers.size === 1) {
      const remaining = activePointers.entries().next().value;
      activePointerId = remaining[0];
      startScreen = remaining[1];
      startOffset = { x: offsetX, y: offsetY };
      interaction = "pan";
      canvas.style.cursor = "grabbing";
      return;
    }
    if (event.pointerId === activePointerId || activePointers.size === 0) {
      activePointerId = null;
      interaction = null;
    }
    updateControls();
    draw();
  }

  canvas.onpointerup = (event) => finishInteraction(event, true);
  canvas.onpointercancel = (event) => finishInteraction(event, false);
  canvas.oncontextmenu = (event) => { if (mode !== "locked") event.preventDefault(); };
  drawButton.onclick = () => setMode("draw");
  panButton.onclick = () => setMode("pan");
  lockButton.onclick = () => setMode("locked");
  resetButton.onclick = resetView;

  image.onload = () => { resizeCanvas(); resetView(); setMode("locked"); };
  image.src = "data:image/jpeg;base64," + data.imageBase64;
  const observer = new ResizeObserver(() => resizeCanvas());
  observer.observe(canvas);
  updateControls();

  return () => {
    observer.disconnect();
    canvas.onwheel = null;
    canvas.onpointerdown = null;
    canvas.onpointermove = null;
    canvas.onpointerup = null;
    canvas.onpointercancel = null;
  };
}
"""


roi_viewer_component = st.components.v2.component(
    "zoomable_dng_roi_viewer",
    html=LOCKED_ROI_VIEWER_HTML,
    css=LOCKED_ROI_VIEWER_CSS,
    js=LOCKED_ROI_VIEWER_JS,
)


def component_box_to_roi(box: dict[str, Any]) -> tuple[int, int, int, int]:
    """Convert the browser component ROI to an original-image integer ROI."""
    return (
        int(box["x"]),
        int(box["y"]),
        int(box["w"]),
        int(box["h"]),
    )


def roi_preview(
    image_bgr: np.ndarray,
    roi_left: tuple[int, int, int, int],
    roi_right: tuple[int, int, int, int],
) -> np.ndarray:
    """Draw confirmed ROI rectangles without changing the source image."""
    preview = image_bgr.copy()
    for roi, color in ((roi_left, (255, 255, 0)), (roi_right, (0, 202, 255))):
        x, y, w, h = roi
        cv2.rectangle(preview, (x, y), (x + w, y + h), color, 8)
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def roi_in_bounds(
    roi: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> bool:
    x, y, w, h = roi
    height, width = image_shape[:2]
    return x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= width and y + h <= height


def _legacy_main() -> None:
    st.set_page_config(
        page_title="智能手机荧光图像银离子定量分析",
        layout="wide",
    )
    st.title("智能手机荧光图像银离子定量分析")
    st.caption("模型定量范围：2–12 μM")

    st.subheader("第 1 步：上传 DNG 图像")
    upload_left, upload_right = st.columns(2)
    with upload_left:
        left_file = st.file_uploader(
            "第一张图：左侧空白 | 右侧样品",
            type=["dng"],
            key="left_file",
        )
    with upload_right:
        right_file = st.file_uploader(
            "第二张图：左侧样品 | 右侧空白",
            type=["dng"],
            key="right_file",
        )

    if left_file is None or right_file is None:
        st.info("请上传第一张和第二张 DNG 图像以继续。")
        return

    left_bytes = left_file.getvalue()
    right_bytes = right_file.getvalue()
    fingerprint = hashlib.sha256(left_bytes + right_bytes).hexdigest()
    if st.session_state.get("pair_fingerprint") != fingerprint:
        st.session_state.pair_fingerprint = fingerprint
        st.session_state.pop("roi_left", None)
        st.session_state.pop("roi_right", None)
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_rois", None)

    try:
        with st.spinner("正在按固定参数读取 DNG 图像…"):
            img_left = read_dng_bytes(left_bytes)
            img_right = read_dng_bytes(right_bytes)
    except Exception as exc:
        st.error(f"DNG 读取失败：{exc}")
        return

    if img_left.shape != img_right.shape:
        st.error(
            "两张图像尺寸不同，无法应用相同坐标："
            f"第一张={img_left.shape[1]}×{img_left.shape[0]}，"
            f"第二张={img_right.shape[1]}×{img_right.shape[0]}"
        )
        return

    preview_left, preview_right = st.columns(2)
    with preview_left:
        st.image(
            cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB),
            caption=f"第一张：左侧空白 / 右侧样品（{left_file.name}）",
            use_container_width=True,
        )
    with preview_right:
        st.image(
            cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB),
            caption=f"第二张：左侧样品 / 右侧空白（{right_file.name}）",
            use_container_width=True,
        )

    st.subheader("第 2 步：选择分析区域")
    st.write(
        "仅在 第一张图上依次选择两个普通矩形框。拖动矩形内部可移动，"
        "拖动角点可调整大小；确认后的原图坐标会直接应用到 第二张 DNG 图像。"
    )
    image_height, image_width = img_left.shape[:2]
    display_image_b64 = roi_display_image_b64(img_left)

    st.markdown("#### 第 1 步：左侧液体中央 左侧分析区域")
    left_component_key = f"zoom_roi_left_{fingerprint}"
    left_component_state = st.session_state.get(left_component_key)
    left_initial_roi = (
        getattr(left_component_state, "roi", None)
        if left_component_state is not None
        else None
    )
    left_result = roi_viewer_component(
        data={
            "imageBase64": display_image_b64,
            "imageWidth": image_width,
            "imageHeight": image_height,
            "boxColor": "#00e5ff",
            "initialRoi": left_initial_roi,
        },
        default={"roi": None},
        on_roi_change=lambda: None,
        key=left_component_key,
        width="stretch",
        height=710,
    )
    current_left = (
        component_box_to_roi(left_result.roi)
        if left_result.roi is not None
        else None
    )
    st.code(f"当前 左侧分析区域 = {current_left}")
    if st.button("确认 左侧分析区域", type="primary", disabled=current_left is None):
        st.session_state.roi_left = current_left
        st.session_state.pop("roi_right", None)
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_rois", None)

    roi_left = st.session_state.get("roi_left")
    if roi_left is None:
        st.info("请调整左侧矩形框，然后点击“确认 左侧分析区域”。")
        return
    st.success(f"已确认 左侧分析区域 = {roi_left}")

    st.markdown("#### 第 2 步：右侧液体中央 右侧分析区域")
    right_component_key = f"zoom_roi_right_{fingerprint}"
    right_component_state = st.session_state.get(right_component_key)
    right_initial_roi = (
        getattr(right_component_state, "roi", None)
        if right_component_state is not None
        else None
    )
    right_result = roi_viewer_component(
        data={
            "imageBase64": display_image_b64,
            "imageWidth": image_width,
            "imageHeight": image_height,
            "boxColor": "#ffca28",
            "initialRoi": right_initial_roi,
        },
        default={"roi": None},
        on_roi_change=lambda: None,
        key=right_component_key,
        width="stretch",
        height=710,
    )
    current_right = (
        component_box_to_roi(right_result.roi)
        if right_result.roi is not None
        else None
    )
    st.code(f"当前 右侧分析区域 = {current_right}")
    if st.button("确认 右侧分析区域", type="primary", disabled=current_right is None):
        st.session_state.roi_right = current_right
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_rois", None)

    roi_right = st.session_state.get("roi_right")
    if roi_right is None:
        st.info("请调整右侧矩形框，然后点击“确认 右侧分析区域”。")
        return
    st.success(f"已确认 右侧分析区域 = {roi_right}")
    if not roi_in_bounds(roi_left, img_left.shape) or not roi_in_bounds(roi_right, img_left.shape):
        st.error("分析区域超出图像边界，请重新框选。")
        return

    st.markdown("#### 分析区域确认预览")
    st.image(
        roi_preview(img_left, roi_left, roi_right),
        caption=(
            f"第一张 DNG 图像：青色 左侧分析区域={roi_left}；黄色 右侧分析区域={roi_right}。"
            "同一组坐标将应用到 第二张 DNG 图像。"
        ),
        use_container_width=True,
    )
    crop_left, crop_right = st.columns(2)
    with crop_left:
        x, y, w, h = roi_left
        st.image(
            cv2.cvtColor(img_left[y : y + h, x : x + w], cv2.COLOR_BGR2RGB),
            caption=f"左侧分析区域 预览：{roi_left}",
            use_container_width=True,
        )
    with crop_right:
        x, y, w, h = roi_right
        st.image(
            cv2.cvtColor(img_left[y : y + h, x : x + w], cv2.COLOR_BGR2RGB),
            caption=f"右侧分析区域 预览：{roi_right}",
            use_container_width=True,
        )

    st.subheader("第 3 步：预测银离子浓度")
    if st.button(
        "预测银离子浓度",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.prediction_result = predict_pair(
            img_left,
            img_right,
            roi_left,
            roi_right,
        )
        st.session_state.prediction_rois = (roi_left, roi_right)

    result = st.session_state.get("prediction_result")
    if result is None:
        st.info("请确认两个分析区域，然后点击“预测银离子浓度”。")
        return

    result_roi_left, result_roi_right = st.session_state["prediction_rois"]
    roi_left, roi_right = result_roi_left, result_roi_right
    st.metric(
        "银离子预测浓度",
        f"{result['predicted_Ag_uM']:.2f} μM",
    )
    if result["range_status"] == "IN_RANGE":
        st.success("处于模型定量范围内（2–12 μM）")
    elif result["range_status"] == "OUT_OF_MODEL_RANGE":
        st.warning("超出模型定量范围")
    else:
        st.error("亮度比值无效，无法预测浓度。")

    with st.expander("计算详情", expanded=False):
        st.write(
            f"模型：浓度 = {MODEL_A} × 校正亮度比值 + {MODEL_B}  "
            f"（模型定量范围： {VALID_MIN:g}–{VALID_MAX:g} μM）"
        )
        st.caption(
            "DNG 处理：使用相机白平衡，"
            "关闭自动亮度，输出位深为 8 位"
        )
        ratio_rows = [
            {"指标": "第一张图亮度比值", "数值": result["Y_ratio_left"]},
            {"指标": "第二张图亮度比值", "数值": result["Y_ratio_right"]},
            {"指标": "校正亮度比值", "数值": result["Y_ratio_corr"]},
            {
                "指标": "左右互换差异（%）",
                "数值": result["position_swap_difference_percent"],
            },
        ]
        st.dataframe(ratio_rows, use_container_width=True, hide_index=True)

        detail_rows = []
        for prefix, label in [
            ("left_blank", "第一张图 / 空白 / 左侧分析区域"),
            ("left_sample", "第一张图 / 样品 / 右侧分析区域"),
            ("right_sample", "第二张图 / 样品 / 左侧分析区域"),
            ("right_blank", "第二张图 / 空白 / 右侧分析区域"),
        ]:
            detail_rows.append(
                {
                    "区域": label,
                    "红色通道": result[f"{prefix}_R"],
                    "绿色通道": result[f"{prefix}_G"],
                    "蓝色通道": result[f"{prefix}_B"],
                    "亮度": result[f"{prefix}_Y"],
                }
            )
        st.dataframe(detail_rows, use_container_width=True, hide_index=True)

    csv_row = {
        "left_file": left_file.name,
        "right_file": right_file.name,
        **result,
        "左侧分析区域_x": roi_left[0],
        "左侧分析区域_y": roi_left[1],
        "左侧分析区域_w": roi_left[2],
        "左侧分析区域_h": roi_left[3],
        "右侧分析区域_x": roi_right[0],
        "右侧分析区域_y": roi_right[1],
        "右侧分析区域_w": roi_right[2],
        "右侧分析区域_h": roi_right[3],
        "model_A": MODEL_A,
        "model_B": MODEL_B,
        "model_valid_min_uM": VALID_MIN,
        "model_valid_max_uM": VALID_MAX,
    }
    csv_row = chinese_export_row(csv_row)
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(csv_row))
    writer.writeheader()
    writer.writerow(csv_row)
    st.download_button(
        "下载结果表格（CSV）",
        data=csv_buffer.getvalue().encode("utf-8-sig"),
        file_name="银离子预测结果.csv",
        mime="text/csv",
    )

    if st.button("开始新的预测", use_container_width=True):
        prediction_state_keys = {
            "left_file",
            "right_file",
            "pair_fingerprint",
            "roi_left",
            "roi_right",
            "prediction_result",
            "prediction_rois",
        }
        for state_key in list(st.session_state):
            if state_key in prediction_state_keys or state_key.startswith("zoom_roi_"):
                del st.session_state[state_key]
        st.rerun()


def chinese_export_row(row: dict[str, Any]) -> dict[str, Any]:
    """将下载表格的字段和状态转换为中文。"""
    labels = {
        "mode": "输入方式", "left_file": "第一张图文件名", "right_file": "第二张图文件名",
        "predicted_Ag_uM": "银离子预测浓度（微摩尔每升）", "range_status": "定量范围状态",
        "Y_ratio_left": "第一张图亮度比值", "Y_ratio_right": "第二张图亮度比值",
        "Y_ratio_corr": "校正亮度比值", "position_swap_difference_percent": "左右互换差异（%）",
        "model_A": "模型斜率", "model_B": "模型截距",
        "model_valid_min_uM": "模型定量下限（微摩尔每升）", "model_valid_max_uM": "模型定量上限（微摩尔每升）",
    }
    for prefix, region in [("left_blank", "第一张图空白"), ("left_sample", "第一张图样品"),
                           ("right_sample", "第二张图样品"), ("right_blank", "第二张图空白")]:
        for channel, name in [("R", "红色通道"), ("G", "绿色通道"), ("B", "蓝色通道"), ("Y", "亮度")]:
            labels[f"{prefix}_{channel}"] = region + name
    for region in ("左侧分析区域", "右侧分析区域"):
        for axis, name in [("x", "横坐标"), ("y", "纵坐标"), ("w", "宽度"), ("h", "高度")]:
            labels[f"{region}_{axis}"] = region + name
    states = {"IN_RANGE": "处于模型定量范围内", "OUT_OF_MODEL_RANGE": "超出模型定量范围", "INVALID_RATIO": "亮度比值无效"}
    return {labels.get(key, key): states.get(value, value) if isinstance(value, str) else value
            for key, value in row.items()}


def main() -> None:
    st.set_page_config(
        page_title="智能手机荧光图像银离子定量分析",
        layout="wide",
    )
    st.title("智能手机荧光图像银离子定量分析")
    st.caption("模型定量范围：2–12 μM")

    mode = st.radio(
        "输入方式",
        ["上传 DNG 图像", "手机拍照"],
        horizontal=True,
        key="input_mode",
    )

    st.subheader("第 1 步：拍摄或上传两张图像")
    if mode == "上传 DNG 图像":
        upload_left, upload_right = st.columns(2)
        with upload_left:
            left_file = st.file_uploader(
                "第一张图：左侧空白 | 右侧样品",
                type=["dng"],
                key="left_file",
            )
        with upload_right:
            right_file = st.file_uploader(
                "第二张图：左侧样品 | 右侧空白",
                type=["dng"],
                key="right_file",
            )
        if left_file is None or right_file is None:
            st.info("请上传两张 DNG 图像以继续。")
            return
        decoder = read_dng_bytes
        decode_label = "正在按固定参数读取 DNG 图像…"
        left_name = left_file.name
        right_name = right_file.name
    else:
        left_file = st.camera_input(
            "拍摄第一张图：左侧空白 | 右侧样品",
            key="camera_left_file",
        )
        if left_file is None:
            st.info("请先用手机相机拍摄第一张图：左侧空白 | 右侧样品。")
            return
        right_file = st.camera_input(
            "拍摄第二张图：左侧样品 | 右侧空白",
            key="camera_right_file",
        )
        if right_file is None:
            st.info("第一张图已完成，请继续拍摄第二张图：左侧样品 | 右侧空白。")
            return
        decoder = read_camera_image_bytes
        decode_label = "正在读取浏览器相机图片…"
        left_name = getattr(left_file, "name", "camera_first.jpg")
        right_name = getattr(right_file, "name", "camera_second.jpg")

    left_bytes = left_file.getvalue()
    right_bytes = right_file.getvalue()
    fingerprint = hashlib.sha256(
        mode.encode("utf-8") + b"\0" + left_bytes + b"\0" + right_bytes
    ).hexdigest()
    if st.session_state.get("pair_fingerprint") != fingerprint:
        st.session_state.pair_fingerprint = fingerprint
        for state_key in ("roi_left", "roi_right", "prediction_result", "prediction_rois"):
            st.session_state.pop(state_key, None)

    try:
        with st.spinner(decode_label):
            img_left = decoder(left_bytes)
            img_right = decoder(right_bytes)
    except Exception as exc:
        st.error(f"图片读取失败：{exc}")
        return

    if img_left.shape != img_right.shape:
        st.error(
            "两张图片尺寸不同，无法应用相同原图坐标："
            f"第一张={img_left.shape[1]}×{img_left.shape[0]}，"
            f"第二张={img_right.shape[1]}×{img_right.shape[0]}"
        )
        return

    preview_left, preview_right = st.columns(2)
    with preview_left:
        st.image(
            cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB),
            caption=f"第一张：左侧空白 | 右侧样品（{left_name}）",
            use_container_width=True,
        )
    with preview_right:
        st.image(
            cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB),
            caption=f"第二张：左侧样品 | 右侧空白（{right_name}）",
            use_container_width=True,
        )

    st.subheader("第 2 步：在第一张图像上选择分析区域")
    st.write(
        "左侧分析区域 和 右侧分析区域 都只在第一张图上选择；保存的是原始图像像素坐标，"
        "并将同一组坐标应用到第二张图。缩放和平移只影响显示。"
    )
    image_height, image_width = img_left.shape[:2]
    display_image_b64 = roi_display_image_b64(img_left)

    st.markdown("#### 1. 左侧分析区域（第一张图左侧空白区域）")
    left_component_key = f"locked_zoom_roi_left_{fingerprint}"
    left_result = roi_viewer_component(
        data={
            "imageBase64": display_image_b64,
            "imageWidth": image_width,
            "imageHeight": image_height,
            "boxColor": "#00e5ff",
            "initialRoi": st.session_state.get("roi_left"),
        },
        default={"roi": None},
        on_roi_change=lambda: None,
        key=left_component_key,
        width="stretch",
        height=730,
    )
    left_box = getattr(left_result, "roi", None)
    if left_box is not None:
        next_roi_left = component_box_to_roi(left_box)
        if st.session_state.get("roi_left") != next_roi_left:
            st.session_state.roi_left = next_roi_left
            st.session_state.pop("prediction_result", None)
            st.session_state.pop("prediction_rois", None)
    roi_left = st.session_state.get("roi_left")
    if roi_left is None:
        st.info("点击“开始框选区域”，拖出矩形；松手后会自动确认并锁定。")
        return
    st.success(f"分析区域已确认 ✓  左侧分析区域 = {roi_left}")

    st.markdown("#### 2. 右侧分析区域（第一张图右侧样品区域）")
    right_component_key = f"locked_zoom_roi_right_{fingerprint}"
    right_result = roi_viewer_component(
        data={
            "imageBase64": display_image_b64,
            "imageWidth": image_width,
            "imageHeight": image_height,
            "boxColor": "#ffca28",
            "initialRoi": st.session_state.get("roi_right"),
        },
        default={"roi": None},
        on_roi_change=lambda: None,
        key=right_component_key,
        width="stretch",
        height=730,
    )
    right_box = getattr(right_result, "roi", None)
    if right_box is not None:
        next_roi_right = component_box_to_roi(right_box)
        if st.session_state.get("roi_right") != next_roi_right:
            st.session_state.roi_right = next_roi_right
            st.session_state.pop("prediction_result", None)
            st.session_state.pop("prediction_rois", None)
    roi_right = st.session_state.get("roi_right")
    if roi_right is None:
        st.info("点击“开始框选区域”，拖出矩形；松手后会自动确认并锁定。")
        return
    st.success(f"分析区域已确认 ✓  右侧分析区域 = {roi_right}")

    if not roi_in_bounds(roi_left, img_left.shape) or not roi_in_bounds(roi_right, img_left.shape):
        st.error("分析区域超出图像边界，请点击“重新框选区域”。")
        return

    st.markdown("#### 分析区域确认预览")
    st.image(
        roi_preview(img_left, roi_left, roi_right),
        caption=(
            f"第一张图：青色 左侧分析区域={roi_left}；黄色 右侧分析区域={roi_right}。"
            "相同原图坐标将应用到第二张图。"
        ),
        use_container_width=True,
    )
    crop_left, crop_right = st.columns(2)
    with crop_left:
        x, y, w, h = roi_left
        st.image(
            cv2.cvtColor(img_left[y:y + h, x:x + w], cv2.COLOR_BGR2RGB),
            caption=f"左侧分析区域预览： {roi_left}",
            use_container_width=True,
        )
    with crop_right:
        x, y, w, h = roi_right
        st.image(
            cv2.cvtColor(img_left[y:y + h, x:x + w], cv2.COLOR_BGR2RGB),
            caption=f"右侧分析区域预览： {roi_right}",
            use_container_width=True,
        )

    st.subheader("第 3 步：预测银离子浓度")
    if st.button("预测银离子浓度", type="primary", use_container_width=True):
        st.session_state.prediction_result = predict_pair(
            img_left,
            img_right,
            roi_left,
            roi_right,
        )
        st.session_state.prediction_rois = (roi_left, roi_right)

    result = st.session_state.get("prediction_result")
    if result is None:
        st.info("两个分析区域已锁定，请点击“预测银离子浓度”继续。")
        return

    result_roi_left, result_roi_right = st.session_state["prediction_rois"]
    roi_left, roi_right = result_roi_left, result_roi_right
    st.metric("银离子预测浓度", f"{result['predicted_Ag_uM']:.2f} μM")
    if result["range_status"] == "IN_RANGE":
        st.success("处于模型定量范围内（2–12 μM）")
    elif result["range_status"] == "OUT_OF_MODEL_RANGE":
        st.warning("超出模型定量范围")
    else:
        st.error("亮度比值无效，无法预测浓度。")

    with st.expander("计算详情", expanded=False):
        st.write(
            f"模型：浓度 = {MODEL_A} × 校正亮度比值 + {MODEL_B} "
            f"（模型定量范围： {VALID_MIN:g}–{VALID_MAX:g} μM）"
        )
        if mode == "上传 DNG 图像":
            st.caption(
                "DNG 处理：使用相机白平衡，"
                "关闭自动亮度，输出位深为 8 位"
            )
        else:
            st.caption("拍照图像处理：解码浏览器照片并读取颜色通道。")
        ratio_rows = [
            {"指标": "第一张图亮度比值", "数值": result["Y_ratio_left"]},
            {"指标": "第二张图亮度比值", "数值": result["Y_ratio_right"]},
            {"指标": "校正亮度比值", "数值": result["Y_ratio_corr"]},
            {
                "指标": "左右互换差异（%）",
                "数值": result["position_swap_difference_percent"],
            },
        ]
        st.dataframe(ratio_rows, use_container_width=True, hide_index=True)

        detail_rows = []
        for prefix, label in [
            ("left_blank", "第一张图 / 空白 / 左侧分析区域"),
            ("left_sample", "第一张图 / 样品 / 右侧分析区域"),
            ("right_sample", "第二张图 / 样品 / 左侧分析区域"),
            ("right_blank", "第二张图 / 空白 / 右侧分析区域"),
        ]:
            detail_rows.append(
                {
                    "区域": label,
                    "红色通道": result[f"{prefix}_R"],
                    "绿色通道": result[f"{prefix}_G"],
                    "蓝色通道": result[f"{prefix}_B"],
                    "亮度": result[f"{prefix}_Y"],
                }
            )
        st.dataframe(detail_rows, use_container_width=True, hide_index=True)

    csv_row = {
        "mode": mode,
        "left_file": left_name,
        "right_file": right_name,
        **result,
        "左侧分析区域_x": roi_left[0],
        "左侧分析区域_y": roi_left[1],
        "左侧分析区域_w": roi_left[2],
        "左侧分析区域_h": roi_left[3],
        "右侧分析区域_x": roi_right[0],
        "右侧分析区域_y": roi_right[1],
        "右侧分析区域_w": roi_right[2],
        "右侧分析区域_h": roi_right[3],
        "model_A": MODEL_A,
        "model_B": MODEL_B,
        "model_valid_min_uM": VALID_MIN,
        "model_valid_max_uM": VALID_MAX,
    }
    csv_row = chinese_export_row(csv_row)
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(csv_row))
    writer.writeheader()
    writer.writerow(csv_row)
    st.download_button(
        "下载结果表格（CSV）",
        data=csv_buffer.getvalue().encode("utf-8-sig"),
        file_name="银离子预测结果.csv",
        mime="text/csv",
    )

    if st.button("开始新的预测", use_container_width=True):
        prediction_state_keys = {
            "left_file",
            "right_file",
            "camera_left_file",
            "camera_right_file",
            "pair_fingerprint",
            "roi_left",
            "roi_right",
            "prediction_result",
            "prediction_rois",
        }
        for state_key in list(st.session_state):
            if state_key in prediction_state_keys or state_key.startswith("locked_zoom_roi_"):
                del st.session_state[state_key]
        st.rerun()


if __name__ == "__main__":
    main()
