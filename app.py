"""
NephroScan v10 — Landing Page + Hue-Shift Button + Refined Analyzer
Run: python app.py  →  http://127.0.0.1:5000
"""

import base64, os, traceback
import cv2, numpy as np
from flask import Flask, jsonify, redirect, render_template_string, request

app = Flask(__name__)

# HSV ranges (OpenCV scale: H 0-179, S 0-255, V 0-255)
# Each entry: (display_name, risk_level, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, hex_color)
COLOR_PROFILES = [
    # Orange / Yellow  →  Low creatinine (dilute urine)
    ("Orange / Yellow",  "Low Creatinine",  8,  40, 100, 255, 100, 255, "#F4A020"),
    # Green            →  Moderate / Normal creatinine
    ("Green",            "Normal",          40,  90,  60, 255,  60, 255, "#3CB371"),
    # Blue / Dark Blue →  High / Concentrated creatinine
    ("Blue / Dark Blue", "High Creatinine", 90, 130,  60, 255,  30, 255, "#1A5DAB"),
]

RISK_META = {
    "Low Creatinine":  {"icon": "⚠", "css": "moderate", "description": "Orange/Yellow detected — Low creatinine level. Urine may be dilute. Ensure adequate hydration and consult a healthcare provider if persistent."},
    "Normal":          {"icon": "✓", "css": "normal",   "description": "Green detected — Moderate/Normal creatinine level. Urine concentration is within the expected healthy range."},
    "High Creatinine": {"icon": "ℹ", "css": "high",     "description": "Blue/Dark Blue detected — High or concentrated creatinine level. May indicate dehydration or kidney stress. Consult a healthcare provider."},
    "Unrecognized":    {"icon": "?", "css": "moderate", "description": "Color did not match Orange/Yellow, Green, or Blue/Dark Blue. Please retake the photo under better lighting."},
}

RECOMMENDATIONS = {
    "Low Creatinine":  "Increase fluid intake gradually. Dilute urine may indicate over-hydration or impaired kidney concentration. Consult a doctor if this persists for more than 48 hours.",
    "Normal":          "No action required. Creatinine levels appear moderate and normal. Maintain adequate hydration and continue routine health monitoring.",
    "High Creatinine": "Increase water intake. High creatinine concentration may indicate dehydration or reduced kidney function. Seek medical evaluation if symptoms persist.",
    "Unrecognized":    "Unable to classify the strip color. Retake the photo in good lighting against a plain white background and ensure the strip is fully visible.",
}

def extract_dominant_hsv(bgr):
    """
    Detect the dipstick region by isolating saturated (coloured) pixels,
    then return the dominant HSV cluster.  Returns None when the image
    contains insufficient coloured content (no strip detected).
    """
    h, w = bgr.shape[:2]
    # Use centre 60 % of the image to reduce border / hand interference
    cy, cx = h // 2, w // 2
    my, mx = int(h * 0.30), int(w * 0.30)
    roi = bgr[cy - my:cy + my, cx - mx:cx + mx]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ── Mask: keep only pixels that look like strip colours ──────────────
    # Exclude near-white background  (low S), blown-out whites (V≥245),
    # and very dark shadows (V≤25).
    sat   = hsv[:, :, 1]
    val   = hsv[:, :, 2]
    mask  = (sat >= 45) & (val >= 30) & (val <= 245)

    colored = hsv[mask].astype(np.float32)

    # Need at least 1 % of ROI pixels to be coloured – otherwise no strip
    min_px = max(60, int(roi.shape[0] * roi.shape[1] * 0.01))
    if len(colored) < min_px:
        return None          # signal: strip not detected

    k = min(3, len(colored))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        colored, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten())
    return tuple(centers[counts.argmax()].astype(int))

def classify_color(h, s, v):
    for name, risk, h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, hex_col in COLOR_PROFILES:
        if h_lo <= h <= h_hi and s_lo <= s <= s_hi and v_lo <= v <= v_hi:
            return {"color_name": name, "risk": risk, "hex": hex_col}
    # Fallback: nearest hue match among the three target colors
    if s < 50:
        return {"color_name": "Unrecognized", "risk": "Unrecognized", "hex": "#AAAAAA"}
    if h < 45 or h > 155:
        return {"color_name": "Orange / Yellow", "risk": "Low Creatinine", "hex": "#F4A020"}
    if 45 <= h < 90:
        return {"color_name": "Green", "risk": "Normal", "hex": "#3CB371"}
    return {"color_name": "Blue / Dark Blue", "risk": "High Creatinine", "hex": "#1A5DAB"}

def hsv_to_rgb(h, s, v):
    px = np.uint8([[[h,s,v]]])
    rgb = cv2.cvtColor(px, cv2.COLOR_HSV2RGB)[0][0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])

def image_to_b64(bgr):
    _, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()

def annotate_image(bgr, h, s, v):
    out = bgr.copy()
    ih, iw = out.shape[:2]
    cy, cx = ih//2, iw//2
    my, mx = ih//4, iw//4
    cv2.rectangle(out, (cx-mx, cy-my), (cx+mx, cy+my), (74,124,89), 2)
    r,g,b = hsv_to_rgb(h,s,v)
    sw = max(40, iw//8); pad = 10
    cv2.rectangle(out, (iw-sw-pad, ih-sw-pad), (iw-pad, ih-pad), (b,g,r), -1)
    cv2.rectangle(out, (iw-sw-pad, ih-sw-pad), (iw-pad, ih-pad), (255,255,255), 1)
    return out

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NephroScan — Clinical Strip Analyzer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Markazi+Text:wght@400..700&family=Space+Grotesk:wght@300..700&display=swap" rel="stylesheet">

<style>
/* ══ Reset ══════════════════════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }

/* ══ Tokens ═════════════════════════════════════════════════════════════════ */
:root {
  --g-950: #060e20;
  --g-900: #0b1735;
  --g-800: #122050;
  --g-700: #1a2e6e;
  --g-600: #1f3a8a;
  --g-500: #2550b8;
  --g-400: #4472d6;
  --g-300: #7aa1ec;
  --g-200: #b2c8f6;
  --g-100: #d8e5fc;
  --g-50:  #eef3fd;

  --bg:      #eef3fb;
  --bg2:     #e4ecf7;
  --bg3:     #d8e3f2;
  --surface: #f6f9ff;

  --ink:  #0d1526;
  --ink2: #243050;
  --ink3: #4a5a80;
  --ink4: #7a8aaa;

  --sep:  #c8d5ee;
  --sep2: #b8c8e6;

  --ok:       #166534; --ok-bg:     #dcfce7; --ok-sep:  #86efac;
  --warn:     #92400e; --warn-bg:   #fef3c7; --warn-sep:#fcd34d;
  --danger:   #991b1b; --danger-bg: #fee2e2; --danger-sep:#fca5a5;

  --r: 3px;
}

/* ══ Shared base ════════════════════════════════════════════════════════════ */
body {
  font-family: 'Inter', sans-serif;
  -webkit-font-smoothing: antialiased;
  background: var(--bg);
  color: var(--ink);
  min-height: 100vh;
  font-size: 14px;
  line-height: 1.6;
  overflow-x: hidden;
}

/* ══════════════════════════════════════════════════════════════════════════
   LANDING PAGE
══════════════════════════════════════════════════════════════════════════ */
#landing-page {
  position: relative; z-index: 1000;
  background: var(--g-950);
  display: flex; flex-direction: column;
  min-height: 100vh;
  transition: opacity .7s ease, visibility .7s ease;
}

#landing-page.hidden-page {
  display: none;
}

/* ── SVG background canvas ─────────────────────────────────────────────── */
#bg-canvas {
  position: absolute; top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none; z-index: 0;
}

/* ── Noise overlay ─────────────────────────────────────────────────────── */
#landing-page::before {
  content: '';
  position: absolute; inset: 0; z-index: 1;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
  background-repeat: repeat;
  background-size: 256px 256px;
  opacity: .6; pointer-events: none;
}

/* ── Top bar ───────────────────────────────────────────────────────────── */
.lp-topbar {
  position: relative; z-index: 10;
  display: flex; align-items: center; justify-content: space-between;
  padding: 1.8rem 3rem;
  border-bottom: 1px solid rgba(255,255,255,.05);
}
.lp-logo {
  display: flex; align-items: center; gap: .85rem;
}
.lp-logo-mark {
  width: 34px; height: 34px; border-radius: 2px;
  background: linear-gradient(135deg, var(--g-500), var(--g-300));
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem;
}
.lp-logo-name {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.4rem; font-weight: 600; color: #fff;
  letter-spacing: .02em;
}
.lp-version {
  font-family: 'JetBrains Mono', monospace;
  font-size: .6rem; color: var(--g-400);
  letter-spacing: .1em; text-transform: uppercase;
  border: 1px solid var(--g-700); padding: .2rem .6rem; border-radius: 2px;
}

/* ── Main hero area ────────────────────────────────────────────────────── */
.lp-hero {
  position: relative; z-index: 10;
  flex: 1;
  min-height: calc(100vh - 130px);
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  text-align: center; padding: 3rem;
}

.lp-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: .65rem; color: var(--g-300);
  text-transform: uppercase; letter-spacing: .2em;
  margin-bottom: 1.8rem;
  display: flex; align-items: center; gap: .8rem;
}
.lp-eyebrow::before, .lp-eyebrow::after {
  content: ''; width: 32px; height: 1px; background: var(--g-600);
}

.lp-headline {
  font-family: 'Space Grotesk', sans-serif;
  font-size: clamp(3rem, 7vw, 5.5rem);
  font-weight: 600; color: #fff;
  line-height: 1.05; letter-spacing: -.02em;
  margin-bottom: .5rem;
}
.lp-headline em {
  font-style: italic; color: var(--g-300);
}

.lp-subhead {
  font-family: 'Space Grotesk', sans-serif;
  font-size: clamp(1.1rem, 2.5vw, 1.6rem);
  font-weight: 400; font-style: italic;
  color: rgba(255,255,255,.45);
  margin-bottom: 2.5rem;
  letter-spacing: .01em;
}

.lp-desc {
  font-size: .88rem; color: rgba(255,255,255,.45);
  max-width: 440px; line-height: 1.8;
  margin-bottom: 3rem;
}

/* ── Get Started button — hue-shifting ─────────────────────────────────── */
.btn-get-started {
  display: inline-flex; align-items: center; gap: .75rem;
  padding: .95rem 2.4rem;
  border: none; cursor: pointer; border-radius: 2px;
  font-family: 'Inter', sans-serif;
  font-size: .88rem; font-weight: 600;
  letter-spacing: .06em; text-transform: uppercase;
  color: #fff;
  position: relative; overflow: hidden;
  transition: transform .2s ease, box-shadow .2s ease;
  /* default green bg — will be overridden by JS hsl() on hover */
  background: linear-gradient(135deg, var(--g-700), var(--g-500));
  box-shadow: 0 0 0 1px rgba(255,255,255,.08), 0 4px 24px rgba(0,0,0,.4);
}
.btn-get-started::before {
  content: '';
  position: absolute; inset: 0;
  background: inherit;
  filter: brightness(1.2);
  opacity: 0; transition: opacity .2s;
}
.btn-get-started:hover { transform: translateY(-2px); box-shadow: 0 0 0 1px rgba(255,255,255,.12), 0 8px 36px rgba(0,0,0,.5); }
.btn-get-started:hover::before { opacity: 1; }
.btn-get-started:active { transform: scale(.98); }
.btn-gs-arrow {
  display: inline-flex; transition: transform .2s;
}
.btn-get-started:hover .btn-gs-arrow { transform: translateX(3px); }

/* ── Feature pills row ─────────────────────────────────────────────────── */
.lp-features {
  position: relative; z-index: 10;
  display: flex; justify-content: center; gap: 0;
  border-top: 1px solid rgba(255,255,255,.05);
  padding: 1.4rem 3rem;
}
.lp-feature {
  padding: .6rem 2rem;
  border-right: 1px solid rgba(255,255,255,.05);
  text-align: center;
}
.lp-feature:last-child { border-right: none; }
.lp-feature-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: .6rem; color: var(--g-400);
  text-transform: uppercase; letter-spacing: .12em;
  margin-bottom: .2rem;
}
.lp-feature-val {
  font-family: 'Space Grotesk', sans-serif;
  font-size: .92rem; font-weight: 500; color: rgba(255,255,255,.7);
}

/* ══════════════════════════════════════════════════════════════════════════
   APP WRAPPER (header + tabs + content)
══════════════════════════════════════════════════════════════════════════ */
#app-wrapper {
  display: none;
}
#app-wrapper.visible {
  display: block;
}

/* ── Progress bar ──────────────────────────────────────────────────────── */
#progress {
  position: fixed; top: 0; left: 0; height: 2px; z-index: 9999;
  background: linear-gradient(90deg, var(--g-600), var(--g-300));
  width: 0; transition: width .3s ease;
}

/* ── Header ────────────────────────────────────────────────────────────── */
header {
  background: var(--g-900);
  border-bottom: 1px solid var(--g-800);
  padding: 0 2.5rem;
  display: flex; align-items: center;
  height: 60px; position: relative; z-index: 100;
}
.brand { display: flex; align-items: center; gap: 1rem; text-decoration: none; }
.brand-mark {
  width: 32px; height: 32px; border-radius: 2px;
  background: linear-gradient(135deg, var(--g-500), var(--g-300));
  display: flex; align-items: center; justify-content: center; font-size: 1rem;
}
.brand-name {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.25rem; color: #fff; font-weight: 600; letter-spacing: .02em;
}
.brand-divider { width: 1px; height: 20px; background: var(--g-700); margin: 0 .4rem; }
.brand-sub {
  font-size: .62rem; color: var(--g-300);
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: .1em; text-transform: uppercase;
}
.header-right { margin-left: auto; display: flex; align-items: center; gap: 1.5rem; }
.header-pill {
  font-family: 'JetBrains Mono', monospace;
  font-size: .62rem; color: var(--g-300);
  letter-spacing: .08em; text-transform: uppercase;
  border: 1px solid var(--g-700); padding: .2rem .65rem; border-radius: 2px;
}
.live-indicator {
  display: flex; align-items: center; gap: .4rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: .62rem; color: var(--g-300);
}
.live-dot {
  width: 6px; height: 6px; border-radius: 50%; background: #4ade80;
  animation: livepulse 2.5s ease infinite;
}
@keyframes livepulse {
  0%,100%{ opacity:1; box-shadow:0 0 0 0 rgba(74,222,128,.4); }
  50%     { opacity:.6; box-shadow:0 0 0 4px rgba(74,222,128,0); }
}

/* ── Tab nav ───────────────────────────────────────────────────────────── */
.tab-nav {
  background: var(--g-800); border-bottom: 1px solid var(--g-700);
  padding: 0 2.5rem; display: flex; position: relative; z-index: 99;
}
.tab-btn {
  padding: .7rem 1.2rem; border: none; background: none; cursor: pointer;
  font-family: 'Inter', sans-serif; font-size: .78rem; font-weight: 500;
  color: var(--g-300); border-bottom: 2px solid transparent;
  transition: color .2s, border-color .2s;
  display: flex; align-items: center; gap: .4rem;
  position: relative; top: 1px; letter-spacing: .02em;
}
.tab-btn:hover { color: #fff; }
.tab-btn.active { color: #fff; border-bottom-color: var(--g-300); }

/* ── Panels ────────────────────────────────────────────────────────────── */
.tab-panel { display: none; }
.tab-panel.active { display: block; }

/* ── Hero ──────────────────────────────────────────────────────────────── */
.hero {
  background:
    radial-gradient(ellipse at 70% 50%, rgba(74,154,106,.18) 0%, transparent 60%),
    radial-gradient(ellipse at 20% 80%, rgba(15,45,26,.5) 0%, transparent 55%),
    linear-gradient(135deg, var(--g-900) 0%, var(--g-800) 40%, var(--g-700) 100%);
  padding: 3.5rem 2.5rem;
  border-bottom: 1px solid var(--g-700);
  position: relative; overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute; inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4'%3E%3Crect width='1' height='1' fill='rgba(255,255,255,0.03)'/%3E%3C/svg%3E");
  background-repeat: repeat; pointer-events: none;
}
.hero::after {
  content: ''; position: absolute; right: 0; top: 0; bottom: 0; width: 1px;
  background: linear-gradient(to bottom, transparent, var(--g-500), transparent);
}
.hero-inner {
  max-width: 1160px; margin: 0 auto;
  display: grid; grid-template-columns: 1fr auto;
  gap: 3rem; align-items: center; position: relative; z-index: 1;
}
@media(max-width:700px){ .hero-inner { grid-template-columns: 1fr; } }
.hero-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: .65rem; font-weight: 500; color: var(--g-300);
  text-transform: uppercase; letter-spacing: .16em; margin-bottom: 1rem;
  display: flex; align-items: center; gap: .6rem;
}
.hero-eyebrow::before { content: ''; width: 24px; height: 1px; background: var(--g-400); }
.hero h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.6rem; font-weight: 600; color: #fff;
  line-height: 1.15; letter-spacing: -.02em; margin-bottom: 1rem;
}
.hero h1 em { font-style: italic; color: var(--g-300); }
.hero-desc { font-size: .88rem; color: rgba(255,255,255,.55); max-width: 460px; line-height: 1.75; }

/* ── Content area ──────────────────────────────────────────────────────── */
.content-area { max-width: 1160px; margin: 0 auto; padding: 0 2.5rem; }

/* ── Section row ───────────────────────────────────────────────────────── */
.section-row {
  display: grid; grid-template-columns: 200px 1fr;
  border-bottom: 1px solid var(--sep); min-height: 48px;
}
.section-label-col {
  padding: .85rem 1.5rem .85rem 0; border-right: 1px solid var(--sep);
  display: flex; align-items: center;
}
.section-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: .62rem; font-weight: 500; color: var(--ink4);
  text-transform: uppercase; letter-spacing: .14em;
}
.section-content-col { padding: .85rem 0 .85rem 1.5rem; display: flex; align-items: center; }

/* ── Analyzer two-col ──────────────────────────────────────────────────── */
.analyzer-grid {
  display: grid; grid-template-columns: 1fr;
  gap: 0; border-bottom: 1px solid var(--sep);
}
.a-left {
  padding: 2rem 0; border-bottom: 1px solid var(--sep);
}
.a-right { padding: 2rem 0; }
/* ── Image panels row (side-by-side desktop, stacked mobile) ─────────── */
.img-panels-row {
  display: flex; gap: 1.25rem; align-items: stretch;
}
.img-panels-row .img-card { flex: 1; min-width: 0; }
/* ── Centered button row below both cards ────────────────────────────── */
.btn-row-centered {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: .85rem;
  margin-top: 1.6rem;
  padding-bottom: .25rem;
}
.btn-run-main {
  padding: .72rem 2.4rem !important;
  font-size: .88rem !important;
  font-weight: 700 !important;
  flex: none !important;
  border-radius: 12px !important;
  box-shadow: 0 4px 20px rgba(37,80,184,.32), 0 2px 6px rgba(37,80,184,.18) !important;
  letter-spacing: .04em !important;
}
.btn-run-main:hover {
  box-shadow: 0 8px 28px rgba(37,80,184,.42), 0 3px 8px rgba(37,80,184,.22) !important;
  transform: translateY(-2px) !important;
}
.btn-reset-main {
  padding: .68rem 1.3rem !important;
  font-size: .84rem !important;
  border-radius: 10px !important;
  flex: none !important;
}
@media (max-width: 680px) {
  .img-panels-row { flex-direction: column; }
  .btn-row-centered { flex-direction: column; gap: .6rem; }
  .btn-run-main, .btn-reset-main { width: 100%; justify-content: center; }
}

/* ── Field label ───────────────────────────────────────────────────────── */
.field-label {
  font-family: 'Space Grotesk', sans-serif;
  font-size: .78rem; font-weight: 700; color: var(--ink2);
  text-transform: uppercase; letter-spacing: .12em; margin-bottom: .5rem;
}

/* ── Card wrapper for upload + preview ──────────────────────────────────── */
.img-card {
  background: linear-gradient(145deg, rgba(246,249,255,.95) 0%, rgba(228,236,247,.9) 100%);
  border: 1px solid rgba(122,161,236,.25);
  border-radius: 18px;
  padding: 1.4rem 1.5rem;
  box-shadow: 0 4px 24px rgba(25,46,110,.10), 0 1px 4px rgba(25,46,110,.06);
  transition: transform .25s ease, box-shadow .25s ease;
  position: relative;
  overflow: hidden;
}
.img-card::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,.6) 0%, transparent 60%);
  border-radius: 18px;
  pointer-events: none;
}
.img-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 36px rgba(25,46,110,.16), 0 2px 8px rgba(25,46,110,.08);
}
.img-card-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: .72rem; font-weight: 700; color: #000;
  text-transform: uppercase; letter-spacing: .14em;
  margin-bottom: .85rem;
  display: flex; align-items: center; gap: .5rem;
}
.img-card-title::before {
  content: '';
  display: inline-block; width: 3px; height: 14px;
  background: linear-gradient(to bottom, var(--g-400), var(--g-700));
  border-radius: 2px;
}

/* ── Upload zone ───────────────────────────────────────────────────────── */
.upload-zone {
  border: 2px dashed rgba(68,114,214,.3);
  background: linear-gradient(145deg, rgba(255,255,255,.7), rgba(238,243,251,.6));
  padding: 2rem 1.25rem; text-align: center; cursor: pointer;
  transition: all .28s cubic-bezier(.4,0,.2,1);
  border-radius: 14px;
  position: relative; overflow: hidden;
}
.upload-zone::after {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(circle at 50% 0%, rgba(68,114,214,.06) 0%, transparent 70%);
  pointer-events: none;
}
.upload-zone:hover, .upload-zone.drag {
  border-color: var(--g-400);
  background: linear-gradient(145deg, rgba(238,243,251,.9), rgba(216,229,252,.7));
  transform: scale(1.01);
  box-shadow: 0 4px 20px rgba(68,114,214,.12);
}
.uz-icon {
  font-size: 2rem; margin-bottom: .65rem;
  display: inline-block;
  background: linear-gradient(135deg, var(--g-500), var(--g-300));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  filter: drop-shadow(0 2px 4px rgba(37,80,184,.2));
  transition: transform .25s ease;
}
.upload-zone:hover .uz-icon { transform: translateY(-3px) scale(1.1); }
.uz-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: .88rem; font-weight: 700; color: var(--ink); margin-bottom: .3rem;
}
.uz-hint {
  font-size: .72rem; color: var(--ink3); line-height: 1.5;
}
#file-input { display: none; }

/* ── Preview card (glassmorphism) ─────────────────────────────────────── */
.preview-frame {
  background: linear-gradient(145deg, rgba(255,255,255,.75), rgba(216,229,252,.5));
  backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(122,161,236,.35);
  border-radius: 14px; min-height: 188px;
  display: flex; align-items: center; justify-content: center;
  overflow: hidden; position: relative;
  box-shadow: inset 0 1px 2px rgba(255,255,255,.8), 0 4px 18px rgba(25,46,110,.10);
  transition: box-shadow .28s ease, transform .28s ease;
}
.preview-frame:hover {
  box-shadow: inset 0 1px 2px rgba(255,255,255,.8), 0 8px 28px rgba(25,46,110,.15);
  transform: translateY(-2px);
}
.preview-frame img { width: 100%; max-height: 260px; object-fit: contain; display: none; border-radius: 10px; }
.preview-empty { padding: 2rem 1rem; text-align: center; }
.preview-empty-icon {
  font-size: 2.2rem; margin-bottom: .5rem; opacity: .35;
  display: block;
}
.preview-empty-text {
  font-family: 'Space Grotesk', sans-serif;
  font-size: .76rem; color: var(--ink4); font-weight: 500;
}
.preview-badge {
  position: absolute; bottom: 10px; right: 10px;
  background: linear-gradient(135deg, rgba(26,46,110,.88), rgba(37,80,184,.85));
  color: #c8d8ff;
  padding: .25rem .7rem; font-family: 'JetBrains Mono', monospace;
  font-size: .56rem; letter-spacing: .06em; text-transform: uppercase;
  border-radius: 20px; display: none;
  border: 1px solid rgba(122,161,236,.3);
  backdrop-filter: blur(4px);
}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.btn-row { display: flex; gap: .65rem; margin-top: 1.1rem; align-items: center; }
.btn-primary {
  padding: .58rem 1.35rem;
  background: linear-gradient(135deg, var(--g-700) 0%, var(--g-500) 100%);
  color: #fff; border: none; border-radius: 10px;
  font-family: 'Space Grotesk', sans-serif; font-size: .8rem; font-weight: 700;
  cursor: pointer; display: inline-flex; align-items: center; justify-content: center; gap: .45rem;
  transition: all .22s cubic-bezier(.4,0,.2,1);
  letter-spacing: .03em;
  box-shadow: 0 2px 12px rgba(37,80,184,.28), 0 1px 3px rgba(37,80,184,.18);
  white-space: nowrap;
}
.btn-primary:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(37,80,184,.38); filter: brightness(1.1); }
.btn-primary:active { transform: scale(.98); filter: brightness(.96); }
.btn-primary:disabled { opacity: .45; cursor: default; transform: none; filter: none; box-shadow: none; }
.btn-secondary {
  padding: .55rem .95rem; background: rgba(255,255,255,.7); border: 1px solid var(--sep2);
  color: var(--ink3); border-radius: 10px;
  font-family: 'Space Grotesk', sans-serif; font-size: .8rem; font-weight: 600;
  cursor: pointer; transition: all .2s ease;
  backdrop-filter: blur(4px);
  white-space: nowrap;
}
.btn-secondary:hover { background: var(--bg2); color: var(--ink); border-color: var(--g-300); transform: translateY(-1px); }
.spinner {
  width: 14px; height: 14px; border: 2px solid rgba(255,255,255,.3);
  border-top-color: #fff; border-radius: 50%;
  animation: spin .65s linear infinite; display: none;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ── Full-page loading overlay ────────────────────────────────────────── */
#analysis-overlay {
  display: none;
  position: fixed; inset: 0; z-index: 9998;
  background: rgba(6,14,32,.55);
  backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
  align-items: center; justify-content: center;
  flex-direction: column; gap: 1.4rem;
}
#analysis-overlay.active { display: flex; }
.overlay-spinner-ring {
  width: 64px; height: 64px;
  border: 4px solid rgba(122,161,236,.2);
  border-top-color: var(--g-300);
  border-right-color: var(--g-400);
  border-radius: 50%;
  animation: spin .75s linear infinite;
  box-shadow: 0 0 28px rgba(68,114,214,.35);
}
.overlay-label {
  font-family: 'Space Grotesk', sans-serif;
  font-size: .92rem; font-weight: 600;
  color: #c8d8ff;
  letter-spacing: .06em;
  text-transform: uppercase;
  animation: overlayPulse 1.5s ease infinite;
}
.overlay-dots::after {
  content: '';
  animation: dots 1.5s steps(4,end) infinite;
}
@keyframes dots {
  0%  { content: ''; }
  25% { content: '.'; }
  50% { content: '..'; }
  75% { content: '...'; }
}
@keyframes overlayPulse {
  0%,100% { opacity: 1; }
  50%      { opacity: .6; }
}

/* ── Empty state ───────────────────────────────────────────────────────── */
.empty-state { padding: 1.5rem 0; color: var(--ink4); font-size: .84rem; font-style: italic; }

/* ── Result section head ───────────────────────────────────────────────── */
.result-section-head {
  padding: 1.2rem 0 .8rem; border-bottom: 1px solid var(--sep); margin-bottom: 1.2rem;
}
.result-section-head h3 {
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 1.0rem; font-weight: 700; color: var(--g-500);
  text-transform: uppercase; letter-spacing: .14em;
}

/* ── HSV bars ──────────────────────────────────────────────────────────── */
.hsv-row {
  display: grid; grid-template-columns: 100px 1fr 64px;
  align-items: center; gap: 1rem; padding: .65rem 0;
  border-bottom: 1px solid var(--sep);
}
.hsv-row:last-child { border-bottom: none; }
.hsv-key {
  font-family: 'Space Grotesk', sans-serif;
  font-size: .78rem; font-weight: 700; color: #000; text-transform: uppercase; letter-spacing: .06em;
}
.hsv-bar-wrap { height: 6px; background: var(--bg3); border-radius: 1px; overflow: hidden; }
.hsv-bar-fill { height: 100%; border-radius: 1px; transition: width .6s cubic-bezier(.4,0,.2,1); }
.hf-h { background: linear-gradient(90deg, var(--g-700), var(--g-300)); }
.hf-s { background: linear-gradient(90deg, var(--g-600), var(--g-300)); }
.hf-v { background: linear-gradient(90deg, #d97706, #fcd34d); }
.hsv-val {
  font-family: 'Space Grotesk', sans-serif;
  font-size: .82rem; font-weight: 700; color: var(--ink); text-align: right;
}

/* ── Spectrum ──────────────────────────────────────────────────────────── */
.spectrum-row {
  display: grid; grid-template-columns: 100px 1fr;
  gap: 1rem; align-items: center; padding: .8rem 0; border-bottom: 1px solid var(--sep);
}
.spectrum-bar {
  height: 10px; border-radius: 2px;
  background: linear-gradient(90deg,#f0f0e8 0%,#f7e96b 14%,#d4a017 28%,#f4820a 42%,#e84040 57%,#5c3317 71%,#3cb371 85%,#8b008b 100%);
  position: relative;
}
.spectrum-cursor {
  position: absolute; top: -4px; width: 2px; height: 18px;
  background: var(--ink); border-radius: 1px; transform: translateX(-50%);
  transition: left .6s cubic-bezier(.4,0,.2,1);
}
.spectrum-cursor::before {
  content: ''; position: absolute; top: -4px; left: 50%; transform: translateX(-50%);
  width: 6px; height: 6px; border-radius: 50%; background: var(--ink);
}

/* ── Color detected ────────────────────────────────────────────────────── */
.color-strip-row {
  display: grid; grid-template-columns: 100px 1fr;
  gap: 1rem; align-items: center; padding: 1rem 0; border-bottom: 1px solid var(--sep);
}
.color-strip-label {
  font-family: 'Space Grotesk', sans-serif;
  font-size: .78rem; font-weight: 700; color: #000; text-transform: uppercase; letter-spacing: .06em;
}
.color-strip-inner { display: flex; align-items: center; gap: 1rem; }
.color-swatch-rect {
  width: 48px; height: 30px; border-radius: 2px;
  border: 1px solid var(--sep); flex-shrink: 0; transition: background .4s;
}
.color-detected-name { font-family: 'Space Grotesk', sans-serif; font-size: .88rem; font-weight: 700; color: #000; margin-bottom: .1rem; }
.color-detected-codes { font-family: 'JetBrains Mono', monospace; font-size: .67rem; color: var(--ink4); }

/* ── Risk ──────────────────────────────────────────────────────────────── */
.risk-grid-row {
  display: grid; grid-template-columns: 100px 1fr;
  gap: 1rem; align-items: flex-start; padding: 1rem 0; border-bottom: 1px solid var(--sep);
}
.risk-badge {
  display: inline-flex; align-items: center; gap: .4rem;
  padding: .3rem .75rem; border-radius: 2px;
  font-size: .72rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
  width: fit-content;
}
.risk-badge.normal   { background:var(--ok-bg);     color:var(--ok);     border:1px solid var(--ok-sep); }
.risk-badge.moderate { background:var(--warn-bg);   color:var(--warn);   border:1px solid var(--warn-sep); }
.risk-badge.high     { background:var(--danger-bg); color:var(--danger); border:1px solid var(--danger-sep); }
.risk-desc-text { font-size: .84rem; color: var(--ink2); line-height: 1.6; }
.recommendation-row { padding: 1rem 0; border-bottom: 1px solid var(--sep); }
.rec-content { margin-top: .6rem; display: flex; gap: .75rem; align-items: flex-start; }
.rec-icon {
  width: 20px; height: 20px; flex-shrink: 0; background: var(--g-100);
  border-radius: 50%; display: flex; align-items: center; justify-content: center;
  font-size: .7rem; margin-top: .1rem;
}
.rec-text { font-size: .84rem; color: var(--ink2); line-height: 1.65; }

/* ── Report ────────────────────────────────────────────────────────────── */
.report-section { padding: 2rem 0; }

/* Outer wrapper card */
.report-table-wrap {
  border: 1px solid rgba(68,114,214,.18);
  border-radius: 10px;
  overflow: hidden;
  background: #fff;
  box-shadow: 0 2px 12px rgba(20,40,120,.06);
}

/* Column header row */
.report-grid-header {
  display: grid;
  grid-template-columns: 220px 1fr;
  padding: .6rem 1.1rem;
  background: linear-gradient(90deg, rgba(68,114,214,.08) 0%, rgba(68,114,214,.04) 100%);
  border-bottom: 1.5px solid rgba(68,114,214,.2);
}
.rg-th {
  font-family: 'Space Grotesk', sans-serif;
  font-size: .57rem;
  color: #5060a0;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .13em;
}

/* Standard data rows */
.report-row {
  display: grid;
  grid-template-columns: 220px 1fr;
  padding: .65rem 1.1rem;
  border-bottom: 1px solid rgba(0,0,0,.055);
  transition: background .15s;
  align-items: start;
}
.report-row:last-child { border-bottom: none; }
.report-row:nth-child(even) { background: rgba(68,114,214,.025); }
.report-row:hover { background: rgba(68,114,214,.065); }

/* Label column — bold solid black */
.rg-key {
  font-family: 'Space Grotesk', sans-serif;
  font-size: .72rem;
  color: #06060f;
  font-weight: 700;
  letter-spacing: .04em;
  text-transform: uppercase;
  padding-right: 1rem;
  line-height: 1.45;
  padding-top: .05rem;
}

/* Value column — readable, lighter weight */
.rg-val {
  font-family: 'Inter', sans-serif;
  font-size: .76rem;
  font-weight: 400;
  color: #2e2e42;
  line-height: 1.55;
  letter-spacing: .008em;
}

/* Special recommendation row */
.report-row-rec {
  padding: 1rem 1.1rem;
  border-top: 1.5px solid rgba(68,114,214,.18);
  background: rgba(68,114,214,.03);
}
.report-row-rec .rg-key {
  font-size: .67rem;
  margin-bottom: .45rem;
  display: block;
  color: #5060a0;
}
.report-row-rec .rg-val-rec {
  font-family: 'Inter', sans-serif;
  font-size: .78rem;
  font-weight: 400;
  color: #1c1c2e;
  line-height: 1.7;
  background: rgba(255,255,255,.7);
  border-left: 3px solid rgba(68,114,214,.5);
  padding: .55rem .85rem;
  border-radius: 0 6px 6px 0;
  letter-spacing: .01em;
}

.disclaimer-bar {
  margin-top: 1.2rem;
  padding: .4rem 0;
  background: none;
  border: none;
  font-size: .68rem;
  line-height: 1.65;
  color: #b8922a;
  opacity: .82;
  letter-spacing: .01em;
}
.btn-dl {
  margin-top: 1.25rem; padding: .68rem 1.3rem;
  background: var(--surface); border: 1px solid var(--sep2);
  color: var(--g-700); border-radius: var(--r);
  font-family: 'Inter', sans-serif; font-size: .8rem; font-weight: 600;
  cursor: pointer; display: inline-flex; align-items: center; gap: .5rem;
  transition: background .2s, border-color .2s, transform .2s; letter-spacing: .02em;
}
.btn-dl:hover { background: var(--g-50); border-color: var(--g-300); transform: translateY(-1px); }

/* ══ LEARN MORE TAB ══════════════════════════════════════════════════════ */

/* Hero */
.lm-hero {
  background:
    radial-gradient(ellipse at 75% 35%, rgba(68,114,214,.22) 0%, transparent 55%),
    linear-gradient(135deg, #0d1e42 0%, #112454 55%, #162d6a 100%);
  padding: 3.2rem 2.5rem 2.8rem;
  border-bottom: 1px solid rgba(68,114,214,.2);
  position: relative; overflow: hidden;
}
.lm-hero::before {
  content: '';
  position: absolute; inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4'%3E%3Crect width='1' height='1' fill='rgba(255,255,255,0.025)'/%3E%3C/svg%3E");
  pointer-events: none;
}
.lm-hero-inner { max-width: 1100px; margin: 0 auto; position: relative; z-index: 1; }
.lm-hero-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: .6rem; color: rgba(125,211,252,.7); text-transform: uppercase;
  letter-spacing: .16em; margin-bottom: .7rem;
  display: flex; align-items: center; gap: .5rem;
}
.lm-hero-eyebrow::before { content:''; width:18px; height:1px; background:rgba(125,211,252,.4); }
.lm-hero-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.2rem; font-weight: 700; color: #fff;
  letter-spacing: -.025em; line-height: 1.1; margin-bottom: .75rem;
}
.lm-hero-sub { font-size: .87rem; color: rgba(255,255,255,.5); max-width: 460px; line-height: 1.7; margin-bottom: 1.8rem; }

/* Jump nav buttons */
.lm-jump-nav { display: flex; gap: .75rem; flex-wrap: wrap; }
.lm-jump-btn {
  display: inline-flex; align-items: center; gap: .45rem;
  padding: .55rem 1.25rem;
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(125,211,252,.25);
  color: #c8deff;
  border-radius: 7px;
  font-family: 'Inter', sans-serif;
  font-size: .78rem; font-weight: 500;
  text-decoration: none; letter-spacing: .02em;
  transition: background .2s, border-color .2s, color .2s, transform .15s;
}
.lm-jump-btn:hover {
  background: rgba(125,211,252,.12);
  border-color: rgba(125,211,252,.5);
  color: #7dd3fc; transform: translateY(-2px);
}

/* Insights tab buttons */
.insights-btn {
  display: inline-flex; align-items: center; gap: .55rem;
  padding: .75rem 1.6rem;
  background: linear-gradient(135deg, #1a3a8f 0%, #2255c4 100%);
  color: #e8efff; font-family: 'Space Grotesk', sans-serif;
  font-size: .88rem; font-weight: 600; letter-spacing: .03em;
  border-radius: 8px; text-decoration: none;
  border: 1px solid rgba(100,150,255,.25);
  box-shadow: 0 4px 18px rgba(26,58,143,.35);
  transition: transform .2s ease, box-shadow .2s ease, background .2s ease;
}
.insights-btn:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 28px rgba(26,58,143,.5);
  background: linear-gradient(135deg, #1e44ab 0%, #2b66e8 100%);
}
.insights-btn-alt {
  background: linear-gradient(135deg, #0f4c75 0%, #1877b8 100%);
  box-shadow: 0 4px 18px rgba(15,76,117,.4);
}
.insights-btn-alt:hover {
  background: linear-gradient(135deg, #125a8a 0%, #1e8cd8 100%);
  box-shadow: 0 8px 28px rgba(15,76,117,.55);
}
.insights-card {
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(68,114,214,.18);
  border-radius: 12px;
  padding: 2rem 2rem 1.75rem;
  max-width: 760px;
}
.insights-card-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: .72rem; font-weight: 500; letter-spacing: .12em;
  text-transform: uppercase; color: var(--g-300); margin-bottom: .6rem;
}
.insights-card-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.2rem; font-weight: 700; color: #e8efff;
  margin-bottom: .85rem;
}
.insights-card-body {
  font-size: .88rem; color: var(--ink2); line-height: 1.7;
  margin-bottom: 1.4rem;
}

/* Sections */
.lm-section {
  padding: 3.5rem 2.5rem;
  border-bottom: 1px solid rgba(68,114,214,.12);
}
.lm-section-alt { background: rgba(6,14,32,.6); }
.lm-section-inner { max-width: 1100px; margin: 0 auto; }
.lm-section-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: .6rem; color: var(--g-500); text-transform: uppercase;
  letter-spacing: .15em; margin-bottom: .55rem;
  display: flex; align-items: center; gap: .5rem;
}
.lm-section-eyebrow::before { content:''; width:18px; height:1px; background:currentColor; opacity:.5; }
.lm-section-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.65rem; font-weight: 700; color: #fff;
  letter-spacing: -.02em; margin-bottom: .55rem;
}
.lm-section-sub {
  font-size: .85rem; color: var(--g-400);
  max-width: 520px; line-height: 1.65; margin-bottom: 2rem;
}

/* Team grid (reused) */
.team-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1rem;
}
.team-card {
  background: var(--g-800);
  border: 1px solid var(--g-700);
  border-radius: 10px;
  padding: 1rem 1.25rem;
  display: flex; align-items: center; gap: .95rem;
  transition: border-color .2s, background .2s, transform .18s;
}
.team-card:hover {
  border-color: rgba(125,211,252,.3);
  background: rgba(125,211,252,.05);
  transform: translateY(-2px);
}
.team-avatar {
  width: 42px; height: 42px; border-radius: 50%;
  background: linear-gradient(135deg, #1e4096, #2960d0);
  display: flex; align-items: center; justify-content: center;
  font-family: 'Space Grotesk', sans-serif;
  font-size: .82rem; font-weight: 700; color: #7dd3fc;
  flex-shrink: 0; border: 1.5px solid rgba(125,211,252,.2);
}
.team-info { min-width: 0; }
.team-name {
  font-family: 'Space Grotesk', sans-serif;
  font-size: .84rem; font-weight: 600; color: #fff;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.team-roll {
  font-family: 'JetBrains Mono', monospace;
  font-size: .68rem; color: var(--g-400); margin-top: .2rem; letter-spacing: .04em;
}

/* Future card (reused + improved) */
.future-card {
  max-width: 740px;
  background: var(--g-800);
  border: 1px solid rgba(68,114,214,.25);
  border-radius: 12px;
  padding: 1.8rem 2rem;
  position: relative; overflow: hidden;
}
.future-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, #1e4096, #7dd3fc, #1e4096);
}
.future-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: .6rem; color: rgba(125,211,252,.75); text-transform: uppercase;
  letter-spacing: .15em; margin-bottom: .75rem;
  display: flex; align-items: center; gap: .5rem;
}
.future-label::before { content:''; width:14px; height:1px; background:rgba(125,211,252,.45); }
.future-body {
  font-size: .9rem; color: var(--g-300); line-height: 1.82;
  border-left: 3px solid rgba(125,211,252,.28);
  padding-left: 1.1rem;
}

/* Responsive */
@media (max-width: 640px) {
  .lm-hero { padding: 2rem 1.25rem 1.8rem; }
  .lm-hero-title { font-size: 1.6rem; }
  .lm-section { padding: 2.5rem 1.25rem; }
  .team-grid { grid-template-columns: 1fr; }
  .lm-jump-nav { gap: .5rem; }
  .future-card { padding: 1.3rem 1.2rem; }
}

/* ══ TUTORIAL ════════════════════════════════════════════════════════════ */
.tut-hero {
  background: radial-gradient(ellipse at 80% 40%, rgba(100,150,240,.18) 0%, transparent 55%),
    linear-gradient(135deg, #1a3580 0%, #1e4096 60%, #2050b0 100%);
  padding: 3rem 2.5rem; border-bottom: 1px solid rgba(122,161,236,.3);
  position: relative; overflow: hidden;
}
.tut-hero::before {
  content: ''; position: absolute; inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4'%3E%3Crect width='1' height='1' fill='rgba(255,255,255,0.03)'/%3E%3C/svg%3E");
  pointer-events: none;
}
.tut-hero-inner { max-width: 1160px; margin: 0 auto; position: relative; z-index: 1; }
.tut-hero-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: .62rem; color: var(--g-300); text-transform: uppercase; letter-spacing: .16em;
  margin-bottom: .8rem; display: flex; align-items: center; gap: .6rem;
}
.tut-hero-eyebrow::before { content:''; width:20px; height:1px; background:var(--g-400); }
.tut-hero h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2.1rem; font-weight: 600; color: #fff; line-height: 1.15;
  letter-spacing: -.02em; margin-bottom: .8rem;
}
.tut-hero h1 em { font-style: italic; color: var(--g-300); }
.tut-hero p { font-size: .88rem; color: rgba(255,255,255,.55); max-width: 500px; line-height: 1.7; }

.steps-section, .ref-section, .faq-wrap { max-width: 1160px; margin: 0 auto; padding: 0 2.5rem; }
.section-divider {
  padding: 1.2rem 0 .8rem; border-bottom: 1px solid var(--sep2); margin-bottom: 0;
}
.section-divider h2 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: .72rem; font-weight: 700; color: var(--g-700);
  text-transform: uppercase; letter-spacing: .12em;
}

.steps-grid {
  display: grid; grid-template-columns: repeat(5, 1fr);
  border-bottom: 1px solid rgba(122,161,236,.25);
  background: rgba(255,255,255,.45);
  border-radius: 14px;
  overflow: hidden;
  box-shadow: 0 2px 16px rgba(25,46,110,.07);
  margin: 1rem 0;
}
@media(max-width:780px){ .steps-grid { grid-template-columns: 1fr; } }
.step-col {
  padding: 1.8rem 1.2rem 1.8rem 0; border-right: 1px solid var(--sep);
  transition: background .2s;
}
.step-col:last-child { border-right: none; padding-right: 0; }
.step-col:not(:first-child) { padding-left: 1.2rem; }
.step-col:hover { background: rgba(180,210,255,.35); }
.step-num {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.6rem; font-weight: 700; color: var(--g-200); line-height: 1; margin-bottom: .8rem;
}
.step-title { font-size: .82rem; font-weight: 700; color: var(--ink); margin-bottom: .4rem; }
.step-desc { font-size: .76rem; color: var(--ink3); line-height: 1.6; }

.cref-table {
  width: 100%; border-collapse: collapse;
  background: rgba(255,255,255,.55);
  border-radius: 14px; overflow: hidden;
  box-shadow: 0 2px 16px rgba(25,46,110,.07);
}
.cref-table th {
  font-family: 'JetBrains Mono', monospace; font-size: .6rem; text-transform: uppercase;
  letter-spacing: .1em; color: var(--ink4); text-align: left; padding: .55rem .5rem;
  border-bottom: 1px solid var(--sep2);
}
.cref-table td { padding: .65rem .5rem; border-bottom: 1px solid var(--sep); font-size: .82rem; vertical-align: middle; }
.cref-table tr:hover td { background: var(--bg2); }
.cchip { display: inline-block; width: 22px; height: 22px; border-radius: 2px; vertical-align: middle; margin-right: .6rem; border: 1px solid rgba(0,0,0,.1); }
.rpill {
  display: inline-flex; align-items: center; gap: .3rem;
  font-family: 'JetBrains Mono', monospace; font-size: .62rem; font-weight: 600;
  padding: .18rem .6rem; border-radius: 2px; letter-spacing: .04em; text-transform: uppercase;
}
.rpill.n { background:var(--ok-bg);     color:var(--ok);     border:1px solid var(--ok-sep); }
.rpill.m { background:var(--warn-bg);   color:var(--warn);   border:1px solid var(--warn-sep); }
.rpill.h { background:var(--danger-bg); color:var(--danger); border:1px solid var(--danger-sep); }

.faq-wrap { padding-bottom: 4rem; padding-top: .5rem; }
.faq-item {
  border-bottom: 1px solid rgba(122,161,236,.25);
  background: rgba(255,255,255,.35);
  transition: background .2s;
}
.faq-item:hover { background: rgba(255,255,255,.55); }
.faq-q {
  width: 100%; padding: 1rem 0; background: none; border: none; cursor: pointer;
  display: flex; justify-content: space-between; align-items: center;
  font-family: 'Inter', sans-serif; font-size: .86rem; font-weight: 500;
  color: var(--ink); text-align: left; transition: color .2s;
}
.faq-q:hover { color: var(--g-600); }
.faq-chev { transition: transform .2s; color: var(--ink4); flex-shrink: 0; }
.faq-item.open .faq-chev { transform: rotate(180deg); }
.faq-a { display: none; padding-bottom: 1rem; font-size: .83rem; line-height: 1.7; color: var(--ink3); }
.faq-item.open .faq-a { display: block; }

/* ── Animations ────────────────────────────────────────────────────────── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
.lp-hero > * {
  animation: fadeInUp .7s ease both;
}
.lp-hero > *:nth-child(1) { animation-delay: .1s; }
.lp-hero > *:nth-child(2) { animation-delay: .2s; }
.lp-hero > *:nth-child(3) { animation-delay: .3s; }
.lp-hero > *:nth-child(4) { animation-delay: .4s; }
.lp-hero > *:nth-child(5) { animation-delay: .5s; }

/* ── Tutorial tab hue-shift ─────────────────────────────────────────── */
.tab-btn-tutorial {
  border-radius: 6px 6px 0 0;
  font-weight: 600 !important;
  transition: color .2s, border-color .2s, background .3s !important;
}
.tab-btn-tutorial:not(.active) {
  background: linear-gradient(135deg, hsl(220,60%,30%), hsl(250,55%,38%));
  color: #c8d8ff !important;
  border-bottom-color: transparent !important;
  animation: tutHueShift 6s linear infinite;
}
.tab-btn-tutorial.active {
  background: rgba(255,255,255,.08) !important;
  color: #fff !important;
  border-bottom-color: var(--g-300) !important;
  animation: none;
}
@keyframes tutHueShift {
  0%   { background: linear-gradient(135deg, hsl(220,60%,28%), hsl(250,55%,36%)); }
  16%  { background: linear-gradient(135deg, hsl(260,58%,28%), hsl(290,55%,36%)); }
  33%  { background: linear-gradient(135deg, hsl(300,55%,26%), hsl(330,52%,34%)); }
  50%  { background: linear-gradient(135deg, hsl(340,58%,28%), hsl(10,55%,36%));  }
  66%  { background: linear-gradient(135deg, hsl(30,60%,28%),  hsl(60,55%,36%));  }
  83%  { background: linear-gradient(135deg, hsl(160,58%,24%), hsl(190,55%,32%)); }
  100% { background: linear-gradient(135deg, hsl(220,60%,28%), hsl(250,55%,36%)); }
}
.hidden { display: none !important; }
</style>
</head>
<body>

<!-- ══════════════════════ LANDING PAGE ══════════════════════════════════ -->
<div id="landing-page">

  <!-- Animated SVG background -->
  <svg id="bg-canvas" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice">
    <defs>
      <radialGradient id="rg1" cx="70%" cy="30%" r="60%">
        <stop offset="0%" stop-color="#2550b8" stop-opacity=".25"/>
        <stop offset="100%" stop-color="#060e20" stop-opacity="0"/>
      </radialGradient>
      <radialGradient id="rg2" cx="20%" cy="80%" r="50%">
        <stop offset="0%" stop-color="#1a2e6e" stop-opacity=".3"/>
        <stop offset="100%" stop-color="#060e20" stop-opacity="0"/>
      </radialGradient>
    </defs>
    <!-- Base gradient fills -->
    <rect width="100%" height="100%" fill="url(#rg1)"/>
    <rect width="100%" height="100%" fill="url(#rg2)"/>

    <!-- Geometric grid lines -->
    <g stroke="#4472d6" stroke-opacity=".06" stroke-width="1" id="grid-lines">
      <!-- Horizontal -->
      <line x1="0" y1="15%" x2="100%" y2="15%"/>
      <line x1="0" y1="30%" x2="100%" y2="30%"/>
      <line x1="0" y1="45%" x2="100%" y2="45%"/>
      <line x1="0" y1="60%" x2="100%" y2="60%"/>
      <line x1="0" y1="75%" x2="100%" y2="75%"/>
      <line x1="0" y1="90%" x2="100%" y2="90%"/>
      <!-- Vertical -->
      <line x1="10%" y1="0" x2="10%" y2="100%"/>
      <line x1="25%" y1="0" x2="25%" y2="100%"/>
      <line x1="40%" y1="0" x2="40%" y2="100%"/>
      <line x1="55%" y1="0" x2="55%" y2="100%"/>
      <line x1="70%" y1="0" x2="70%" y2="100%"/>
      <line x1="85%" y1="0" x2="85%" y2="100%"/>
    </g>

    <!-- Diagonal accent lines -->
    <g stroke="#7aa1ec" stroke-opacity=".04" stroke-width="1">
      <line x1="0" y1="0" x2="40%" y2="100%"/>
      <line x1="60%" y1="0" x2="100%" y2="100%"/>
      <line x1="80%" y1="0" x2="20%" y2="100%"/>
    </g>

    <!-- Large decorative circles -->
    <circle cx="78%" cy="22%" r="260" fill="none" stroke="#4472d6" stroke-opacity=".07" stroke-width="1"/>
    <circle cx="78%" cy="22%" r="180" fill="none" stroke="#4472d6" stroke-opacity=".09" stroke-width="1"/>
    <circle cx="78%" cy="22%" r="100" fill="none" stroke="#7aa1ec" stroke-opacity=".1" stroke-width="1"/>
    <circle cx="78%" cy="22%" r="40"  fill="#4472d6" fill-opacity=".06"/>

    <circle cx="15%" cy="72%" r="200" fill="none" stroke="#1f3a8a" stroke-opacity=".06" stroke-width="1"/>
    <circle cx="15%" cy="72%" r="120" fill="none" stroke="#1f3a8a" stroke-opacity=".07" stroke-width="1"/>

    <!-- Small crosshair marks -->
    <g stroke="#7aa1ec" stroke-opacity=".15" stroke-width="1">
      <line x1="78%" y1="calc(22% - 12px)" x2="78%" y2="calc(22% + 12px)"/>
      <line x1="calc(78% - 12px)" y1="22%" x2="calc(78% + 12px)" y2="22%"/>
      <line x1="15%" y1="calc(72% - 10px)" x2="15%" y2="calc(72% + 10px)"/>
      <line x1="calc(15% - 10px)" y1="72%" x2="calc(15% + 10px)" y2="72%"/>
    </g>

    <!-- Floating data-viz lines (right side) -->
    <g transform="translate(72%, 55%)" stroke-opacity=".12" fill="none">
      <polyline points="0,0 30,-15 60,5 90,-20 120,0 150,-8" stroke="#7aa1ec" stroke-width="1.5"/>
      <polyline points="0,20 30,8 60,25 90,5 120,18 150,10" stroke="#4472d6" stroke-width="1"/>
      <circle cx="30" cy="-15" r="2.5" fill="#7aa1ec" fill-opacity=".4"/>
      <circle cx="90" cy="-20" r="2.5" fill="#7aa1ec" fill-opacity=".4"/>
    </g>

    <!-- Scan line effect -->
    <rect x="0" y="0" width="100%" height="2px" fill="url(#scanGrad)" opacity=".15">
      <animate attributeName="y" from="-2" to="100%" dur="4s" repeatCount="indefinite"/>
    </rect>
    <defs>
      <linearGradient id="scanGrad" x1="0" y1="0" x2="1" y2="0">
        <stop offset="0%" stop-color="transparent"/>
        <stop offset="50%" stop-color="#7aa1ec"/>
        <stop offset="100%" stop-color="transparent"/>
      </linearGradient>
    </defs>
  </svg>

  <!-- Top bar -->
  <div class="lp-topbar">
    <div class="lp-logo">
      <div class="lp-logo-mark">🧪</div>
      <div class="lp-logo-name">NephroScan</div>
    </div>
    <div class="lp-version">Clinical Analyzer · v10.0</div>
  </div>

  <!-- Hero -->
  <div class="lp-hero">
    <div class="lp-eyebrow">Urine Test Strip Analysis System</div>

    <h1 class="lp-headline">
      Kidney Strip<br><em>Diagnostic</em>
    </h1>

    <p class="lp-subhead">Color-space analysis for clinical insight</p>

    <p class="lp-desc">
      Upload a photograph of your test strip. The system extracts the dominant color,
      maps it across the HSV spectrum, and delivers an instant risk classification
      against validated clinical reference thresholds.
    </p>

    <button class="btn-get-started" id="btn-get-started" onclick="enterApp()">
      Get Started
      <span class="btn-gs-arrow">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
      </span>
    </button>
  </div>

  <!-- Feature strip -->
  <div class="lp-features">
    <div class="lp-feature">
      <div class="lp-feature-label">Technology</div>
      <div class="lp-feature-val">Computer Vision</div>
    </div>
    <div class="lp-feature">
      <div class="lp-feature-label">Color Space</div>
      <div class="lp-feature-val">HSV Analysis</div>
    </div>
    <div class="lp-feature">
      <div class="lp-feature-label">Processing</div>
      <div class="lp-feature-val">Real-time</div>
    </div>
    <div class="lp-feature">
      <div class="lp-feature-label">Classification</div>
      <div class="lp-feature-val">3-Level Risk</div>
    </div>
    <div class="lp-feature">
      <div class="lp-feature-label">Report</div>
      <div class="lp-feature-val">PDF Export</div>
    </div>
  </div>
</div>

<!-- ══════════════════════ APP ════════════════════════════════════════════ -->
<div id="app-wrapper">
<div id="progress"></div>

<!-- Header -->
<header>
  <a class="brand" href="#" onclick="goHome()">
    <div class="brand-mark">🧪</div>
    <div class="brand-name">NephroScan</div>
    <div class="brand-divider"></div>
    <span class="brand-sub">Clinical Analyzer</span>
  </a>
  <div class="header-right">
    <div class="live-indicator"><div class="live-dot"></div>System Online</div>
    <div class="header-pill">v10.0 · Prototype</div>
  </div>
</header>

<!-- Tab nav -->
<nav class="tab-nav">
  <button class="tab-btn active" onclick="switchTab('analyzer',this)">
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    Analyzer
  </button>
  <button class="tab-btn" id="tab-learnmore-btn" onclick="switchTab('learnmore',this)">
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
    Learn More
  </button>
  <button class="tab-btn tab-btn-tutorial" id="tab-tutorial-btn" onclick="switchTab('tutorial',this)" style="margin-left:auto;">
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
    Tutorial
  </button>
</nav>

<!-- ════ ANALYZER TAB ═══════════════════════════════════════════════════ -->
<div id="tab-analyzer" class="tab-panel active">

  <div class="hero">
    <div class="hero-inner">
      <div>
        <div class="hero-eyebrow">Urine Test Strip · Color Analysis</div>
        <h1>Kidney Strip <em>Diagnostic</em><br>Analyzer</h1>
        <p class="hero-desc">Upload a photograph of your urine test strip. The system extracts the dominant color and classifies the result against clinical reference thresholds.</p>
      </div>
    </div>
  </div>

  <div class="content-area" style="padding-top:0;padding-bottom:4rem;">

    <div class="section-row">
      <div class="section-label-col"><span class="section-label" style="color:#000;font-weight:700;font-family:'Space Grotesk',sans-serif;font-size:.82rem;letter-spacing:.08em;">Input &amp; Results</span></div>
      <div class="section-content-col" style="font-size:.82rem;color:#000;font-weight:700;font-family:'Space Grotesk',sans-serif;">
        Upload an image, then click Run Analysis to generate a diagnostic report
      </div>
    </div>

    <div class="analyzer-grid">

      <!-- LEFT -->
      <div class="a-left">
        <div class="img-panels-row">
          <!-- Upload Card -->
          <div class="img-card">
            <div class="img-card-title">Strip Image</div>
            <div class="upload-zone" id="drop-zone"
                 onclick="document.getElementById('file-input').click()">
              <div class="uz-icon">⬆</div>
              <div class="uz-title">Upload strip image</div>
              <div class="uz-hint">Click to browse or drag &amp; drop<br>PNG · JPG · BMP · WEBP</div>
              <input type="file" id="file-input" accept="image/*">
            </div>
          </div>
          <!-- Preview Card -->
          <div class="img-card">
            <div class="img-card-title">Image Preview</div>
            <div class="preview-frame" id="preview-frame">
              <div class="preview-empty" id="preview-empty">
                <span class="preview-empty-icon">🔬</span>
                <div class="preview-empty-text">No image loaded yet</div>
              </div>
              <img id="preview-img" alt="Strip preview">
              <div class="preview-badge" id="preview-badge">ROI Annotated</div>
            </div>
          </div>
        </div>

        <!-- Run Analysis button centered below both cards -->
        <div class="btn-row-centered">
          <button class="btn-primary btn-run-main" id="analyze-btn" onclick="analyzeImage()" disabled>
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            <span id="btn-txt">Run Analysis</span>
            <div class="spinner" id="spinner"></div>
          </button>
          <button class="btn-secondary btn-reset-main" onclick="resetAll()">↺ Reset</button>
        </div>
      </div>

      <!-- RIGHT / CLASSIFICATION -->
      <div class="a-right">
        <div id="empty-state" class="empty-state">
          No analysis results yet. Upload a strip image and click Run Analysis.
        </div>

        <div id="results-wrap" class="hidden">

          <div class="result-section-head"><h3>HSV Color Measurements</h3></div>

          <div class="hsv-row">
            <div class="hsv-key">Hue (H)</div>
            <div class="hsv-bar-wrap"><div class="hsv-bar-fill hf-h" id="bar-h" style="width:0%"></div></div>
            <div class="hsv-val" id="val-h">—</div>
          </div>
          <div class="hsv-row">
            <div class="hsv-key">Saturation (S)</div>
            <div class="hsv-bar-wrap"><div class="hsv-bar-fill hf-s" id="bar-s" style="width:0%"></div></div>
            <div class="hsv-val" id="val-s">—</div>
          </div>
          <div class="hsv-row">
            <div class="hsv-key">Value (V)</div>
            <div class="hsv-bar-wrap"><div class="hsv-bar-fill hf-v" id="bar-v" style="width:0%"></div></div>
            <div class="hsv-val" id="val-v">—</div>
          </div>

          <div class="result-section-head" style="margin-top:1.5rem;"><h3>Color Spectrum Position</h3></div>

          <div class="spectrum-row">
            <div class="field-label" style="margin:0;">Hue Position</div>
            <div>
              <div class="spectrum-bar">
                <div class="spectrum-cursor" id="spectrum-cursor" style="left:0%"></div>
              </div>
              <div style="display:flex;justify-content:space-between;margin-top:.3rem;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:var(--ink4);">0°</span>
                <span style="font-family:'JetBrains Mono',monospace;font-size:.58rem;color:var(--ink4);">180°</span>
              </div>
            </div>
          </div>

          <div class="color-strip-row">
            <div class="color-strip-label">Detected Color</div>
            <div class="color-strip-inner">
              <div class="color-swatch-rect" id="det-swatch"></div>
              <div>
                <div class="color-detected-name" id="det-name">—</div>
                <div class="color-detected-codes" id="det-codes">—</div>
              </div>
            </div>
          </div>

          <div class="result-section-head" style="margin-top:1.5rem;"><h3>Risk Classification</h3></div>

          <div class="risk-grid-row">
            <div class="field-label" style="margin:0;">Risk Level</div>
            <div><div class="risk-badge" id="risk-badge"></div></div>
          </div>
          <div class="risk-grid-row">
            <div class="field-label" style="margin:0;">Assessment</div>
            <div class="risk-desc-text" id="risk-desc"></div>
          </div>
          <div class="recommendation-row">
            <div class="field-label">Recommendation</div>
            <div class="rec-content">
              <div class="rec-icon">→</div>
              <div class="rec-text" id="rec-text"></div>
            </div>
          </div>

        </div>
      </div>
    </div>

    <!-- Report -->
    <div id="report-section" style="display:none;">
      <div class="section-row">
        <div class="section-label-col"><span class="section-label">Diagnostic Report</span></div>
        <div class="section-content-col">
          <button class="btn-dl" onclick="downloadReport()">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            Export PDF Report
          </button>
        </div>
      </div>
      <div class="report-section">
        <div class="report-table-wrap">
          <div class="report-grid-header">
            <div class="rg-th">Parameter</div>
            <div class="rg-th">Value</div>
          </div>
          <div id="report-rows"></div>
        </div>
        <div class="disclaimer-bar">
          ⚠ Prototype — not for clinical use. NephroScan is a computer-vision demonstration. Always consult a qualified healthcare provider.
        </div>
      </div>
    </div>

  </div>
</div>

<!-- ════ LEARN MORE TAB ══════════════════════════════════════════════════ -->
<div id="tab-learnmore" class="tab-panel" style="background:linear-gradient(180deg,#07101f 0%,#0b1628 60%,#0d1a30 100%);min-height:100vh;">

  <!-- Hero banner -->
  <div class="lm-hero">
    <div class="lm-hero-inner">
      <div class="lm-hero-eyebrow">Explore NephroScan</div>
      <h1 class="lm-hero-title">Learn More</h1>
      <p class="lm-hero-sub">Discover the team behind NephroScan and our vision for the future of kidney health monitoring.</p>
      <!-- Sub-section jump buttons -->
      <div class="lm-jump-nav">
        <a href="#lm-about" class="lm-jump-btn" onclick="scrollToLmSection('lm-about',event)">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
          About Us
        </a>
        <a href="#lm-future" class="lm-jump-btn" onclick="scrollToLmSection('lm-future',event)">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
          Future
        </a>
        <a href="#lm-insights" class="lm-jump-btn" onclick="scrollToLmSection('lm-insights',event)">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>
          Insights
        </a>
      </div>
    </div>
  </div>

  <!-- ── About Us ───────────────────────────────────────────────────────── -->
  <section id="lm-about" class="lm-section">
    <div class="lm-section-inner">
      <div class="lm-section-eyebrow">Team</div>
      <h2 class="lm-section-title">About Us</h2>
      <p class="lm-section-sub">Meet the students building NephroScan — at the intersection of computer vision and clinical diagnostics.</p>
      <div class="team-grid">
        <div class="team-card"><div class="team-avatar">SR</div><div class="team-info"><div class="team-name">Saianish Reddy</div><div class="team-roll">SE24UCSE083</div></div></div>
        <div class="team-card"><div class="team-avatar">JS</div><div class="team-info"><div class="team-name">Jureddi Sathya Sabitha</div><div class="team-roll">SE24UCSE109</div></div></div>
        <div class="team-card"><div class="team-avatar">KN</div><div class="team-info"><div class="team-name">Kamal Nampally</div><div class="team-roll">SE24UCSE110</div></div></div>
        <div class="team-card"><div class="team-avatar">BS</div><div class="team-info"><div class="team-name">Busam Shikhar Reddy</div><div class="team-roll">SE24UCSE111</div></div></div>
        <div class="team-card"><div class="team-avatar">VM</div><div class="team-info"><div class="team-name">Vanka Mahesh Laxmi Sai Nag</div><div class="team-roll">SE24UCSE112</div></div></div>
        <div class="team-card"><div class="team-avatar">SJ</div><div class="team-info"><div class="team-name">Sai Jeevan</div><div class="team-roll">SE24UCSE236</div></div></div>
        <div class="team-card"><div class="team-avatar">AS</div><div class="team-info"><div class="team-name">Abhinav Santosh</div><div class="team-roll">SE24UCSE022</div></div></div>
        <div class="team-card"><div class="team-avatar">ML</div><div class="team-info"><div class="team-name">M. Likith</div><div class="team-roll">SE24UCSE062</div></div></div>
      </div>
    </div>
  </section>

  <!-- ── Future ─────────────────────────────────────────────────────────── -->
  <section id="lm-future" class="lm-section lm-section-alt">
    <div class="lm-section-inner">
      <div class="lm-section-eyebrow" style="color:#7dd3fc;">Vision</div>
      <h2 class="lm-section-title">Our Future Goals</h2>
      <div class="future-card">
        <div class="future-label">Road Ahead</div>
        <div class="future-body">
          Our future goal is to develop an intelligent model that continuously tracks user data over a period of time, monitors health and lifestyle patterns, and provides personalized diet recommendations along with hydration planning to improve overall wellness.
        </div>
      </div>
    </div>
  </section>

  <!-- ── Insights ────────────────────────────────────────────────────── -->
  <section id="lm-insights" class="lm-section">
    <div class="lm-section-inner">
      <div class="lm-section-eyebrow">Documentation &amp; Research</div>
      <h2 class="lm-section-title">Insights</h2>
      <p class="lm-section-sub" style="margin-bottom:2.5rem;">Explore the full academic research, technical thesis, and system workflow documentation behind NephroScan.</p>

      <!-- Thesis / Report card -->
      <div class="insights-card">
        <div class="insights-card-eyebrow">Academic Work</div>
        <h3 class="insights-card-title">Thesis / Report</h3>
        <p class="insights-card-body">
          From initial problem scoping and biomarker research to iterative hardware evaluation and full-stack web deployment, our project journey is captured in this comprehensive technical thesis. It documents every design decision, research finding, and development milestone — along with the complete architecture of the NephroScan platform and its clinical rationale.
        </p>
        <a href="https://docs.google.com/document/d/1IZGsjANnYJvwx3bX8Zwo38AxzUTsrmn2/edit?usp=sharing&ouid=100245952038616943949&rtpof=true&sd=true"
           target="_blank" rel="noopener" class="insights-btn">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
          View Thesis / Report
        </a>
      </div>

      <!-- Flowchart card -->
      <div class="insights-card" style="margin-top:1.5rem;">
        <div class="insights-card-eyebrow" style="color:#7dd3fc;">Workflow Diagrams</div>
        <h3 class="insights-card-title">Flowchart</h3>
        <p class="insights-card-body">
          These flowcharts illustrate the complete NephroScan system — from the high-level project journey and website navigation structure, through the backend image analysis pipeline, hardware evaluation decision tree, HSV color classification logic, and a step-by-step user tutorial workflow. Together they provide a clear visual map of how the system was designed, built, and operates end-to-end.
        </p>
        <a href="https://docs.google.com/document/d/1pWQoiswio5RDxMOjn3VH5v8KKhxEJeDZTWp58vFKEpk/edit?usp=sharing"
           target="_blank" rel="noopener" class="insights-btn insights-btn-alt">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex-shrink:0"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
          View Flowchart
        </a>
      </div>
    </div>
  </section>

</div>

<!-- ════ TUTORIAL TAB ═══════════════════════════════════════════════════ -->
<div id="tab-tutorial" class="tab-panel" style="background:linear-gradient(180deg,#dce8fa 0%,#e8f0fb 60%,#eef3fc 100%);min-height:100vh;">
  <div class="tut-hero">
    <div class="tut-hero-inner">
      <div class="tut-hero-eyebrow">User Guide</div>
      <h1>How to Use <em>NephroScan</em></h1>
      <p>Step-by-step protocol for capturing, uploading, and interpreting creatinine kidney test strip results accurately. The system detects three key colors — Orange/Yellow, Green, and Blue/Dark Blue — each mapped to a specific creatinine level.</p>
    </div>
  </div>

  <div class="steps-section">
    <div class="section-divider" style="padding:1.2rem 0 .8rem;border-bottom:1px solid var(--sep2);">
      <h2>Analysis Protocol — Creatinine Detection</h2>
    </div>
    <div class="steps-grid">
      <div class="step-col">
        <div class="step-num">01</div>
        <div class="step-title">Prepare Strip</div>
        <div class="step-desc">Dip in urine sample for exactly 2 seconds. Remove and hold horizontal. Wait 30–60 s before photographing.</div>
      </div>
      <div class="step-col">
        <div class="step-num">02</div>
        <div class="step-title">Photograph</div>
        <div class="step-desc">Place on white background. Use natural diffused light. Shoot directly overhead. Avoid flash or shadows.</div>
      </div>
      <div class="step-col">
        <div class="step-num">03</div>
        <div class="step-title">Upload Image</div>
        <div class="step-desc">Click the upload zone or drag and drop. Supported: PNG, JPG, BMP, WEBP. Preview appears immediately.</div>
      </div>
      <div class="step-col">
        <div class="step-num">04</div>
        <div class="step-title">Run Analysis</div>
        <div class="step-desc">Click Run Analysis. The system applies color clustering and matches the dominant color against 3 creatinine-mapped profiles: Orange/Yellow (Low), Green (Normal), Blue/Dark Blue (High).</div>
      </div>
      <div class="step-col">
        <div class="step-num">05</div>
        <div class="step-title">Review &amp; Export</div>
        <div class="step-desc">Read the risk classification and recommendation. Export a PDF diagnostic report for records or clinical review.</div>
      </div>
    </div>
  </div>

  <div class="ref-section">
    <div class="section-divider" style="padding:1.2rem 0 .8rem;border-bottom:1px solid var(--sep2);">
      <h2>Color Reference Guide</h2>
    </div>
    <table class="cref-table">
      <thead>
        <tr>
          <th style="width:48px;">Color</th>
          <th>Color Name</th>
          <th>Creatinine Level</th>
          <th>Clinical Note</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><span class="cchip" style="background:linear-gradient(135deg,#F4A020,#F7E040);border:1px solid #e0a000;"></span></td>
          <td><strong>Orange / Yellow</strong></td>
          <td><span class="rpill m">⚠ Low Creatinine</span></td>
          <td style="color:var(--ink3);font-size:.8rem">Indicates dilute or low creatinine urine (&lt; normal range). May reflect over-hydration or impaired kidney concentration. Consult a healthcare provider if persistent.</td>
        </tr>
        <tr>
          <td><span class="cchip" style="background:#3CB371;"></span></td>
          <td><strong>Green</strong></td>
          <td><span class="rpill n">✓ Normal Creatinine</span></td>
          <td style="color:var(--ink3);font-size:.8rem">Indicates moderate/normal creatinine level. Urine concentration is within the healthy expected range. No immediate action required.</td>
        </tr>
        <tr>
          <td><span class="cchip" style="background:#1A5DAB;"></span></td>
          <td><strong>Blue / Dark Blue</strong></td>
          <td><span class="rpill h">ℹ High Creatinine</span></td>
          <td style="color:var(--ink3);font-size:.8rem">Indicates high or concentrated creatinine level. May reflect dehydration or reduced kidney function. Increase water intake and seek medical evaluation if symptoms persist.</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="faq-wrap">
    <div class="section-divider" style="padding:1.2rem 0 .8rem;border-bottom:1px solid var(--sep2);margin-bottom:0;">
      <h2>Frequently Asked Questions</h2>
    </div>
    <div class="faq-item">
      <button class="faq-q" onclick="toggleFaq(this)">How accurate is NephroScan?<svg class="faq-chev" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg></button>
      <div class="faq-a">NephroScan is a <strong>prototype</strong> for demonstrating computer-vision techniques. Accuracy is affected by lighting, camera quality, and strip brand. It is not validated for clinical use.</div>
    </div>
    <div class="faq-item">
      <button class="faq-q" onclick="toggleFaq(this)">What image conditions give the best results?<svg class="faq-chev" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg></button>
      <div class="faq-a">Shoot on a plain white background under natural, diffused light (no flash). Camera directly overhead, strip filling the centre of the frame. Minimum recommended resolution: 800×600 px.</div>
    </div>
    <div class="faq-item">
      <button class="faq-q" onclick="toggleFaq(this)">Is my image stored or shared?<svg class="faq-chev" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg></button>
      <div class="faq-a">No. Images are processed in-memory on the local server and are never written to disk or transmitted to any third party. Each request is stateless.</div>
    </div>
    <div class="faq-item">
      <button class="faq-q" onclick="toggleFaq(this)">What does a High Risk result mean?<svg class="faq-chev" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg></button>
      <div class="faq-a">It indicates the detected hue falls outside the normal yellow–amber range and corresponds to colours associated with blood, bile pigments, infection, or metabolic abnormalities. It is a prompt to seek medical evaluation — not a diagnosis.</div>
    </div>
  </div>
</div>

</div><!-- /app-wrapper -->

<script>
/* ── Landing page ────────────────────────────────────────────────────────── */
function enterApp() {
  document.getElementById('landing-page').classList.add('hidden-page');
  const app = document.getElementById('app-wrapper');
  app.classList.add('visible');
  window.scrollTo({ top: 0 });
}

function goHome(e) {
  if (e) e.preventDefault();
  document.getElementById('app-wrapper').classList.remove('visible');
  document.getElementById('landing-page').classList.remove('hidden-page');
  window.scrollTo({ top: 0 });
}

/* ── Hue-shifting button ─────────────────────────────────────────────────── */
const gsBtn = document.getElementById('btn-get-started');
let hueAngle = 145; // start at green
let hueRAF = null;
let targetHue = 145;
let currentHue = 145;

gsBtn.addEventListener('mouseenter', () => {
  cancelAnimationFrame(hueRAF);
  function animateHue() {
    currentHue = (currentHue + 1.2) % 360;
    const h = currentHue;
    // derive two hsl values for gradient
    const h1 = h, h2 = (h + 30) % 360;
    gsBtn.style.background = `linear-gradient(135deg, hsl(${h1},55%,28%) 0%, hsl(${h2},60%,42%) 100%)`;
    hueRAF = requestAnimationFrame(animateHue);
  }
  animateHue();
});

gsBtn.addEventListener('mouseleave', () => {
  cancelAnimationFrame(hueRAF);
  // Smoothly return to green
  function returnToGreen() {
    const diff = 145 - currentHue;
    if (Math.abs(diff) < 1) {
      currentHue = 145;
      gsBtn.style.background = `linear-gradient(135deg, var(--g-700), var(--g-500))`;
      return;
    }
    // shortest path around the circle
    let step = diff > 0 ? Math.min(diff, 3) : Math.max(diff, -3);
    // handle wrap-around
    if (Math.abs(diff) > 180) step = -step;
    currentHue = (currentHue + step + 360) % 360;
    const h1 = currentHue, h2 = (currentHue + 30) % 360;
    gsBtn.style.background = `linear-gradient(135deg, hsl(${h1},55%,28%) 0%, hsl(${h2},60%,42%) 100%)`;
    hueRAF = requestAnimationFrame(returnToGreen);
  }
  returnToGreen();
});

/* ── Tab switching ───────────────────────────────────────────────────────── */
function switchTab(id, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  btn.classList.add('active');
}

/* ── FAQ ─────────────────────────────────────────────────────────────────── */
function toggleFaq(btn) { btn.closest('.faq-item').classList.toggle('open'); }

/* ── File handling ───────────────────────────────────────────────────────── */
const dz = document.getElementById('drop-zone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => {
  e.preventDefault(); dz.classList.remove('drag');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) setFile(f);
});
document.getElementById('file-input').addEventListener('change', e => {
  if (e.target.files[0]) setFile(e.target.files[0]);
});

function setFile(f) {
  window._file = f;
  const reader = new FileReader();
  reader.onload = ev => {
    const img = document.getElementById('preview-img');
    img.src = ev.target.result; img.style.display = 'block';
    document.getElementById('preview-empty').style.display = 'none';
  };
  reader.readAsDataURL(f);
  document.getElementById('analyze-btn').disabled = false;
  resetResults();
}

function resetResults() {
  document.getElementById('empty-state').classList.remove('hidden');
  document.getElementById('results-wrap').classList.add('hidden');
  document.getElementById('report-section').style.display = 'none';
  document.getElementById('preview-badge').style.display = 'none';
  window._result = null;
}

function resetAll() {
  window._file = null;
  document.getElementById('file-input').value = '';
  const img = document.getElementById('preview-img');
  img.src = ''; img.style.display = 'none';
  document.getElementById('preview-empty').style.display = 'flex';
  document.getElementById('analyze-btn').disabled = true;
  resetResults();
}

/* ── Analyze ─────────────────────────────────────────────────────────────── */
async function analyzeImage() {
  if (!window._file) return;
  const btn = document.getElementById('analyze-btn');
  const sp  = document.getElementById('spinner');
  const txt = document.getElementById('btn-txt');
  const prg = document.getElementById('progress');
  btn.disabled = true; sp.style.display = 'block';
  txt.textContent = 'Analyzing…'; prg.style.width = '45%';
  document.getElementById('analysis-overlay').classList.add('active');
  const fd = new FormData(); fd.append('image', window._file);
  try {
    const res = await fetch('/flask/analyze', { method: 'POST', body: fd });
    prg.style.width = '88%';
    const data = await res.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    window._result = data;
    if (data.strip_detected === false) { showStripWarning(); return; }
    renderResults(data);
    prg.style.width = '100%';
    setTimeout(() => prg.style.width = '0', 600);
  } catch(e) {
    alert('Request failed: ' + e.message);
  } finally {
    btn.disabled = false; sp.style.display = 'none';
    txt.textContent = 'Run Analysis';
    document.getElementById('analysis-overlay').classList.remove('active');
  }
}

/* ── Render results ──────────────────────────────────────────────────────── */
function renderResults(d) {
  if (d.annotated_image) {
    const img = document.getElementById('preview-img');
    img.src = d.annotated_image; img.style.display = 'block';
    document.getElementById('preview-empty').style.display = 'none';
    document.getElementById('preview-badge').style.display = 'block';
  }
  document.getElementById('empty-state').classList.add('hidden');
  document.getElementById('results-wrap').classList.remove('hidden');
  const h = d.hsv.h, s = d.hsv.s, v = d.hsv.v;
  setTimeout(() => {
    document.getElementById('bar-h').style.width = Math.round(h/180*100)+'%';
    document.getElementById('bar-s').style.width = Math.round(s/255*100)+'%';
    document.getElementById('bar-v').style.width = Math.round(v/255*100)+'%';
    document.getElementById('spectrum-cursor').style.left = Math.round(h/180*100)+'%';
  }, 60);
  document.getElementById('val-h').textContent = h + ' / 180';
  document.getElementById('val-s').textContent = s + ' / 255';
  document.getElementById('val-v').textContent = v + ' / 255';
  document.getElementById('det-swatch').style.background = d.hex;
  document.getElementById('det-name').textContent = d.color_name;
  document.getElementById('det-codes').textContent = d.hex.toUpperCase() + ' · RGB(' + d.rgb.r + ', ' + d.rgb.g + ', ' + d.rgb.b + ')';
  const cssMap  = { Normal:'normal', 'Moderate Risk':'moderate', 'High Risk':'high' };
  const iconMap = { Normal:'✓', 'Moderate Risk':'⚠', 'High Risk':'✗' };
  const css = cssMap[d.risk] || 'normal';
  const badge = document.getElementById('risk-badge');
  badge.className = 'risk-badge ' + css;
  badge.textContent = (iconMap[d.risk]||'?') + '  ' + d.risk;
  document.getElementById('risk-desc').textContent = d.risk_description;
  document.getElementById('rec-text').textContent = d.recommendation;
  const fields = [
    ['Detected Color', d.color_name],
    ['Risk Level',     d.risk],
    ['Hex Color',      d.hex.toUpperCase()],
    ['RGB Values',     'R:'+d.rgb.r+'  G:'+d.rgb.g+'  B:'+d.rgb.b],
    ['Hue (H)',        h+' / 180'],
    ['Saturation (S)', s+' / 255'],
    ['Value (V)',      v+' / 255'],
    ['Recommendation', d.recommendation],
  ];
  document.getElementById('report-rows').innerHTML = fields.map(([k,v2]) => {
    if (k === 'Recommendation') {
      return `<div class="report-row-rec">
        <span class="rg-key">&#9656; Recommendation</span>
        <div class="rg-val-rec">${v2}</div>
      </div>`;
    }
    return `<div class="report-row">
      <div class="rg-key">${k}</div>
      <div class="rg-val">${v2}</div>
    </div>`;
  }).join('');
  document.getElementById('report-section').style.display = 'block';
}

/* ── Learn More in-page scroll ──────────────────────────────────────────── */
function scrollToLmSection(id, e) {
  if (e) e.preventDefault();
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── Download PDF ────────────────────────────────────────────────────────── */
function downloadReport() {
  const d = window._result; if (!d) return;
  const rC = { Normal:'#166534', 'Moderate Risk':'#92400e', 'High Risk':'#991b1b' };
  const rB = { Normal:'#dcfce7', 'Moderate Risk':'#fef3c7', 'High Risk':'#fee2e2' };
  const rL = { Normal:'#86efac', 'Moderate Risk':'#fcd34d', 'High Risk':'#fca5a5' };
  const now = new Date().toLocaleString();
  const html = `<!DOCTYPE html><html><head><meta charset="UTF-8"><title>NephroScan Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:'Inter',Arial,sans-serif;margin:0;padding:44px 52px;color:#0d0d1a;background:#fff;font-size:13px;line-height:1.6;}
  /* Header */
  .hdr{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:2.5px solid #1a3a8f;padding-bottom:16px;margin-bottom:28px;}
  .brand{font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:700;color:#1a3a8f;letter-spacing:-.02em;}
  .sub{font-size:8.5px;color:#6070a0;letter-spacing:.12em;text-transform:uppercase;margin-top:4px;font-family:'Inter',sans-serif;}
  .meta{text-align:right;font-family:'Inter',monospace;font-size:8.5px;color:#8090b0;line-height:1.9;}
  /* Risk banner */
  .risk-box{padding:14px 18px;border-radius:6px;margin-bottom:22px;background:${rB[d.risk]};border-left:4px solid ${rL[d.risk]};}
  .risk-lv{font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:700;color:${rC[d.risk]};margin-bottom:5px;letter-spacing:.01em;}
  .risk-d{font-size:11px;color:${rC[d.risk]};line-height:1.55;font-weight:400;}
  /* Swatch */
  .swatch-row{display:flex;align-items:center;gap:14px;margin-bottom:22px;padding-bottom:18px;border-bottom:1px solid #dde3f0;}
  .sw{width:44px;height:28px;border-radius:4px;background:${d.hex};border:1px solid rgba(0,0,0,.1);box-shadow:0 1px 4px rgba(0,0,0,.08);}
  .sw-label{font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:700;color:#0d0d1a;}
  .sw-sub{font-size:10px;color:#7080a0;font-family:'Inter',monospace;margin-top:2px;letter-spacing:.03em;}
  /* Section heading */
  .sec-h{font-family:'Space Grotesk',sans-serif;font-size:8px;font-weight:700;text-transform:uppercase;letter-spacing:.15em;color:#4060b0;margin-bottom:0;padding-bottom:7px;border-bottom:1.5px solid #dde3f0;}
  /* Data table */
  table{width:100%;border-collapse:collapse;margin-bottom:22px;margin-top:0;}
  tr{border-bottom:1px solid #eaedf5;}
  tr:last-child{border-bottom:none;}
  tr:nth-child(even){background:#f7f8fc;}
  td{padding:9px 10px;vertical-align:middle;}
  /* LEFT column — bold black labels */
  td:first-child{
    font-family:'Space Grotesk',sans-serif;
    font-size:10px;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:.09em;
    color:#0a0a18;
    width:36%;
    border-right:1px solid #eaedf5;
  }
  /* RIGHT column — lighter values */
  td:last-child{
    font-family:'Inter',sans-serif;
    font-size:12px;
    font-weight:400;
    color:#2a2a3e;
    padding-left:16px;
  }
  /* Recommendation row special */
  .rec-label{font-family:'Space Grotesk',sans-serif;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.09em;color:#0a0a18;}
  .rec-val{font-family:'Inter',sans-serif;font-size:11.5px;font-weight:400;color:#1a1a2e;line-height:1.7;border-left:3px solid #4060b0;padding:8px 12px;background:#f4f6ff;border-radius:0 5px 5px 0;margin-top:6px;}
  /* Disclaimer */
  .disc{margin-top:18px;padding:10px 14px;border-left:3px solid #f59e0b;background:#fffbeb;font-size:10px;line-height:1.7;color:#92400e;border-radius:0 5px 5px 0;}
  /* Footer */
  .foot{margin-top:24px;padding-top:12px;border-top:1px solid #dde3f0;font-size:8.5px;color:#8090b0;text-align:center;letter-spacing:.07em;font-family:'Inter',sans-serif;}
</style></head><body>
<div class="hdr">
  <div><div class="brand">NephroScan</div><div class="sub">Clinical Strip Analyzer — Diagnostic Report</div></div>
  <div class="meta">Generated: ${now}<br>v10.0 · Prototype</div>
</div>
<div class="risk-box">
  <div class="risk-lv">${d.risk.toUpperCase()}</div>
  <div class="risk-d">${d.risk_description}</div>
</div>
<div class="swatch-row">
  <div class="sw"></div>
  <div>
    <div class="sw-label">${d.color_name}</div>
    <div class="sw-sub">${d.hex.toUpperCase()} &nbsp;·&nbsp; RGB(${d.rgb.r}, ${d.rgb.g}, ${d.rgb.b})</div>
  </div>
</div>
<div class="sec-h">Color Analysis Data</div>
<table>
  <tr><td>Detected Color</td><td>${d.color_name}</td></tr>
  <tr><td>Risk / Creatinine Level</td><td style="color:${rC[d.risk]||'#1a3a8f'};font-weight:600;">${d.risk}</td></tr>
  <tr><td>Hex Color</td><td style="font-family:'Inter',monospace;">${d.hex.toUpperCase()}</td></tr>
  <tr><td>RGB Values</td><td style="font-family:'Inter',monospace;">R: ${d.rgb.r} &nbsp; G: ${d.rgb.g} &nbsp; B: ${d.rgb.b}</td></tr>
  <tr><td>Hue (H)</td><td>${d.hsv.h} / 180</td></tr>
  <tr><td>Saturation (S)</td><td>${d.hsv.s} / 255</td></tr>
  <tr><td>Value (V)</td><td>${d.hsv.v} / 255</td></tr>
</table>
<div class="sec-h" style="margin-bottom:8px;">Recommendation</div>
<div class="rec-val">${d.recommendation}</div>
<div class="disc" style="margin-top:18px;">⚠ &nbsp;<strong>Prototype — not for clinical use.</strong> NephroScan is a computer-vision demonstration only. Always consult a qualified healthcare provider.</div>
<div class="foot">NEPHROSCAN v10.0 &nbsp;·&nbsp; CLINICAL STRIP ANALYZER &nbsp;·&nbsp; PROTOTYPE &nbsp;·&nbsp; NOT FOR CLINICAL USE</div>
</body></html>`;
  const w = window.open('','_blank');
  if (!w) { alert('Please allow pop-ups to export the report.'); return; }
  w.document.write(html); w.document.close();
  w.addEventListener('load', () => setTimeout(() => w.print(), 200));
}
</script>

<!-- ── Analysis loading overlay ─────────────────────────────────────── -->
<div id="analysis-overlay" aria-live="polite">
  <div class="overlay-spinner-ring"></div>
  <div class="overlay-label">Analyzing<span class="overlay-dots"></span></div>
</div>

<!-- ── Strip-not-detected warning popup ──────────────────────────── -->
<div id="strip-warning-backdrop" onclick="closeStripWarning()" style="display:none;position:fixed;inset:0;background:rgba(6,14,32,.55);backdrop-filter:blur(3px);z-index:9000;align-items:center;justify-content:center;"></div>
<div id="strip-warning-modal" style="display:none;position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:9001;background:#fff;border-radius:14px;box-shadow:0 24px 64px rgba(0,0,0,.28);padding:36px 32px 28px;max-width:420px;width:90%;text-align:center;">
  <div style="font-size:2.4rem;margin-bottom:12px;">&#9888;&#65039;</div>
  <h3 style="font-family:'Space Grotesk',sans-serif;font-size:1.1rem;font-weight:700;color:#0d0d1a;margin-bottom:12px;">Unrecognized Strip Detected</h3>
  <p style="font-size:.9rem;color:#4a5570;line-height:1.65;margin-bottom:24px;">Unrecognized strip detected. Please upload a clearer image with proper lighting and zoom in on the test strip.</p>
  <button onclick="closeStripWarning()" style="background:#1a3a8f;color:#fff;border:none;border-radius:8px;padding:10px 28px;font-size:.9rem;font-family:'Space Grotesk',sans-serif;font-weight:600;cursor:pointer;letter-spacing:.02em;">Try Again</button>
</div>

<script>
function showStripWarning() {
  document.getElementById('strip-warning-backdrop').style.display = 'flex';
  document.getElementById('strip-warning-modal').style.display   = 'block';
}
function closeStripWarning() {
  document.getElementById('strip-warning-backdrop').style.display = 'none';
  document.getElementById('strip-warning-modal').style.display   = 'none';
}
document.addEventListener('keydown', function(e){ if(e.key==='Escape') closeStripWarning(); });
</script>

</body>
</html>
"""

@app.route("/")
def root():
    return redirect("/flask/")

@app.route("/flask/")
def index():
    return render_template_string(HTML)

@app.route("/flask/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    np_arr = np.frombuffer(request.files["image"].read(), np.uint8)
    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "Could not decode image"}), 400
    try:
        result = extract_dominant_hsv(bgr)
        if result is None:
            # Not enough coloured pixels — strip not detected
            return jsonify({
                "hsv":              {"h": 0, "s": 0, "v": 0},
                "rgb":              {"r": 180, "g": 180, "b": 180},
                "hex":              "#AAAAAA",
                "color_name":       "Unrecognized",
                "risk":             "Unrecognized",
                "risk_description": RISK_META["Unrecognized"]["description"],
                "recommendation":   RECOMMENDATIONS["Unrecognized"],
                "annotated_image":  image_to_b64(bgr),
                "strip_detected":   False,
            })
        h, s, v = result
        cls     = classify_color(h, s, v)
        ann     = annotate_image(bgr, h, s, v)
        r, g, b = hsv_to_rgb(h, s, v)
        risk    = cls["risk"]
        return jsonify({
            "hsv":              {"h": int(h), "s": int(s), "v": int(v)},
            "rgb":              {"r": r, "g": g, "b": b},
            "hex":              cls["hex"],
            "color_name":       cls["color_name"],
            "risk":             risk,
            "risk_description": RISK_META[risk]["description"],
            "recommendation":   RECOMMENDATIONS[risk],
            "annotated_image":  image_to_b64(ann),
            "strip_detected":   risk != "Unrecognized",
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🧪  NephroScan v10 →  http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
