"""Erzeugt eine eigenständige HTML-Animation mit Slider und LaTeX-Formel.

Voraussetzung: pip install numpy
Start: python3 minimal_animation.py
Die Formel wird mit MathJax gerendert (Internetverbindung erforderlich).
"""
import json
from pathlib import Path
import numpy as np


def f(x, a):
    """Hier kannst du deine eigene Python-Funktion einsetzen."""
    return np.exp(a * x) *np.sin(1/a * x)

# Einstellungen
x_min, x_max = -3.0, 3.0
slider_min, slider_max = 0.01, 1.0
x = np.linspace(x_min, x_max, 301)
parameters = np.linspace(slider_min, slider_max, 81)
latex_formula = r"f_a(x)=e^{ax}\sin\left(\frac{1}{a}x\right)"  # LaTeX-Beschriftung über dem Graphen

curves = [np.asarray(f(x, float(a)), dtype=float).tolist() for a in parameters]
all_y = np.asarray(curves, dtype=float)
if not np.all(np.isfinite(all_y)):
    raise ValueError("Die Funktionswerte müssen endlich sein (kein NaN/Inf).")
y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
padding = 0.1 * (y_max - y_min) if y_max > y_min else max(1.0, abs(y_min)*0.1)
y_min -= padding
y_max += padding

html = r'''<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Interaktive mathematische Funktion</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; padding: 18px; font: 16px system-ui, sans-serif; background: #fff; color: #222; }
    main { width: min(100%, 680px); margin: auto; }
    .formula { text-align: center; font-size: 22px; min-height: 56px; padding: 8px 0 14px; }
    canvas { display: block; width: 100%; height: auto; }
    .controls { display: flex; gap: 14px; align-items: center; padding: 0 28px 10px 48px; }
    input { flex: 1; accent-color: #2563eb; cursor: pointer; min-width: 0; }
    output { font-variant-numeric: tabular-nums; min-width: 76px; text-align: right; }
  </style>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head>
<body>
<main>
  <div class="formula" id="formula">\[__FORMULA__\]</div>
  <canvas id="graph" width="680" height="410" aria-label="Graph der Funktion"></canvas>
  <div class="controls">
    <label for="slider">a</label>
    <input id="slider" type="range" min="0" max="__SLIDER_MAX__" step="1" value="__SLIDER_INITIAL__">
    <output id="value" for="slider"></output>
  </div>
</main>
<script>
  const xs = __XS__;
  const parameters = __PARAMETERS__;
  const curves = __CURVES__;
  const canvas = document.getElementById('graph');
  const ctx = canvas.getContext('2d');
  const slider = document.getElementById('slider');
  const label = document.getElementById('value');
  const W = canvas.width, H = canvas.height;
  const margin = {left: 62, right: 22, top: 20, bottom: 36};
  const xmin = __XMIN__, xmax = __XMAX__, ymin = __YMIN__, ymax = __YMAX__;
  const X = x => margin.left + (x - xmin)/(xmax-xmin)*(W-margin.left-margin.right);
  const Y = y => H-margin.bottom - (y-ymin)/(ymax-ymin)*(H-margin.top-margin.bottom);
  function draw() {
    const i = Number(slider.value);
    label.textContent = parameters[i].toFixed(2);
    ctx.clearRect(0, 0, W, H);
    ctx.font = '13px system-ui, sans-serif';
    ctx.lineWidth = 1;
    const yStep = (ymax-ymin)/5;
    for(let k=0; k<=5; k++) {
      const y = ymin+k*yStep;
      ctx.strokeStyle = '#e5e7eb';
      ctx.beginPath(); ctx.moveTo(X(xmin),Y(y)); ctx.lineTo(X(xmax),Y(y)); ctx.stroke();
      ctx.fillStyle = '#666'; ctx.textAlign = 'right';
      ctx.fillText(Number(y.toPrecision(3)).toString(), X(xmin)-9, Y(y)+4);
    }
    const xStep = (xmax-xmin)/6;
    for(let k=0; k<=6; k++) {
      const x = xmin + k*xStep;
      ctx.strokeStyle = '#e5e7eb';
      ctx.beginPath(); ctx.moveTo(X(x),Y(ymin)); ctx.lineTo(X(x),Y(ymax)); ctx.stroke();
      ctx.fillStyle = '#666'; ctx.textAlign = 'center';
      ctx.fillText(Number(x.toPrecision(3)).toString(), X(x), Y(ymin)+22);
    }
    ctx.strokeStyle = '#666'; ctx.lineWidth = 1.4;
    if(xmin<=0 && xmax>=0) {
      ctx.beginPath(); ctx.moveTo(X(0),Y(ymin)); ctx.lineTo(X(0),Y(ymax)); ctx.stroke();
    }
    if(ymin<=0 && ymax>=0) {
      ctx.beginPath(); ctx.moveTo(X(xmin),Y(0)); ctx.lineTo(X(xmax),Y(0)); ctx.stroke();
    }
    ctx.save();
    ctx.beginPath(); ctx.rect(X(xmin),Y(ymax),X(xmax)-X(xmin),Y(ymin)-Y(ymax)); ctx.clip();
    ctx.strokeStyle = '#2563eb'; ctx.lineWidth = 3; ctx.lineJoin = 'round';
    ctx.beginPath();
    curves[i].forEach((y,j) => {
      if(j===0) ctx.moveTo(X(xs[j]),Y(y)); else ctx.lineTo(X(xs[j]),Y(y));
    });
    ctx.stroke(); ctx.restore();
  }
  slider.addEventListener('input', draw);
  draw();
</script>
</body>
</html>
'''

replacements = {
    '__FORMULA__': latex_formula,
    '__XS__': json.dumps(x.tolist()),
    '__PARAMETERS__': json.dumps(parameters.tolist()),
    '__CURVES__': json.dumps(curves, separators=(',', ':')),
    '__XMIN__': json.dumps(x_min),
    '__XMAX__': json.dumps(x_max),
    '__YMIN__': json.dumps(y_min),
    '__YMAX__': json.dumps(y_max),
    '__SLIDER_MAX__': str(len(parameters)-1),
    '__SLIDER_INITIAL__': str(int(np.argmin(abs(parameters)))),
}
for placeholder, value in replacements.items():
    html = html.replace(placeholder, value)

output = Path(__file__).with_name('minimal_animation.html')
output.write_text(html, encoding='utf-8')
print(f'Erstellt: {output}')
