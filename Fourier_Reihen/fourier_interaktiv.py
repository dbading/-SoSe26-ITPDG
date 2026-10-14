"""Erstellt eine eigenständige HTML-Animation der Fourier-Partialsummen.

Benötigt nur NumPy: python3 -m pip install numpy
Aufruf: python3 fourier_interaktiv.py
Danach fourier_interaktiv.html im Browser öffnen.
"""
import json
from pathlib import Path
import numpy as np 


def parameters():
    # Diese Einstellungen stammen aus deiner ursprünglichen Animation.
    return {
        "l": -1.0,
        "r": 1.0,
        # "f": lambda x: x,
        "f": lambda x: np.where( x < 0, 1, 0),  # Die Funktion, die approximiert werden soll.
        "n_terms": 20,
        "n_plot": 1200,
        "n_integration": 5000,
        "formula_html": "f(x) = x",  # Nur die Beschriftung, unabhängig von f.
    }


def compute_fourier_coefficients(f, l, r, n_terms, n_integration=5000):
    """Fourier-Koeffizienten gemäß deiner Originalimplementierung."""
    L = r - l
    t = np.linspace(l, r, n_integration, endpoint=False)
    ft = f(t)
    phase_t = 2 * np.pi * (t - l) / L

    a0 = (2.0 / L) * np.trapezoid(ft, t)
    a = np.zeros(n_terms + 1)
    b = np.zeros(n_terms + 1)
    for n in range(1, n_terms + 1):
        a[n] = (2.0 / L) * np.trapezoid(ft * np.cos(n * phase_t), t)
        b[n] = (2.0 / L) * np.trapezoid(ft * np.sin(n * phase_t), t)
    return a0, a, b


def build_partial_sums(x, coeffs, l, r):
    """S_0, S_1, ..., S_N aus deiner Originalimplementierung."""
    a0, a, b = coeffs
    L = r - l
    phase_x = 2 * np.pi * (x - l) / L
    current = np.full_like(x, a0 / 2.0, dtype=float)
    partial_sums = [current.copy()]
    for n in range(1, len(a)):
        current = current + a[n] * np.cos(n * phase_x) + b[n] * np.sin(n * phase_x)
        partial_sums.append(current.copy())
    return partial_sums


def main():
    params = parameters()
    l, r = params["l"], params["r"]
    x = np.linspace(l, r, params["n_plot"])
    f_x = np.asarray(params["f"](x), dtype=float)
    coeffs = compute_fourier_coefficients(
        params["f"], l, r, params["n_terms"], params["n_integration"]
    )
    partial_sums = build_partial_sums(x, coeffs, l, r)

    # Einheitliche Achsen über ALLE Approximationsschritte hinweg.
    all_min = min(float(f_x.min()), *(float(s.min()) for s in partial_sums))
    all_max = max(float(f_x.max()), *(float(s.max()) for s in partial_sums))
    padding = 0.1 * (all_max - all_min) if all_max > all_min else 1.0
    ymin, ymax = all_min - padding, all_max + padding

    # JSON-Payload kompakt halten und gleichzeitig ausreichende Genauigkeit bewahren.
    def nums(values):
        return [round(float(v), 8) for v in values]

    substitutions = {
        "__XS__": json.dumps(nums(x), separators=(",", ":")),
        "__ORIGINAL__": json.dumps(nums(f_x), separators=(",", ":")),
        "__CURVES__": json.dumps([nums(s) for s in partial_sums], separators=(",", ":")),
        "__XMIN__": json.dumps(l),
        "__XMAX__": json.dumps(r),
        "__YMIN__": json.dumps(ymin),
        "__YMAX__": json.dumps(ymax),
        "__MAX_N__": json.dumps(params["n_terms"]),
        "__FORMULA__": params["formula_html"],
    }

    html = r'''<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fourier-Reihe – interaktive Approximation</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; padding: 16px; font: 16px system-ui, sans-serif; background: #fff; color: #202124; }
    main { width: min(100%, 720px); margin: auto; }
    .formula { text-align: center; font: italic 22px Georgia, serif; margin: 4px 0 8px; }
    canvas { display: block; width: 100%; height: auto; }
    .legend { display: flex; justify-content: center; gap: 22px; flex-wrap: wrap; font-size: 13px; margin-top: 2px; }
    .swatch { display: inline-block; width: 22px; height: 3px; vertical-align: middle; margin-right: 6px; }
    .controls { display: flex; gap: 14px; align-items: center; padding: 22px 30px 12px 52px; }
    input[type=range] { flex: 1; min-width: 0; accent-color: #2563eb; cursor: pointer; }
    output { min-width: 58px; text-align: right; font-variant-numeric: tabular-nums; }
  </style>
</head>
<body>
<main>
  <div class="formula">__FORMULA__</div>
  <canvas id="graph" width="720" height="430" aria-label="Originalfunktion und Fourier-Approximation"></canvas>
  <div class="legend">
    <span><span class="swatch" style="background:#6b7280"></span>Originalfunktion</span>
    <span><span class="swatch" style="background:#2563eb"></span>Fourier-Approximation</span>
  </div>
  <div class="controls">
    <label for="slider">N</label>
    <input id="slider" type="range" min="0" max="__MAX_N__" step="1" value="0">
    <output id="value" for="slider">0</output>
  </div>
</main>
<script>
  // Die Werte stammen vollständig aus dem Python-Skript.
  const xs = __XS__;
  const original = __ORIGINAL__;
  const curves = __CURVES__;
  const xmin = __XMIN__, xmax = __XMAX__;
  const ymin = __YMIN__, ymax = __YMAX__;

  const canvas = document.getElementById('graph');
  const ctx = canvas.getContext('2d');
  const slider = document.getElementById('slider');
  const label = document.getElementById('value');
  const W = canvas.width, H = canvas.height;
  const margin = {left: 62, right: 25, top: 18, bottom: 44};
  const X = x => margin.left + (x - xmin) / (xmax - xmin) * (W - margin.left - margin.right);
  const Y = y => H - margin.bottom - (y - ymin) / (ymax - ymin) * (H - margin.top - margin.bottom);

  function tidy(v) { return Math.abs(v) < 1e-9 ? '0' : Number(v.toFixed(2)).toString(); }

  function drawLine(values, color, width) {
    ctx.beginPath();
    for (let j = 0; j < xs.length; j++) {
      if (j === 0) ctx.moveTo(X(xs[j]), Y(values[j]));
      else ctx.lineTo(X(xs[j]), Y(values[j]));
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.stroke();
  }

  function draw() {
    const n = Number(slider.value);
    label.textContent = String(n);
    ctx.clearRect(0, 0, W, H);
    ctx.font = '13px system-ui, sans-serif';
    ctx.lineWidth = 1;

    // Automatisch beschriftete, FESTE Achsen.
    for (let k = 0; k <= 6; k++) {
      const y = ymin + (ymax - ymin) * k / 6;
      ctx.strokeStyle = '#e5e7eb';
      ctx.beginPath(); ctx.moveTo(X(xmin), Y(y)); ctx.lineTo(X(xmax), Y(y)); ctx.stroke();
      ctx.fillStyle = '#6b7280'; ctx.textAlign = 'right';
      ctx.fillText(tidy(y), X(xmin) - 10, Y(y) + 4);
    }
    for (let k = 0; k <= 6; k++) {
      const x = xmin + (xmax - xmin) * k / 6;
      ctx.strokeStyle = '#e5e7eb';
      ctx.beginPath(); ctx.moveTo(X(x), Y(ymin)); ctx.lineTo(X(x), Y(ymax)); ctx.stroke();
      ctx.fillStyle = '#6b7280'; ctx.textAlign = 'center';
      ctx.fillText(tidy(x), X(x), Y(ymin) + 20);
    }
    ctx.strokeStyle = '#858585'; ctx.lineWidth = 1.2;
    if (xmin <= 0 && xmax >= 0) {
      ctx.beginPath(); ctx.moveTo(X(0), Y(ymin)); ctx.lineTo(X(0), Y(ymax)); ctx.stroke();
    }
    if (ymin <= 0 && ymax >= 0) {
      ctx.beginPath(); ctx.moveTo(X(xmin), Y(0)); ctx.lineTo(X(xmax), Y(0)); ctx.stroke();
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(X(xmin), Y(ymax), X(xmax)-X(xmin), Y(ymin)-Y(ymax));
    ctx.clip();
    drawLine(original, '#6b7280', 2.5);
    drawLine(curves[n], '#2563eb', 3);
    ctx.restore();
  }

  slider.addEventListener('input', draw);
  draw();
</script>
</body>
</html>
'''
    for key, value in substitutions.items():
        html = html.replace(key, value)

    output = Path(__file__).with_name("fourier_interaktiv.html")
    output.write_text(html, encoding="utf-8")
    print(f"Erstellt: {output} ({output.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
