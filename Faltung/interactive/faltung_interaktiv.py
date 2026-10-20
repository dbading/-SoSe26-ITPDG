"""Erzeugt faltung_interaktiv.html als eigenständige, interaktive Webseite.

Basierend auf Faltung.py: oben f(t), g(x-t) und das Produkt,
unten die Faltung (f*g)(x). Der Slider verschiebt g von links nach rechts.

Installation: python3 -m pip install numpy
Start:        python3 faltung_interaktiv.py
Anschließend faltung_interaktiv.html im Browser öffnen.

Die Funktionen und Grenzen lassen sich in parameters() anpassen.
Die erstellte HTML-Datei benötigt weder Python noch Internet.
"""
import json
from pathlib import Path
import numpy as np


def M_1(x):
    return np.where(np.abs(x) < 1, 1 - np.abs(x), 0.0)


def M_k(k, x):
    """Gleiche numerische Faltung wie im Original Faltung.py."""
    if k == 1:
        return M_1(x)
    return np.convolve(M_1(x), M_k(k - 1, x), mode="same") * (x[1] - x[0])

Dict_of_functions = {
    "M_1": M_1,
    "M_2": lambda x: M_k(2, x),
    "M_3": lambda x: M_k(3, x),
    "M_4": lambda x: M_k(4, x),
    "M_5": lambda x: M_k(5, x),
}

def parameters():
    return {
        "f_l": -5.0, "f_r": 5.0,
        "f": Dict_of_functions["M_2"],  # Die Funktion, die gefaltet werden soll.
        "g_l": -1.0, "g_r": 1.0,
        "g": lambda x: 1.0,
        "x_min": -3.5, "x_max": 3.5,
        "n_curve": 601,
        "n_integration": 1200,
        "n_t_plot": 1000,
        "n_g_samples": 2501,
        "formula_html": "(f * g)(x) = ∫ f(t) · g(x − t) dt",
    }


def supported(func, x, left, right):
    x = np.asarray(x, dtype=float)
    result = np.asarray(func(x), dtype=float)
    if result.ndim == 0:
        result = np.full_like(x, float(result))
    else:
        result = np.broadcast_to(result, x.shape)
    return np.where((x >= left) & (x <= right), result, 0.0)


def compute_convolution(p, shifts):
    t = np.linspace(p["f_l"], p["f_r"], p["n_integration"])
    ft = supported(p["f"], t, p["f_l"], p["f_r"])
    result = np.empty_like(shifts)
    for start in range(0, len(shifts), 100):
        batch = shifts[start:start + 100, None]
        gt = supported(p["g"], batch - t[None, :], p["g_l"], p["g_r"])
        result[start:start + len(batch)] = np.trapezoid(ft[None, :] * gt, t, axis=1)
    return result


def main():
    p = parameters()
    shifts = np.linspace(p["x_min"], p["x_max"], p["n_curve"])
    t_min = min(p["f_l"], p["x_min"] - p["g_r"]) - 0.25
    t_max = max(p["f_r"], p["x_max"] - p["g_l"]) + 0.25
    ts = np.linspace(t_min, t_max, p["n_t_plot"])
    ft = supported(p["f"], ts, p["f_l"], p["f_r"])
    us = np.linspace(p["g_l"], p["g_r"], p["n_g_samples"])
    gu = supported(p["g"], us, p["g_l"], p["g_r"])
    convolution = compute_convolution(p, shifts)
    top_extent = max(1e-8, np.max(np.abs(ft)), np.max(np.abs(gu)), np.max(np.abs(ft)) * np.max(np.abs(gu)))
    bottom_min = min(0.0, float(np.min(convolution)))
    bottom_max = max(0.0, float(np.max(convolution)))
    bottom_pad = max(0.08, (bottom_max - bottom_min) * 0.15)

    def numlist(a):
        return [round(float(x), 7) for x in a]

    replacements = {
        "__T__": json.dumps(numlist(ts)),
        "__F__": json.dumps(numlist(ft)),
        "__U__": json.dumps(numlist(us)),
        "__G__": json.dumps(numlist(gu)),
        "__SHIFTS__": json.dumps(numlist(shifts)),
        "__CONV__": json.dumps(numlist(convolution)),
        "__TMIN__": json.dumps(t_min),
        "__TMAX__": json.dumps(t_max),
        "__XMIN__": json.dumps(p["x_min"]),
        "__XMAX__": json.dumps(p["x_max"]),
        "__GL__": json.dumps(p["g_l"]),
        "__GR__": json.dumps(p["g_r"]),
        "__TOPMIN__": json.dumps(-0.16 * top_extent if np.min(ft) >= 0 and np.min(gu) >= 0 else -1.16 * top_extent),
        "__TOPMAX__": json.dumps(1.16 * top_extent),
        "__BOTMIN__": json.dumps(bottom_min - bottom_pad),
        "__BOTMAX__": json.dumps(bottom_max + bottom_pad),
        "__FORMULA__": p["formula_html"],
    }

    html = r'''<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Faltung – interaktive Animation</title>
<style>
*{box-sizing:border-box}
body{margin:0;padding:16px;font:16px system-ui,-apple-system,sans-serif;background:#fff;color:#202124}
main{width:min(100%,780px);margin:auto}
.formula{text-align:center;font:italic 21px Georgia,serif;margin:5px 0 15px}
.caption{font-size:14px;color:#374151;margin:14px 0 2px;font-weight:600}
canvas{display:block;width:100%;height:auto}
.legend{display:flex;justify-content:center;flex-wrap:wrap;gap:10px 22px;font-size:13px;margin:5px 0 12px}
.swatch{display:inline-block;width:20px;height:3px;vertical-align:middle;margin-right:6px}
.controls{display:flex;align-items:center;gap:12px;padding:20px 14px 8px}
input[type=range]{flex:1;min-width:0;accent-color:#2563eb;cursor:pointer;touch-action:pan-y}
output{min-width:64px;text-align:right;font-variant-numeric:tabular-nums}
.hint{color:#6b7280;text-align:center;font-size:12px;margin:6px 0 0}
</style>
</head>
<body><main>
<div class="formula">__FORMULA__</div>
<div class="caption">Funktionen und Überlappung</div>
<canvas id="top" width="900" height="340" aria-label="f(t), verschobenes g(x-t) und Produkt"></canvas>
<div class="legend">
<span><i class="swatch" style="background:#64748b"></i>f(t)</span>
<span><i class="swatch" style="background:#2563eb"></i>g(x − t)</span>
<span><i class="swatch" style="background:#16a34a"></i>f(t) · g(x − t)</span>
</div>
<div class="caption">Faltungsintegral</div>
<canvas id="bottom" width="900" height="245" aria-label="Faltungskurve und aktueller Integralwert"></canvas>
<div class="controls">
<label for="slider">x</label>
<input id="slider" type="range" min="__XMIN__" max="__XMAX__" step="any" value="__XMIN__">
<output id="value" for="slider"></output>
</div>
<div class="hint">Schiebe den Regler von ganz links nach ganz rechts.</div>
</main>
<script>
const t=__T__, f=__F__, u=__U__, g=__G__, shifts=__SHIFTS__, conv=__CONV__;
const xmin=__XMIN__, xmax=__XMAX__, tmin=__TMIN__, tmax=__TMAX__;
const gl=__GL__, gr=__GR__, topmin=__TOPMIN__, topmax=__TOPMAX__;
const botmin=__BOTMIN__, botmax=__BOTMAX__;
const topCanvas=document.getElementById('top'),bottomCanvas=document.getElementById('bottom');
const slider=document.getElementById('slider'),out=document.getElementById('value');
const margin={left:65,right:24,top:20,bottom:38};
function interp(xs,ys,v){
  if(v<=xs[0])return ys[0];
  if(v>=xs[xs.length-1])return ys[ys.length-1];
  const ix=Math.min(xs.length-2,Math.floor((v-xs[0])/(xs[xs.length-1]-xs[0])*(xs.length-1)));
  const a=(v-xs[ix])/(xs[ix+1]-xs[ix]);
  return ys[ix]*(1-a)+ys[ix+1]*a;
}
function gAt(z){return z<gl||z>gr?0:interp(u,g,z)}
function xy(canvas,l,r,b,h){
  const w=canvas.width-margin.left-margin.right,hh=canvas.height-margin.top-margin.bottom;
  return {X:v=>margin.left+(v-l)/(r-l)*w,Y:v=>canvas.height-margin.bottom-(v-b)/(h-b)*hh};
}
function curve(ctx,xs,ys,X,Y,color,width){
  ctx.beginPath();
  xs.forEach((v,i)=>{if(i===0)ctx.moveTo(X(v),Y(ys[i]));else ctx.lineTo(X(v),Y(ys[i]))});
  ctx.strokeStyle=color;ctx.lineWidth=width;ctx.lineJoin='round';ctx.stroke();
}
function tidy(v){return Math.abs(v)<1e-8?'0':Number(v.toFixed(2)).toString()}
function axes(canvas,l,r,b,h){
  const ctx=canvas.getContext('2d'),{X,Y}=xy(canvas,l,r,b,h);
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.font='13px system-ui,sans-serif';ctx.lineWidth=1;
  for(let k=0;k<=5;k++){
    const yy=b+(h-b)*k/5;
    ctx.strokeStyle='#e5e7eb';ctx.beginPath();ctx.moveTo(X(l),Y(yy));ctx.lineTo(X(r),Y(yy));ctx.stroke();
    ctx.fillStyle='#6b7280';ctx.textAlign='right';ctx.fillText(tidy(yy),X(l)-9,Y(yy)+4);
  }
  for(let k=0;k<=6;k++){
    const xx=l+(r-l)*k/6;
    ctx.strokeStyle='#e5e7eb';ctx.beginPath();ctx.moveTo(X(xx),Y(b));ctx.lineTo(X(xx),Y(h));ctx.stroke();
    ctx.fillStyle='#6b7280';ctx.textAlign='center';ctx.fillText(tidy(xx),X(xx),Y(b)+20);
  }
  ctx.strokeStyle='#94a3b8';ctx.lineWidth=1.1;
  if(l<=0&&r>=0){ctx.beginPath();ctx.moveTo(X(0),Y(b));ctx.lineTo(X(0),Y(h));ctx.stroke()}
  if(b<=0&&h>=0){ctx.beginPath();ctx.moveTo(X(l),Y(0));ctx.lineTo(X(r),Y(0));ctx.stroke()}
  return {ctx,X,Y};
}
function clip(ctx,X,Y,l,r,b,h){ctx.beginPath();ctx.rect(X(l),Y(h),X(r)-X(l),Y(b)-Y(h));ctx.clip()}
function draw(){
  const x=Number(slider.value);out.textContent=tidy(x);
  const gt=t.map(v=>gAt(x-v)), product=gt.map((v,i)=>v*f[i]);
  {
    const {ctx,X,Y}=axes(topCanvas,tmin,tmax,topmin,topmax);
    ctx.save();clip(ctx,X,Y,tmin,tmax,topmin,topmax);
    // Fill signed product area against the horizontal axis.
    ctx.beginPath();ctx.moveTo(X(t[0]),Y(0));
    t.forEach((v,i)=>ctx.lineTo(X(v),Y(product[i])));
    ctx.lineTo(X(t[t.length-1]),Y(0));ctx.closePath();
    ctx.fillStyle='rgba(22,163,74,.20)';ctx.fill();
    curve(ctx,t,f,X,Y,'#64748b',2.5);
    curve(ctx,t,gt,X,Y,'#2563eb',2.7);
    curve(ctx,t,product,X,Y,'#16a34a',1.7);
    ctx.restore();
  }
  {
    const {ctx,X,Y}=axes(bottomCanvas,xmin,xmax,botmin,botmax);
    ctx.save();clip(ctx,X,Y,xmin,xmax,botmin,botmax);
    curve(ctx,shifts,conv,X,Y,'#d1d5db',2.2);
    const indices=shifts.filter(v=>v<x), vals=indices.map(v=>interp(shifts,conv,v));
    const integral=interp(shifts,conv,x);
    indices.push(x);vals.push(integral);
    curve(ctx,indices,vals,X,Y,'#2563eb',3);
    ctx.setLineDash([4,5]);ctx.strokeStyle='#94a3b8';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(X(x),Y(botmin));ctx.lineTo(X(x),Y(integral));ctx.stroke();ctx.setLineDash([]);
    ctx.beginPath();ctx.arc(X(x),Y(integral),5,0,Math.PI*2);ctx.fillStyle='#2563eb';ctx.fill();
    ctx.restore();
  }
}
slider.addEventListener('input',draw);
draw();
</script></body></html>
'''
    for key, value in replacements.items():
        html = html.replace(key, value)
    dest = Path(__file__).with_name('faltung_interaktiv.html')
    dest.write_text(html, encoding='utf-8')
    print(f'Erstellt: {dest} ({dest.stat().st_size/1024:.0f} KB)')


if __name__ == '__main__':
    main()
