import json
import numpy as np


def cloud_view(points, colors, key='cloud'):
    import streamlit.components.v1 as components
    if not len(points): return
    step = max(1, int(np.ceil(len(points) / 20000)))
    p = np.asarray(points[::step], float); c = np.asarray(colors[::step], int)
    p -= np.median(p, axis=0)
    p /= max(float(np.percentile(np.abs(p), 98)), 1e-6)
    payload = json.dumps(np.column_stack([p, c]).round(4).tolist(), allow_nan=False)
    components.html('''<!doctype html><html><body style="margin:0;background:#0b111a;color:#9caabd;font:13px system-ui">
<canvas id="c" style="width:100%;height:440px;cursor:grab;touch-action:none" aria-label="Interactive point cloud; drag to rotate and scroll to zoom"></canvas>
<div style="padding:0 14px 10px">Drag to orbit · Scroll to zoom · Relative scale · Preview sampled to 20,000 points</div>
<script>const pts=PAYLOAD, canvas=document.getElementById('c'),ctx=canvas.getContext('2d');
let yaw=.2,pitch=.05,zoom=1,drag=false,px=0,py=0;
function draw(){canvas.width=canvas.clientWidth*devicePixelRatio;canvas.height=440*devicePixelRatio;
const w=canvas.width,h=canvas.height,sc=Math.min(w,h)*.35*zoom;
ctx.fillStyle='#0b111a';ctx.fillRect(0,0,w,h);let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
let q=pts.map(p=>{let x=cy*p[0]+sy*p[2],z=-sy*p[0]+cy*p[2];return [x,cp*p[1]-sp*z,sp*p[1]+cp*z,p[3],p[4],p[5]]}).sort((a,b)=>b[2]-a[2]);
q.forEach(p=>{ctx.fillStyle=`rgb(${p[3]},${p[4]},${p[5]})`;ctx.fillRect(w/2+p[0]*sc,h/2+p[1]*sc,2*devicePixelRatio,2*devicePixelRatio)})}
canvas.onpointerdown=e=>{drag=true;px=e.clientX;py=e.clientY;canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(drag){yaw+=(e.clientX-px)*.008;pitch+=(e.clientY-py)*.008;px=e.clientX;py=e.clientY;draw()}};
canvas.onpointerup=()=>drag=false;canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.2,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw()};
window.onresize=draw;draw();</script></body></html>'''.replace('PAYLOAD', payload), height=480)
