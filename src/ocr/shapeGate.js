/**
 * Shape-gate verifier (spec 9.5): deterministic, no learning.
 * Pure JS except rankTemplates/extractDigitSlots which take cv Mats.
 */

const INK_THR = 128;

function columnRuns(px, W, H, minRows, minWidth) {
  minRows = minRows ?? Math.max(2, Math.round(H * 0.12));
  minWidth = minWidth ?? 2;
  const active = new Array(W).fill(false);
  for (let x = 0; x < W; x++) {
    let ink = 0;
    for (let y = 0; y < H; y++) {
      if (px[y * W + x] < INK_THR) { ink++; if (ink >= minRows) break; }
    }
    active[x] = ink >= minRows;
  }
  const runs = [];
  let s = -1;
  for (let x = 0; x <= W; x++) {
    const a = x < W ? active[x] : false;
    if (a && s < 0) s = x;
    else if (!a && s >= 0) { if (x - s >= minWidth) runs.push({ x0: s, x1: x }); s = -1; }
  }
  return runs;
}

function runHeight(px, W, H, x0, x1) {
  let top = H, bot = -1;
  for (let x = x0; x < x1; x++) {
    for (let y = 0; y < H; y++) { if (px[y * W + x] < INK_THR) { if (y < top) top = y; if (y > bot) bot = y; break; } }
    for (let y = H - 1; y >= 0; y--) { if (px[y * W + x] < INK_THR) { if (y < top) top = y; if (y > bot) bot = y; break; } }
  }
  if (bot < 0) return { y: 0, height: H };
  return { y: top, height: bot - top + 1 };
}

export function splitSlotsProjection(px, W, H, n) {
  if (!px || W <= 0 || H <= 0 || !(n > 0)) return [];
  let runs = columnRuns(px, W, H).map((r) => ({ ...r }));
  if (!runs.length) return [];
  while (runs.length > n) {
    let bi = 0, bg = Infinity;
    for (let i = 0; i < runs.length - 1; i++) { const g = runs[i+1].x0 - runs[i].x1; if (g < bg) { bg = g; bi = i; } }
    runs.splice(bi, 2, { x0: runs[bi].x0, x1: runs[bi+1].x1 });
  }
  while (runs.length < n) {
    let wi = 0, ww = -1;
    for (let i = 0; i < runs.length; i++) { const w = runs[i].x1 - runs[i].x0; if (w > ww) { ww = w; wi = i; } }
    const r = runs[wi];
    if (r.x1 - r.x0 < 4) break;
    const mid = Math.floor((r.x0 + r.x1) / 2);
    runs.splice(wi, 1, { x0: r.x0, x1: mid }, { x0: mid, x1: r.x1 });
  }
  return runs.slice(0, n).map((r) => ({ x0: r.x0, x1: r.x1 }));
}

export function splitDateSlots(px, W, H) {
  const fallback = (fixed) => {
    const w = W / 8, digits = [];
    for (let i = 0; i < 8; i++) digits.push({ x0: Math.round(i*w), x1: Math.round((i+1)*w) });
    return { fixed, digits, slashes: [] };
  };
  if (!px || W <= 0 || H <= 0) return fallback(false);
  const runs = columnRuns(px, W, H);
  if (runs.length !== 10) return fallback(false);
  const span = runs[9].x1 - runs[0].x0;
  if (span < W * 0.3) return fallback(false);
  const slashes = [runs[4], runs[7]].map((r) => ({ x0: r.x0, x1: r.x1 }));
  const digits = runs.filter((_, i) => i !== 4 && i !== 7).map((r) => ({ x0: r.x0, x1: r.x1 }));
  if (digits.length !== 8) return fallback(false);
  return { fixed: true, digits, slashes };
}

export function resizeNearest(px, w, h, nw, nh) {
  const out = new Float32Array(nw * nh);
  for (let y = 0; y < nh; y++) {
    const sy = Math.min(h-1, Math.floor((y*h)/nh));
    for (let x = 0; x < nw; x++) { const sx = Math.min(w-1, Math.floor((x*w)/nw)); out[y*nw+x] = px[sy*w+sx]; }
  }
  return out;
}

export function ncc(a, b) {
  const n = Math.min(a.length, b.length);
  if (!n) return -1;
  let ma = 0, mb = 0;
  for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
  ma /= n; mb /= n;
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < n; i++) { const da = a[i]-ma, db = b[i]-mb; dot += da*db; na += da*da; nb += db*db; }
  const d = Math.sqrt(na*nb);
  if (!(d > 1e-12)) return -1;
  return dot / d;
}

export function huFromPixels(px, w, h) {
  let m00=0,m10=0,m01=0;
  for (let y=0;y<h;y++) for (let x=0;x<w;x++) { const v=px[y*w+x]<128?1:0; m00+=v; m10+=x*v; m01+=y*v; }
  if (!(m00>0)) return [0,0,0,0,0,0,0];
  const cx=m10/m00, cy=m01/m00;
  let mu20=0,mu11=0,mu02=0,mu30=0,mu21=0,mu12=0,mu03=0;
  for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
    if (!(px[y*w+x]<128)) continue;
    const dx=x-cx, dy=y-cy;
    mu20+=dx*dx; mu11+=dx*dy; mu02+=dy*dy;
    mu30+=dx*dx*dx; mu21+=dx*dx*dy; mu12+=dx*dy*dy; mu03+=dy*dy*dy;
  }
  const n20=mu20/Math.pow(m00,2), n11=mu11/Math.pow(m00,2), n02=mu02/Math.pow(m00,2);
  const n30=mu30/Math.pow(m00,2.5), n21=mu21/Math.pow(m00,2.5), n12=mu12/Math.pow(m00,2.5), n03=mu03/Math.pow(m00,2.5);
  const hu=new Array(7);
  hu[0]=n20+n02;
  hu[1]=(n20-n02)*(n20-n02)+4*n11*n11;
  hu[2]=(n30-3*n12)*(n30-3*n12)+(3*n21-n03)*(3*n21-n03);
  hu[3]=(n30+n12)*(n30+n12)+(n21+n03)*(n21+n03);
  hu[4]=(n30-3*n12)*(n30+n12)*((n30+n12)*(n30+n12)-3*(n21+n03)*(n21+n03))+(3*n21-n03)*(n21+n03)*(3*(n30+n12)*(n30+n12)-(n21+n03)*(n21+n03));
  hu[5]=(n20-n02)*((n30+n12)*(n30+n12)-(n21+n03)*(n21+n03))+4*n11*(n30+n12)*(n21+n03);
  hu[6]=(3*n21-n03)*(n30+n12)*((n30+n12)*(n30+n12)-3*(n21+n03)*(n21+n03))-(n30-3*n12)*(n21+n03)*(3*(n30+n12)*(n30+n12)-(n21+n03)*(n21+n03));
  return hu.map((v)=>{ if(v===0) return 0; const s=v<0?-1:1; return -s*Math.log10(Math.abs(v)+1e-300); });
}

export function huDistance(a,b){ let s=0; for(let i=0;i<7;i++) s+=(a[i]-b[i])*(a[i]-b[i]); return Math.sqrt(s); }

const registry={ hollow:[], slash:[], narrow:[], looped:[] };

export function registerShapeTemplates(obj={}){
  for (const k of Object.keys(registry)){
    const list=obj[k]; if(!Array.isArray(list)) continue;
    for (const t of list){ if(!t||!t.data||!(t.w>0)||!(t.h>0)) continue; registry[k].push({w:t.w,h:t.h,data:Float32Array.from(t.data)}); }
  }
}
export function clearShapeTemplates(){ for(const k of Object.keys(registry)) registry[k]=[]; }
export function shapeRegistryEmpty(){ return Object.values(registry).every((l)=>l.length===0); }

function synthTemplate(kind, seed){
  seed=seed||0;
  const W=24,H=32, px=new Float32Array(W*H).fill(255);
  const ink=(x,y)=>{ if(x>=0&&x<W&&y>=0&&y<H) px[y*W+x]=0; };
  if(kind==="hollow"){
    const rx=7-(seed%2), ry=6-(seed%2);
    for(let a=0;a<64;a++){ const t=(a/64)*Math.PI*2;
      ink(Math.round(12+Math.cos(t)*rx),Math.round(16+Math.sin(t)*ry));
      ink(Math.round(12+Math.cos(t)*(rx-2)),Math.round(16+Math.sin(t)*(ry-2))); }
  } else if(kind==="narrow"){
    const x0=10+(seed%3);
    for(let y=3;y<H-3;y++) for(let x=x0;x<x0+3;x++) ink(x,y);
  } else if(kind==="slash"){
    for(let y=3;y<H-3;y++){ const x=Math.round(18-((y-3)/(H-6))*12)+(seed%2); ink(x,y); ink(x+1,y); }
  } else {
    for(let y=6;y<H-6;y++) for(let x=6;x<W-6;x++){ const dx=(x-12)/6, dy=(y-16)/10; if(dx*dx+dy*dy<1) ink(x,y); }
  }
  return {w:W,h:H,data:px};
}

export async function loadEmbeddedShapeTemplates(){
  if(!shapeRegistryEmpty()) return;
  registerShapeTemplates({hollow:[synthTemplate("hollow",0),synthTemplate("hollow",1)],slash:[synthTemplate("slash",0),synthTemplate("slash",1)],narrow:[synthTemplate("narrow",0),synthTemplate("narrow",1)],looped:[synthTemplate("looped",0)]});
}

/** Node-only loader lives in tests/helpers/loadRealShapes.mjs (kept out of
 * the browser bundle: any static or dynamic node:/canvas import in this
 * shipped module breaks `vite build`). */
export async function loadRealShapeTemplates(){
  throw new Error("loadRealShapeTemplates is Node/test-only; import it from tests/helpers/loadRealShapes.mjs");
}

export function rankTemplates(cv,query,templates){
  const labels=Object.keys(templates||{});
  if(!labels.length||!query) return [];
  const out=[];
  for(const label of labels){
    const tpl=templates[label];
    if(!tpl||tpl.cols<=0||tpl.rows<=0) continue;
    try{
      let t=tpl, owned=false;
      if(tpl.cols>query.cols||tpl.rows>query.rows){ t=new cv.Mat(); cv.resize(tpl,t,new cv.Size(query.cols,query.rows),0,0,cv.INTER_AREA); owned=true; }
      const res=new cv.Mat();
      try{
        const method=cv.TM_CCOEFF_NORMED??cv.TM_CCORR_NORMED??5;
        cv.matchTemplate(query,t,res,method);
        const mm=cv.minMaxLoc(res,new cv.Mat());
        out.push([label,mm.maxVal]);
      }finally{ res.delete(); }
      if(owned) t.delete();
    }catch{}
  }
  out.sort((a,b)=>b[1]-a[1]);
  return out;
}

export function extractDigitSlots(cv,probe){
  if(!probe||probe.cols<=0||probe.rows<=0) return null;
  try{
    const W=probe.cols,H=probe.rows,px=probe.data;
    let runs=columnRuns(px,W,H);
    const merged=[];
    for(const r of runs){ const p=merged[merged.length-1]; if(p&&r.x0-p.x1<=2) p.x1=r.x1; else merged.push({...r}); }
    runs=merged;
    if(runs.length!==10) return null;
    const digits=runs.filter((_,i)=>i!==4&&i!==7);
    return digits.map((r)=>{ const hh=runHeight(px,W,H,r.x0,r.x1); return {x:r.x0,y:hh.y,width:r.x1-r.x0,height:hh.height}; });
  }catch{ return null; }
}

export function checkLineGate(digitsStr,slots){
  const reasons=[];
  if(!digitsStr||!slots||digitsStr.length!==8||slots.length!==8) return {conflict:false,reasons,abstained:true};
  const hs=slots.map((s)=>s.height||0);
  const sorted=[...hs].sort((a,b)=>a-b);
  const median=sorted[4]||1;
  if(!(median>0)) return {conflict:false,reasons,abstained:true};
  for(let i=0;i<8;i++){
    const hr=hs[i]/median, d=digitsStr[i];
    if(d==="0"&&hr>0.7) reasons.push("pos"+i+":0-tall-"+hr.toFixed(2));
    if(d==="5"&&hr<0.55) reasons.push("pos"+i+":5-short-"+hr.toFixed(2));
  }
  return {conflict:reasons.length>0,reasons};
}

function slotScores(spx,sw,sh){
  const slot=Float32Array.from(spx);
  const nccScores={}, huDists={};
  const slotHu=huFromPixels(slot,sw,sh);
  for(const e of Object.entries(registry)){
    const cls=e[0], list=e[1];
    if(!list.length) continue;
    let bn=-Infinity, bh=Infinity;
    for(const t of list){
      const a=resizeNearest(slot,sw,sh,24,32);
      const b=resizeNearest(t.data,t.w,t.h,24,32);
      bn=Math.max(bn,ncc(a,b));
      bh=Math.min(bh,huDistance(slotHu,huFromPixels(t.data,t.w,t.h)));
    }
    nccScores[cls]=bn; huDists[cls]=bh;
  }
  const se=Object.entries(nccScores).sort((a,b)=>b[1]-a[1])[0];
  const he=Object.entries(huDists).sort((a,b)=>a[1]-b[1])[0];
  return {nccScores,huDists,top:se?se[0]:null,huTop:he?he[0]:null};
}

export function verifySlotShape(spx,sw,sh,digit){
  const d=String(digit);
  const want=d==="0"?"hollow":d==="1"?"narrow":null;
  if(!want) return {conflict:false,abstained:true,top:null,huTop:null};
  if(shapeRegistryEmpty()) return {conflict:false,abstained:true,top:null,huTop:null};
  if(!spx||!(sw>3)||!(sh>3)) return {conflict:false,abstained:true,top:null,huTop:null};
  try{
    const r=slotScores(spx,sw,sh);
    return {conflict:r.top!==want&&r.huTop!==want,top:r.top,huTop:r.huTop,nccScores:r.nccScores,huDists:r.huDists};
  }catch{ return {conflict:false,abstained:true,top:null,huTop:null}; }
}
