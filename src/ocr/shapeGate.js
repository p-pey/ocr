/**
 * Shape-gate verifier (spec 9.5 + MASTER SPEC §6 "Shape Gate"):
 * deterministic topological checker, no learning.
 * Pure JS except rankTemplates/extractDigitSlots/countHolesCv which take cv Mats.
 *
 * MASTER SPEC ground truth (card font):
 *  ۰ (0): hollow diamond/circle, exactly 1 hole.
 *  ۹ (9): exactly 1 hole in upper 55% + descending right-side stem in lower 45%.
 *  ۵ (5): exactly 1 hole near centre, no descending stem below loop.
 *  ۸ (8): 0 holes, caret/^ shape (pointed peak, split legs).
 *  ۶ (6): 0 holes, open top-left concavity.
 *  ۱ (1): 0 holes, straight vertical stroke, aspect W/H < 0.40.
 *  ۳ (3): 0 holes, horizontal bar with 3 upward prongs.
 * If CNN predicts a digit but topology fails, penalise/zero that class.
 * Thresholds (ink level, height ratios, hole rules, stem, aspect) live in
 * engineConfig.js §N (SHAPE_GATE) — edit there, not here.
 */
import { SHAPE_GATE as GATE_CFG } from "./engineConfig.js";

const INK_THR = GATE_CFG.inkThreshold;

function columnRuns(px, W, H, minRows, minWidth) {
  // Defaults from §N: rows scale with line height so thin/small print
  // still yields runs; raise minWidth to split touching digits less.
  minRows =
    minRows ??
    GATE_CFG.minInkRowsPerColumn ??
    Math.max(2, Math.round(H * 0.12));
  minWidth = minWidth ?? GATE_CFG.minRunWidthPx;
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
  const T = GATE_CFG.inkThreshold;
  let m00=0,m10=0,m01=0;
  for (let y=0;y<h;y++) for (let x=0;x<w;x++) { const v=px[y*w+x]<T?1:0; m00+=v; m10+=x*v; m01+=y*v; }
  if (!(m00>0)) return [0,0,0,0,0,0,0];
  const cx=m10/m00, cy=m01/m00;
  let mu20=0,mu11=0,mu02=0,mu30=0,mu21=0,mu12=0,mu03=0;
  for (let y=0;y<h;y++) for (let x=0;x<w;x++) {
    if (!(px[y*w+x]<T)) continue;
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
    // MASTER SPEC §3 memory safety: every Mat freed (incl. minMaxLoc mask).
    let mask=null;
    try{
      let t=tpl, owned=false;
      if(tpl.cols>query.cols||tpl.rows>query.rows){ t=new cv.Mat(); cv.resize(tpl,t,new cv.Size(query.cols,query.rows),0,0,cv.INTER_AREA); owned=true; }
      const res=new cv.Mat();
      try{
        const method=cv.TM_CCOEFF_NORMED??cv.TM_CCORR_NORMED??5;
        mask=new cv.Mat();
        cv.matchTemplate(query,t,res,method);
        const mm=cv.minMaxLoc(res,mask);
        out.push([label,mm.maxVal]);
      }finally{ res.delete(); try{mask?.delete?.();}catch{} mask=null; }
      if(owned) t.delete();
    }catch{}
    finally { try{mask?.delete?.();}catch{} }
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
    for(const r of runs){ const p=merged[merged.length-1]; if(p&&r.x0-p.x1<=GATE_CFG.runMergeGapPx) p.x1=r.x1; else merged.push({...r}); }
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
    // §N: ۰ is a small dot (short); ۵ is a full-height loop (tall).
    if(d==="0"&&hr>GATE_CFG.zeroMaxHeightRatio) reasons.push("pos"+i+":0-tall-"+hr.toFixed(2));
    if(d==="5"&&hr<GATE_CFG.fiveMinHeightRatio) reasons.push("pos"+i+":5-short-"+hr.toFixed(2));
  }
  return {conflict:reasons.length>0,reasons};
}

/* MASTER SPEC S6: pure-pixel hole counter (flood fill from borders). */
function toInkMask(spx, sw, sh, thr) {
  const t = thr ?? GATE_CFG.inkThreshold;
  const mask = new Uint8Array(sw * sh);
  for (let i = 0; i < mask.length; i++) mask[i] = spx[i] < t ? 1 : 0;
  return mask;
}

export function countHoles(spx, sw, sh, thr) {
  if (!spx || !(sw > 0) || !(sh > 0)) return { holes: 0, holeBoxes: [], ink: 0, inkRatio: 0 };
  const mask = toInkMask(spx, sw, sh, thr);
  let ink = 0;
  for (let i = 0; i < mask.length; i++) ink += mask[i];
  const seen = new Uint8Array(sw * sh);
  const qx = new Int32Array(sw * sh);
  const qy = new Int32Array(sw * sh);
  let qh = 0, qt = 0;
  const push = (x, y) => {
    if (x < 0 || y < 0 || x >= sw || y >= sh) return;
    const i = y * sw + x;
    if (seen[i] || mask[i]) return;
    seen[i] = 1; qx[qt] = x; qy[qt] = y; qt++;
  };
  for (let x = 0; x < sw; x++) { push(x, 0); push(x, sh - 1); }
  for (let y = 0; y < sh; y++) { push(0, y); push(sw - 1, y); }
  while (qh < qt) {
    const x = qx[qh], y = qy[qh]; qh++;
    push(x + 1, y); push(x - 1, y); push(x, y + 1); push(x, y - 1);
  }
  const holeBoxes = [];
  for (let y = 0; y < sh; y++) for (let x = 0; x < sw; x++) {
    const i = y * sw + x;
    if (mask[i] || seen[i]) continue;
    let x0 = x, x1 = x, y0 = y, y1 = y, count = 0, sx = 0, sy = 0;
    let lh = 0, lt = 0;
    qx[lt] = x; qy[lt] = y; lt++; seen[i] = 1;
    while (lh < lt) {
      const cx = qx[lh], cy = qy[lh]; lh++;
      count++; sx += cx; sy += cy;
      if (cx < x0) x0 = cx; if (cx > x1) x1 = cx;
      if (cy < y0) y0 = cy; if (cy > y1) y1 = cy;
      const nb = [[cx+1,cy],[cx-1,cy],[cx,cy+1],[cx,cy-1]];
      for (let k = 0; k < 4; k++) {
        const nx = nb[k][0], ny = nb[k][1];
        if (nx < 0 || ny < 0 || nx >= sw || ny >= sh) continue;
        const ni = ny * sw + nx;
        if (seen[ni] || mask[ni]) continue;
        seen[ni] = 1; qx[lt] = nx; qy[lt] = ny; lt++;
      }
    }
    // §N minHoleAreaPx: smaller white regions are dust, not a loop.
    if (count >= GATE_CFG.minHoleAreaPx) holeBoxes.push({ x0, y0, x1, y1, cx: sx/count, cy: sy/count, count });
  }
  return { holes: holeBoxes.length, holeBoxes, ink, inkRatio: ink/(sw*sh) };
}

export function countHolesTopo(spx, sw, sh, thr) {
  return countHoles(spx, sw, sh, thr);
}

/* MASTER SPEC S6: OpenCV twin via RETR_CCOMP (delete every Mat). */
export function countHolesCv(cv, digitMat) {
  let bin = null, contours = null, hierarchy = null;
  try {
    bin = new cv.Mat();
    cv.threshold(digitMat, bin, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    contours = new cv.MatVector();
    hierarchy = new cv.Mat();
    cv.findContours(bin, contours, hierarchy, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE);
    let holes = 0;
    const boxes = [];
    for (let i = 0; i < contours.size(); i++) {
      let parent = -1;
      try {
        const d = hierarchy.data32S;
        parent = d ? d[i * 4 + 3] : -1;
      } catch { parent = -1; }
      if (parent !== -1) {
        holes++;
        const c = contours.get(i);
        try {
          const r = cv.boundingRect(c);
          boxes.push({ x0: r.x, y0: r.y, x1: r.x + r.width, y1: r.y + r.height, cx: r.x + r.width/2, cy: r.y + r.height/2, count: r.width*r.height });
        } finally { try { c.delete(); } catch {} }
      }
    }
    return { holes, holeBoxes: boxes };
  } catch {
    return { holes: 0, holeBoxes: [] };
  } finally {
    try { bin?.delete?.(); } catch {}
    try { contours?.delete?.(); } catch {}
    try { hierarchy?.delete?.(); } catch {}
  }
}

function stemRightSide(spx, sw, sh) {
  // §N: measures ink in the lower half — ۹ has a descending right-side
  // stem, ۵ does not. rightHeavy = right ink dominates left ink.
  const y0 = Math.floor(sh * GATE_CFG.nineLoopMaxCenterYFraction);
  let left = 0, right = 0;
  for (let y = y0; y < sh; y++) for (let x = 0; x < sw; x++) {
    if (spx[y*sw+x] < GATE_CFG.inkThreshold) { if (x >= sw/2) right++; else left++; }
  }
  return { left, right, rightHeavy: right > left*GATE_CFG.stemRightHeavyRatio && right > GATE_CFG.stemMinRightInkPx };
}

export function verifyDigitTopo(spx, sw, sh, digit) {
  const d = String(digit);
  if (!spx || !(sw > 3) || !(sh > 3)) return { conflict: false, abstained: true, reasons: ["tiny-slot"] };
  if (!["0","1","3","5","6","8","9"].includes(d)) return { conflict: false, abstained: true, reasons: [], holes: 0 };
  const r = countHoles(spx, sw, sh);
  const reasons = [];
  const hole = r.holeBoxes[0] || null;
  if (d === "0") { if (r.holes !== 1) reasons.push("0-holes-"+r.holes); }
  else if (d === "9") {
    if (r.holes !== 1) reasons.push("9-holes-"+r.holes);
    else if (hole && hole.cy > sh*GATE_CFG.nineLoopMaxCenterYFraction) reasons.push("9-loop-low");
    if (!stemRightSide(spx,sw,sh).rightHeavy) reasons.push("9-no-right-stem");
  } else if (d === "5") {
    if (r.holes !== 1) reasons.push("5-holes-"+r.holes);
    else if (hole && (hole.cy < sh*GATE_CFG.fiveLoopMinCenterYFraction || hole.cy > sh*GATE_CFG.fiveLoopMaxCenterYFraction)) reasons.push("5-loop-offcenter");
    const st = stemRightSide(spx,sw,sh);
    if (st.rightHeavy && st.right > GATE_CFG.stemMaxRightInkForFivePx) reasons.push("5-has-descending-stem");
  } else if (d === "8" || d === "6") { if (r.holes !== 0) reasons.push(d+"-holes-"+r.holes); }
  else if (d === "1") {
    if (r.holes !== 0) reasons.push("1-holes-"+r.holes);
    const a = sw/Math.max(1,sh);
    if (!(a < GATE_CFG.oneMaxAspect)) reasons.push("1-aspect-"+a.toFixed(2));
  } else if (d === "3") { if (r.holes !== 0) reasons.push("3-holes-"+r.holes); }
  return { conflict: reasons.length > 0, abstained: false, reasons, holes: r.holes };
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
      // Compare at the fixed template size (§N templateCompare*Px).
      const a=resizeNearest(slot,sw,sh,GATE_CFG.templateCompareWidthPx,GATE_CFG.templateCompareHeightPx);
      const b=resizeNearest(t.data,t.w,t.h,GATE_CFG.templateCompareWidthPx,GATE_CFG.templateCompareHeightPx);
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
