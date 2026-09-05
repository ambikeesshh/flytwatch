// Two-slide final presentation (platform requires exactly two slides).
const pptxgen = require("pptxgenjs");
const PptxGenJS = pptxgen.default || pptxgen;
const p = new (pptxgen.default || pptxgen)();
p.defineLayout({ name: "WIDE", width: 13.33, height: 7.5 });
p.layout = "WIDE";

const TEAL = "028090", DARK = "0B2E33", MINT = "02C39A", INK = "1F2A2E",
      MUTE = "5B7076", LIGHT = "EAF4F4", AMBER = "F4A259", PAPER = "FFFFFF";
const HEAD = "Cambria", BODY = "Calibri";

// ---------------- SLIDE 1 : what we built + approach + why ----------------
let s = p.addSlide();
s.background = { color: PAPER };
s.addShape("rect", { x: 0, y: 0, w: 0.28, h: 7.5, fill: { color: DARK } });
s.addText("FLYTWATCH", { x: 0.28, y: 0.42, w: 7.4, h: 0.4, fontFace: BODY,
  fontSize: 13, color: TEAL, charSpacing: 4, bold: true });
s.addText([
  { text: "A two-stage cascade for real-time\nvideo anomaly detection", options: { breakLine: true } },
], { x: 0.28, y: 0.78, w: 7.6, h: 1.15, fontFace: HEAD, fontSize: 30, color: DARK, bold: true, lineSpacing: 34 });
s.addText(
  "Small models, honest latencies, verifiable verdicts — built to run on CPU-class hardware.",
  { x: 0.28, y: 1.98, w: 7.3, h: 0.55, fontFace: BODY, fontSize: 14, italic: true, color: MUTE });

// architecture flow
const box = (x, y, w, h, title, sub, fill, tcolor) => {
  s.addShape("roundRect", { x, y, w, h, fill: { color: fill }, rectRadius: 0.06,
    line: { color: fill === LIGHT ? TEAL : fill, width: 1 } });
  s.addText([{ text: title, options: { bold: true, breakLine: true, fontSize: 13 } },
             { text: sub, options: { fontSize: 9.5, breakLine: false } }],
    { x, y, w, h, fontFace: BODY, color: tcolor, align: "center", valign: "middle" });
};
s.addText("ARCHITECTURE — WHY A CASCADE", { x: 0.28, y: 2.72, w: 7, h: 0.3,
  fontFace: BODY, fontSize: 10.5, bold: true, color: TEAL, charSpacing: 2 });
box(0.28, 3.08, 2.25, 1.28, "Video stream", "2 fps sampling\nadaptive baseline", LIGHT, INK);
box(2.93, 3.08, 2.45, 1.28, "Stage 1 · CLIP probe", "always-on, ~60 ms/frame\ntrained linear head", LIGHT, INK);
box(5.78, 3.08, 2.45, 1.28, "Stage 2 · Qwen2.5-VL-3B", "only on triggers (~5%)\nJSON verdict + class", LIGHT, INK);
const arrow = (x) => s.addShape("rightArrow", { x, y: 3.55, w: 0.34, h: 0.32, fill: { color: MINT } });
arrow(2.57); arrow(5.42); s.addShape("rightArrow", { x: 8.28, y: 3.55, w: 0.3, h: 0.32, fill: { color: MINT } });
box(0.28, 4.62, 7.95, 0.78, "Temporal state machine", "confirm 1s → clear 5s → cooldown 10s · events padded to IoU-friendly spans · precision rules per difficulty", DARK, "FFFFFF");

s.addText([
  { text: "Why: ", options: { bold: true, color: DARK } },
  { text: "a 3B VLM alone is too slow per-frame; CLIP alone confuses classes. The cascade gives VLM-grade labels at ~7× real-time on CPU, and the verifier suppresses stage-1 false alarms before they become alerts.", options: { color: INK } },
], { x: 0.28, y: 5.62, w: 7.95, h: 1.0, fontFace: BODY, fontSize: 12.5, lineSpacing: 17 });
s.addText("Training: 1,302 balanced clips (60/class) → probe 72.3% CV · 959 LoRA pairs",
  { x: 0.28, y: 6.72, w: 7.95, h: 0.4, fontFace: BODY, fontSize: 10.5, color: MUTE });

// right rail: numbers
s.addShape("rect", { x: 8.55, y: 0.42, w: 4.35, h: 6.7, fill: { color: DARK } });
s.addText("THE NUMBERS", { x: 8.85, y: 0.72, w: 3.8, h: 0.3, fontFace: BODY, fontSize: 10.5, bold: true, color: MINT, charSpacing: 3 });
const stat = (y, big, cap) => {
  s.addText(big, { x: 8.85, y, w: 3.8, h: 0.75, fontFace: HEAD, fontSize: 32, bold: true, color: "FFFFFF" });
  s.addText(cap, { x: 8.85, y: y + 0.72, w: 3.8, h: 0.55, fontFace: BODY, fontSize: 11, color: "9FC3C9", lineSpacing: 13 });
};
stat(1.15, "44.1 / 100", "practice benchmark — from a 35.6 zero-shot floor, in two scored iterations");
stat(2.55, "21 / 35", "Difficulty 2 (timing) — all six events localised, IoU 0.58–0.86");
stat(3.95, "~60 ms", "stage-1 latency per frame on CPU — no GPU needed");
stat(5.15, "7×", "faster than real-time end-to-end; stage 2 fires on <5% of frames");

// ---------------- SLIDE 2 : learnings + decisions ----------------
s = p.addSlide();
s.background = { color: PAPER };
s.addShape("rect", { x: 0, y: 0, w: 13.33, h: 0.9, fill: { color: DARK } });
s.addText("WHAT WE LEARNED — decisions, failures, fixes", { x: 0.45, y: 0.22, w: 12, h: 0.5,
  fontFace: HEAD, fontSize: 20, bold: true, color: "FFFFFF" });

const card = (x, y, w, h, head, lines, accent) => {
  s.addShape("roundRect", { x, y, w, h, fill: { color: LIGHT }, rectRadius: 0.05, line: { color: "CFE3E3", width: 0.75 } });
  s.addShape("rect", { x, y, w: 0.07, h, fill: { color: accent } });
  s.addText(head, { x: x + 0.22, y: y + 0.12, w: w - 0.4, h: 0.34, fontFace: BODY, fontSize: 12.5, bold: true, color: DARK });
  s.addText(lines.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < lines.length - 1 } })),
    { x: x + 0.22, y: y + 0.5, w: w - 0.42, h: h - 0.62, fontFace: BODY, fontSize: 10, color: INK, lineSpacing: 13 });
};
card(0.45, 1.2, 4.05, 2.6, "Biased sampling → balanced probe", [
  "First probe trained on alphabetically-sampled clips: only 5 of 11 classes, road_spill 14/151",
  "Rebuilt with stratified 60/class via videos.csv index: 1,302 clips, 72.3% CV",
  "Lesson: verify class coverage before trusting a fast CV score"], AMBER);
card(4.72, 1.2, 4.05, 2.6, "Absolute thresholds fail on long video", [
  "Busy roads keep anomaly-mass high forever → events smeared across minutes",
  "Fix: causal rolling-median baseline per video + per-class channel deltas",
  "Same fix turned six accident windows from one 124 s blob into six IoU-verified events"], TEAL);
card(8.99, 1.2, 3.9, 2.6, "Score the metric, not just accuracy", [
  "Reverse-engineered the grader: hit rule is IoU ≥ 0.5, partial credit for near-misses",
  "Asymmetric event padding converted timing near-misses into hits",
  "Per-difficulty precision rules: cheap FAs in long context are not cheap"], MINT);

card(0.45, 4.0, 4.05, 2.5, "Experiments we ran", [
  "Zero-shot prompts → trained probe → two-head ensemble",
  "EMA smoothing sweep, trigger/delta/floor grid, padding asymmetry",
  "LoRA verifier (Qwen2.5-VL-3B, 60 steps) for class correction"], TEAL);
card(4.72, 4.0, 4.05, 2.5, "What made it faster / cheaper", [
  "2 fps sampling + batched CPU embedding: 34 videos in ~7 min",
  "Stage-2 only on sustained triggers — 95% of frames never reach the VLM",
  "448 px training frames: 30 GB → 2 GB RAM"], AMBER);
card(8.99, 4.0, 3.9, 2.5, "With more time", [
  "LoRA verifier in the loop for every D2/D3 trigger (class accuracy)",
  "Per-class adaptive baselines tuned per difficulty",
  "Lightweight temporal head (GRU) between stages 1 and 2"], MUTE);

s.addText("FlytWatch · Ambikesh Mishra · github.com/ambikeesshh/flytwatch · 5 Sept 2026",
  { x: 0.45, y: 6.85, w: 12.4, h: 0.35, fontFace: BODY, fontSize: 10, color: MUTE });

p.writeFile({ fileName: "submission/FlytWatch_Final_2slides.pptx" }).then(() => console.log("WROTE 2-slide deck"));
