/* =========================================================================
   Palatino Burgundy — Pencil Annotation
   Self-contained Apple Pencil / stylus drawing layer for HTML documents.

   Usage:
     1. <link rel="stylesheet" href="pencil.css">
     2. <script src="pencil.js" defer></script>
     3. Add data-pencil="on" to <body> (or call PalatinoPencil.enable()).

   Enables:
     - Floating burgundy toolbar (Pen / Highlighter / Eraser / Undo / Clear / Close)
     - Pointer Events filtered to pen + mouse (finger scrolls)
     - Pressure-aware smooth strokes via quadratic midpoint interpolation
     - Per-document localStorage persistence (key: "pb-pencil::" + location.pathname)

   Rendering model:
     - Canvas is position:fixed and sized to the viewport (window.innerWidth
       / innerHeight), not the full document.
     - Strokes are stored in document-space (pageX/pageY) and rendered via a
       context transform that translates by -scrollX,-scrollY each frame.
     - A passive scroll listener redraws through requestAnimationFrame.
     - devicePixelRatio is clamped to MAX_DPR (=1.5) to keep the GPU texture
       small on retina screens; ink quality is preserved at that ratio.

   No external dependencies.
   ========================================================================= */

(function (root) {
  'use strict';

  // ------- Config -------
  var BURGUNDY   = '#7c3043';
  var AMBER      = '#d4a017';
  var CREAM      = '#f5f4f0';
  var CREAM_PAPER= '#faf9f7';
  var BORDER     = '#e0ddd5';

  // Cap device-pixel-ratio used for the annotation canvas. Retina screens
  // often report 2–3; annotation strokes don't need full retina sharpness
  // and the canvas texture memory scales with dpr^2. 1.5 is a sweet spot.
  var MAX_DPR = 1.5;

  var TOOLS = {
    pen:         { color: BURGUNDY,  width: 2.4, alpha: 1.00, composite: 'source-over' },
    highlighter: { color: AMBER,     width: 14,  alpha: 0.28, composite: 'multiply'    },
    eraser:      { color: 'rgba(0,0,0,1)', width: 18, alpha: 1, composite: 'destination-out' }
  };

  var state = {
    enabled: false,
    tool: 'pen',
    strokes: [],   // persisted shape list (coords are document-space)
    current: null, // in-progress stroke
    redoStack: [],
    canvas: null,
    ctx: null,
    toolbar: null,
    dpr: 1,
    scrollRaf: 0,
    storageKey: 'pb-pencil::' + location.pathname
  };

  // ------- Mount -------
  function enable() {
    if (state.enabled) return;
    state.enabled = true;
    document.body.setAttribute('data-pencil', 'on');
    mountCanvas();
    mountToolbar();
    loadStrokes();
    redraw();
    window.addEventListener('resize', onResize, { passive: true });
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  function disable() {
    if (!state.enabled) return;
    state.enabled = false;
    document.body.setAttribute('data-pencil', 'off');
    window.removeEventListener('resize', onResize);
    window.removeEventListener('scroll', onScroll);
    if (state.scrollRaf) {
      cancelAnimationFrame(state.scrollRaf);
      state.scrollRaf = 0;
    }
    if (state.canvas) state.canvas.remove();
    if (state.toolbar) state.toolbar.remove();
    state.canvas = state.ctx = state.toolbar = null;
  }

  function mountCanvas() {
    var c = document.createElement('canvas');
    c.className = 'pb-pencil-canvas';
    document.body.appendChild(c);
    state.canvas = c;
    state.ctx = c.getContext('2d');
    sizeCanvas();

    c.addEventListener('pointerdown', onPointerDown);
    c.addEventListener('pointermove', onPointerMove);
    c.addEventListener('pointerup',   onPointerUp);
    c.addEventListener('pointercancel', onPointerUp);
    c.addEventListener('pointerleave',  onPointerUp);
  }

  function sizeCanvas() {
    var c = state.canvas;
    if (!c) return;
    // Canvas is fixed-positioned and covers only the viewport. Strokes
    // remain stored in document-space; redraw() translates by scroll.
    var dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR);
    var w = window.innerWidth;
    var h = window.innerHeight;
    c.style.width  = w + 'px';
    c.style.height = h + 'px';
    c.width  = Math.floor(w * dpr);
    c.height = Math.floor(h * dpr);
    state.dpr = dpr;
    state.ctx.lineCap = 'round';
    state.ctx.lineJoin = 'round';
  }

  function onResize() { sizeCanvas(); redraw(); }

  function onScroll() {
    if (state.scrollRaf) return;
    state.scrollRaf = requestAnimationFrame(function () {
      state.scrollRaf = 0;
      redraw();
    });
  }

  // ------- Toolbar -------
  function mountToolbar() {
    var bar = document.createElement('div');
    bar.className = 'pb-pencil-toolbar';
    bar.setAttribute('role', 'toolbar');
    bar.setAttribute('aria-label', 'Pencil tools');

    bar.innerHTML = [
      btn('pen',         'Pen',         penGlyph()),
      btn('highlighter', 'Highlighter', hiGlyph()),
      btn('eraser',      'Eraser',      erGlyph()),
      '<span class="pb-sep"></span>',
      actionBtn('undo',  'Undo',  undoGlyph()),
      actionBtn('clear', 'Clear', clearGlyph()),
      '<span class="pb-sep"></span>',
      actionBtn('save',  'Save annotations (.json)', saveGlyph()),
      actionBtn('load',  'Load annotations',         loadGlyph()),
      '<span class="pb-sep"></span>',
      actionBtn('close', 'Exit pencil mode', closeGlyph())
    ].join('');

    document.body.appendChild(bar);
    state.toolbar = bar;

    bar.addEventListener('click', function (e) {
      var t = e.target.closest('button');
      if (!t) return;
      var kind = t.dataset.tool || t.dataset.action;
      if (t.dataset.tool)        { setTool(t.dataset.tool); }
      else if (kind === 'undo')  { undo(); }
      else if (kind === 'clear') { clearAll(); }
      else if (kind === 'save')  { saveToFile(); }
      else if (kind === 'load')  { loadFromFile(); }
      else if (kind === 'close') { disable(); }
    });

    setTool('pen');
  }

  function btn(tool, label, glyph) {
    return '<button type="button" data-tool="' + tool + '" aria-label="' + label + '" title="' + label + '">' + glyph + '</button>';
  }
  function actionBtn(action, label, glyph) {
    return '<button type="button" data-action="' + action + '" aria-label="' + label + '" title="' + label + '">' + glyph + '</button>';
  }

  function setTool(tool) {
    state.tool = tool;
    var btns = state.toolbar.querySelectorAll('button[data-tool]');
    for (var i = 0; i < btns.length; i++) {
      btns[i].classList.toggle('active', btns[i].dataset.tool === tool);
    }
  }

  // ------- Pointer handling -------
  function shouldDraw(e) {
    // Pen always draws. Mouse draws. Finger is ignored (lets page scroll).
    return e.pointerType === 'pen' || e.pointerType === 'mouse';
  }

  function getPoint(e) {
    var rect = state.canvas.getBoundingClientRect();
    // pressure: pen gives 0..1; mouse gives 0.5 while pressed. Normalize.
    var p = e.pressure;
    if (e.pointerType === 'mouse' || p === 0 || p === 0.5 || p == null) p = 0.5;
    return {
      x: e.clientX - rect.left + window.scrollX - (rect.left + window.scrollX - rect.left),
      y: e.clientY - rect.top  + window.scrollY - (rect.top  + window.scrollY - rect.top),
      p: p
    };
  }
  // Simpler coords: the canvas is positioned absolute at 0,0 of document, so:
  function pt(e) {
    return {
      x: e.pageX,
      y: e.pageY,
      p: (e.pointerType === 'pen' && e.pressure > 0) ? e.pressure : 0.5
    };
  }

  function onPointerDown(e) {
    if (!shouldDraw(e)) return;
    e.preventDefault();
    state.canvas.setPointerCapture(e.pointerId);
    var cfg = TOOLS[state.tool];
    state.current = {
      tool: state.tool,
      color: cfg.color,
      baseWidth: cfg.width,
      alpha: cfg.alpha,
      composite: cfg.composite,
      points: [pt(e)]
    };
    state.redoStack.length = 0;
  }

  function onPointerMove(e) {
    if (!state.current) return;
    if (!shouldDraw(e)) return;
    e.preventDefault();
    // Coalesced events give buttery-smooth pen input on iPad/Safari.
    var events = (e.getCoalescedEvents && e.getCoalescedEvents()) || [e];
    for (var i = 0; i < events.length; i++) {
      state.current.points.push(pt(events[i]));
    }
    // The viewport canvas is re-composited each frame against the stored
    // strokes + the current in-progress one, so a full redraw stays
    // consistent with scroll-translation (incremental live-draw would
    // leave stale pixels at the old scroll position).
    redraw();
  }

  function onPointerUp(e) {
    if (!state.current) return;
    if (state.current.points.length > 1) {
      state.strokes.push(state.current);
      saveStrokes();
    }
    state.current = null;
    redraw(); // composite final cleanly
  }

  // ------- Drawing -------
  function drawStroke(s, live) {
    var ctx = state.ctx;
    ctx.save();
    ctx.globalAlpha = s.alpha;
    ctx.globalCompositeOperation = s.composite;
    ctx.strokeStyle = s.color;

    var pts = s.points;
    if (pts.length < 2) {
      // single tap — dot
      ctx.beginPath();
      ctx.fillStyle = s.color;
      var r = s.baseWidth * 0.6;
      ctx.arc(pts[0].x, pts[0].y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      return;
    }

    // Quadratic mid-point smoothing, variable width via pressure.
    for (var i = 1; i < pts.length; i++) {
      var p0 = pts[i - 1], p1 = pts[i];
      var w = s.baseWidth * (0.4 + 1.1 * p1.p);
      ctx.lineWidth = w;
      ctx.beginPath();
      if (i === 1) {
        ctx.moveTo(p0.x, p0.y);
      } else {
        var pp = pts[i - 2];
        var midA = { x: (pp.x + p0.x) / 2, y: (pp.y + p0.y) / 2 };
        ctx.moveTo(midA.x, midA.y);
      }
      var midB = { x: (p0.x + p1.x) / 2, y: (p0.y + p1.y) / 2 };
      ctx.quadraticCurveTo(p0.x, p0.y, midB.x, midB.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  function redraw() {
    var ctx = state.ctx;
    if (!ctx) return;
    // Clear the entire device-pixel buffer.
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, state.canvas.width, state.canvas.height);
    // Strokes are stored in document-space; the canvas is viewport-fixed,
    // so translate by -scroll. Translation values are in device pixels.
    var dpr = state.dpr || 1;
    var sx = window.scrollX || window.pageXOffset || 0;
    var sy = window.scrollY || window.pageYOffset || 0;
    ctx.setTransform(dpr, 0, 0, dpr, -sx * dpr, -sy * dpr);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (var i = 0; i < state.strokes.length; i++) {
      drawStroke(state.strokes[i], false);
    }
    // Keep the in-progress stroke visible during scroll/pointermove.
    if (state.current) drawStroke(state.current, true);
  }

  // ------- Actions -------
  function undo() {
    if (!state.strokes.length) return;
    state.redoStack.push(state.strokes.pop());
    saveStrokes();
    redraw();
  }

  function clearAll() {
    if (!state.strokes.length) return;
    if (!confirm('Clear all annotations on this page?')) return;
    state.redoStack = state.strokes.slice();
    state.strokes = [];
    saveStrokes();
    redraw();
  }

  // ------- Persistence -------
  function saveStrokes() {
    try { localStorage.setItem(state.storageKey, JSON.stringify(state.strokes)); }
    catch (e) {}
  }
  function loadStrokes() {
    try {
      var raw = localStorage.getItem(state.storageKey);
      state.strokes = raw ? JSON.parse(raw) : [];
    } catch (e) { state.strokes = []; }
  }

  // ------- Save / load to file -------
  function saveToFile() {
    var payload = {
      kind: 'palatino-pencil',
      version: 1,
      pathname: location.pathname,
      savedAt: new Date().toISOString(),
      strokes: state.strokes
    };
    var blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    var base = (location.pathname.split('/').pop() || 'document').replace(/\.[^.]+$/, '');
    a.href = url;
    a.download = base + '-annotations.json';
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(url); a.remove(); }, 0);
  }

  function loadFromFile() {
    var inp = document.createElement('input');
    inp.type = 'file';
    inp.accept = 'application/json,.json';
    inp.addEventListener('change', function () {
      var f = inp.files && inp.files[0];
      if (!f) return;
      var r = new FileReader();
      r.onload = function () {
        try {
          var data = JSON.parse(r.result);
          var arr = Array.isArray(data) ? data : data.strokes;
          if (!Array.isArray(arr)) throw new Error('no strokes');
          var replace = !state.strokes.length ||
            confirm('Replace current annotations with the loaded file? Cancel to merge.');
          state.strokes = replace ? arr : state.strokes.concat(arr);
          saveStrokes();
          redraw();
        } catch (e) {
          alert('Could not load: ' + e.message);
        }
      };
      r.readAsText(f);
    });
    inp.click();
  }

  // ------- Icons (burgundy-stroke, inherit currentColor) -------
  function svg(path, extra) {
    return '<svg viewBox="0 0 20 20" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">' + path + (extra || '') + '</svg>';
  }
  function penGlyph()   { return svg('<path d="M3 17 L5 12 L13 4 L16 7 L8 15 Z"/><path d="M11 6 L14 9"/>'); }
  function hiGlyph()    { return svg('<path d="M4 15 L9 10 L14 15"/><rect x="3" y="14" width="12" height="4" rx="1"/>'); }
  function erGlyph()    { return svg('<path d="M4 14 L10 8 L14 12 L10 16 L6 16 Z"/><path d="M4 18 L14 18"/>'); }
  function undoGlyph()  { return svg('<path d="M6 7 L3 10 L6 13"/><path d="M3 10 H12 A5 5 0 0 1 12 17 H8"/>'); }
  function clearGlyph() { return svg('<path d="M5 6 H15"/><path d="M7 6 V16 A1 1 0 0 0 8 17 H12 A1 1 0 0 0 13 16 V6"/><path d="M8 6 V4 H12 V6"/>'); }
  function closeGlyph() { return svg('<path d="M5 5 L15 15"/><path d="M15 5 L5 15"/>'); }
  function saveGlyph()  { return svg('<path d="M4 4 H13 L16 7 V16 A0 0 0 0 1 16 16 H4 Z"/><path d="M6 4 V8 H12 V4"/><path d="M7 16 V12 H13 V16"/>'); }
  function loadGlyph()  { return svg('<path d="M4 14 V16 H16 V14"/><path d="M10 4 V12"/><path d="M6 9 L10 13 L14 9"/>'); }

  // ------- Expose -------
  root.PalatinoPencil = {
    enable: enable,
    disable: disable,
    isEnabled: function () { return state.enabled; },
    setTool: setTool,
    undo: undo,
    clear: clearAll,
    export: function () { return JSON.parse(JSON.stringify(state.strokes)); },
    import: function (arr) { state.strokes = Array.isArray(arr) ? arr : []; saveStrokes(); redraw(); },
    saveToFile: saveToFile,
    loadFromFile: loadFromFile
  };

  // Auto-enable if <body data-pencil="on"> is set at load time.
  function autoBoot() {
    if (document.body && document.body.getAttribute('data-pencil') === 'on') {
      enable();
    }
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoBoot);
  } else {
    autoBoot();
  }

})(window);
