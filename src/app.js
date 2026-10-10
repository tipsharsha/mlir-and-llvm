(function () {
  'use strict';

  var KEY = 'tm-mlir-app:v1';
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var esc = function (s) {
    return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; });
  };
  var today = function () {
    var d = new Date();
    return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0');
  };

  var manifest, state, serverMode = false, chapters = [], widgets = [], saveTimer = null, laterOpen = false;
  var emptyState = function () { return { steps: {}, done: {}, read: {}, last: '' }; };

  // ---------------------------------------------------------------- state
  function hasData(s) { return !!s && (Object.keys(s.steps || {}).length > 0 || Object.keys(s.read || {}).length > 0); }

  function legacyState() {
    var a = null, b = null;
    try { a = JSON.parse(localStorage.getItem('tm-mlir-001:v1')); } catch (e) { /* none */ }
    try { b = JSON.parse(localStorage.getItem('tm-mlir-002:v1')); } catch (e) { /* none */ }
    if (!a && !b) return null;
    return { steps: (a && a.steps) || {}, done: (a && a.done) || {}, read: (b && b.read) || {}, last: '' };
  }

  function loadState() {
    return fetch('/api/state', { cache: 'no-store' }).then(function (r) {
      return r.ok ? r.json() : null;
    }).catch(function () { return null; }).then(function (server) {
      var local = null;
      try { local = JSON.parse(localStorage.getItem(KEY)); } catch (e) { /* none */ }
      serverMode = server !== null && typeof server === 'object';
      var src = hasData(server) ? server : hasData(local) ? local : (legacyState() || server || local || {});
      state = Object.assign(emptyState(), src);
      if (serverMode && !hasData(server) && hasData(state)) save(true);
    });
  }

  function save(now) {
    try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* storage unavailable */ }
    if (!serverMode) return;
    clearTimeout(saveTimer);
    var go = function (keep) {
      saveTimer = null;
      fetch('/api/state', {
        method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(state), keepalive: !!keep
      }).catch(function () { /* the next change saves again */ });
    };
    if (now) go(); else saveTimer = setTimeout(go, 250);
  }
  window.addEventListener('pagehide', function () {
    if (serverMode && saveTimer) { clearTimeout(saveTimer); saveTimer = null; fetch('/api/state', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(state), keepalive: true }).catch(function () {}); }
  });

  // ---------------------------------------------------------------- content index
  function readMinutes(node) {
    var c = node.cloneNode(true);
    $$('.a', c).forEach(function (n) { n.remove(); });
    var words = (c.textContent || '').split(/\s+/).filter(Boolean).length;
    return Math.max(1, Math.round(words / 180));
  }

  function buildIndex() {
    chapters = [];
    manifest.tabs.forEach(function (tab) {
      (tab.chapters || []).forEach(function (ch) {
        ch.tab = tab;
        var root = ch._tpl.content;
        ch.tasks = $$('[data-task]', root).map(function (t) {
          return {
            id: t.dataset.task, title: t.dataset.title, time: t.dataset.time, anchor: t.id,
            steps: $$('li.step:not(.opt)', t).map(function (li) { return t.dataset.task + ':' + li.querySelector('.n').textContent.trim(); })
          };
        });
        ch.reads = $$('[data-read]', root).map(function (a) {
          return { id: a.dataset.read, title: a.dataset.title, ref: a.dataset.ref, anchor: a.id, min: readMinutes(a.querySelector('.task-body')) };
        });
        chapters.push(ch);
      });
    });
  }

  function chProg(ch) {
    var d = 0, t = 0;
    ch.tasks.forEach(function (x) { x.steps.forEach(function (k) { t++; if (state.steps[k]) d++; }); });
    ch.reads.forEach(function (r) { t++; if (state.read[r.id]) d++; });
    return { d: d, t: t };
  }
  function tabProg(tab) {
    var d = 0, t = 0;
    (tab.chapters || []).forEach(function (ch) { var p = chProg(ch); d += p.d; t += p.t; });
    return { d: d, t: t };
  }
  function overall() {
    var d = 0, t = 0;
    manifest.tabs.forEach(function (tab) { var p = tabProg(tab); d += p.d; t += p.t; });
    return { d: d, t: t };
  }

  // ---------------------------------------------------------------- routing
  function href(tab, ch, anchor) {
    return '#/' + tab.id + (ch ? '/' + ch.id : '') + (anchor ? '/' + anchor : '');
  }
  function resolve(hash) {
    var p = String(hash || '').replace(/^#\/?/, '').split('/').filter(Boolean).map(decodeURIComponent);
    var tab = manifest.tabs.filter(function (t) { return t.id === (p[0] || 'home'); })[0] || manifest.tabs[0];
    var list = tab.chapters || [];
    var ch = list.filter(function (c) { return c.id === p[1]; })[0] || list[0] || null;
    return { tab: tab, ch: ch, anchor: p[2] || '' };
  }
  function route() {
    var r = resolve(location.hash);
    render(r.tab, r.ch, r.anchor);
  }

  // ---------------------------------------------------------------- rendering
  function render(tab, ch, anchor) {
    document.body.classList.remove('nav-open');
    widgets = [];
    var content = $('#content');
    content.innerHTML = '';
    var sheet = document.createElement('main');
    sheet.className = 'sheet';
    if (tab.planned) {
      sheet.innerHTML = plannedHtml(tab);
    } else {
      if (!ch.hideHeader) {
        sheet.insertAdjacentHTML('beforeend',
          '<div class="ch-head"><span class="num">' + esc(ch.num || '') + '</span><span class="wk">' + esc(tab.label + ' · ' + tab.sub) + '</span></div><h2>' + esc(ch.title) + '</h2>');
      }
      var body = document.createElement('div');
      body.className = 'chapter-body';
      body.appendChild(ch._tpl.content.cloneNode(true));
      sheet.appendChild(body);
      hydrate(body);
      var nav = chapterNav(ch);
      if (nav) sheet.appendChild(nav);
    }
    content.appendChild(sheet);

    var label = tab.planned ? tab.label + ' · ' + tab.sub : (ch.hideHeader ? tab.label : tab.label + ' · ' + ch.title);
    $('#crumb').textContent = label;
    document.title = label + ' · TM-MLIR';
    if (tab.id !== 'home' && ch) { state.last = href(tab, ch); save(); }
    refreshChrome(tab, ch);

    var target = anchor ? document.getElementById(anchor) : null;
    if (target) {
      target.scrollIntoView();
      target.classList.add('flash');
    } else {
      window.scrollTo(0, 0);
    }
  }

  function plannedHtml(tab) {
    var n = parseInt(tab.id.split('-')[1], 10), row = null;
    var sched = chapters.filter(function (c) { return c.tab.id === 'program' && c.id === 'schedule'; })[0];
    if (sched) {
      $$('tbody tr', sched._tpl.content).forEach(function (tr) {
        if (parseInt(tr.cells[0].textContent, 10) === n) row = tr;
      });
    }
    var cell = function (i) { return row ? row.cells[i].innerHTML : '—'; };
    return '<div class="ch-head"><span class="num">PLANNED</span><span class="wk">' + esc(tab.label) + '</span></div>' +
      '<h2>' + esc(tab.label + ' · ' + tab.sub) + '</h2>' +
      '<div class="note"><span class="nh">NOTE:</span>This week is not in this issue. A later revision adds its chapters and tasks.</div>' +
      '<table><tbody>' +
      '<tr><th>Module</th><td>' + cell(1) + '</td></tr>' +
      '<tr><th>Subject</th><td>' + cell(2) + '</td></tr>' +
      '<tr><th>Main references</th><td>' + cell(3) + '</td></tr>' +
      '<tr><th>Output</th><td>' + cell(4) + '</td></tr></tbody></table>' +
      '<p><a href="#/program/schedule">See the program schedule</a></p>';
  }

  function chapterNav(ch) {
    var flat = chapters.filter(function (c) { return !c.hideHeader; });
    var i = flat.indexOf(ch);
    if (i < 0) return null;
    var prev = flat[i - 1], next = flat[i + 1];
    var link = function (c, cls, dir) {
      return '<a class="' + cls + '" href="' + href(c.tab, c) + '"><span class="dir">' + dir + '</span>' + esc(c.tab.label + ' · ' + c.title) + '</a>';
    };
    var nav = document.createElement('nav');
    nav.className = 'chapter-nav';
    nav.setAttribute('aria-label', 'Chapters');
    nav.innerHTML = (prev ? link(prev, 'cn-prev', '← Previous') : '<span></span>') + (next ? link(next, 'cn-next', 'Next →') : '');
    return nav;
  }

  // ---------------------------------------------------------------- chrome (sidebar, top bar)
  function tabHtml(tab, cur) {
    var p = tabProg(tab), active = cur && cur.tab === tab;
    var cnt = tab.planned ? '' : (p.t ? p.d + '/' + p.t : '');
    return '<a class="tab' + (active ? ' active' : '') + (tab.planned ? ' planned' : '') + (p.t && p.d === p.t ? ' done' : '') +
      '" href="#/' + tab.id + '"><span><span class="t1">' + esc(tab.label) + '</span><span class="t2">' + esc(tab.sub) + '</span></span><span class="cnt">' + cnt + '</span></a>';
  }
  function chapsHtml(tab, cur) {
    if (!tab.chapters || tab.chapters.length < 2) return '';
    return '<div class="chaps">' + tab.chapters.map(function (c) {
      var p = chProg(c);
      return '<a class="chap' + (cur.ch === c ? ' active' : '') + (p.t && p.d === p.t ? ' done' : '') + '" href="' + href(tab, c) +
        '"><span class="cnum">' + esc(c.num || '') + '</span><span>' + esc(c.title) + '</span><i class="tick">✓</i></a>';
    }).join('') + '</div>';
  }

  function renderNav(cur) {
    var html = '', lastGroup = '', planned = [];
    var flush = function () {
      if (!planned.length) return;
      var open = laterOpen || planned.indexOf(cur.tab) >= 0;
      html += '<details class="later"' + (open ? ' open' : '') + '><summary>Later weeks · planned</summary>' +
        planned.map(function (t) { return tabHtml(t, cur); }).join('') + '</details>';
      planned = [];
    };
    manifest.tabs.forEach(function (tab) {
      if (tab.planned) { planned.push(tab); return; }
      flush();
      if (tab.group !== lastGroup) { html += '<div class="nav-label">' + esc(tab.group) + '</div>'; lastGroup = tab.group; }
      html += tabHtml(tab, cur);
      if (cur.tab === tab) html += chapsHtml(tab, cur);
    });
    flush();
    $('#nav').innerHTML = html;
    $('#save-note').textContent = serverMode ? 'Progress is saved in data/progress.json.' : 'Progress is saved in this browser only. Run npm start to save it in a file.';
  }

  function refreshChrome(tab, ch) {
    if (tab) { refreshChrome.cur = { tab: tab, ch: ch }; }
    var cur = refreshChrome.cur;
    if (!cur) return;
    renderNav(cur);
    var o = overall();
    $('#top-bar').style.width = (o.t ? 100 * o.d / o.t : 0) + '%';
    $('#top-label').textContent = o.d + ' / ' + o.t;
    widgets.forEach(function (draw) { draw(); });
  }

  // ---------------------------------------------------------------- hydrate chapter content
  function stepCounts(task) {
    var all = $$('li.step:not(.opt)', task);
    return [all.filter(function (li) { return li.classList.contains('done'); }).length, all.length];
  }
  function updateTask(task, so) {
    var id = task.dataset.task, c = stepCounts(task), complete = c[1] > 0 && c[0] === c[1];
    if (complete && !state.done[id]) state.done[id] = today();
    if (!complete) delete state.done[id];
    so.classList.toggle('complete', complete);
    so.innerHTML = '<span>Steps complete: ' + c[0] + ' / ' + c[1] + '</span><span>' + (complete ? 'Task complete · ' + state.done[id] : 'Task open') + '</span>';
  }

  function hydrateTask(task) {
    var id = task.dataset.task;
    var so = document.createElement('div');
    so.className = 'signoff';
    $$('li.step', task).forEach(function (li) {
      var n = li.querySelector('.n').textContent.trim(), key = id + ':' + n;
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.setAttribute('aria-label', 'Step ' + n + ' complete');
      cb.checked = !!state.steps[key];
      li.classList.toggle('done', cb.checked);
      cb.addEventListener('change', function () {
        if (cb.checked) state.steps[key] = 1; else delete state.steps[key];
        li.classList.toggle('done', cb.checked);
        updateTask(task, so);
        save();
        refreshChrome();
      });
      li.insertBefore(cb, li.firstChild);
    });
    task.appendChild(so);
    updateTask(task, so);
  }

  function hydrateRead(card) {
    var id = card.dataset.read;
    card.querySelector('.time').textContent = 'About ' + readMinutes(card.querySelector('.task-body')) + ' min';
    var so = document.createElement('div');
    so.className = 'signoff';
    var label = document.createElement('label');
    var cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = !!state.read[id];
    cb.setAttribute('aria-label', 'Mark ' + card.dataset.ref + ' as read');
    label.appendChild(cb);
    label.appendChild(document.createTextNode('I have read this digest'));
    var st = document.createElement('span');
    so.appendChild(label);
    so.appendChild(st);
    card.appendChild(so);
    var paint = function () {
      so.classList.toggle('complete', cb.checked);
      st.textContent = cb.checked ? 'Read · ' + state.read[id] : 'Open';
    };
    cb.addEventListener('change', function () {
      if (cb.checked) state.read[id] = today(); else delete state.read[id];
      paint(); save(); refreshChrome();
    });
    paint();
  }

  function addCopy(pre) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'copy';
    b.textContent = 'Copy';
    b.addEventListener('click', function () {
      var text = pre.querySelector('code').textContent.replace(/\s+$/, '') + '\n';
      var ok = function () { b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy'; }, 1200); };
      var fallback = function () {
        var ta = document.createElement('textarea');
        ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); } catch (e) { /* ignore */ }
        document.body.removeChild(ta);
        ok();
      };
      if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(ok, fallback); else fallback();
    });
    pre.appendChild(b);
  }

  function hydrate(root) {
    $$('.task[data-task]', root).forEach(hydrateTask);
    $$('article[data-read]', root).forEach(hydrateRead);
    $$('pre.cmd, pre.code:not(.nocopy)', root).forEach(addCopy);
    $$('[data-widget]', root).forEach(function (w) {
      if (w.dataset.widget === 'record') recordWidget(w);
      if (w.dataset.widget === 'home') homeWidget(w);
    });
  }

  // ---------------------------------------------------------------- widgets
  function recordWidget(tbl) {
    var tab = manifest.tabs.filter(function (t) { return t.id === tbl.dataset.tab; })[0];
    var draw = function () {
      var rows = '';
      tab.chapters.forEach(function (ch) {
        ch.tasks.forEach(function (t) {
          var d = state.done[t.id], n = t.steps.filter(function (k) { return state.steps[k]; }).length;
          rows += '<tr' + (d ? ' class="ok"' : '') + '><td class="mono"><a href="' + href(tab, ch, t.anchor) + '">' + esc(t.id) + '</a></td><td>' + esc(t.title) +
            '</td><td>' + esc(t.time || '') + '</td><td>' + n + ' / ' + t.steps.length + '</td><td>' + (d ? 'COMPLETE' : 'OPEN') + '</td><td class="mono">' + (d || '—') + '</td></tr>';
        });
        ch.reads.forEach(function (r) {
          var d = state.read[r.id];
          rows += '<tr' + (d ? ' class="ok"' : '') + '><td class="mono"><a href="' + href(tab, ch, r.anchor) + '">' + esc(r.ref) + '</a></td><td>' + esc(r.title) +
            '</td><td>' + r.min + ' min</td><td>' + (d ? '1 / 1' : '0 / 1') + '</td><td>' + (d ? 'READ' : 'OPEN') + '</td><td class="mono">' + (d || '—') + '</td></tr>';
        });
      });
      tbl.tBodies[0].innerHTML = rows;
    };
    widgets.push(draw);
    draw();
  }

  function firstUnfinished() {
    var found = chapters.filter(function (c) {
      var p = chProg(c);
      return /^week-/.test(c.tab.id) && p.t > 0 && p.d < p.t;
    })[0];
    return found || chapters.filter(function (c) { return c.tab.id === 'week-0'; })[0];
  }

  function homeWidget(host) {
    var draw = function () {
      var o = overall(), pct = o.t ? Math.round(100 * o.d / o.t) : 0;
      var last = state.last && /^#\/[^/]+\/[^/]+/.test(state.last) ? resolve(state.last) : null;
      var next = firstUnfinished();
      var started = o.d > 0 || !!last;
      var cont = last && last.ch ? last : { tab: next.tab, ch: next };
      var cards = manifest.tabs.filter(function (t) { return /^week-/.test(t.id); }).map(function (t) {
        var p = tabProg(t);
        var body = t.planned ? '<span class="badge later">Planned</span>' :
          '<span class="badge now">Issued</span><div class="bar"><i style="width:' + (p.t ? 100 * p.d / p.t : 0) + '%"></i></div><span class="cnt">' + p.d + ' / ' + p.t + '</span>';
        return '<a class="wcard' + (t.planned ? ' planned' : '') + '" href="#/' + t.id + '"><span class="wn">' + esc(t.label.toUpperCase()) + '</span><b>' + esc(t.sub) + '</b>' + body + '</a>';
      }).join('');
      host.innerHTML = '<div class="dash"><div class="dash-top">' +
        '<div class="dash-card"><div class="lab">' + (started ? 'Continue where you stopped' : 'Start here') + '</div><div class="big">' + esc(cont.tab.label + ' · ' + cont.ch.title) +
        '</div><a class="btn-primary" href="' + href(cont.tab, cont.ch) + '">' + (started ? 'Continue' : 'Start') + '</a>' +
        (started && next && next !== cont.ch ? ' <a href="' + href(next.tab, next) + '" style="margin-left:10px;font-size:12px">Next unfinished: ' + esc(next.tab.label + ' · ' + next.title) + '</a>' : '') + '</div>' +
        '<div class="dash-card"><div class="lab">Overall progress</div><div class="big">' + o.d + ' / ' + o.t + ' · ' + pct + '%</div><div class="bar"><i style="width:' + pct + '%"></i></div>' +
        '<div class="sub">Steps and reading cards that you completed</div></div></div>' +
        '<h3>Weeks</h3><div class="weeks">' + cards + '</div>' +
        '<p class="noprint"><button type="button" class="btn" id="reset-progress">Clear all progress</button></p></div>';
      $('#reset-progress', host).addEventListener('click', function () {
        if (confirm('Clear all progress? This cannot be undone.')) {
          var keep = state.last;
          state = emptyState();
          state.last = keep;
          save(true);
          route();
        }
      });
    };
    widgets.push(draw);
    draw();
  }

  // ---------------------------------------------------------------- boot
  function fail(html) {
    $('#content').innerHTML = '<main class="sheet"><div class="notice">' + html + '</div></main>';
  }

  function boot() {
    if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
    if (location.protocol === 'file:') {
      fail('<p><b>This app needs its small local server.</b> The browser does not let a page open the chapter files from a folder.</p>' +
        '<p>Open a terminal in the project folder and run <code>npm start</code>. The app then opens in your browser.</p>');
      return;
    }
    fetch('content/manifest.json', { cache: 'no-store' }).then(function (r) {
      if (!r.ok) throw new Error('manifest.json: HTTP ' + r.status);
      return r.json();
    }).then(function (m) {
      manifest = m;
      var jobs = [];
      manifest.tabs.forEach(function (tab) {
        (tab.chapters || []).forEach(function (ch) {
          jobs.push(fetch('content/' + ch.file, { cache: 'no-store' }).then(function (r) {
            if (!r.ok) throw new Error(ch.file + ': HTTP ' + r.status);
            return r.text();
          }).then(function (html) {
            ch._tpl = document.createElement('template');
            ch._tpl.innerHTML = html;
          }));
        });
      });
      return Promise.all(jobs);
    }).then(loadState).then(function () {
      buildIndex();
      $('#nav').addEventListener('toggle', function (e) { if (e.target.matches && e.target.matches('details.later')) laterOpen = e.target.open; }, true);
      $('#menu-btn').addEventListener('click', function () { document.body.classList.toggle('nav-open'); });
      $('#scrim').addEventListener('click', function () { document.body.classList.remove('nav-open'); });
      window.addEventListener('hashchange', route);
      route();
    }).catch(function (e) {
      fail('<p><b>The content did not load.</b></p><p>' + esc(e.message) + '</p><p>Run <code>npm start</code> in the project folder, then reload this page.</p>');
    });
  }

  boot();
})();
