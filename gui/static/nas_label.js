/* [NAS 라벨링] 탭 - NAS 원본 클립을 하나씩 보며 라벨링한다.
 *
 * 서버: nas_labeling.py (/api/nl/*). index.html 의 $, $$, esc, toast,
 * opts, showModal, closeModal 을 그대로 쓴다.
 *
 * 실시간 표시: 3초마다 poll 로 (1) 그 사이 저장된 클립, (2) 누가 어느 클립을
 * 작업 중인지를 받아 목록/개수를 고친다. 작업 중 표시는 클립을 연 동안
 * 30초마다 갱신하고, 끊기면 서버에서 90초 뒤 저절로 풀린다.
 */
(() => {
const NL = {
  me: null, schema: null, datasets: [], ds: null,
  seq: 0, rows: [], offset: 0, limit: 100, total: 0, claims: {},
  cur: null,          // {uuid, rev, label, row, history, idx}
  form: null,         // 입력 중인 값
  manual: {},         // 묶음 점수를 사람이 직접 바꿨나 (아니면 카테고리 최댓값을 따라간다)
  dirty: false, inited: false, booting: false,
};
const LS_DS = 'edgegui.nl.ds', LS_LAST = 'edgegui.nl.last.';
const AXIS_KO = { illumination: '조도', precipitation: '강수',
  road_surface: '노면', atmospheric_obscurants: '대기 가림' };
const RSF_KO = { rain: '비', snow: '눈', fog: '안개' };
const tabOn = () => !$('#tab-nlabel').classList.contains('hide');
const lsGet = k => { try { return localStorage.getItem(k); } catch (e) { return null; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} };

async function nlapi(path, body) {
  const o = body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body) };
  const r = await fetch('/api/nl/' + path, o);
  let j;
  try { j = await r.json(); } catch (e) { throw new Error(`HTTP ${r.status}`); }
  if (r.status === 401 && path !== 'login') showLogin('로그인이 필요합니다');
  if (j.error) {
    const e = new Error(j.error); e.status = r.status; e.data = j; throw e;
  }
  return j;
}

/* ---------- 로그인 ---------- */
async function init() {
  if (NL.inited) return;
  NL.inited = true;
  try {
    const j = await nlapi('me');
    if (!j.me) return showLogin();
    await boot();
  } catch (e) {
    NL.inited = false;
    $('#nl-login').classList.remove('hide');
    $('#nl-login-msg').textContent = e.message;
  }
}
function showLogin(msg) {
  NL.me = null;
  $('#nl-app').classList.add('hide');
  $('#nl-login').classList.remove('hide');
  $('#nl-login-msg').textContent = msg || '';
  setTimeout(() => $('#nl-user').focus(), 0);
}
$('#nl-login-form').onsubmit = async ev => {
  ev.preventDefault();
  try {
    await nlapi('login', { user: $('#nl-user').value.trim(), password: $('#nl-pw').value });
    $('#nl-pw').value = '';
    await boot();
  } catch (e) { $('#nl-login-msg').textContent = e.message; }
};
$('#nl-logout').onclick = async () => {
  if (!confirmLeave()) return;
  releaseClaim();
  try { await nlapi('logout', {}); } catch (e) {}
  NL.cur = null; NL.dirty = false;
  showLogin('로그아웃했습니다');
};

async function boot() {
  if (NL.booting) return;
  NL.booting = true;
  try {
    const b = await nlapi('bootstrap');
    NL.me = b.me; NL.schema = b.schema; NL.datasets = b.datasets;
    $('#nl-login').classList.add('hide');
    $('#nl-app').classList.remove('hide');
    $('#nl-me').textContent = `${NL.me.name} (${NL.me.user})${NL.me.admin ? ' · 관리자' : ''}`;
    opts($('#nl-ds'), NL.datasets.map(d => [d.id, d.name]));
    const saved = lsGet(LS_DS);
    if (saved && NL.datasets.some(d => d.id === saved)) $('#nl-ds').value = saved;
    buildForm();
    await selectDataset($('#nl-ds').value);
  } finally { NL.booting = false; }
}

/* ---------- 데이터셋 / 목록 ---------- */
async function selectDataset(id) {
  releaseClaim();
  NL.ds = id; NL.cur = null; NL.offset = 0; NL.seq = 0; NL.dirty = false;
  lsSet(LS_DS, id);
  const d = NL.datasets.find(x => x.id === id) || {};
  $('#nl-path').innerHTML = `NAS: ${esc(d.nas_path)}<br>서버: ${esc(d.local_path)}`
    + `<br>라벨 파일: ${esc(d.labels_file)}`;
  renderEmptyClip();
  await loadList();
  // 마지막으로 보던 클립을 다시 연다(새로고침해도 이어서 작업).
  const last = lsGet(LS_LAST + id);
  if (last) openClip(last, { force: true, quiet: true });
}
$('#nl-ds').onchange = async ev => {
  if (!confirmLeave()) { ev.target.value = NL.ds; return; }
  await selectDataset(ev.target.value);
};

function listQuery() {
  return new URLSearchParams({
    ds: NL.ds, status: $('#nl-status').value, sort: $('#nl-sort').value,
    q: $('#nl-q').value.trim(), chunk: $('#nl-chunk').value.trim(),
    labeler: $('#nl-labeler').value || '', offset: NL.offset, limit: NL.limit,
  }).toString();
}
let listRetry = null;
async function loadList() {
  clearTimeout(listRetry);
  let j;
  try { j = await nlapi('clips?' + listQuery()); }
  catch (e) {
    if (e.data && e.data.indexing) {
      $('#nl-rows').innerHTML = `<tr><td colspan="4" class="muted">${esc(e.message)}
        - 클립 수가 많으면 처음 한 번은 수 분 걸릴 수 있습니다. 자동으로 다시 시도합니다.</td></tr>`;
      listRetry = setTimeout(loadList, 3000);
      return;
    }
    if (e.status !== 401) toast(e.message);
    return;
  }
  NL.rows = j.rows; NL.total = j.total; NL.offset = j.offset;
  NL.seq = Math.max(NL.seq, j.seq);
  renderCounts(j.counts);
  renderRows();
}
['#nl-status', '#nl-sort', '#nl-labeler'].forEach(s =>
  $(s).onchange = () => { NL.offset = 0; loadList(); });
$('#nl-reload').onclick = () => { NL.offset = 0; loadList(); };
['#nl-q', '#nl-chunk'].forEach(s => $(s).onkeydown = ev => {
  if (ev.key === 'Enter') { NL.offset = 0; loadList(); }
});
$('#nl-prevp').onclick = () => {
  if (NL.offset <= 0) return;
  NL.offset = Math.max(0, NL.offset - NL.limit); loadList();
};
$('#nl-nextp').onclick = () => {
  if (NL.offset + NL.limit >= NL.total) return;
  NL.offset += NL.limit; loadList();
};

function renderCounts(c) {
  if (!c) return;
  const pctv = c.total ? (c.labeled / c.total * 100).toFixed(c.labeled / c.total < 0.01 ? 2 : 1) : '0';
  const n = x => Number(x || 0).toLocaleString();
  $('#nl-counts').innerHTML = `
    <div><b>${n(c.total)}</b><span>전체 클립</span></div>
    <div><b>${n(c.unlabeled)}</b><span>미라벨</span></div>
    <div><b style="color:var(--ok)">${n(c.labeled)}</b><span>라벨 완료 (${pctv}%)</span></div>
    <div><b style="color:var(--warn)">${n(c.claimed)}</b><span>작업 중</span></div>
    <div><b>${n(c.mine)}</b><span>내가 라벨한</span></div>`;
  const sel = $('#nl-labeler'), keep = sel.value;
  const ls = Object.entries(c.labelers || {}).sort((a, b) => b[1] - a[1]);
  opts(sel, [['', '전체'], ...ls.map(([u, k]) => [u, `${u} (${k})`])]);
  sel.value = ls.some(([u]) => u === keep) ? keep : '';
}

function fmtTs(ts, full) {
  if (!ts) return '';
  const m = ts.match(/^(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)/);
  if (!m) return ts;
  return full ? `${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}:${m[6]}` : `${m[2]}-${m[3]} ${m[4]}:${m[5]}`;
}
function badge(r) {
  let s = r.labeled ? '<span class="st ok">라벨 완료</span>' : '<span class="st">미라벨</span>';
  const c = r.claimed_by;
  if (c) s += ` <span class="st wk" title="${esc(c)} 님이 보고 있습니다">✎ ${esc(c)}</span>`;
  return s;
}
const rowsSig = () => JSON.stringify(NL.rows.map(r => [r.uuid, r.rev, r.claimed_by]));
function renderRows() {
  NL._sig = rowsSig();
  const cur = NL.cur && NL.cur.uuid;
  $('#nl-rows').innerHTML = NL.rows.length ? NL.rows.map(r => `
    <tr data-u="${esc(r.uuid)}" class="${r.uuid === cur ? 'cur' : ''}">
      <td class="num muted">${(r.idx + 1).toLocaleString()}</td>
      <td class="mono" title="${esc(r.uuid)} · chunk ${r.chunk}">${esc(r.uuid.slice(0, 8))}…</td>
      <td>${badge(r)}</td>
      <td title="${r.labeled ? '최초 ' + esc(r.created_by) + ' ' + fmtTs(r.created_at, 1) : ''}">${r.labeled
        ? `${esc(r.updated_by)} <span class="muted">${fmtTs(r.updated_at)}${r.rev > 1 ? ` · r${r.rev}` : ''}</span>` : ''}</td>
    </tr>`).join('')
    : '<tr><td colspan="4" class="muted">조건에 맞는 클립이 없습니다.</td></tr>';
  $$('#nl-rows tr[data-u]').forEach(tr => tr.onclick = () => openClip(tr.dataset.u));
  const a = NL.total ? NL.offset + 1 : 0, b = Math.min(NL.offset + NL.limit, NL.total);
  $('#nl-pageinfo').textContent = `${a.toLocaleString()}–${b.toLocaleString()} / ${NL.total.toLocaleString()}`;
}

/* ---------- 클립 열기 ---------- */
function confirmLeave() {
  return !NL.dirty || confirm('저장하지 않은 라벨 변경이 있습니다. 버리고 이동할까요?');
}
function renderEmptyClip() {
  $('#nl-player').innerHTML = '<p class="muted">왼쪽 목록에서 클립을 고르거나 [다음 미라벨]을 누르세요.</p>'
    + '<div class="nl-ctrl"><button class="primary" id="nl-next0">다음 미라벨 클립 열기</button></div>';
  $('#nl-next0').onclick = () => goNext();
  $('#nl-hist').innerHTML = '클립을 열면 표시됩니다.';
  fillForm(null);
}

async function openClip(uuid, opt = {}) {
  if (!opt.force && !confirmLeave()) return false;
  let d;
  try { d = await nlapi(`clip?ds=${encodeURIComponent(NL.ds)}&uuid=${encodeURIComponent(uuid)}`); }
  catch (e) { if (!opt.quiet) toast(e.message); return false; }
  const same = NL.cur && NL.cur.uuid === uuid;
  NL.cur = { uuid, rev: d.row.rev, label: d.label, row: d.row, history: d.history, idx: d.idx };
  lsSet(LS_LAST + NL.ds, uuid);
  if (!same) { renderPlayer(); claim(); }
  renderHead();
  fillForm(d.label);
  renderHistory();
  $$('#nl-rows tr').forEach(tr => tr.classList.toggle('cur', tr.dataset.u === uuid));
  // scrollIntoView 는 페이지 전체까지 굴려 버린다 - 목록 상자 안에서만 맞춘다.
  const tr = $(`#nl-rows tr[data-u="${uuid}"]`), box = tr && tr.closest('.scroll');
  if (box && !same) {
    const t = tr.offsetTop, h = tr.offsetHeight;
    if (t < box.scrollTop + 30 || t + h > box.scrollTop + box.clientHeight)
      box.scrollTop = Math.max(0, t - box.clientHeight / 2);
  }
  return true;
}

function renderPlayer() {
  const u = NL.cur.uuid;
  const src = `/api/nl/video?ds=${encodeURIComponent(NL.ds)}&uuid=${encodeURIComponent(u)}`;
  const rate = lsGet('edgegui.nl.rate') || '1';
  $('#nl-player').innerHTML = `
    <div class="nl-head" id="nl-head"></div>
    <video id="nl-video" controls autoplay muted playsinline preload="auto" src="${src}"></video>
    <div class="nl-ctrl">
      <button id="nl-prev" title="목록 순서상 이전 클립">◀ 이전 클립</button>
      <button id="nl-next" title="목록 순서상 다음 클립">다음 클립 ▶</button>
      <button class="primary" id="nl-nextu" title="남이 작업 중인 클립은 건너뜁니다">다음 미라벨 ▶▶</button>
      <label class="muted" style="margin-left:auto">속도
        <select id="nl-rate">${['0.25', '0.5', '1', '1.5', '2', '4'].map(v =>
          `<option ${v === rate ? 'selected' : ''}>${v}</option>`).join('')}</select></label>
      <label class="muted"><input type="checkbox" id="nl-loop" checked> 반복</label>
      <a class="pill" href="${src}" download="${esc(u)}.mp4">다운로드</a>
    </div>
    <div class="nl-sub2"><kbd>Space</kbd> 재생/정지 · <kbd>←</kbd><kbd>→</kbd> 2초 이동 ·
      <kbd>Ctrl+S</kbd> 저장 · <kbd>Ctrl+Enter</kbd> 저장 후 다음 미라벨</div>`;
  const v = $('#nl-video');
  v.playbackRate = +rate; v.loop = true;
  v.onerror = () => {
    const err = v.error && v.error.code;
    $('#nl-head').insertAdjacentHTML('afterend', `<div class="nl-warn">영상을 재생하지 못했습니다
      (code ${err}). HEVC(H.265) 영상이라 브라우저가 이 코덱을 지원해야 합니다 -
      Windows 의 Chrome/Edge 를 권장합니다. [다운로드]로 받아 볼 수도 있습니다.</div>`);
  };
  $('#nl-rate').onchange = ev => { v.playbackRate = +ev.target.value; lsSet('edgegui.nl.rate', ev.target.value); };
  $('#nl-loop').onchange = ev => { v.loop = ev.target.checked; };
  $('#nl-prev').onclick = () => goNext({ any: true, dir: 'prev' });
  $('#nl-next').onclick = () => goNext({ any: true });
  $('#nl-nextu').onclick = () => goNext();
}

function renderHead() {
  const c = NL.cur, r = c.row;
  const other = NL.claims[c.uuid] && NL.claims[c.uuid] !== NL.me.user ? NL.claims[c.uuid]
    : (r.claimed_by && r.claimed_by !== NL.me.user ? r.claimed_by : null);
  const info = r.labeled
    ? `최초 ${esc(r.created_by)} ${fmtTs(r.created_at, 1)} · 최종 ${esc(r.updated_by)} ${fmtTs(r.updated_at, 1)} · rev ${r.rev}`
    : (r.rev ? `라벨 삭제됨 (rev ${r.rev})` : '아직 라벨 없음');
  $('#nl-head').innerHTML = `
    <span class="mono">${esc(c.uuid)}</span>
    <button class="pill" id="nl-copy" title="uuid 복사">복사</button>
    ${badge({ labeled: r.labeled })}
    <span class="muted">#${(c.idx + 1).toLocaleString()} · chunk ${r.chunk}</span>
    <div class="nl-sub2" style="flex-basis:100%;margin:0">${info}</div>
    ${other ? `<div class="nl-warn" style="flex-basis:100%">✎ ${esc(other)} 님이 이 클립을 보고 있습니다.
      함께 저장하면 나중에 저장한 쪽이 충돌 경고를 받습니다.</div>` : ''}
    <div id="nl-banner" style="flex-basis:100%"></div>`;
  $('#nl-copy').onclick = async () => {
    try { await navigator.clipboard.writeText(c.uuid); toast('복사했습니다'); }
    catch (e) { prompt('uuid', c.uuid); }
  };
}

async function goNext(o = {}) {
  if (!confirmLeave()) return;
  const q = new URLSearchParams({ ds: NL.ds });
  if (NL.cur) q.set('after', NL.cur.uuid);
  if (o.any) q.set('any', '1');
  if (o.dir) q.set('dir', o.dir);
  try {
    const j = await nlapi('next?' + q);
    if (!j.uuid) return toast('남은 미라벨 클립이 없습니다');
    await openClip(j.uuid, { force: true });
  } catch (e) { toast(e.message); }
}

/* ---------- 작업 중 표시 ---------- */
let claimTimer = null;
function claim() {
  clearInterval(claimTimer);
  const send = () => NL.cur && nlapi('claim', { ds: NL.ds, uuid: NL.cur.uuid })
    .then(j => { NL.claims = j.claims; }).catch(() => {});
  send();
  claimTimer = setInterval(send, 30000);
}
function releaseClaim() {
  clearInterval(claimTimer);
  if (!NL.cur || !NL.me) return;
  const body = JSON.stringify({ ds: NL.ds, uuid: NL.cur.uuid, release: true });
  try { navigator.sendBeacon('/api/nl/claim', new Blob([body], { type: 'application/json' })); }
  catch (e) {}
}
window.addEventListener('beforeunload', ev => {
  releaseClaim();
  if (NL.dirty) { ev.preventDefault(); ev.returnValue = ''; }
});

/* ---------- 라벨 입력 ---------- */
function scaleMap(text) {
  // prompts.DIFFICULTY_SCALES 는 "   0 = ..." 줄들이다.
  const m = {};
  String(text || '').split('\n').forEach(l => {
    const x = l.match(/^\s*(\d)\s*=\s*(.*)$/);
    if (x) m[x[1]] = x[2];
  });
  return m;
}
function catRubric(c, g) {
  const own = NL.schema.category_rubrics[c];
  return own ? own.scale : (NL.schema.group_rubrics[g] || {}).scale || {};
}
function seg(key, vals, title) {
  return `<span class="seg" data-k="${esc(key)}">${vals.map(v =>
    `<button type="button" data-v="${v}" title="${esc(title ? title(v) : '')}">${v === null ? '−' : v}</button>`).join('')}</span>`;
}

function buildForm() {
  const S = NL.schema;
  const groups = Object.entries(S.groups).map(([g, cats]) => {
    const gr = (S.group_rubrics[g] || {}).scale || {};
    return `<div class="nl-group">
      <div class="nl-gh"><b>${esc(g)}</b>
        ${seg('g:' + g, [0, 1, 2, 3, 4], v => gr[v] ? `${v}: ${gr[v]}` : `${v}: (기준표에 없는 값)`)}</div>
      <div class="nl-rub" id="nl-grub-${cssKey(g)}"></div>
      ${cats.map(c => {
        const rb = catRubric(c, g);
        return `<div class="nl-line" data-cat="${esc(c)}"><span class="nm" title="${esc(c)}">${esc(c)}</span>
          ${seg('c:' + c, [null, 1, 2, 3, 4], v => v === null ? '없음' : (rb[v] ? `${v}: ${rb[v]}` : `${v}`))}</div>`;
      }).join('')}
    </div>`;
  }).join('');
  const weather = S.weather_axes.map(a => {
    const sm = scaleMap(S.weather_scales[a]);
    return `<div class="nl-line"><span class="nm" title="${esc(a)}">${esc(AXIS_KO[a] || a)}
        <span class="muted mono" style="font-size:10px">${esc(a)}</span></span>
      ${seg('w:' + a, [0, 1, 2, 3, 4], v => sm[v] ? `${v}: ${sm[v]}` : String(v))}</div>
      <div class="nl-rub" id="nl-wrub-${a}"></div>`;
  }).join('');
  $('#nl-form').innerHTML = `
    <div class="nl-sub2">카테고리는 나온 것만 1~4로 고릅니다(−는 없음). 묶음 점수는 카테고리 최댓값이
      자동으로 들어가며, 직접 바꿀 수 있습니다. 버튼에 마우스를 올리면 기준표가 보입니다.</div>
    ${groups}
    <div class="nl-sec">비 / 눈 / 안개</div>
    <div>${S.rain_snow_fog.map(x =>
      `<span class="pill" data-rsf="${x}">${RSF_KO[x] || x} <span class="muted">${x}</span></span>`).join(' ')}</div>
    <div class="nl-sec">날씨 4축 (0~4)</div>
    ${weather}
    <div class="nl-sec">note</div>
    <textarea id="nl-note" rows="3" placeholder="특이사항"></textarea>
    <div class="nl-btns">
      <button class="primary" id="nl-save">저장 <kbd>Ctrl+S</kbd></button>
      <button class="primary" id="nl-savenext">저장 후 다음 미라벨 <kbd>Ctrl+↵</kbd></button>
      <button id="nl-revert">되돌리기</button>
      <button class="danger" id="nl-del" style="margin-left:auto">라벨 삭제</button>
    </div>`;
  $$('#nl-form .seg button').forEach(b => b.onclick = () => {
    const k = b.parentElement.dataset.k, raw = b.dataset.v;
    const v = raw === 'null' ? null : +raw;
    setField(k, v);
  });
  $$('#nl-form [data-rsf]').forEach(p => p.onclick = () => {
    const x = p.dataset.rsf, s = NL.form.rsf;
    s.has(x) ? s.delete(x) : s.add(x);
    markDirty(); renderForm();
  });
  $('#nl-note').oninput = ev => { NL.form.note = ev.target.value; markDirty(); };
  $('#nl-save').onclick = () => save(false);
  $('#nl-savenext').onclick = () => save(true);
  $('#nl-revert').onclick = () => {
    if (!NL.cur || (NL.dirty && !confirm('입력한 내용을 버리고 저장된 상태로 되돌릴까요?'))) return;
    fillForm(NL.cur.label);
  };
  $('#nl-del').onclick = delLabel;
}
function cssKey(s) { return s.replace(/[^A-Za-z0-9]/g, '_'); }

function groupMax(g) {
  return Math.max(0, ...NL.schema.groups[g].map(c => NL.form.cats[c] || 0));
}
function setField(k, v) {
  const f = NL.form;
  if (!NL.cur) return toast('먼저 클립을 여세요');
  if (k.startsWith('c:')) {
    const c = k.slice(2);
    if (v === null) delete f.cats[c]; else f.cats[c] = v;
    for (const [g, cs] of Object.entries(NL.schema.groups))
      if (cs.includes(c) && !NL.manual[g]) f.groups[g] = groupMax(g);
  } else if (k.startsWith('g:')) {
    const g = k.slice(2);
    f.groups[g] = v;
    NL.manual[g] = v !== groupMax(g);
  } else if (k.startsWith('w:')) {
    f.weather[k.slice(2)] = v;
  }
  markDirty(); renderForm();
}

function fillForm(label) {
  const S = NL.schema;
  if (!S) return;
  const L = label || {};
  NL.form = {
    cats: { ...(L.categories || {}) },
    groups: Object.fromEntries(Object.keys(S.groups).map(g => [g, L[g] ?? 0])),
    rsf: new Set(L.rain_snow_fog || []),
    weather: Object.fromEntries(S.weather_axes.map(a => [a, (L.weather || {})[a] ?? 0])),
    note: L.note || '',
  };
  NL.manual = Object.fromEntries(Object.keys(S.groups).map(g => [g, NL.form.groups[g] !== groupMax(g)]));
  $('#nl-note').value = NL.form.note;
  NL.dirty = false;
  // 옛 분류 체계에만 있는 카테고리는 화면에 칸이 없다 - 그대로 저장된다는 것만 알린다.
  const known = new Set(Object.values(S.groups).flat());
  const legacy = Object.keys(NL.form.cats).filter(c => !known.has(c));
  if (legacy.length) toast('현재 분류 체계에 없는 카테고리(그대로 유지됨): ' + legacy.join(', '), 5000);
  renderForm();
}

function renderForm() {
  const f = NL.form, S = NL.schema;
  const on = (key, v) => $$(`#nl-form .seg[data-k="${CSS.escape(key)}"] button`).forEach(b =>
    { b.classList.toggle('on', b.dataset.v === String(v)); b.classList.toggle('zero', b.dataset.v === '0' || b.dataset.v === 'null'); });
  for (const [g, cats] of Object.entries(S.groups)) {
    on('g:' + g, f.groups[g]);
    cats.forEach(c => {
      on('c:' + c, f.cats[c] ?? null);
      const line = $(`#nl-form .nl-line[data-cat="${CSS.escape(c)}"]`);
      if (line) line.classList.toggle('has', f.cats[c] != null);
    });
    const gr = (S.group_rubrics[g] || {}).scale || {}, mx = groupMax(g), el = $('#nl-grub-' + cssKey(g));
    const txt = gr[f.groups[g]] ? `${f.groups[g]}: ${gr[f.groups[g]]}` : '';
    el.className = 'nl-rub' + (f.groups[g] !== mx ? ' diff' : '');
    el.textContent = f.groups[g] !== mx
      ? `카테고리 최댓값(${mx})과 다르게 매겼습니다. ${txt}` : txt;
    el.title = txt;
  }
  S.weather_axes.forEach(a => {
    on('w:' + a, f.weather[a]);
    const sm = scaleMap(S.weather_scales[a]);
    const wr = $('#nl-wrub-' + a);
    wr.textContent = wr.title = f.weather[a] ? (sm[f.weather[a]] || '') : '';
  });
  $$('#nl-form [data-rsf]').forEach(p => p.classList.toggle('on', f.rsf.has(p.dataset.rsf)));
  const dis = !NL.cur;
  ['#nl-save', '#nl-savenext', '#nl-revert'].forEach(s => $(s).disabled = dis);
  $('#nl-del').disabled = dis || !(NL.cur && NL.cur.row.labeled);
  updateDirty();
}
function markDirty() { NL.dirty = true; updateDirty(); }
function updateDirty() {
  const r = NL.cur && NL.cur.row;
  $('#nl-dirty').textContent = !NL.cur ? '' : NL.dirty ? '● 저장 안 됨'
    : (r.labeled ? `저장됨 · rev ${r.rev}` : '아직 저장 전 (기본값: 전부 0)');
}

function collect() {
  const f = NL.form, out = { categories: { ...f.cats } };
  Object.keys(NL.schema.groups).forEach(g => { out[g] = f.groups[g]; });
  out.rain_snow_fog = NL.schema.rain_snow_fog.filter(x => f.rsf.has(x));
  out.weather = { ...f.weather };
  out.note = $('#nl-note').value;
  return out;
}

async function save(next) {
  if (!NL.cur) return;
  const uuid = NL.cur.uuid;
  try {
    const j = await nlapi('save', { ds: NL.ds, uuid, label: collect(), base_rev: NL.cur.rev });
    toast(j.unchanged ? '바뀐 내용이 없습니다' : `저장했습니다 (rev ${j.rev})`);
    NL.dirty = false;
    if (next) { await openClip(uuid, { force: true }); await goNext(); }
    else await openClip(uuid, { force: true });
    poll();
  } catch (e) {
    if (e.status === 409) {
      const c = e.data.conflict || {};
      const ok = confirm(`${c.by || '다른 사용자'} 님이 ${fmtTs(c.at, 1)}에 이 클립을 먼저 수정했습니다 (rev ${c.rev}).\n\n`
        + '[확인] 내 입력을 버리고 최신 라벨을 불러옵니다.\n'
        + '[취소] 내 입력을 유지합니다. 한 번 더 저장하면 그 수정을 덮어씁니다(이력에는 둘 다 남습니다).');
      if (ok) await openClip(uuid, { force: true });
      else { NL.cur.rev = c.rev; banner(`최신 rev ${c.rev} 기준으로 바꿨습니다. 다시 저장하면 덮어씁니다.`); }
    } else toast(e.message, 4000);
  }
}

async function delLabel() {
  if (!NL.cur || !NL.cur.row.labeled) return;
  if (!confirm('이 클립의 라벨을 삭제할까요? 미라벨 상태로 돌아가며 이력에는 남습니다.')) return;
  try {
    await nlapi('delete', { ds: NL.ds, uuid: NL.cur.uuid, base_rev: NL.cur.rev });
    toast('삭제했습니다');
    await openClip(NL.cur.uuid, { force: true });
    poll();
  } catch (e) {
    if (e.status === 409) { toast('다른 사용자가 먼저 수정했습니다 - 다시 불러옵니다', 4000); openClip(NL.cur.uuid, { force: true }); }
    else toast(e.message);
  }
}

function banner(msg, act) {
  const el = $('#nl-banner');
  if (!el) return;
  el.innerHTML = msg ? `<div class="nl-warn">${esc(msg)} ${act ? '<button id="nl-banbtn">다시 불러오기</button>' : ''}</div>` : '';
  if (act) $('#nl-banbtn').onclick = act;
}

/* ---------- 이력 ---------- */
function fmtV(v) {
  if (v === null || v === undefined) return '<span class="muted">없음</span>';
  if (typeof v === 'string') return v === '' ? '<span class="muted">(빈 값)</span>' : `"${esc(v)}"`;
  return esc(String(v));
}
function fieldKo(f) {
  if (f.startsWith('weather.')) return '날씨·' + (AXIS_KO[f.slice(8)] || f.slice(8));
  if (f.startsWith('categories.')) return f.slice(11);
  if (f === 'rain_snow_fog') return '비/눈/안개';
  return f;
}
function renderHistory() {
  const h = NL.cur.history || [];
  if (!h.length) { $('#nl-hist').innerHTML = '<span class="muted">아직 기록이 없습니다.</span>'; return; }
  const ACT = { create: '생성', update: '수정', delete: '삭제' };
  $('#nl-hist').innerHTML = h.map(ev => {
    // 생성은 모든 칸이 "없음 → 값"이라 기본값(0, 빈 값)까지 늘어놓으면 길기만 하다.
    const ch = ev.action === 'create'
      ? ev.changes.filter(c => c.to !== 0 && c.to !== '' && c.to !== null)
      : ev.changes;
    return `<div class="nl-ev ${ev.action}">
      <div><b>${ACT[ev.action] || ev.action}</b> · rev ${ev.rev} · ${esc(ev.user)} ·
        <span class="muted">${fmtTs(ev.ts, 1)}</span>${ev.import ? ` · <span class="muted">가져오기 ${esc(ev.import)}</span>` : ''}</div>
      <ul>${ch.length ? ch.map(c => `<li><span class="mono">${esc(fieldKo(c.field))}</span>
          ${fmtV(c.from)} → ${fmtV(c.to)}</li>`).join('')
        : `<li>${ev.action === 'create' ? '전부 0 (특이사항 없음)' : '바뀐 칸 없음'}</li>`}</ul>
    </div>`;
  }).join('');
}

/* ---------- 실시간 갱신 ---------- */
let polling = false;
async function poll() {
  if (polling || !NL.me || !NL.ds || !tabOn() || document.hidden) return;
  polling = true;
  try {
    const j = await nlapi(`poll?ds=${encodeURIComponent(NL.ds)}&since=${NL.seq}`);
    NL.claims = j.claims || {};
    if (j.counts) renderCounts(j.counts);
    const ch = new Map((j.changed || []).map(r => [r.uuid, r]));
    NL.rows = NL.rows.map(r => {
      const c = ch.get(r.uuid);
      const n = c ? { ...r, ...c, idx: r.idx } : { ...r };
      n.claimed_by = NL.claims[r.uuid] || null;
      return n;
    });
    // 입력 중에 표를 다시 그리면 스크롤/포커스가 흔들릴 수 있어, 바뀐 게 있을 때만.
    if (rowsSig() !== NL._sig) renderRows();
    NL.seq = j.seq;
    if (NL.cur && ch.has(NL.cur.uuid)) {
      const r = ch.get(NL.cur.uuid);
      if (r.rev !== NL.cur.rev) {
        if (!NL.dirty) await openClip(NL.cur.uuid, { force: true, quiet: true });
        else banner(`${r.updated_by || '다른 사용자'} 님이 방금 이 클립을 수정했습니다 (rev ${r.rev}).`,
          () => { NL.dirty = false; openClip(NL.cur.uuid, { force: true }); });
      }
    }
    if (NL.cur && $('#nl-head')) {
      const other = NL.claims[NL.cur.uuid];
      const was = NL._other;
      NL._other = other && other !== NL.me.user ? other : null;
      if (NL._other !== was) renderHead();
    }
  } catch (e) { /* 일시적인 네트워크 오류는 다음 주기에 */ }
  finally { polling = false; }
}
setInterval(poll, 3000);
document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });

/* ---------- 계정 ---------- */
$('#nl-acct').onclick = async () => {
  let users = '';
  if (NL.me.admin) {
    try {
      const j = await nlapi('users');
      users = `<h3 style="margin-top:16px">계정 목록</h3>
        <table><thead><tr><th>아이디</th><th>이름</th><th>권한</th></tr></thead><tbody>
        ${j.users.map(u => `<tr><td class="mono">${esc(u.user)}</td><td>${esc(u.name)}</td>
          <td>${u.admin ? '관리자' : '라벨러'}</td></tr>`).join('')}</tbody></table>
        <h3 style="margin-top:16px">계정 추가 / 비밀번호 초기화</h3>
        <div class="row">
          <label>아이디<input type="text" id="nl-a-user" size="12"></label>
          <label>이름<input type="text" id="nl-a-name" size="10"></label>
          <label>비밀번호<input type="password" id="nl-a-pw" size="12"></label>
          <label class="chk"><input type="checkbox" id="nl-a-admin">관리자</label>
          <label class="chk"><input type="checkbox" id="nl-a-reset">기존 계정 비밀번호 초기화</label>
          <button class="primary" id="nl-a-go">적용</button>
        </div>`;
    } catch (e) { users = `<p class="muted">${esc(e.message)}</p>`; }
  }
  showModal(`<h2>계정 - ${esc(NL.me.name)} (${esc(NL.me.user)})</h2>
    <h3>비밀번호 변경</h3>
    <div class="row">
      <label>현재 비밀번호<input type="password" id="nl-p-old"></label>
      <label>새 비밀번호<input type="password" id="nl-p-new"></label>
      <label>새 비밀번호 확인<input type="password" id="nl-p-new2"></label>
      <button class="primary" id="nl-p-go">변경</button>
    </div>
    ${users}
    <div class="row" style="margin-top:14px"><button onclick="closeModal()" style="margin-left:auto">닫기</button></div>`);
  $('#nl-p-go').onclick = async () => {
    if ($('#nl-p-new').value !== $('#nl-p-new2').value) return toast('새 비밀번호가 서로 다릅니다');
    try {
      await nlapi('passwd', { old: $('#nl-p-old').value, new: $('#nl-p-new').value });
      toast('비밀번호를 바꿨습니다'); closeModal();
    } catch (e) { toast(e.message); }
  };
  if ($('#nl-a-go')) $('#nl-a-go').onclick = async () => {
    try {
      await nlapi('useradd', { user: $('#nl-a-user').value.trim(), name: $('#nl-a-name').value.trim(),
        password: $('#nl-a-pw').value, admin: $('#nl-a-admin').checked, reset: $('#nl-a-reset').checked });
      toast('적용했습니다'); closeModal(); $('#nl-acct').click();
    } catch (e) { toast(e.message); }
  };
};

/* ---------- 단축키 ---------- */
document.addEventListener('keydown', ev => {
  if (!tabOn() || !NL.cur || $('#modal').innerHTML) return;
  const mod = ev.ctrlKey || ev.metaKey;
  if (mod && (ev.key === 's' || ev.key === 'S')) { ev.preventDefault(); save(false); return; }
  if (mod && ev.key === 'Enter') { ev.preventDefault(); save(true); return; }
  const t = ev.target.tagName;
  if (['INPUT', 'TEXTAREA', 'SELECT', 'VIDEO', 'BUTTON'].includes(t) && ev.key === ' ') return;
  if (['INPUT', 'TEXTAREA', 'SELECT', 'VIDEO'].includes(t)) return;
  const v = $('#nl-video');
  if (!v) return;
  if (ev.key === ' ') { ev.preventDefault(); v.paused ? v.play() : v.pause(); }
  else if (ev.key === 'ArrowLeft') { ev.preventDefault(); v.currentTime = Math.max(0, v.currentTime - 2); }
  else if (ev.key === 'ArrowRight') { ev.preventDefault(); v.currentTime = v.currentTime + 2; }
});

/* 탭이 처음 보일 때 시작한다(새로고침으로 이 탭이 복원된 경우 포함). */
$$('nav button').find(b => b.dataset.tab === 'nlabel')
  .addEventListener('click', () => init());
const startIfShown = () => { if (tabOn()) init(); };
setTimeout(startIfShown, 0);
setInterval(startIfShown, 1000);
})();
