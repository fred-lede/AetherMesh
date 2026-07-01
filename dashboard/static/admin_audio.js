(function() {
  'use strict';

  function isoToShort(iso) {
    if (!iso) return '-';
    try { return new Date(iso).toLocaleDateString() + ' ' + new Date(iso).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}); }
    catch(e) { return iso; }
  }

  async function loadTtsSettings() {
    var panel = document.getElementById('tts-settings-panel');
    if (!panel) return;
    try {
      var resp = await fetch('/api/audio/status');
      if (resp.status === 401) { redirectLogin(); return; }
      if (!resp.ok) { panel.innerHTML = '<span class="pill warn">Failed to load</span>'; return; }
      var s = await resp.json();
      var tts = s.tts || {};
      var enabled = tts.enabled;
      var pillCls = enabled ? 'ok' : 'disabled';
      panel.innerHTML = '<div class="settings-grid">' +
        '<div class="label">Status</div><div class="value"><span class="pill ' + pillCls + '">' + (enabled ? 'Enabled' : 'Disabled') + '</span></div><div></div>' +
        '<div class="label">Model</div><div class="value"><code>' + escapeHtml(tts.model || '—') + '</code></div><div></div>' +
        '<div class="label">Device</div><div class="value"><code>' + escapeHtml(tts.device || '—') + '</code></div><div></div>' +
        '<div class="label">Dtype</div><div class="value"><code>' + escapeHtml(tts.dtype || '—') + '</code></div><div></div>' +
        '<div class="label">Voices</div><div class="value"><code>' + (tts.voices_count != null ? tts.voices_count : '—') + '</code></div><div></div>' +
      '</div>' +
      (enabled ? '<div class="hint">Use <code>POST /v1/audio/speech</code> with <code>voice</code> = voice ID. Register voices below or via <code>POST /v1/voices</code>.</div>' : '');
      document.getElementById('voice-upload-panel').style.display = enabled ? '' : 'none';
    } catch(e) {
      panel.innerHTML = '<span class="pill warn">Error</span>';
    }
  }

  async function loadAsrSettings() {
    var panel = document.getElementById('asr-settings-panel');
    if (!panel) return;
    try {
      var resp = await fetch('/api/audio/status');
      if (!resp.ok) { panel.innerHTML = '<span class="pill warn">Failed to load</span>'; return; }
      var s = await resp.json();
      var asr = s.asr || {};
      var enabled = asr.enabled;
      var pillCls = enabled ? 'ok' : 'disabled';
      panel.innerHTML = '<div class="settings-grid">' +
        '<div class="label">Status</div><div class="value"><span class="pill ' + pillCls + '">' + (enabled ? 'Enabled' : 'Disabled') + '</span></div><div></div>' +
        '<div class="label">Model</div><div class="value"><code>' + escapeHtml(asr.model || '—') + '</code></div><div></div>' +
        '<div class="label">Device</div><div class="value"><code>' + escapeHtml(asr.device || '—') + (asr.device_index != null && asr.device_index > 0 ? ' (index ' + asr.device_index + ')' : '') + '</code></div><div></div>' +
        '<div class="label">Compute</div><div class="value"><code>' + escapeHtml(asr.compute_type || '—') + '</code></div><div></div>' +
      '</div>' +
      (enabled ? '<div class="hint">Use <code>POST /v1/audio/transcriptions</code> or <code>/v1/audio/translations</code> to transcribe audio files.</div>' : '');
    } catch(e) {
      panel.innerHTML = '<span class="pill warn">Error</span>';
    }
  }

  async function loadVoices() {
    var panel = document.getElementById('voices-panel');
    if (!panel) return;
    try {
      var resp = await fetch('/api/audio/voices');
      if (resp.status === 401) { redirectLogin(); return; }
      if (resp.status === 503) { panel.innerHTML = '<span class="pill disabled">TTS not enabled</span>'; return; }
      if (!resp.ok) { panel.innerHTML = '<span class="pill warn">Failed to load voices</span>'; return; }
      var voices = await resp.json();
      if (!voices || voices.length === 0) {
        panel.innerHTML = '<span class="pill">No voices registered</span>';
        return;
      }
      panel.innerHTML = '<table class="table"><thead><tr>' +
        '<th>ID</th><th>Name</th><th>Language</th><th>Duration</th><th>Created</th><th>Preview</th><th></th>' +
      '</tr></thead><tbody>' +
        voices.map(function(v) {
          var idShort = v.voice_id ? v.voice_id.slice(0, 8) : '—';
          return '<tr data-vid="' + escapeHtml(v.voice_id) + '">' +
            '<td><code title="' + escapeHtml(v.voice_id) + '">' + escapeHtml(idShort) + '…</code></td>' +
            '<td><input class="voice-name-edit" value="' + escapeHtml(v.name || '') + '" data-field="name"></td>' +
            '<td><input class="voice-lang-edit" value="' + escapeHtml(v.language || '') + '" data-field="language"></td>' +
            '<td>' + (v.duration_seconds != null ? v.duration_seconds + 's' : '—') + '</td>' +
            '<td>' + isoToShort(v.created_at) + '</td>' +
            '<td><audio class="audio-player" controls preload="none" src="/api/audio/voices/' + escapeHtml(v.voice_id) + '/preview"></audio></td>' +
            '<td class="actions">' +
              '<button class="btn btn-sm btn-secondary" onclick="window.saveVoice(\'' + escapeHtml(v.voice_id) + '\')">Save</button>' +
              '<button class="btn btn-sm btn-danger" onclick="window.deleteVoice(\'' + escapeHtml(v.voice_id) + '\')">Delete</button>' +
            '</td>' +
          '</tr>';
        }).join('') +
      '</tbody></table>';
    } catch(e) {
      panel.innerHTML = '<span class="pill warn">Error</span>';
    }
  }

  function _findBtn(label) {
    var btns = document.querySelectorAll('button');
    for (var i = 0; i < btns.length; i++) {
      if (btns[i].textContent.trim() === label) return btns[i];
    }
    return null;
  }

  function _idShort(id) {
    return id ? id.slice(0, 8) + '…' : '—';
  }

  window.registerVoice = async function() {
    var nameEl = document.getElementById('voice-name');
    var langEl = document.getElementById('voice-language');
    var fileEl = document.getElementById('voice-file');
    var name = (nameEl.value || '').trim();
    var lang = langEl.value;
    var file = fileEl.files && fileEl.files[0];
    if (!name) { alert('Voice name is required.'); return; }
    if (!file) { alert('Audio file is required.'); return; }
    var btn = _findBtn('Register');
    var restoreBtn = setButtonBusy(btn, 'Registering…');
    setOperationStatus('Registering voice ' + name + '…', 'warn');
    try {
      var fd = new FormData();
      fd.append('name', name);
      if (lang) fd.append('language', lang);
      fd.append('file', file);
      var resp = await fetch('/api/audio/voices', { method: 'POST', body: fd });
      if (!resp.ok) {
        var err = await resp.json().catch(function() { return {detail:'Failed'}; });
        throw new Error(err.detail || 'Registration failed');
      }
      var result = await resp.json();
      setOperationStatus('Voice registered: ' + _idShort(result.voice_id), 'ok');
      nameEl.value = '';
      langEl.value = '';
      fileEl.value = '';
      loadVoices();
      loadTtsSettings();
    } catch(error) {
      setOperationStatus('Failed: ' + summarizeError(error), 'bad');
    } finally {
      restoreBtn();
    }
  };

  window.saveVoice = async function(voiceId) {
    var row = document.querySelector('tr[data-vid="' + voiceId + '"]');
    if (!row) return;
    var nameInput = row.querySelector('input[data-field="name"]');
    var langInput = row.querySelector('input[data-field="language"]');
    var body = {};
    if (nameInput) body.name = nameInput.value;
    if (langInput) body.language = langInput.value;
    var btn = row.querySelector('.btn-secondary');
    var restoreBtn = setButtonBusy(btn, 'Saving…');
    setOperationStatus('Updating voice…', 'warn');
    try {
      var resp = await fetch('/api/audio/voices/' + voiceId, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        var err = await resp.json().catch(function() { return {detail:'Failed'}; });
        throw new Error(err.detail || 'Update failed');
      }
      setOperationStatus('Voice updated.', 'ok');
      loadVoices();
    } catch(error) {
      setOperationStatus('Failed: ' + summarizeError(error), 'bad');
    } finally {
      restoreBtn();
    }
  };

  window.deleteVoice = async function(voiceId) {
    if (!confirm('Delete voice ' + _idShort(voiceId) + '? This cannot be undone.')) return;
    var row = document.querySelector('tr[data-vid="' + voiceId + '"]');
    var btn = row ? row.querySelector('.btn-danger') : null;
    var restoreBtn = setButtonBusy(btn, 'Deleting…');
    setOperationStatus('Deleting voice…', 'warn');
    try {
      var resp = await fetch('/api/audio/voices/' + voiceId, { method: 'DELETE' });
      if (!resp.ok && resp.status !== 204) {
        var err = await resp.json().catch(function() { return {detail:'Failed'}; });
        throw new Error(err.detail || 'Delete failed');
      }
      setOperationStatus('Voice deleted.', 'ok');
      loadVoices();
      loadTtsSettings();
    } catch(error) {
      setOperationStatus('Failed: ' + summarizeError(error), 'bad');
    } finally {
      restoreBtn();
    }
  };

  document.addEventListener('DOMContentLoaded', function() {
    loadTtsSettings();
    loadAsrSettings();
    loadVoices();
  });
})();
