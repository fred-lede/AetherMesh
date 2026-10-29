(function() {
  'use strict';

  function isoToShort(iso) {
    if (!iso) return '-';
    try { return new Date(iso).toLocaleDateString() + ' ' + new Date(iso).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}); }
    catch(e) { return iso; }
  }

  function activateAudioTab(tab) {
    if (!tab) return;
    var tabs = document.querySelectorAll('.audio-tab');
    var panels = document.querySelectorAll('.tab-panel');
    var target = tab.getAttribute('data-audio-tab');
    tabs.forEach(function(t) { t.classList.toggle('active', t === tab); });
    panels.forEach(function(p) {
      p.classList.toggle('active', p.getAttribute('data-audio-panel') === target);
    });
    try { if (target) localStorage.setItem('aethermesh.audio.tab', target); } catch(e) {}
  }

  window.audioTabClick = function(e, el) {
    try { if (e) { e.preventDefault(); e.stopPropagation(); } } catch(ex) {}
    activateAudioTab(el);
  };

  function initAudioTabs() {
    var tabs = document.querySelectorAll('.audio-tab');
    var container = document.querySelector('.audio-tabs');
    if (!tabs.length) return;

    if (container) {
      container.addEventListener('click', function(e) {
        var tab = e.target.closest('.audio-tab');
        if (!tab || !container.contains(tab)) return;
        try { e.preventDefault(); e.stopPropagation(); } catch(ex) {}
        activateAudioTab(tab);
      }, true);
    } else {
      tabs.forEach(function(tab) {
        if (tab.__audioTabBound) return;
        tab.__audioTabBound = true;
        tab.addEventListener('click', function(e) {
          try { e.preventDefault(); e.stopPropagation(); } catch(ex) {}
          activateAudioTab(tab);
        }, true);
      });
    }

    try {
      var saved = localStorage.getItem('aethermesh.audio.tab');
      if (saved) {
        var savedTab = document.querySelector('.audio-tab[data-audio-tab="' + saved + '"]');
        if (savedTab) {
          activateAudioTab(savedTab);
        }
      }
    } catch(e) {}
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
            '<td><input class="voice-lang-edit" value="' + (v.language || 'auto') + '" data-field="language"></td>' +
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

  function _idShort(id) {
    return id ? id.slice(0, 8) + '…' : '—';
  }

  function _uploadMsg(msg, cls) {
    var el = document.getElementById('voice-upload-msg');
    if (!el) return;
    el.textContent = msg;
    el.className = 'upload-msg-' + (cls || '');
    el.style.display = 'inline';
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
    var btn = document.getElementById('btn-register-voice');
    var restoreBtn = setButtonBusy(btn, 'Registering…');
    _uploadMsg('Uploading…', 'busy');
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
      _uploadMsg('Registered: ' + _idShort(result.voice_id), 'ok');
      nameEl.value = '';
      langEl.value = '';
      fileEl.value = '';
      loadVoices();
      loadTtsSettings();
    } catch(error) {
      _uploadMsg('Failed: ' + summarizeError(error), 'bad');
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

  var OLLAMA_TTS_MODEL_NAMES = [];
  var OLLAMA_TTS_MODEL_SET = {};

  function loadOllamaTtsModels() {
    return fetch('/api/models')
      .then(function(resp) { return resp.ok ? resp.json() : null; })
      .then(function(data) {
        OLLAMA_TTS_MODEL_NAMES = [];
        OLLAMA_TTS_MODEL_SET = {};
        var models = (data && data.models) || [];
        models.forEach(function(m) {
          var provider = String(m.provider || '').toLowerCase();
          var caps = (m.capabilities || []).map(function(c) { return String(c).toLowerCase(); });
          if (provider === 'ollama' && caps.indexOf('tts') !== -1) {
            OLLAMA_TTS_MODEL_NAMES.push(m.name);
            OLLAMA_TTS_MODEL_SET[m.name] = true;
            OLLAMA_TTS_MODEL_SET[m.name.toLowerCase()] = true;
          }
        });
        OLLAMA_TTS_MODEL_NAMES.sort(function(a, b) { return a.localeCompare(b); });
      })
      .catch(function() {});
  }

  function isOllamaTtsModel(model) {
    if (!model) return false;
    var m = String(model);
    var stripped = m.replace(/^ollama\//i, '');
    if (OLLAMA_TTS_MODEL_SET[m] || OLLAMA_TTS_MODEL_SET[stripped]) return true;
    if (OLLAMA_TTS_MODEL_SET[m.toLowerCase()] || OLLAMA_TTS_MODEL_SET[stripped.toLowerCase()]) return true;
    return false;
  }

  var MODEL_TEXT_LIMITS = {
    default: 500,
    xtts: 150
  };

  function _effectiveLimit(model, maxCharsInput) {
    var base = isOllamaTtsModel(model) ? MODEL_TEXT_LIMITS.default : MODEL_TEXT_LIMITS.xtts;
    if (maxCharsInput) {
      var v = parseInt(maxCharsInput, 10);
      if (!isNaN(v) && v > 0) return v;
    }
    return base;
  }

  function onSpeechInputChanged() {
    var inputEl = document.getElementById('speech-input');
    var countEl = document.getElementById('speech-char-count');
    var modelEl = document.getElementById('speech-model');
    var maxCharsEl = document.getElementById('speech-max-chars');
    if (!inputEl || !countEl) return;
    var model = modelEl ? modelEl.value : '';
    var limit = _effectiveLimit(model, maxCharsEl ? maxCharsEl.value : '');
    var len = (inputEl.value || '').length;
    countEl.textContent = len + ' / ' + limit + ' chars';
    countEl.style.color = len > limit ? 'var(--danger, #f87171)' : 'var(--muted)';
  }

  function updateCharLimit() {
    onSpeechInputChanged();
  }

  window.onSpeechInputChanged = onSpeechInputChanged;

  function onSpeechMaxCharsChanged() {
    onSpeechInputChanged();
  }

  window.onSpeechMaxCharsChanged = onSpeechMaxCharsChanged;

  function onSpeechModelChanged() {
    var modelEl = document.getElementById('speech-model');
    var voiceEl = document.getElementById('speech-voice');
    if (!modelEl || !voiceEl) return;
    voiceEl.value = '';
    updateCharLimit();
    loadSpeechOptions();
  }

  function populateSpeechModelSelect(selected) {
    var modelSelect = document.getElementById('speech-model');
    if (!modelSelect) return;
    var prev = selected || modelSelect.value;
    modelSelect.innerHTML = '';
    OLLAMA_TTS_MODEL_NAMES.forEach(function(name) {
      var opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name + ' (Ollama TTS)';
      modelSelect.appendChild(opt);
    });
    var xttsOpt = document.createElement('option');
    xttsOpt.value = 'xtts';
    xttsOpt.textContent = 'xtts (local XTTS)';
    modelSelect.appendChild(xttsOpt);
    if (OLLAMA_TTS_MODEL_SET[prev] || OLLAMA_TTS_MODEL_SET[prev.toLowerCase()]) {
      modelSelect.value = prev;
    } else if (prev === 'xtts') {
      modelSelect.value = 'xtts';
    } else {
      modelSelect.value = OLLAMA_TTS_MODEL_NAMES[0] || 'xtts';
    }
  }

  async function loadSpeechOptions() {
    var voiceSelect = document.getElementById('speech-voice');
    if (!voiceSelect) return;
    var modelEl = document.getElementById('speech-model');
    var model = modelEl ? modelEl.value : '';
    voiceSelect.innerHTML = '<option value="">Default</option>';
    if (isOllamaTtsModel(model)) {
      try {
        var resp = await fetch('/v1/tts/voices?model=' + encodeURIComponent(model));
        if (resp.ok) {
          var data = await resp.json();
          var voices = Array.isArray(data.voices) ? data.voices : [];
          voices.forEach(function(name) {
            var opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            voiceSelect.appendChild(opt);
          });
        }
      } catch(e) {}
      return;
    }
    try {
      var resp = await fetch('/api/audio/voices');
      if (!resp.ok) return;
      var voices = await resp.json();
      if (Array.isArray(voices)) {
        voices.forEach(function(v) {
          var opt = document.createElement('option');
          opt.value = v.voice_id || v.name || '';
          var label = v.name || v.voice_id || '';
          if (v.language) label += ' [' + v.language + ']';
          opt.textContent = label;
          voiceSelect.appendChild(opt);
        });
      }
    } catch(e) {}
  }

  window.loadSpeechOptions = loadSpeechOptions;

  window.generateSpeech = async function() {
    var textEl = document.getElementById('speech-input');
    var modelEl = document.getElementById('speech-model');
    var voiceEl = document.getElementById('speech-voice');
    var formatEl = document.getElementById('speech-format');
    var speedEl = document.getElementById('speech-speed');
    var instrEl = document.getElementById('speech-instructions');
    var btn = document.getElementById('btn-generate-speech');
    var msgEl = document.getElementById('speech-generate-msg');
    var statusEl = document.getElementById('speech-status');
    var audioEl = document.getElementById('speech-audio');
    if (!textEl) return;
    var input = (textEl.value || '').trim();
    if (!input) {
      if (msgEl) { msgEl.textContent = 'Enter text to synthesize.'; msgEl.className = 'speech-status warn'; }
      if (statusEl) { statusEl.textContent = 'Missing input text.'; statusEl.className = 'speech-status warn'; }
      return;
    }
    var model = modelEl ? modelEl.value : '';
    var voice = voiceEl ? voiceEl.value : '';
    var responseFormat = formatEl ? formatEl.value : 'mp3';
    var speed = speedEl ? Number(speedEl.value) : 1.0;
    var instructions = instrEl ? (instrEl.value || '').trim() : '';
    var maxCharsEl = document.getElementById('speech-max-chars');
    var maxChars = maxCharsEl ? (maxCharsEl.value || '').trim() : '';
    var restoreBtn = setButtonBusy(btn, 'Generating…');
    if (msgEl) { msgEl.textContent = 'Calling /v1/audio/speech...'; msgEl.className = 'speech-status'; }
    if (statusEl) { statusEl.textContent = 'Requesting audio...'; statusEl.className = 'speech-status'; }
    try {
      var body = { input: input, model: model, response_format: responseFormat, speed: isNaN(speed) ? 1.0 : speed };
      if (voice) body.voice = voice;
      if (instructions) body.instructions = instructions;
      if (maxChars) body.max_chars = parseInt(maxChars, 10);
      try {
        localStorage.setItem('aethermesh.audio.speech.model', model);
        localStorage.setItem('aethermesh.audio.speech.voice', voice);
        localStorage.setItem('aethermesh.audio.speech.format', responseFormat);
        localStorage.setItem('aethermesh.audio.speech.speed', String(speed));
      } catch(e) {}
      var resp = await fetch('/v1/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      if (!resp.ok) {
        var errText = '';
        try {
          var errJson = await resp.json();
          errText = errJson.detail || errJson.message || JSON.stringify(errJson);
        } catch(e) {
          try { errText = await resp.text(); } catch(_e) {}
        }
        throw new Error((resp.status) + ' ' + (resp.statusText || '') + (errText ? ': ' + errText : ''));
      }
      var blob = await resp.blob();
      var contentType = resp.headers.get('content-type') || ('audio/' + responseFormat);
      var url = URL.createObjectURL(blob);
      if (audioEl) {
        audioEl.src = url;
        audioEl.play().catch(function(){});
      }
      var fileName = 'speech_' + Date.now() + '.' + responseFormat;
      var downloadBtn = document.getElementById('btn-download-speech');
      if (downloadBtn) {
        downloadBtn.style.display = 'inline-block';
        downloadBtn.onclick = function() {
          var a = document.createElement('a');
          a.href = url;
          a.download = fileName;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
        };
      }
      if (msgEl) { msgEl.textContent = 'Done'; msgEl.className = 'speech-status ok'; }
      if (statusEl) { statusEl.textContent = 'Generated ' + Math.round(blob.size/1024) + ' KB (' + contentType + ')'; statusEl.className = 'speech-status ok'; }
    } catch(error) {
      var errMsg = summarizeError ? summarizeError(error) : (error && error.message ? error.message : String(error));
      if (msgEl) { msgEl.textContent = 'Failed'; msgEl.className = 'speech-status bad'; }
      if (statusEl) { statusEl.textContent = errMsg; statusEl.className = 'speech-status bad'; }
      var downloadBtnFail = document.getElementById('btn-download-speech');
      if (downloadBtnFail) downloadBtnFail.style.display = 'none';
    } finally {
      if (restoreBtn) restoreBtn();
    }
  };

  function initSpeechDefaults() {
    var modelEl = document.getElementById('speech-model');
    var voiceEl = document.getElementById('speech-voice');
    var formatEl = document.getElementById('speech-format');
    var speedEl = document.getElementById('speech-speed');
    var speedVal = document.getElementById('speech-speed-value');
    loadOllamaTtsModels().then(function() {
      populateSpeechModelSelect(null);
      var m = '';
      var v = '';
      var f = '';
      var s = '';
      try {
        m = localStorage.getItem('aethermesh.audio.speech.model') || '';
        v = localStorage.getItem('aethermesh.audio.speech.voice') || '';
        f = localStorage.getItem('aethermesh.audio.speech.format') || '';
        s = localStorage.getItem('aethermesh.audio.speech.speed') || '';
      } catch(e) {}
      if (m && OLLAMA_TTS_MODEL_SET[m]) {
        modelEl.value = m;
      } else if (m === 'xtts') {
        modelEl.value = 'xtts';
      }
      if (v && voiceEl) voiceEl.value = v;
      if (f && formatEl) formatEl.value = f;
      if (s && speedEl) { speedEl.value = s; if (speedVal) speedVal.textContent = Number(s).toFixed(2); }
      updateCharLimit();
      loadSpeechOptions();
    });
  }

  document.addEventListener('DOMContentLoaded', function() {
    initAudioTabs();
    loadTtsSettings();
    loadAsrSettings();
    loadVoices();
    initSpeechDefaults();
  });
})();
