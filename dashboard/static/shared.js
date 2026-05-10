function escapeHtml(str) {
  if (str == null) return '';
  const div = document.createElement('div');
  div.textContent = String(str);
  return div.innerHTML;
}

function timeAgo(ts) {
  if (ts == null) return '-';
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function summarizeError(error) {
  if (!error) return 'unknown error';
  if (error.message) return error.message.length > 80 ? error.message.slice(0, 80) + '...' : error.message;
  return String(error).length > 80 ? String(error).slice(0, 80) + '...' : String(error);
}

let _statusTimeout = null;
function setOperationStatus(msg, cls) {
  const el = document.getElementById('operation-status');
  if (!el) return;
  el.textContent = msg;
  el.className = 'status-' + (cls || '');
  el.style.display = 'block';
  if (_statusTimeout) clearTimeout(_statusTimeout);
  if (!cls || cls === 'ok' || cls === 'bad' || !cls) {
    _statusTimeout = setTimeout(() => { el.style.display = 'none'; }, cls === 'bad' ? 8000 : 4000);
  }
}

function setButtonBusy(el, text) {
  if (!el) el = { disabled: false, textContent: '' };
  const prev = el.disabled;
  el.disabled = true;
  const prevText = el.textContent;
  if (text) el.textContent = text;
  return function restore() {
    el.disabled = prev;
    if (text) el.textContent = prevText;
  };
}

function redirectLogin() {
  window.location.href = '/login';
}
