(function() {
  'use strict';

  async function loadProfile() {
    const panel = document.getElementById('profile-panel');
    if (!panel) return;
    try {
      const resp = await fetch('/api/auth/me');
      if (resp.status === 401) { redirectLogin(); return; }
      if (!resp.ok) { panel.innerHTML = '<span class="pill warn">Failed to load profile</span>'; return; }
      const me = await resp.json();
      panel.innerHTML = `
        <div class="stat-row"><span class="label">Email</span><span class="value">${escapeHtml(me.email)}</span></div>
        <div class="stat-row"><span class="label">Name</span><span class="value">${escapeHtml(me.display_name)}</span></div>
        <div class="stat-row"><span class="label">Role</span><span class="value"><span class="pill ${me.role === 'admin' ? 'warn' : 'ok'}">${escapeHtml(me.role)}</span></span></div>`;
    } catch(e) {
      panel.innerHTML = '';
    }
  }

  function showChangePasswordModal() {
    const oldPwd = prompt('Current password:');
    if (!oldPwd) return;
    const newPwd = prompt('New password (min 8 characters):');
    if (!newPwd || newPwd.length < 8) { alert('Password must be at least 8 characters.'); return; }
    const confirmPwd = prompt('Confirm new password:');
    if (newPwd !== confirmPwd) { alert('Passwords do not match.'); return; }
    const restoreButton = setButtonBusy(document.querySelector('.change-pwd-btn'), 'Changing...');
    setOperationStatus('Changing password...', 'warn');
    (async () => {
      try {
        const resp = await fetch('/api/auth/me/change-password', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ old_password: oldPwd, new_password: newPwd }),
        });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed to change password');
        setOperationStatus('Password changed.', 'ok');
      } catch (error) {
        setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
      } finally { restoreButton(); }
    })();
  }

  let _keysVisible = false;
  async function toggleMyApiKeys() {
    const panel = document.getElementById('my-api-keys-panel');
    if (!panel) return;
    _keysVisible = !_keysVisible;
    if (!_keysVisible) { panel.style.display = 'none'; return; }
    panel.style.display = 'block';
    try {
      const resp = await fetch('/api/auth/me/api-keys');
      if (resp.status === 401) { redirectLogin(); return; }
      if (!resp.ok) { panel.innerHTML = '<span class="pill warn">Failed to load</span>'; return; }
      const keys = await resp.json();
      if (!keys || keys.length === 0) {
        panel.innerHTML = '<span class="pill">No API keys</span> <button class="btn btn-sm" onclick="createMyApiKey()">Generate</button>';
        return;
      }
      panel.innerHTML = `<table class="table"><thead><tr>
        <th>Prefix</th><th>Name</th><th>Status</th><th>Created</th><th>Last Used</th><th></th>
      </tr></thead><tbody>
        ${keys.map(k => `<tr>
          <td><code>${escapeHtml(k.key_prefix)}...</code></td>
          <td>${escapeHtml(k.name || '-')}</td>
          <td><span class="pill ${k.is_active ? 'ok' : 'disabled'}">${k.is_active ? 'Active' : 'Revoked'}</span></td>
          <td>${k.created_at ? timeAgo(k.created_at) : '-'}</td>
          <td>${k.last_used_at ? timeAgo(k.last_used_at) : 'Never'}</td>
          <td>${k.is_active ? `<button class="btn btn-sm btn-danger" onclick="revokeMyApiKey(${k.id})">Revoke</button>` : ''}</td>
        </tr>`).join('')}
      </tbody></table>
      <button class="btn btn-sm" onclick="createMyApiKey()" style="margin-top:4px">Generate</button>`;
    } catch(e) { panel.innerHTML = '<span class="pill warn">Error</span>'; }
  }

  async function createMyApiKey() {
    const name = prompt('Label (optional):');
    if (name === null) return;
    const restoreButton = setButtonBusy(document.querySelector('.gen-key-btn'), 'Generating...');
    setOperationStatus('Generating...', 'warn');
    try {
      const resp = await fetch('/api/auth/me/api-keys', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name || '' }),
      });
      if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed');
      const result = await resp.json();
      alert(`API Key:\n\n${result.raw_key}\n\nShown once only.`);
      setOperationStatus('Key generated.', 'ok');
      toggleMyApiKeys();
    } catch (error) {
      setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
    } finally { restoreButton(); }
  }

  async function revokeMyApiKey(keyId) {
    if (!confirm('Revoke this key?')) return;
    const restoreButton = setButtonBusy(null, 'Revoking...');
    setOperationStatus('Revoking...', 'warn');
    try {
      const resp = await fetch(`/api/auth/me/api-keys/${keyId}`, { method: 'DELETE' });
      if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed');
      setOperationStatus('Revoked.', 'ok');
      toggleMyApiKeys();
    } catch (error) {
      setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
    } finally { restoreButton(); }
  }

  async function loadTokenUsage() {
    const summaryEl = document.getElementById('token-usage-summary');
    const chartCanvas = document.getElementById('token-usage-chart');
    if (!summaryEl) return;
    try {
      const [summary, records] = await Promise.all([
        fetch('/api/auth/me/token-usage/summary').then(r => r.ok ? r.json() : null),
        fetch('/api/auth/me/token-usage?limit=50').then(r => r.ok ? r.json() : []),
      ]);
      if (!summary) { summaryEl.innerHTML = '<span class="pill">No usage data</span>'; return; }
      const fmt = (n) => n.toLocaleString();
      summaryEl.innerHTML = `
        <div class="stat-row"><span class="label">Total Input Tokens</span><span class="value">${fmt(summary.total_input_tokens)}</span></div>
        <div class="stat-row"><span class="label">Total Output Tokens</span><span class="value">${fmt(summary.total_output_tokens)}</span></div>
        <div class="stat-row"><span class="label">Total Tokens</span><span class="value">${fmt(summary.total_tokens)}</span></div>
        <div class="stat-row"><span class="label">Requests</span><span class="value">${fmt(summary.record_count)}</span></div>`;
      if (!chartCanvas || !records || records.length === 0) return;
      const dayTotals = {};
      records.forEach(r => {
        const day = new Date(r.created_at * 1000).toISOString().slice(0, 10);
        dayTotals[day] = (dayTotals[day] || 0) + (r.total_tokens || 0);
      });
      const days = Object.keys(dayTotals).sort().slice(-7);
      const data = days.map(d => dayTotals[d]);
      if (window._tokenChart) { window._tokenChart.destroy(); }
      const ctx = chartCanvas.getContext('2d');
      window._tokenChart = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: days.map(d => d.slice(5)),
          datasets: [{ label: 'Tokens', data, backgroundColor: 'rgba(94, 234, 212, 0.5)', borderColor: '#5eead4', borderWidth: 1 }],
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: false } },
          scales: { y: { beginAtZero: true, ticks: { color: '#9fb6c8' }, grid: { color: 'rgba(150,210,255,0.1)' } }, x: { ticks: { color: '#9fb6c8' } } },
        },
      });
    } catch(e) { summaryEl.innerHTML = '<span class="pill warn">Error loading usage</span>'; }
  }

  document.addEventListener('DOMContentLoaded', () => {
    loadProfile();
    loadTokenUsage();
    document.querySelector('.change-pwd-btn')?.addEventListener('click', showChangePasswordModal);
    document.querySelector('.toggle-keys-btn')?.addEventListener('click', toggleMyApiKeys);
    window.createMyApiKey = createMyApiKey;
    window.revokeMyApiKey = revokeMyApiKey;
  });
})();
