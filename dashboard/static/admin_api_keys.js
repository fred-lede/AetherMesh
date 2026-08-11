(function() {
  'use strict';

  async function loadApiKeys() {
    const panel = document.getElementById('api-keys-panel');
    if (!panel) return;
    try {
      const resp = await fetch('/api/security/api-keys');
      if (resp.status === 403) { panel.innerHTML = '<span class="pill">Admin access required</span>'; return; }
      if (resp.status === 401) { redirectLogin(); return; }
      if (!resp.ok) { panel.innerHTML = '<span class="pill warn">Failed to load API keys</span>'; return; }
      const data = await resp.json();
      const keys = data && Array.isArray(data.keys) ? data.keys : [];
      const unkeyed = (data && data.unkeyed_usage) || { total_tokens: 0, total_input_tokens: 0, total_output_tokens: 0, record_count: 0 };
      const keyedTotal = keys.reduce((s, k) => s + ((k.token_usage || {}).total_tokens || 0), 0);
      const unkeyedPill = unkeyed.total_tokens > 0
        ? `<span class="pill warn" title="Input: ${unkeyed.total_input_tokens.toLocaleString()}, Output: ${unkeyed.total_output_tokens.toLocaleString()}, Requests: ${unkeyed.record_count.toLocaleString()}">Unkeyed (env key / login): ${unkeyed.total_tokens.toLocaleString()}</span>`
        : '';
      const summaryLine = (keyedTotal + unkeyed.total_tokens > 0)
        ? `<div style="margin:10px 0 2px">
            <span class="pill ok">Keyed: ${keyedTotal.toLocaleString()}</span>
            ${unkeyedPill}
            <span class="pill">Total: ${(keyedTotal + unkeyed.total_tokens).toLocaleString()}</span>
          </div>`
        : '';
      if (!keys || keys.length === 0) {
        panel.innerHTML = `<span class="pill">No API keys configured</span>
          ${summaryLine}
          <button class="btn btn-sm" onclick="createApiKey()" style="margin-top:8px">Generate Key</button>`;
        return;
      }
      panel.innerHTML = `<table class="table"><thead><tr>
        <th>Prefix</th><th>Name</th><th>Owner</th><th>Status</th><th>Tokens</th><th>Created</th><th>Last Used</th><th></th>
      </tr></thead><tbody>
        ${keys.map(k => {
          const tu = k.token_usage || { total_tokens: 0, total_input_tokens: 0, total_output_tokens: 0, record_count: 0 };
          const title = `Input: ${tu.total_input_tokens.toLocaleString()}, Output: ${tu.total_output_tokens.toLocaleString()}, Requests: ${tu.record_count.toLocaleString()}`;
          return `<tr>
          <td><code>${escapeHtml(k.key_prefix)}...</code></td>
          <td>${escapeHtml(k.name || '-')}</td>
          <td>${escapeHtml(k.owner_display_name || k.owner_email || '—')}</td>
          <td><span class="pill ${k.is_active ? 'ok' : 'disabled'}">${k.is_active ? 'Active' : 'Revoked'}</span></td>
          <td title="${title}">${tu.total_tokens.toLocaleString()}</td>
          <td>${k.created_at ? timeAgo(k.created_at) : '-'}</td>
          <td>${k.last_used_at ? timeAgo(k.last_used_at) : 'Never'}</td>
          <td>${k.is_active ? `<button class="btn btn-sm btn-danger" onclick="revokeApiKey(${k.id})">Revoke</button>` : ''}</td>
        </tr>`;
        }).join('')}
      </tbody></table>
      <button class="btn btn-sm" onclick="createApiKey()" style="margin-top:8px">Generate Key</button>
      ${summaryLine}`;
    } catch(e) {
      panel.innerHTML = '<span class="pill warn">Error</span>';
    }
  }

  window.createApiKey = async function() {
    const name = prompt('Label (optional):');
    if (name === null) return;
    const restoreButton = setButtonBusy(null, 'Generating...');
    setOperationStatus('Generating...', 'warn');
    try {
      const resp = await fetch('/api/security/api-keys', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name || '' }),
      });
      if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed');
      const result = await resp.json();
      alert(`API Key:\n\n${result.raw_key}\n\nShown once only.`);
      setOperationStatus('Key generated.', 'ok');
      loadApiKeys();
    } catch (error) {
      setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
    } finally { restoreButton(); }
  };

  window.revokeApiKey = async function(keyId) {
    if (!confirm('Revoke this API key? This cannot be undone.')) return;
    const restoreButton = setButtonBusy(null, 'Revoking...');
    setOperationStatus('Revoking API key...', 'warn');
    try {
      const resp = await fetch(`/api/security/api-keys/${keyId}`, { method: 'DELETE' });
      if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed to revoke key');
      setOperationStatus('API key revoked.', 'ok');
      loadApiKeys();
    } catch (error) {
      setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
    } finally { restoreButton(); }
  };

  document.addEventListener('DOMContentLoaded', loadApiKeys);
})();
