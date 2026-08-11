(function() {
  'use strict';

  async function loadUsers() {
    const panel = document.getElementById('users-panel');
    if (!panel) return;
    try {
      const resp = await fetch('/api/users');
      if (resp.status === 403) { panel.innerHTML = '<span class="pill">Admin access required</span>'; return; }
      if (resp.status === 401) { redirectLogin(); return; }
      if (!resp.ok) { panel.innerHTML = '<span class="pill warn">Failed to load users</span>'; return; }
      const users = await resp.json();
      if (!users || users.length === 0) {
        panel.innerHTML = '<span class="pill">No users</span>';
        return;
      }
      panel.innerHTML = `<table class="table"><thead><tr>
        <th>Email</th><th>Name</th><th>Role</th><th>Status</th><th>Tokens</th><th>Created</th><th>Last Login</th><th></th>
      </tr></thead><tbody>
        ${users.map(u => {
          const tu = u.token_usage || { total_tokens: 0, total_input_tokens: 0, total_output_tokens: 0, record_count: 0 };
          const tokenText = tu.total_tokens > 0 ? `${tu.total_tokens.toLocaleString()} · ${tu.record_count.toLocaleString()} req` : '0';
          const tokenTitle = `In ${tu.total_input_tokens.toLocaleString()} / Out ${tu.total_output_tokens.toLocaleString()} · ${tu.record_count.toLocaleString()} requests`;
          return `<tr>
          <td>${escapeHtml(u.email)}</td>
          <td>${escapeHtml(u.display_name)}</td>
          <td><span class="pill ${u.role === 'admin' ? 'warn' : 'ok'}">${escapeHtml(u.role)}</span></td>
          <td><span class="pill ${u.is_active ? 'ok' : 'disabled'}">${u.is_active ? 'Active' : 'Disabled'}</span></td>
          <td title="${escapeHtml(tokenTitle)}">${tokenText}</td>
          <td>${u.created_at ? timeAgo(u.created_at) : '-'}</td>
          <td>${u.last_login_at ? timeAgo(u.last_login_at) : 'Never'}</td>
          <td>
            <button class="btn btn-sm" onclick="editUser(${u.id}, '${escapeHtml(u.email)}', '${escapeHtml(u.display_name)}', '${u.role}', ${u.is_active})">Edit</button>
            <button class="btn btn-sm btn-danger" onclick="deleteUser(${u.id}, '${escapeHtml(u.email)}')">Delete</button>
          </td>
        </tr>`;
        }).join('')}
      </tbody></table>
      <button class="btn btn-sm" onclick="showCreateUserModal()" style="margin-top:8px">Add User</button>`;
    } catch(e) {
      panel.innerHTML = '<span class="pill warn">Error loading users</span>';
    }
  }

  window.showCreateUserModal = function() {
    const email = prompt('Email address:');
    if (!email) return;
    const password = prompt('Password (min 8 characters):');
    if (!password || password.length < 8) { alert('Password must be at least 8 characters.'); return; }
    const displayName = prompt('Display name (optional):') || email.split('@')[0];
    const role = prompt('Role (admin or user):') || 'user';
    createUser({ email, password, display_name: displayName, role });
  };

  async function createUser(data) {
    const restoreButton = setButtonBusy(null, 'Creating...');
    setOperationStatus('Creating user...', 'warn');
    try {
      const resp = await fetch('/api/users', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });
      if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed to create user');
      setOperationStatus('User created.', 'ok');
      loadUsers();
    } catch (error) {
      setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
    } finally { restoreButton(); }
  }

  window.editUser = function(id, email, displayName, role, isActive) {
    const newDisplayName = prompt('Display name:', displayName);
    if (newDisplayName === null) return;
    const newRole = prompt('Role (admin or user):', role);
    if (newRole === null) return;
    const newPassword = prompt('New password (leave blank to keep current):');
    const body = { display_name: newDisplayName, role: newRole };
    if (newPassword && newPassword.length >= 8) body.password = newPassword;
    if (newPassword && newPassword.length < 8) { alert('Password must be at least 8 characters.'); return; }
    const restoreButton = setButtonBusy(null, 'Updating...');
    setOperationStatus('Updating user...', 'warn');
    (async () => {
      try {
        const resp = await fetch(`/api/users/${id}`, {
          method: 'PATCH', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed to update user');
        setOperationStatus('User updated.', 'ok');
        loadUsers();
      } catch (error) {
        setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
      } finally { restoreButton(); }
    })();
  };

  window.deleteUser = function(id, email) {
    if (!confirm(`Delete user ${email}?`)) return;
    const restoreButton = setButtonBusy(null, 'Deleting...');
    setOperationStatus('Deleting user...', 'warn');
    (async () => {
      try {
        const resp = await fetch(`/api/users/${id}`, { method: 'DELETE' });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed to delete user');
        setOperationStatus('User deleted.', 'ok');
        loadUsers();
      } catch (error) {
        setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
      } finally { restoreButton(); }
    })();
  };

  document.addEventListener('DOMContentLoaded', loadUsers);
})();
