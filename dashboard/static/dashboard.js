    const topMetrics = document.getElementById('top-metrics');
    const providers = document.getElementById('providers');
    const cloudProviders = document.getElementById('cloud-providers');
    const alerts = document.getElementById('alerts');
    const nodesTable = document.getElementById('nodes-table');
    const gpusTable = document.getElementById('gpus-table');
    const workersTable = document.getElementById('workers-table');
    const modelsTable = document.getElementById('models-table');
    const tasksTable = document.getElementById('tasks-table');
    const errorCodesTable = document.getElementById('error-codes-table');
    const endpointLatencyTable = document.getElementById('endpoint-latency-table');
    const gpuOsPanel = document.getElementById('gpu-os-panel');
    const multiAgentStats = document.getElementById('multi-agent-stats');
    const securityStats = document.getElementById('security-stats');
    const observabilityStats = document.getElementById('observability-stats');
    const updated = document.getElementById('updated');
    const requestMetricsCards = document.getElementById('request-metrics-cards');
    const providerMetricsTable = document.getElementById('provider-metrics-table');
    const routingProviders = document.getElementById('routing-providers');
    const modelAliasTable = document.getElementById('model-alias-table');
    const routingOverridesTable = document.getElementById('routing-overrides-table');
    const routingAuditTable = document.getElementById('routing-audit-table');
    const routingModeLabel = document.getElementById('routing-mode-label');
    const routingModeDetail = document.getElementById('routing-mode-detail');
    const routingFallbackDetail = document.getElementById('routing-fallback-detail');
    const localOnlyButton = document.getElementById('local-only-button');
    const enableAllProvidersButton = document.getElementById('enable-all-providers-button');
    const providerHealthGrid = document.getElementById('provider-health-grid');
    const overallHealth = document.getElementById('overall-health');
    const routingOperationStatus = document.getElementById('routing-operation-status');
    let operationStatusTimer = null;

    // Chart instances
    let latencyChart = null;
    let requestsChart = null;
    const chartHistory = { latency: [], requests: [], labels: [], gpu: {}, model: {}, workerLoad: {}, requestTotal: null };
    const MAX_HISTORY = 30; // 30 data points (5min / 10s intervals)
    let gpuChart = null;
    let modelChart = null;
    let workerChart = null;
    let latencyDistChart = null;
    let tokensChart = null;
    let ttftChart = null;
    let memoryChart = null;
    let powerChart = null;
    let tempChart = null;
    let errorChart = null;
    let activeChartGroup = 'health';

    function statusClass(status) {
      if (status === true) return 'ok';
      if (status === false) return 'bad';
      const value = String(status || '').toLowerCase();
      if (value === 'healthy' || value === 'ok' || value === '1' || value === 'true') return 'ok';
      if (value === 'degraded' || value === 'stale' || value === 'warn' || value === 'warning') return 'warn';
      return 'bad';
    }

    function formatTime(ts) {
      if (!ts) return '-';
      return new Date(ts * 1000).toLocaleString();
    }

    function timeAgo(ts) {
      const numeric = Number(ts || 0);
      if (!numeric) return 'never';
      const seconds = Math.max(0, Math.floor(Date.now() / 1000 - numeric));
      if (seconds < 60) return `${seconds}s ago`;
      const minutes = Math.floor(seconds / 60);
      if (minutes < 60) return `${minutes}m ago`;
      const hours = Math.floor(minutes / 60);
      if (hours < 48) return `${hours}h ago`;
      return `${Math.floor(hours / 24)}d ago`;
    }

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
      }[char]));
    }

    function escapeJsString(value) {
      return String(value ?? '').replace(/[\\']/g, (char) => `\\${char}`).replace(/\n/g, '\\n').replace(/\r/g, '\\r');
    }

    function formatPercent(value) {
      const numeric = Number(value);
      if (!Number.isFinite(numeric)) return '0.0%';
      return `${(numeric * 100).toFixed(1)}%`;
    }

    function formatLatency(value) {
      const numeric = Number(value || 0);
      return numeric > 0 ? `${numeric.toFixed(0)} ms` : '-';
    }

    function shortDiagnostic(value) {
      const text = String(value || '').replace(/\s+/g, ' ').trim();
      if (!text) return '';
      return text.length > 96 ? `${text.slice(0, 93)}...` : text;
    }

    function formatAuditAction(action) {
      const labels = {
        provider_enabled_changed: 'Provider toggle',
        local_only_mode_changed: 'Routing mode',
        model_override_set: 'Override set',
        model_override_cleared: 'Override removed',
      };
      return labels[action] || action;
    }

    function formatAuditDetails(event) {
      const details = event.details || {};
      if (event.action === 'provider_enabled_changed') {
        return `${details.provider || '-'} -> ${details.enabled ? 'enabled' : 'disabled'}`;
      }
      if (event.action === 'local_only_mode_changed') {
        return details.enabled ? 'Local only enabled' : 'All providers enabled';
      }
      if (event.action === 'model_override_set') {
        return `${details.model || '-'} -> ${details.provider || '-'}`;
      }
      if (event.action === 'model_override_cleared') {
        return `${details.model || '-'} removed${details.previous_provider ? ` from ${details.previous_provider}` : ''}`;
      }
      return JSON.stringify(details);
    }

    function summarizeError(error) {
      const message = String(error?.message || error || 'Unknown dashboard error');
      if (message.includes('Connection refused') || message.includes('Failed to establish a new connection')) {
        return 'Control plane unavailable on http://127.0.0.1:9200.';
      }
      if (message.includes('Overview API returned')) {
        return message;
      }
      return message.length > 140 ? `${message.slice(0, 137)}...` : message;
    }

    function setOperationStatus(message, level = 'ok') {
      if (!routingOperationStatus) return;
      if (operationStatusTimer) {
        clearTimeout(operationStatusTimer);
        operationStatusTimer = null;
      }
      routingOperationStatus.className = `operation-status ${level}`;
      routingOperationStatus.textContent = message;
      if (level === 'ok') {
        operationStatusTimer = setTimeout(() => {
          routingOperationStatus.className = 'operation-status';
          routingOperationStatus.textContent = '';
        }, 5000);
      }
    }

    async function parseApiError(response) {
      let detail = `${response.status} ${response.statusText}`.trim();
      try {
        const data = await response.json();
        detail = data.detail || data.message || detail;
      } catch(e) {
        try {
          const text = await response.text();
          if (text) detail = text;
        } catch(_e) {}
      }
      return detail;
    }

    async function mutateDashboard(endpoint, options = {}) {
      const response = await fetch(endpoint, options);
      if (!response.ok) {
        throw new Error(await parseApiError(response));
      }
      if (response.status === 204) return {};
      try {
        return await response.json();
      } catch(e) {
        return {};
      }
    }

    function setButtonBusy(button, busyLabel = 'Working...') {
      if (!button) return () => {};
      const previousText = button.textContent;
      button.disabled = true;
      button.textContent = busyLabel;
      return () => {
        button.disabled = false;
        button.textContent = previousText;
      };
    }

    function metricCard(title, value, detail) {
      return `<div class="card"><h2>${title}</h2><div class="metric">${value}</div><div>${detail}</div></div>`;
    }

    function setChartGroup(group) {
      activeChartGroup = group;
      document.querySelectorAll('[data-chart-tab]').forEach((button) => {
        button.classList.toggle('active', button.dataset.chartTab === group);
      });
      document.querySelectorAll('[data-chart-group]').forEach((card) => {
        const groups = String(card.dataset.chartGroup || '').split(/\s+/);
        card.hidden = !groups.includes(group);
      });
      [
        latencyChart, requestsChart, gpuChart, modelChart, workerChart,
        latencyDistChart, tokensChart, ttftChart, memoryChart, powerChart,
        tempChart, errorChart,
      ].forEach((chart) => {
        if (chart) {
          try { chart.resize(); } catch(e) {}
        }
      });
    }

    function healthCell(label, value, detail, cls = '') {
      return `
        <div class="health-cell">
          <span>${label}</span>
          <strong class="${cls}">${value}</strong>
          ${detail ? `<div class="health-reason">${detail}</div>` : ''}
        </div>
      `;
    }

    function buildProviderSummary(data, providerMetrics, routingProvidersData, cloudList) {
      const providerStatusMap = data.provider_status || {};
      const providerDiagnostics = data.provider_diagnostics || {};
      const cloudByName = Object.fromEntries((cloudList || []).map((item) => [item.name, item]));
      const providerNames = new Set([
        ...Object.keys(providerStatusMap),
        ...Object.keys(providerMetrics || {}),
        ...Object.keys(routingProvidersData || {}),
        ...Object.keys(providerDiagnostics || {}),
        ...Object.keys(cloudByName),
      ]);
      let down = 0;
      let degraded = 0;
      let disabled = 0;

      providerNames.forEach((provider) => {
        const routeState = routingProvidersData[provider] || {};
        const status = providerStatus(provider, routeState, providerStatusMap, cloudByName[provider], providerDiagnostics[provider] || {});
        if (routeState.enabled === false) disabled += 1;
        if (status.cls === 'bad') down += 1;
        if (status.cls === 'warn') degraded += 1;
      });

      return { total: providerNames.size, down, degraded, disabled };
    }

    function renderOverallHealth(data, metrics, requestMetrics, providerSummary) {
      const alertsList = data.alerts || [];
      const actionableAlerts = alertsList.filter((alert) => String(alert.level || '').toLowerCase() !== 'ok');
      const criticalAlerts = alertsList.filter((alert) => String(alert.level || '').toLowerCase() === 'critical');
      const workers = data.workers || [];
      const deadWorkers = workers.filter((worker) => String(worker.status || '').toLowerCase() === 'dead').length;
      const degradedWorkers = workers.filter((worker) => ['degraded', 'stale'].includes(String(worker.status || '').toLowerCase())).length;
      const queueDepth = Number(metrics.queue_length || 0);
      const p95 = Number(metrics.request_latency_ms_p95 || requestMetrics.p95_latency_ms || 0);
      const errorRate = Number(requestMetrics.error_rate ?? 0);
      const queueLimit = Number(metrics.max_worker_queue_size || 0);
      const queuePressure = queueLimit > 0 ? queueDepth / queueLimit : 0;

      let level = 'ok';
      const reasons = [];
      if (criticalAlerts.length || deadWorkers || providerSummary.down) {
        level = 'bad';
        if (criticalAlerts.length) reasons.push(`${criticalAlerts.length} critical alert${criticalAlerts.length === 1 ? '' : 's'}`);
        if (deadWorkers) reasons.push(`${deadWorkers} dead worker${deadWorkers === 1 ? '' : 's'}`);
        if (providerSummary.down) reasons.push(`${providerSummary.down} provider${providerSummary.down === 1 ? '' : 's'} down`);
      }
      if (level !== 'bad' && (actionableAlerts.length || degradedWorkers || providerSummary.degraded || errorRate >= 0.05 || queuePressure >= 0.8 || p95 >= 5000)) {
        level = 'warn';
        if (actionableAlerts.length) reasons.push(`${actionableAlerts.length} active alert${actionableAlerts.length === 1 ? '' : 's'}`);
        if (degradedWorkers) reasons.push(`${degradedWorkers} degraded worker${degradedWorkers === 1 ? '' : 's'}`);
        if (providerSummary.degraded) reasons.push(`${providerSummary.degraded} provider${providerSummary.degraded === 1 ? '' : 's'} degraded`);
        if (errorRate >= 0.05) reasons.push(`${formatPercent(errorRate)} request errors`);
        if (queuePressure >= 0.8) reasons.push(`queue pressure ${Math.round(queuePressure * 100)}%`);
        if (p95 >= 5000) reasons.push(`p95 ${formatLatency(p95)}`);
      }
      if (!reasons.length) reasons.push('No active blockers detected.');

      const title = level === 'bad' ? 'Critical' : level === 'warn' ? 'Degraded' : 'Healthy';
      const alertCls = actionableAlerts.length ? (criticalAlerts.length ? 'bad' : 'warn') : 'ok';
      const providerCls = providerSummary.down ? 'bad' : providerSummary.degraded ? 'warn' : 'ok';
      const workerCls = deadWorkers ? 'bad' : degradedWorkers ? 'warn' : 'ok';
      const queueCls = queuePressure >= 1 ? 'bad' : queuePressure >= 0.8 ? 'warn' : 'ok';
      const latencyCls = p95 >= 5000 ? 'warn' : 'ok';
      const errorCls = errorRate >= 0.05 ? 'warn' : 'ok';

      overallHealth.innerHTML = `
        <div class="health-cell">
          <div class="health-primary">
            <span class="health-dot ${level}"></span>
            <div>
              <div class="health-title ${level}">${title}</div>
              <div class="health-reason">${escapeHtml(reasons.slice(0, 3).join(' · '))}</div>
            </div>
          </div>
        </div>
        ${healthCell('Alerts', actionableAlerts.length, criticalAlerts.length ? `${criticalAlerts.length} critical` : '', alertCls)}
        ${healthCell('Providers', providerSummary.total ? `${Math.max(0, providerSummary.total - providerSummary.down)}/${providerSummary.total}` : '-', providerSummary.total ? (providerSummary.disabled ? `${providerSummary.disabled} disabled` : '') : 'no telemetry', providerCls)}
        ${healthCell('Workers', workers.length ? `${Math.max(0, workers.length - deadWorkers)}/${workers.length}` : '-', workers.length ? (degradedWorkers ? `${degradedWorkers} degraded` : '') : 'none registered', workerCls)}
        ${healthCell('Queue', queueDepth, queueLimit ? `${Math.round(queuePressure * 100)}% of limit` : '', queueCls)}
        ${healthCell('P95 Latency', formatLatency(p95), '', latencyCls)}
        ${healthCell('Error Rate', formatPercent(errorRate), '', errorCls)}
      `;
    }

    function renderDashboardError(error) {
      const rawMessage = String(error?.message || error || 'Unknown dashboard error');
      const summary = escapeHtml(summarizeError(error));
      const message = escapeHtml(rawMessage);
      overallHealth.innerHTML = `
        <div class="health-cell">
          <div class="health-primary">
            <span class="health-dot bad"></span>
            <div>
              <div class="health-title bad">Dashboard Degraded</div>
              <div class="health-reason">${summary}</div>
            </div>
          </div>
        </div>
        ${healthCell('Alerts', 1, 'dashboard fetch failed', 'bad')}
        ${healthCell('Providers', '-', 'unavailable', 'warn')}
        ${healthCell('Workers', '-', 'unavailable', 'warn')}
        ${healthCell('Queue', '-', 'unavailable', 'warn')}
        ${healthCell('P95 Latency', '-', '', 'warn')}
        ${healthCell('Error Rate', '-', '', 'warn')}
      `;
      topMetrics.innerHTML = [
        metricCard('Dashboard', 'Error', 'Control plane or overview API unavailable'),
        metricCard('Next Step', 'Check', 'Start control plane and worker services'),
      ].join('');
      alerts.innerHTML = `<span class="pill"><strong class="bad">critical</strong><span>${message}</span></span>`;
      providerHealthGrid.innerHTML = '<span class="pill">Provider health unavailable until overview API responds</span>';
      providers.innerHTML = '<span class="pill">No provider telemetry available</span>';
      cloudProviders.innerHTML = '';
      requestMetricsCards.innerHTML = '';
      providerMetricsTable.innerHTML = '<tr><td colspan="6">No provider metrics available.</td></tr>';
      routingProviders.innerHTML = '<span class="pill">Routing controls unavailable</span>';
      routingOverridesTable.innerHTML = '<tr><td colspan="3">No routing data available.</td></tr>';
      updated.textContent = `Dashboard error: ${summary}`;
      setChartGroup(activeChartGroup);
    }

    function diagnosticAgeSeconds(timestamp) {
      const value = Number(timestamp || 0);
      if (!value) return Infinity;
      return Math.max(0, (Date.now() / 1000) - value);
    }

    function providerStatus(provider, routeState, providerStatusMap, cloudState, diagnostic = {}) {
      const enabled = routeState?.enabled !== false;
      const routeHealthy = routeState?.healthy;
      const telemetryHealthy = providerStatusMap[provider];
      const cloudHealthy = cloudState?.ok;
      const configured = !cloudState || cloudState.status !== 'not_configured';
      const consecutiveErrors = Number(diagnostic.consecutive_errors || 0);
      const lastError = String(diagnostic.last_error_message || '').toLowerCase();
      const staleError = diagnosticAgeSeconds(diagnostic.last_error_at) >= 600;
      const hasPassingProbe = cloudHealthy === true || routeHealthy === true;
      const canRecover = staleError && hasPassingProbe;
      const cooldownRemaining = Number(routeState?.cooldown_remaining_s || 0);
      const lastSuccessAt = Number(diagnostic.last_success_at || 0);
      const lastErrorAt = Number(diagnostic.last_error_at || 0);

      if (cooldownRemaining > 0) {
        return { cls: 'warn', label: 'cooldown', reason: `Temporarily skipped for ${Math.ceil(cooldownRemaining)}s after provider failure.` };
      }

      if (!enabled) return { cls: 'bad', label: 'disabled', reason: 'Routing is manually disabled.' };
      if (configured === false) return { cls: 'warn', label: 'not configured', reason: cloudState?.message || 'API key is not configured.' };
      if (hasPassingProbe && consecutiveErrors === 0 && (!lastErrorAt || lastSuccessAt >= lastErrorAt || routeHealthy === true || cloudHealthy === true)) {
        return { cls: 'ok', label: 'healthy', reason: 'Provider is enabled and passing available health checks.' };
      }
      if (lastError.includes('model_not_found') || lastError.includes('404') || lastError.includes('not found')) {
        return { cls: 'warn', label: 'model unavailable', reason: 'Requested model was not available from this provider; local fallback should handle new requests.' };
      }
      if (lastError.includes('bad gateway') || lastError.includes('502')) {
        return { cls: 'warn', label: 'upstream gateway', reason: 'Provider gateway returned 502; local fallback should handle new requests.' };
      }
      if (lastError.includes('provider_rate_limited') || lastError.includes('rate limit') || lastError.includes('429')) {
        if (canRecover) {
          return { cls: 'warn', label: 'recovering', reason: 'Last rate-limit error is stale and the latest health probe is passing.' };
        }
        return { cls: 'warn', label: 'throttled', reason: 'Provider is rate limited; local fallback should handle new requests.' };
      }
      if (lastError.includes('provider_timeout') || lastError.includes('read timed out') || lastError.includes('timeout')) {
        if (canRecover) {
          return { cls: 'warn', label: 'recovering', reason: 'Last timeout is stale and the latest health probe is passing.' };
        }
        return { cls: 'warn', label: 'timeout', reason: 'Provider timed out; local fallback should handle new requests.' };
      }
      if (consecutiveErrors > 0 && canRecover) {
        return { cls: 'warn', label: 'recovering', reason: 'Recent provider failures are stale and health telemetry is passing.' };
      }
      if (consecutiveErrors >= 3) {
        return { cls: 'bad', label: 'failing', reason: `${consecutiveErrors} consecutive provider failures.` };
      }
      if (routeHealthy === false || telemetryHealthy === false || cloudHealthy === false) {
        const reason = cloudState?.message || cloudState?.status || 'Health telemetry reports failure.';
        return { cls: 'bad', label: 'unhealthy', reason };
      }
      if (consecutiveErrors > 0) {
        return { cls: 'warn', label: 'degraded', reason: `${consecutiveErrors} recent provider failure${consecutiveErrors === 1 ? '' : 's'}.` };
      }
      if (routeHealthy === undefined && telemetryHealthy === undefined && cloudHealthy === undefined) {
        return { cls: 'warn', label: 'no telemetry', reason: 'No provider health signal has been reported yet.' };
      }
      return { cls: 'ok', label: 'healthy', reason: 'Provider is enabled and passing available health checks.' };
    }

    function renderProviderHealthCards(data, providerMetrics, routingProvidersData, cloudList) {
      const providerStatusMap = data.provider_status || {};
      const providerDiagnostics = data.provider_diagnostics || {};
      const cloudByName = Object.fromEntries((cloudList || []).map((item) => [item.name, item]));
      const providerNames = new Set([
        ...Object.keys(providerStatusMap),
        ...Object.keys(providerMetrics || {}),
        ...Object.keys(routingProvidersData || {}),
        ...Object.keys(providerDiagnostics || {}),
        ...Object.keys(cloudByName),
      ]);

      if (providerNames.size === 0) {
        providerHealthGrid.innerHTML = '<span class="pill">No provider telemetry yet</span>';
        return;
      }

      providerHealthGrid.innerHTML = [...providerNames].sort().map((provider) => {
        const stats = providerMetrics[provider] || {};
        const routeState = routingProvidersData[provider] || {};
        const cloudState = cloudByName[provider];
        const diagnostic = providerDiagnostics[provider] || {};
        const status = providerStatus(provider, routeState, providerStatusMap, cloudState, diagnostic);
        const enabled = routeState.enabled !== false;
        const requestCount = Number(stats.requests || 0);
        const tokenText = `${Number(stats.total_input_tokens || 0)} in / ${Number(stats.total_output_tokens || 0)} out`;
        const consecutiveErrors = Number(diagnostic.consecutive_errors || 0);
        const lastError = shortDiagnostic(diagnostic.last_error_message);
        const lastSuccessText = timeAgo(diagnostic.last_success_at);
        const lastErrorText = diagnostic.last_error_at ? timeAgo(diagnostic.last_error_at) : 'none';
        const cloudDetail = cloudState
          ? `${cloudState.model_count ?? 0} models${cloudState.base_url ? ` · ${cloudState.base_url}` : ''}`
          : 'local / routing telemetry';

        return `
          <div class="provider-card">
            <div class="provider-top">
              <div>
                <div class="provider-name">${escapeHtml(provider)}</div>
                <div class="provider-sub">${escapeHtml(cloudDetail)}</div>
              </div>
              <span class="${status.cls}">${status.label}</span>
            </div>
            <div class="muted-line">${escapeHtml(status.reason)}</div>
            <div class="provider-matrix">
              <div class="provider-stat"><span>Requests</span><strong>${requestCount}</strong></div>
              <div class="provider-stat"><span>Error Rate</span><strong class="${Number(stats.error_rate || 0) > 0.05 ? 'bad' : 'ok'}">${formatPercent(stats.error_rate)}</strong></div>
              <div class="provider-stat"><span>Avg Latency</span><strong>${formatLatency(stats.avg_latency_ms || routeState.latency_ms || cloudState?.latency_ms)}</strong></div>
              <div class="provider-stat"><span>Tokens</span><strong>${escapeHtml(tokenText)}</strong></div>
            </div>
            <div class="provider-diagnostics">
              <div>Last success: <strong>${escapeHtml(lastSuccessText)}</strong>${diagnostic.last_success_model ? ` · ${escapeHtml(diagnostic.last_success_model)}` : ''}</div>
              <div>Last error: <strong class="${consecutiveErrors ? 'bad' : ''}">${escapeHtml(lastErrorText)}</strong>${lastError ? ` · ${escapeHtml(lastError)}` : ''}</div>
              <div>Consecutive failures: <strong class="${consecutiveErrors ? 'bad' : ''}">${consecutiveErrors}</strong></div>
            </div>
            <div class="provider-actions">
              <span class="muted-line">Route: ${enabled ? 'enabled' : 'disabled'}</span>
              <button onclick="probeProvider(event, '${escapeHtml(escapeJsString(provider))}')" class="btn">Probe</button>
              <button onclick="toggleProvider(event, '${escapeHtml(escapeJsString(provider))}', ${!enabled})" class="btn ${enabled ? 'btn-disable' : 'btn-enable'}">${enabled ? 'Disable' : 'Enable'}</button>
            </div>
          </div>
        `;
      }).join('');
    }

    function renderOverview(data) {
      const metrics = data.metrics || {};

      const vision5m = (metrics.vision_error_rate_5m && metrics.vision_error_rate_5m.model) || { requests: 0, errors: 0, error_rate: 0 };

      topMetrics.innerHTML = [
        metricCard('Requests', metrics.request_total || 0, `avg ${metrics.request_latency_ms_avg || 0} ms, p95 ${metrics.request_latency_ms_p95 || 0} ms, p99 ${metrics.request_latency_ms_p99 || 0} ms`),
        metricCard('Queue Depth', metrics.queue_length || 0, `Redis backend: ${metrics.redis_backend || 'memory'}`),
        metricCard('Vision Errors (5m)', `${(vision5m.error_rate * 100).toFixed(1)}%`, `${vision5m.errors || 0}/${vision5m.requests || 0} failed`),
        metricCard('Nodes', (data.nodes || []).length, `${(data.gpus || []).length} GPUs discovered`),
        metricCard('Workers', (data.workers || []).length, `${(data.models || []).length} models registered`),
      ].join('');

      const cloudList = data.cloud_providers || [];
      if (cloudList.length === 0) {
        cloudProviders.innerHTML = '<span class="pill">No cloud providers configured</span>';
      } else {
        cloudProviders.innerHTML = cloudList.map(cp => {
          const statusCls = cp.ok ? 'ok' : (cp.status === 'not_configured' ? 'warn' : 'bad');
          const detail = cp.ok
            ? `${cp.model_count || 0} models • ${cp.latency_ms || 0}ms`
            : (cp.message || cp.status);
          return `<span class="pill"><strong>${cp.name}</strong><span class="${statusCls}">${cp.ok ? 'healthy' : cp.status}</span><span style="color: var(--muted); font-size: 0.85rem;">${detail}</span></span>`;
        }).join('');
      }

      const webSearchEl = document.getElementById('web-search-providers');
      if (data.web_search_providers) {
        const wsList = data.web_search_providers || [];
        webSearchEl.innerHTML = '<span class="subsection-title" style="font-size:0.85rem;">Web Search</span>' +
          wsList.map(ws => {
            const cls = ws.configured ? 'ok' : 'warn';
            const label = ws.configured ? 'configured' : 'not set';
            return `<span class="pill"><strong>${ws.name}</strong><span class="${cls}">${label}</span></span>`;
          }).join('');
      }

      alerts.innerHTML = (data.alerts || []).map(alert => (
        `<span class="pill"><strong class="${statusClass(alert.level)}">${alert.level}</strong><span>${alert.message}</span></span>`
      )).join('') || '<span class="pill">No alerts</span>';

      const rm = data.request_metrics || {};
      const pm = data.provider_metrics || {};
      const routingProviderData = (data.routing_status || {}).providers || {};
      const providerSummary = buildProviderSummary(data, pm, routingProviderData, cloudList);
      renderOverallHealth(data, metrics, rm, providerSummary);

      document.getElementById('request-metrics-summary').textContent =
        `${rm.total_requests || 0} requests tracked • ${formatPercent(rm.error_rate)} error rate`;
      requestMetricsCards.innerHTML = [
        metricCard('Total Requests', rm.total_requests || 0, `Streaming: ${rm.streaming_requests || 0} | Non-streaming: ${rm.non_streaming_requests || 0}`),
        metricCard('Tokens', `${rm.total_input_tokens || 0} in / ${rm.total_output_tokens || 0} out`, `Ratio: ${((rm.total_output_tokens || 0) / Math.max(1, rm.total_input_tokens || 1) * 100).toFixed(0)}%`),
        metricCard('Avg Latency', `${rm.avg_latency_ms || 0} ms`, `P50: ${rm.p50_latency_ms || 0}ms | P95: ${rm.p95_latency_ms || 0}ms | P99: ${rm.p99_latency_ms || 0}ms`),
        metricCard('Error Rate', formatPercent(rm.error_rate), `${rm.total_errors || 0}/${rm.total_requests || 0} failed`),
      ].join('');

      renderProviderHealthCards(data, pm, routingProviderData, cloudList);
      providerMetricsTable.innerHTML = Object.entries(pm).map(([provider, stats]) => `
        <tr>
          <td><strong>${provider}</strong></td>
          <td>${stats.requests || 0}</td>
          <td class="${(stats.error_rate || 0) > 0.05 ? 'bad' : 'ok'}">${((stats.error_rate || 0) * 100).toFixed(1)}%</td>
          <td>${stats.avg_latency_ms || 0} ms</td>
          <td>${stats.total_input_tokens || 0}</td>
          <td>${stats.total_output_tokens || 0}</td>
        </tr>
      `).join('') || '<tr><td colspan="6">No provider metrics yet.</td></tr>';

      const rs = data.routing_status || {};
      const routingProvs = rs.providers || {};
      const localOnly = Boolean(rs.local_only);
      routingModeLabel.textContent = localOnly ? 'Local only mode' : 'Automatic routing';
      routingModeDetail.textContent = localOnly
        ? 'Only local Ollama routing is enabled. Cloud providers are paused.'
        : 'Local and cloud providers may be used according to routing rules.';
      const fallback = rs.fallback || {};
      const fallbackModel = fallback.resolved_model || fallback.ollama_default_model || 'auto';
      const fallbackWorker = fallback.worker?.base_url ? ` @ ${fallback.worker.base_url}` : '';
      const fallbackSource = fallback.source === 'configured' ? 'configured' : 'auto';
      routingFallbackDetail.textContent = `Cloud fallback: ${fallbackModel}${fallbackWorker} (${fallbackSource})`;
      localOnlyButton.disabled = localOnly;
      enableAllProvidersButton.disabled = !localOnly;
      routingProviders.innerHTML = Object.entries(routingProvs).map(([name, p]) => {
        const enabled = p.enabled;
        const healthy = p.healthy;
        const cooldownRemaining = Number(p.cooldown_remaining_s || 0);
        const inCooldown = cooldownRemaining > 0;
        const statusCls = enabled ? (inCooldown || healthy === false ? 'warn' : 'ok') : 'bad';
        const statusText = enabled ? (inCooldown ? 'cooldown' : (healthy ? 'healthy' : 'unhealthy')) : 'disabled';
        const latencyText = inCooldown ? `${Math.ceil(cooldownRemaining)}s left` : `${p.latency_ms || 0}ms`;
        return `
          <div class="routing-control-card">
            <div>
              <div class="routing-provider-name">${escapeHtml(name)}</div>
              <div class="routing-provider-meta">
                <span class="${statusCls}">${statusText}</span>
                <span class="muted-line">${latencyText}</span>
              </div>
            </div>
            <button onclick="toggleProvider(event, '${escapeHtml(escapeJsString(name))}', ${!enabled})" class="btn ${enabled ? 'btn-disable' : 'btn-enable'}">${enabled ? 'Disable' : 'Enable'}</button>
          </div>
        `;
      }).join('');
      routingProviders.innerHTML = routingProviders.innerHTML
        ? `<div class="routing-control-grid">${routingProviders.innerHTML}</div>`
        : '<span class="pill">No routing providers configured</span>';

      const modelAliases = rs.model_aliases || [];
      modelAliasTable.innerHTML = modelAliases.map((entry) => {
        const configured = entry.configured !== false;
        const providerClass = configured ? 'ok' : 'bad';
        const workerLabel = entry.worker_label || entry.worker?.base_url || 'not configured';
        const capabilities = Array.isArray(entry.capabilities) && entry.capabilities.length
          ? entry.capabilities.join(', ')
          : 'none';
        return `
          <tr>
            <td class="mono">${escapeHtml(entry.gateway_model || '')}</td>
            <td class="mono">${escapeHtml(entry.prefixed_alias || entry.alias || '')}</td>
            <td class="mono">${escapeHtml(entry.target_model || '')}</td>
            <td><span class="${providerClass}">${escapeHtml(entry.provider || 'unknown')}</span></td>
            <td class="mono">${escapeHtml(workerLabel)}</td>
            <td>${escapeHtml(capabilities)}</td>
          </tr>
        `;
      }).join('') || '<tr><td colspan="6">No model aliases configured.</td></tr>';

      const overrides = rs.model_overrides || {};
      routingOverridesTable.innerHTML = Object.entries(overrides).map(([model, provider]) => `
        <tr>
          <td class="mono">${escapeHtml(model)}</td>
          <td><span class="pill"><strong>${escapeHtml(provider)}</strong></span></td>
          <td><button onclick="removeOverride(event, '${escapeHtml(escapeJsString(model))}')" class="btn btn-danger">Remove</button></td>
        </tr>
      `).join('') || '<tr><td colspan="3">No model overrides set. Requests are routed automatically.</td></tr>';

      const auditEvents = rs.audit_events || [];
      routingAuditTable.innerHTML = auditEvents.slice().reverse().map((event) => `
        <tr>
          <td>${escapeHtml(timeAgo(event.timestamp))}</td>
          <td class="mono">${escapeHtml(event.actor || 'system')}</td>
          <td>${escapeHtml(formatAuditAction(event.action || 'unknown'))}</td>
          <td>${escapeHtml(formatAuditDetails(event))}</td>
        </tr>
      `).join('') || '<tr><td colspan="4">No routing activity recorded yet.</td></tr>';

      nodesTable.innerHTML = (data.nodes || []).map(node => `
        <tr>
          <td class="mono">${node.node_id}</td>
          <td class="mono">${node.ip}</td>
          <td>${(node.gpus || []).map(gpu => gpu.name).join('<br>') || '-'}</td>
          <td>${(node.workers || []).join(', ') || '-'}</td>
          <td class="${statusClass(node.status)}">${node.status}</td>
        </tr>
      `).join('') || '<tr><td colspan="5">No nodes registered yet.</td></tr>';

      gpusTable.innerHTML = (data.gpus || []).map(gpu => {
        const utilization = Number(gpu.utilization || 0);
        const powerWatts = Number(gpu.power_watts ?? gpu.metadata?.power_watts ?? 0);
        const temperature = Number(gpu.temperature || 0);
        return `
        <tr>
          <td class="mono">${gpu.node_id || 'local'}</td>
          <td>${gpu.name}</td>
          <td>${gpu.memory || 0}</td>
          <td>
            <div>${utilization}%</div>
            <div class="status-bar"><span style="width:${Math.min(utilization, 100)}%"></span></div>
          </td>
          <td>${powerWatts.toFixed(1)}W</td>
          <td>${temperature.toFixed(1)}C</td>
        </tr>
      `;
      }).join('') || '<tr><td colspan="6">No GPU telemetry available.</td></tr>';

      workersTable.innerHTML = (data.workers || []).map(worker => {
        const runtime = worker.metadata || {};
        const psCount = Number(runtime.ps_count || 0);
        const psModels = Array.isArray(runtime.ps_models) ? runtime.ps_models : [];
        const processors = Array.isArray(runtime.processors) ? runtime.processors : [];
        const psError = runtime.error ? String(runtime.error) : '';
        let runtimeLabel = 'idle';
        let runtimeClass = 'ok';
        if (psCount > 0 && Number(worker.queue_size || 0) > 0) {
          runtimeLabel = 'running';
          runtimeClass = 'warn';
        } else if (psCount > 0) {
          runtimeLabel = 'loaded';
          runtimeClass = 'ok';
        }
        const processorsText = processors.length ? processors.join(', ') : '-';
        const psCell = psError
          ? `<span class="warn">${psError}</span>`
          : `${psCount} active${psModels.length ? `<br>${psModels.join(', ')}` : ''}${processors.length ? `<br>${processors.join(', ')}` : ''}`;
        return `
        <tr>
          <td class="mono">${worker.worker_id}</td>
          <td class="mono">${worker.base_url}</td>
          <td>${worker.gpu_name} (#${worker.gpu_id})</td>
          <td>${worker.queue_size}</td>
          <td>${(worker.models || []).join(', ') || 'dynamic'}</td>
          <td><span class="${runtimeClass}">${runtimeLabel}</span></td>
          <td>${processorsText}</td>
          <td>${psCell}</td>
          <td>${formatTime(worker.last_heartbeat)}</td>
          <td class="${statusClass(worker.status)}">${worker.status}</td>
        </tr>
      `;
      }).join('') || '<tr><td colspan="10">No workers registered yet.</td></tr>';

      modelsTable.innerHTML = (data.models_enriched || data.models || []).map(model => {
        const configured = model.workers_configured || (model.worker_bindings || []).map(b => `${b.node_id}:${b.port}`);
        const online = model.workers_online || [];
        const workersDisplay = configured.length
          ? `${configured.join(', ')} (${online.length}/${configured.length})`
          : '-';
        return `
        <tr>
          <td>${model.name}</td>
          <td>${model.provider}</td>
          <td>${workersDisplay}</td>
          <td>${(model.capabilities || []).join(', ') || '-'}</td>
        </tr>
      `;
      }).join('') || '<tr><td colspan="4">No models configured.</td></tr>';

      tasksTable.innerHTML = (data.tasks || []).slice(0, 20).map(task => `
        <tr>
          <td class="mono"><a href="/task/${encodeURIComponent(task.task_id)}">${task.task_id}</a></td>
          <td class="${statusClass(task.status)}">${task.status}</td>
          <td>${task.payload?.model || '-'}</td>
          <td>${task.retries || 0}</td>
          <td>${formatTime(task.updated_at)}</td>
        </tr>
      `).join('') || '<tr><td colspan="5">No queued tasks.</td></tr>';


      const errorCodeRows = Object.entries(metrics.error_code_usage || {})
        .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0));
      errorCodesTable.innerHTML = errorCodeRows.map(([code, count]) => `
        <tr>
          <td class="mono">${code}</td>
          <td>${count}</td>
        </tr>
      `).join('') || '<tr><td colspan="2">No error codes recorded.</td></tr>';

      const endpointLatencyRows = Object.entries(metrics.endpoint_latency || {})
        .sort((a, b) => Number((b[1] || {}).p95_ms || 0) - Number((a[1] || {}).p95_ms || 0));
      endpointLatencyTable.innerHTML = endpointLatencyRows.map(([endpoint, item]) => `
        <tr>
          <td class="mono">${endpoint}</td>
          <td>${Number(item?.requests || 0)}</td>
          <td>${Number(item?.avg_ms || 0).toFixed(1)}</td>
          <td>${Number(item?.p95_ms || 0).toFixed(1)}</td>
          <td>${Number(item?.p99_ms || 0).toFixed(1)}</td>
        </tr>
      `).join('') || '<tr><td colspan="5">No endpoint latency metrics yet.</td></tr>';

      updated.textContent = `Last update: ${new Date().toLocaleTimeString()}`;

      // Render GPU OS
      const gpuOs = data.gpu_os || {};
      const devices = gpuOs.devices || [];
      const scheduler = gpuOs.scheduler || {};
      gpuOsPanel.innerHTML = devices.length === 0
        ? '<span class="pill">No GPU devices registered</span>'
        : `<div class="runtime-grid">${devices.map(d => {
            const vramPct = d.vram_used_pct || 0;
            const barColor = vramPct > 80 ? 'var(--bad)' : vramPct > 50 ? 'var(--warn)' : 'var(--good)';
            return `<div class="runtime-card">
              <h4>${escapeHtml(d.name)} (${d.device_id})</h4>
              <div class="stat-row"><span class="label">VRAM</span><span class="value">${d.used_vram_gb} / ${d.total_vram_gb} GB</span></div>
              <div class="vram-bar"><div class="fill" style="width:${vramPct}%;background:${barColor}"></div></div>
              <div class="stat-row"><span class="label">Utilization</span><span class="value">${d.utilization}%</span></div>
              <div class="stat-row"><span class="label">Temp</span><span class="value">${d.temperature}°C</span></div>
              <div class="stat-row"><span class="label">Models</span><span class="value">${(d.models_loaded || []).join(', ') || 'none'}</span></div>
              <div class="stat-row"><span class="label">Health</span><span class="value ${d.healthy ? 'ok' : 'bad'}">${d.healthy ? 'healthy' : 'unhealthy'}</span></div>
            </div>`;
          }).join('')}
          <div class="runtime-card">
            <h4>Scheduler</h4>
            <div class="stat-row"><span class="label">Max VRAM</span><span class="value">${scheduler.max_vram_gb || 0} GB</span></div>
            <div class="stat-row"><span class="label">Loaded Models</span><span class="value">${scheduler.loaded_count || 0}</span></div>
            ${(scheduler.models || []).map(m => `<div class="stat-row"><span class="label">${escapeHtml(m.name)}</span><span class="value">${m.vram_gb} GB, ${m.use_count} uses</span></div>`).join('')}
          </div>
        </div>`;

      // Render Multi-Agent
      const ma = data.multi_agent || {};
      multiAgentStats.innerHTML = `<div class="stat-row"><span class="label">Registered Agents</span><span class="value">${(ma.agents || []).length}</span></div>
        ${(ma.agents || []).map(a => `<div class="stat-row"><span class="label">${escapeHtml(a)}</span></div>`).join('')}
        ${(ma.agents || []).length === 0 ? '<span class="pill">No agents registered</span>' : ''}`;

      // Render Security
      const sec = data.security || {};
      const auth = sec.api_key_auth || {};
      securityStats.innerHTML = `<div class="stat-row"><span class="label">API Key Auth</span><span class="value ${auth.enabled ? 'ok' : 'warn'}">${auth.enabled ? 'enabled' : 'disabled'}</span></div>
        <div class="stat-row"><span class="label">Keys Configured</span><span class="value">${auth.key_count || 0}</span></div>
        <div class="stat-row"><span class="label">Rate Limit Buckets</span><span class="value">${sec.rate_limiter_buckets || 0}</span></div>
        <div class="stat-row"><span class="label">Max Text Length</span><span class="value">${(sec.max_text_length || 0).toLocaleString()} chars</span></div>
        <div class="stat-row"><span class="label">Max Messages</span><span class="value">${sec.max_messages || 0}</span></div>`;

      // Render Observability
      const obs = data.observability || {};
      const obsMetrics = obs.metrics || {};
      const counterKeys = Object.keys(obsMetrics).filter(k => k.startsWith('counter:'));
      const histogramKeys = Object.keys(obsMetrics).filter(k => k.startsWith('histogram:'));
      observabilityStats.innerHTML = `<div class="stat-row"><span class="label">Counters</span><span class="value">${counterKeys.length}</span></div>
        <div class="stat-row"><span class="label">Histograms</span><span class="value">${histogramKeys.length}</span></div>
        ${counterKeys.slice(0, 5).map(k => `<div class="stat-row"><span class="label">${escapeHtml(k.slice(8))}</span><span class="value">${obsMetrics[k]}</span></div>`).join('')}`;

      // Render API Keys
      (async () => {
        const panel = document.getElementById('api-keys-panel');
        if (!panel) return;
        try {
          const resp = await fetch('/api/security/api-keys');
          if (resp.status === 403) { panel.innerHTML = '<span class="pill">Admin access required</span>'; return; }
          if (!resp.ok) { panel.innerHTML = '<span class="pill warn">Failed to load API keys</span>'; return; }
          const keys = await resp.json();
          if (!keys || keys.length === 0) {
            panel.innerHTML = `<span class="pill">No API keys configured</span>
              <button class="btn btn-sm" onclick="createApiKey()" style="margin-top:8px">Generate Key</button>`;
            return;
          }
          panel.innerHTML = `<table class="table"><thead><tr>
            <th>Prefix</th><th>Name</th><th>Owner</th><th>Status</th><th>Created</th><th>Last Used</th><th></th>
          </tr></thead><tbody>
            ${keys.map(k => `<tr>
              <td><code>${escapeHtml(k.key_prefix)}...</code></td>
              <td>${escapeHtml(k.name || '-')}</td>
              <td>${escapeHtml(k.owner_display_name || k.owner_email || '—')}</td>
              <td><span class="pill ${k.is_active ? 'ok' : 'disabled'}">${k.is_active ? 'Active' : 'Revoked'}</span></td>
              <td>${k.created_at ? timeAgo(k.created_at) : '-'}</td>
              <td>${k.last_used_at ? timeAgo(k.last_used_at) : 'Never'}</td>
              <td>${k.is_active ? `<button class="btn btn-sm btn-danger" onclick="revokeApiKey(${k.id})">Revoke</button>` : ''}</td>
            </tr>`).join('')}
          </tbody></table>
          <button class="btn btn-sm" onclick="createApiKey()" style="margin-top:8px">Generate Key</button>`;
        } catch(e) {
          panel.innerHTML = '<span class="pill warn">Error loading API keys</span>';
        }
      })();

      // Render Users
      (async () => {
        const panel = document.getElementById('users-panel');
        if (!panel) return;
        try {
          const resp = await fetch('/api/users');
          if (resp.status === 403) { panel.innerHTML = '<span class="pill">Admin access required</span>'; return; }
          if (!resp.ok) { panel.innerHTML = '<span class="pill warn">Failed to load users</span>'; return; }
          const users = await resp.json();
          panel.innerHTML = `<table class="table"><thead><tr>
            <th>Email</th><th>Name</th><th>Role</th><th>Status</th><th>Created</th><th>Last Login</th><th></th>
          </tr></thead><tbody>
            ${users.map(u => `<tr>
              <td>${escapeHtml(u.email)}</td>
              <td>${escapeHtml(u.display_name)}</td>
              <td><span class="pill ${u.role === 'admin' ? 'warn' : 'ok'}">${escapeHtml(u.role)}</span></td>
              <td><span class="pill ${u.is_active ? 'ok' : 'disabled'}">${u.is_active ? 'Active' : 'Disabled'}</span></td>
              <td>${u.created_at ? timeAgo(u.created_at) : '-'}</td>
              <td>${u.last_login_at ? timeAgo(u.last_login_at) : 'Never'}</td>
              <td>
                <button class="btn btn-sm" onclick="editUser(${u.id}, '${escapeHtml(u.email)}', '${escapeHtml(u.display_name)}', '${u.role}', ${u.is_active})">Edit</button>
                <button class="btn btn-sm btn-danger" onclick="deleteUser(${u.id}, '${escapeHtml(u.email)}')">Delete</button>
              </td>
            </tr>`).join('')}
          </tbody></table>
          <button class="btn btn-sm" onclick="showCreateUserModal()" style="margin-top:8px">Add User</button>`;
        } catch(e) {
          panel.innerHTML = '<span class="pill warn">Error loading users</span>';
        }
      })();

      // Update chart history
      const now = new Date();
      const timeLabel = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const avgLatency = Number(metrics.request_latency_ms_avg || 0);
      const requestTotal = Number(metrics.request_total || 0);

      // Calculate delta requests (approximate)
      const lastTotal = chartHistory.requestTotal ?? requestTotal;
      const deltaRequests = Math.max(0, requestTotal - lastTotal);
      chartHistory.requestTotal = requestTotal;

      // Collect GPU utilization per worker (use worker_id for unique key)
      const workersData = data.workers || [];
      workersData.forEach((w) => {
        const key = w.worker_id || `${w.gpu_name}-${w.gpu_id}`;
        if (!chartHistory.gpu[key]) chartHistory.gpu[key] = [];
        chartHistory.gpu[key].push(Number(w.gpu_utilization || 0));
        if (chartHistory.gpu[key].length > MAX_HISTORY) chartHistory.gpu[key].shift();
      });

      // Collect model usage from metrics
      const modelUsage = metrics.model_usage || {};
      Object.keys(modelUsage).forEach(model => {
        if (!chartHistory.model[model]) chartHistory.model[model] = 0;
        chartHistory.model[model] = Number(modelUsage[model] || 0);
      });

      // Collect worker load from workers
      workersData.forEach(w => {
        const key = w.worker_id || 'unknown';
        const load = Number(w.queue_size || 0);
        chartHistory.workerLoad[key] = load;
      });

      // Latency distribution (store current values for distribution chart)
      chartHistory.latencyDist = {
        avg: avgLatency,
        p50: Number(metrics.request_latency_ms_p50 || Math.round(avgLatency * 0.8)),
        p95: Number(metrics.request_latency_ms_p95 || 0),
        p99: Number(metrics.request_latency_ms_p99 || 0),
      };

      // Tokens per second
      chartHistory.tokensPerSec = Number(metrics.tokens_per_second || 0);

      // Time to first token (TTFT)
      chartHistory.ttft = Number(metrics.time_to_first_token_ms || 0);

      // Error rate (from vision_error_rate_5m)
      const visionStats = metrics.vision_error_rate_5m || {};
      const allStats = visionStats.all_requests || {};
      chartHistory.errorRate = Number(allStats.error_rate || 0) * 100;

      // GPU memory per worker (use worker_id for unique key)
      workersData.forEach((w) => {
        const key = w.worker_id || `${w.gpu_name}-${w.gpu_id}`;
        if (!chartHistory.memory) chartHistory.memory = {};
        chartHistory.memory[key] = Number(w.gpu_memory || 0);
      });

      // GPU power draw (use worker_id for unique key)
      workersData.forEach((w) => {
        const key = w.worker_id || `${w.gpu_name}-${w.gpu_id}`;
        if (!chartHistory.power) chartHistory.power = {};
        chartHistory.power[key] = Number(w.power_watts || w.gpu_utilization * 5 || 0);
      });

      // GPU temperature (use worker_id for unique key)
      workersData.forEach((w) => {
        const key = w.worker_id || `${w.gpu_name}-${w.gpu_id}`;
        if (!chartHistory.temp) chartHistory.temp = {};
        chartHistory.temp[key] = Number(w.temperature || 0);
      });

      chartHistory.labels.push(timeLabel);
      chartHistory.latency.push(avgLatency);
      chartHistory.requests.push(deltaRequests);

      // Keep only recent history
      if (chartHistory.labels.length > MAX_HISTORY) {
        chartHistory.labels.shift();
        chartHistory.latency.shift();
        chartHistory.requests.shift();
      }

      updateCharts();
    }

    function _keysChanged(oldKeys, newKeys) {
      return !oldKeys || oldKeys.length !== newKeys.length || oldKeys.some((k, i) => k !== newKeys[i]);
    }

    function updateCharts() {
      const anim = { duration: 300 };
      const chartBase = (overrides = {}) => ({ responsive: true, maintainAspectRatio: false, animation: anim, ...overrides });
      const noLegend = { legend: { display: false } };
      const scalesY = {
        y: { display: true, grid: { color: 'rgba(150, 210, 255, 0.1)' }, ticks: { color: '#9fb6c8', font: { size: 10 } } },
        x: { display: false }
      };
      const scalesXY = {
        y: { display: true, grid: { color: 'rgba(150, 210, 255, 0.1)' }, ticks: { color: '#9fb6c8' } },
        x: { display: true, grid: { color: 'rgba(150, 210, 255, 0.1)' }, ticks: { color: '#9fb6c8' } }
      };

      // --- Latency chart (line) ---
      const latCtx = document.getElementById('latency-chart');
      if (latCtx) {
        if (!latencyChart) {
          latencyChart = new Chart(latCtx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Latency (ms)', data: [], borderColor: '#5eead4', backgroundColor: 'rgba(94, 234, 212, 0.1)', fill: true, tension: 0.3, pointRadius: 2 }] },
            options: chartBase({ plugins: noLegend, scales: scalesY })
          });
        }
        latencyChart.data.labels = chartHistory.labels;
        latencyChart.data.datasets[0].data = chartHistory.latency;
        latencyChart.update();
      }

      // --- Requests chart (bar) ---
      const reqCtx = document.getElementById('requests-chart');
      if (reqCtx) {
        if (!requestsChart) {
          requestsChart = new Chart(reqCtx, {
            type: 'bar',
            data: { labels: [], datasets: [{ label: 'Requests', data: [], backgroundColor: 'rgba(96, 165, 250, 0.7)' }] },
            options: chartBase({ plugins: noLegend, scales: scalesY })
          });
        }
        requestsChart.data.labels = chartHistory.labels;
        requestsChart.data.datasets[0].data = chartHistory.requests;
        requestsChart.update();
      }

      // --- GPU utilization chart (line, dynamic datasets) ---
      const gpuCtx = document.getElementById('gpu-chart');
      if (gpuCtx) {
        const gpuKeys = Object.keys(chartHistory.gpu);
        if (gpuChart && gpuKeys.length > 0 && !_keysChanged(gpuChart._keys, gpuKeys)) {
          gpuKeys.forEach((key, idx) => { gpuChart.data.datasets[idx].data = chartHistory.gpu[key]; });
          gpuChart.update();
        } else if (gpuKeys.length > 0) {
          if (gpuChart) gpuChart.destroy();
          const gpuColors = ['#5eead4', '#60a5fa', '#f59e0b', '#f87171', '#34d399'];
          gpuChart = new Chart(gpuCtx, {
            type: 'line',
            data: { labels: chartHistory.labels, datasets: gpuKeys.map((key, idx) => ({ label: key, data: chartHistory.gpu[key], borderColor: gpuColors[idx % gpuColors.length], backgroundColor: 'transparent', tension: 0.3, pointRadius: 2 })) },
            options: chartBase({ plugins: { legend: { display: true, position: 'bottom', labels: { color: '#9fb6c8', boxWidth: 12, padding: 8 } } }, scales: { x: { display: false }, y: { display: true, min: 0, max: 100, grid: { color: 'rgba(150, 210, 255, 0.1)' }, ticks: { color: '#9fb6c8', font: { size: 10 }, callback: v => v + '%' } } } })
          });
          gpuChart._keys = gpuKeys;
        }
      }

      // --- Model usage chart (doughnut, dynamic) ---
      const modelCtx = document.getElementById('model-chart');
      if (modelCtx) {
        const modelKeys = Object.keys(chartHistory.model);
        if (modelChart && modelKeys.length > 0 && !_keysChanged(modelChart._keys, modelKeys)) {
          modelChart.data.datasets[0].data = modelKeys.map(k => chartHistory.model[k]);
          modelChart.update();
        } else if (modelKeys.length > 0) {
          if (modelChart) modelChart.destroy();
          const modelColors = ['#5eead4', '#60a5fa', '#f59e0b', '#f87171', '#34d399', '#a78bfa'];
          modelChart = new Chart(modelCtx, {
            type: 'doughnut',
            data: { labels: modelKeys, datasets: [{ data: modelKeys.map(k => chartHistory.model[k]), backgroundColor: modelColors.slice(0, modelKeys.length) }] },
            options: chartBase({ plugins: { legend: { display: true, position: 'bottom', labels: { color: '#9fb6c8', boxWidth: 12, padding: 8 } } } })
          });
          modelChart._keys = modelKeys;
        }
      }

      // --- Worker load chart (horizontal bar, dynamic) ---
      const workerCtx = document.getElementById('worker-chart');
      if (workerCtx) {
        const workerKeys = Object.keys(chartHistory.workerLoad);
        if (workerChart && workerKeys.length > 0 && !_keysChanged(workerChart._keys, workerKeys)) {
          workerChart.data.datasets[0].data = workerKeys.map(k => chartHistory.workerLoad[k]);
          workerChart.update();
        } else if (workerKeys.length > 0) {
          if (workerChart) workerChart.destroy();
          workerChart = new Chart(workerCtx, {
            type: 'bar',
            data: { labels: workerKeys, datasets: [{ label: 'Queue', data: workerKeys.map(k => chartHistory.workerLoad[k]), backgroundColor: 'rgba(96, 165, 250, 0.7)' }] },
            options: chartBase({ indexAxis: 'y', plugins: noLegend, scales: { x: { display: true, grid: { color: 'rgba(150, 210, 255, 0.1)' }, ticks: { color: '#9fb6c8' } }, y: { display: true, grid: { color: 'rgba(150, 210, 255, 0.1)' }, ticks: { color: '#9fb6c8', font: { size: 10 } } } } })
          });
          workerChart._keys = workerKeys;
        }
      }

      // --- Latency distribution chart (bar, fixed 4 categories) ---
      const latDistCtx = document.getElementById('latency-dist-chart');
      if (latDistCtx) {
        const dist = chartHistory.latencyDist;
        if (dist) {
          const labels = ['Avg', 'P50', 'P95', 'P99'];
          const values = [dist.avg || 0, dist.p50 || 0, dist.p95 || 0, dist.p99 || 0];
          if (!latencyDistChart) {
            latencyDistChart = new Chart(latDistCtx, {
              type: 'bar',
              data: { labels: labels, datasets: [{ label: 'Latency (ms)', data: values, backgroundColor: ['#60a5fa', '#5eead4', '#f59e0b', '#f87171'] }] },
              options: chartBase({ plugins: noLegend, scales: { y: { display: true, grid: { color: 'rgba(150, 210, 255, 0.1)' }, ticks: { color: '#9fb6c8', callback: v => v + 'ms' } }, x: { display: true, grid: { color: 'rgba(150, 210, 255, 0.1)' }, ticks: { color: '#9fb6c8' } } } })
            });
          } else {
            latencyDistChart.data.datasets[0].data = values;
            latencyDistChart.update();
          }
        }
      }

      // --- Tokens per second chart (doughnut, fixed 2 items) ---
      const tokensCtx = document.getElementById('tokens-chart');
      if (tokensCtx && chartHistory.tokensPerSec !== undefined) {
        if (!tokensChart) {
          tokensChart = new Chart(tokensCtx, {
            type: 'doughnut',
            data: { labels: ['Tokens/sec', 'Idle'], datasets: [{ data: [chartHistory.tokensPerSec, Math.max(0, 2000 - chartHistory.tokensPerSec)], backgroundColor: ['#5eead4', 'rgba(94, 234, 212, 0.2)'] }] },
            options: chartBase({ plugins: noLegend })
          });
        } else {
          tokensChart.data.datasets[0].data = [chartHistory.tokensPerSec, Math.max(0, 2000 - chartHistory.tokensPerSec)];
          tokensChart.update();
        }
      }

      // --- Time to first token chart (changes type: bar with data, doughnut idle) ---
      const ttftCtx = document.getElementById('ttft-chart');
      if (ttftCtx) {
        if (chartHistory.ttft > 0) {
          const ttftColor = chartHistory.ttft < 500 ? '#34d399' : chartHistory.ttft < 1000 ? '#5eead4' : chartHistory.ttft < 2000 ? '#f59e0b' : '#f87171';
          if (ttftChart && ttftChart.config.type !== 'bar') { ttftChart.destroy(); ttftChart = null; }
          if (!ttftChart) {
            ttftChart = new Chart(ttftCtx, {
              type: 'bar',
              data: { labels: ['TTFT'], datasets: [{ data: [chartHistory.ttft], backgroundColor: [ttftColor] }] },
              options: chartBase({ indexAxis: 'y', plugins: { legend: { display: false }, title: { display: true, text: chartHistory.ttft.toFixed(0) + 'ms', color: ttftColor } }, scales: { x: { display: true, min: 0, max: Math.max(5000, chartHistory.ttft * 1.2), grid: { color: 'rgba(150, 210, 255, 0.1)' }, ticks: { color: '#9fb6c8' } }, y: { display: false } } })
            });
          } else {
            ttftChart.data.datasets[0].data = [chartHistory.ttft];
            ttftChart.data.datasets[0].backgroundColor = [ttftColor];
            ttftChart.options.plugins.title.text = chartHistory.ttft.toFixed(0) + 'ms';
            ttftChart.options.plugins.title.color = ttftColor;
            ttftChart.options.scales.x.max = Math.max(5000, chartHistory.ttft * 1.2);
            ttftChart.update();
          }
        } else if (ttftCtx) {
          if (ttftChart && ttftChart.config.type !== 'doughnut') { ttftChart.destroy(); ttftChart = null; }
          if (!ttftChart) {
            ttftChart = new Chart(ttftCtx, {
              type: 'doughnut',
              data: { labels: ['Idle'], datasets: [{ data: [1], backgroundColor: ['rgba(150, 210, 255, 0.2)'] }] },
              options: chartBase({ plugins: noLegend })
            });
          }
        }
      }

      // --- GPU Memory chart (bar, dynamic) ---
      const memCtx = document.getElementById('memory-chart');
      if (memCtx && chartHistory.memory) {
        const memKeys = Object.keys(chartHistory.memory);
        const memValues = memKeys.map(k => chartHistory.memory[k]);
        if (memoryChart && !_keysChanged(memoryChart._keys, memKeys)) {
          memoryChart.data.datasets[0].data = memValues;
          memoryChart.data.datasets[0].backgroundColor = memValues.map(v => v > 20000 ? '#f87171' : v > 15000 ? '#f59e0b' : '#5eead4');
          memoryChart.update();
        } else if (memKeys.length > 0) {
          if (memoryChart) memoryChart.destroy();
          memoryChart = new Chart(memCtx, {
            type: 'bar',
            data: { labels: memKeys, datasets: [{ label: 'Memory (MB)', data: memValues, backgroundColor: memValues.map(v => v > 20000 ? '#f87171' : v > 15000 ? '#f59e0b' : '#5eead4') }] },
            options: chartBase({ plugins: noLegend, scales: scalesXY })
          });
          memoryChart._keys = memKeys;
        }
      }

      // --- Power Draw chart (bar, dynamic) ---
      const powCtx = document.getElementById('power-chart');
      if (powCtx && chartHistory.power) {
        const powKeys = Object.keys(chartHistory.power);
        const powValues = powKeys.map(k => chartHistory.power[k]);
        if (powerChart && !_keysChanged(powerChart._keys, powKeys)) {
          powerChart.data.datasets[0].data = powValues;
          powerChart.data.datasets[0].backgroundColor = powValues.map(v => v > 300 ? '#f87171' : v > 200 ? '#f59e0b' : '#60a5fa');
          powerChart.update();
        } else if (powKeys.length > 0) {
          if (powerChart) powerChart.destroy();
          powerChart = new Chart(powCtx, {
            type: 'bar',
            data: { labels: powKeys, datasets: [{ label: 'Power (W)', data: powValues, backgroundColor: powValues.map(v => v > 300 ? '#f87171' : v > 200 ? '#f59e0b' : '#60a5fa') }] },
            options: chartBase({ plugins: noLegend, scales: scalesXY })
          });
          powerChart._keys = powKeys;
        }
      }

      // --- Temperature chart (bar, dynamic) ---
      const tempCtx = document.getElementById('temp-chart');
      if (tempCtx && chartHistory.temp) {
        const tempKeys = Object.keys(chartHistory.temp);
        const tempValues = tempKeys.map(k => chartHistory.temp[k]);
        if (tempChart && !_keysChanged(tempChart._keys, tempKeys)) {
          tempChart.data.datasets[0].data = tempValues;
          tempChart.data.datasets[0].backgroundColor = tempValues.map(v => v > 85 ? '#f87171' : v > 75 ? '#f59e0b' : '#34d399');
          tempChart.update();
        } else if (tempKeys.length > 0) {
          if (tempChart) tempChart.destroy();
          tempChart = new Chart(tempCtx, {
            type: 'bar',
            data: { labels: tempKeys, datasets: [{ label: 'Temp (C)', data: tempValues, backgroundColor: tempValues.map(v => v > 85 ? '#f87171' : v > 75 ? '#f59e0b' : '#34d399') }] },
            options: chartBase({ plugins: noLegend, scales: { y: { display: true, max: 100, grid: { color: 'rgba(150, 210, 255, 0.1)' }, ticks: { color: '#9fb6c8' } }, x: { display: true, grid: { color: 'rgba(150, 210, 255, 0.1)' }, ticks: { color: '#9fb6c8' } } } })
          });
          tempChart._keys = tempKeys;
        }
      }

      // --- Error rate chart (doughnut, fixed 2 items) ---
      const errCtx = document.getElementById('error-chart');
      if (errCtx && chartHistory.errorRate !== undefined) {
        const errColor = chartHistory.errorRate < 1 ? '#34d399' : chartHistory.errorRate < 5 ? '#f59e0b' : '#f87171';
        if (!errorChart) {
          errorChart = new Chart(errCtx, {
            type: 'doughnut',
            data: { labels: ['Success', 'Errors'], datasets: [{ data: [Math.max(0, 100 - chartHistory.errorRate), chartHistory.errorRate], backgroundColor: ['#34d399', errColor] }] },
            options: chartBase({ plugins: { legend: { display: false }, title: { display: true, text: chartHistory.errorRate.toFixed(1) + '%', color: errColor } } })
          });
        } else {
          errorChart.data.datasets[0].data = [Math.max(0, 100 - chartHistory.errorRate), chartHistory.errorRate];
          errorChart.data.datasets[0].backgroundColor = ['#34d399', errColor];
          errorChart.options.plugins.title.text = chartHistory.errorRate.toFixed(1) + '%';
          errorChart.options.plugins.title.color = errColor;
          errorChart.update();
        }
      }

      setChartGroup(activeChartGroup);
    }

    async function refresh() {
      const response = await fetch('/api/overview');
      if (!response.ok) {
        let message = `Overview API returned ${response.status}`;
        try {
          const errorData = await response.json();
          message = errorData.detail || message;
        } catch(e) {}
        throw new Error(message);
      }
      renderOverview(await response.json());
    }

    async function toggleProvider(event, provider, enable) {
      const endpoint = enable ? `/api/providers/${encodeURIComponent(provider)}/enable` : `/api/providers/${encodeURIComponent(provider)}/disable`;
      const restoreButton = setButtonBusy(event?.currentTarget, enable ? 'Enabling...' : 'Disabling...');
      setOperationStatus(`${enable ? 'Enabling' : 'Disabling'} ${provider}...`, 'warn');
      try {
        await mutateDashboard(endpoint, { method: 'POST' });
        setOperationStatus(`${provider} ${enable ? 'enabled' : 'disabled'}.`, 'ok');
        await refresh();
      } catch (error) {
        setOperationStatus(`Failed to ${enable ? 'enable' : 'disable'} ${provider}: ${summarizeError(error)}`, 'bad');
      } finally {
        restoreButton();
      }
    }

    async function probeProvider(event, provider) {
      const restoreButton = setButtonBusy(event?.currentTarget, 'Probing...');
      setOperationStatus(`Probing ${provider} health...`, 'warn');
      try {
        const result = await mutateDashboard(`/api/providers/${encodeURIComponent(provider)}/probe`, { method: 'POST' });
        if (!result.ok) {
          const message = result.result?.message || result.result?.status || 'Probe failed.';
          throw new Error(message);
        }
        setOperationStatus(`${provider} probe passed.`, 'ok');
        await refresh();
      } catch (error) {
        setOperationStatus(`${provider} probe failed: ${summarizeError(error)}`, 'bad');
      } finally {
        restoreButton();
      }
    }

    async function setLocalOnlyMode(event, enabled) {
      const endpoint = enabled ? '/api/routing/local-only/enable' : '/api/routing/local-only/disable';
      const restoreButton = setButtonBusy(event?.currentTarget, enabled ? 'Pausing cloud...' : 'Enabling all...');
      setOperationStatus(enabled ? 'Switching to local only mode...' : 'Enabling all routing providers...', 'warn');
      try {
        await mutateDashboard(endpoint, { method: 'POST' });
        setOperationStatus(enabled ? 'Local only mode enabled.' : 'All routing providers enabled.', 'ok');
        await refresh();
      } catch (error) {
        setOperationStatus(`Failed to update routing mode: ${summarizeError(error)}`, 'bad');
      } finally {
        restoreButton();
      }
    }

    async function addOverride(event) {
      const model = document.getElementById('override-model').value.trim();
      const provider = document.getElementById('override-provider').value;
      if (!model) {
        setOperationStatus('Enter a model name before adding an override.', 'bad');
        return;
      }
      const restoreButton = setButtonBusy(event?.currentTarget || document.getElementById('add-override-button'), 'Adding...');
      setOperationStatus(`Setting ${model} to route through ${provider}...`, 'warn');
      try {
        await mutateDashboard('/api/routing/overrides', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model, provider }),
        });
        document.getElementById('override-model').value = '';
        setOperationStatus(`${model} now routes through ${provider}.`, 'ok');
        await refresh();
      } catch (error) {
        setOperationStatus(`Failed to set override for ${model}: ${summarizeError(error)}`, 'bad');
      } finally {
        restoreButton();
      }
    }

    async function removeOverride(event, model) {
      const restoreButton = setButtonBusy(event?.currentTarget, 'Removing...');
      setOperationStatus(`Removing override for ${model}...`, 'warn');
      try {
        await mutateDashboard(`/api/routing/overrides/${encodeURIComponent(model)}`, { method: 'DELETE' });
        setOperationStatus(`Override removed for ${model}.`, 'ok');
        await refresh();
      } catch (error) {
        setOperationStatus(`Failed to remove override for ${model}: ${summarizeError(error)}`, 'bad');
      } finally {
        restoreButton();
      }
    }

    window.createApiKey = async function() {
      const name = prompt('Enter a label for this API key (optional):');
      if (name === null) return;
      const restoreButton = setButtonBusy(null, 'Generating...');
      setOperationStatus('Generating API key...', 'warn');
      try {
        const resp = await fetch('/api/security/api-keys', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: name || '' }),
        });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed to create key');
        const result = await resp.json();
        alert(`API Key generated!\n\n${result.raw_key}\n\nThis key will only be shown once. Copy it now.`);
        setOperationStatus('API key generated.', 'ok');
        await refresh();
      } catch (error) {
        setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
      } finally {
        restoreButton();
      }
    };

    window.revokeApiKey = async function(keyId) {
      if (!confirm('Revoke this API key? This cannot be undone.')) return;
      const restoreButton = setButtonBusy(null, 'Revoking...');
      setOperationStatus('Revoking API key...', 'warn');
      try {
        const resp = await fetch(`/api/security/api-keys/${keyId}`, { method: 'DELETE' });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed to revoke key');
        setOperationStatus('API key revoked.', 'ok');
        await refresh();
      } catch (error) {
        setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
      } finally {
        restoreButton();
      }
    };

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
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data),
        });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed to create user');
        setOperationStatus(`User ${data.email} created.`, 'ok');
        await refresh();
      } catch (error) {
        setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
      } finally {
        restoreButton();
      }
    }

    window.editUser = function(userId, email, displayName, role, isActive) {
      const newDisplayName = prompt('Display name:', displayName);
      if (newDisplayName === null) return;
      const newRole = prompt('Role (admin or user):', role);
      if (newRole === null) return;
      const newPassword = prompt('New password (leave blank to keep current):');
      const body = { display_name: newDisplayName, role: newRole };
      if (newPassword) body.password = newPassword;

      const restoreButton = setButtonBusy(null, 'Saving...');
      setOperationStatus(`Updating user ${email}...`, 'warn');
      (async () => {
        try {
          const resp = await fetch(`/api/users/${userId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
          });
          if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed to update user');
          setOperationStatus(`User ${email} updated.`, 'ok');
          await refresh();
        } catch (error) {
          setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
        } finally {
          restoreButton();
        }
      })();
    };

    window.toggleUserStatus = async function(userId, email, currentActive) {
      const action = currentActive ? 'disable' : 'enable';
      if (!confirm(`${action.charAt(0).toUpperCase() + action.slice(1)} user ${email}?`)) return;
      const restoreButton = setButtonBusy(null, 'Updating...');
      setOperationStatus(`${action.charAt(0).toUpperCase() + action.slice(1)}ing user...`, 'warn');
      try {
        const resp = await fetch(`/api/users/${userId}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ is_active: !currentActive }),
        });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed to update user');
        setOperationStatus(`User ${email} ${action}d.`, 'ok');
        await refresh();
      } catch (error) {
        setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
      } finally {
        restoreButton();
      }
    };

    window.deleteUser = async function(userId, email) {
      if (!confirm(`Delete user ${email}? This cannot be undone.`)) return;
      const restoreButton = setButtonBusy(null, 'Deleting...');
      setOperationStatus(`Deleting user ${email}...`, 'warn');
      try {
        const resp = await fetch(`/api/users/${userId}`, { method: 'DELETE' });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed to delete user');
        setOperationStatus(`User ${email} deleted.`, 'ok');
        await refresh();
      } catch (error) {
        setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
      } finally {
        restoreButton();
      }
    };

    // ── Self-service ─────────────────────────────────────────────

    window.showChangePasswordModal = function() {
      const oldPwd = prompt('Current password:');
      if (!oldPwd) return;
      const newPwd = prompt('New password (min 8 characters):');
      if (!newPwd || newPwd.length < 8) { alert('Password must be at least 8 characters.'); return; }
      const confirmPwd = prompt('Confirm new password:');
      if (newPwd !== confirmPwd) { alert('Passwords do not match.'); return; }
      const restoreButton = setButtonBusy(null, 'Changing...');
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
    };

    window.showMyApiKeys = async function() {
      const panel = document.getElementById('my-api-keys-panel');
      if (!panel) return;
      const isHidden = !panel.style.display || panel.style.display === 'none';
      if (isHidden) {
        panel.style.display = 'block';
      } else {
        panel.style.display = 'none';
        return;
      }
      try {
        const resp = await fetch('/api/auth/me/api-keys');
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
    };

    window.createMyApiKey = async function() {
      const name = prompt('Label (optional):');
      if (name === null) return;
      const restoreButton = setButtonBusy(null, 'Generating...');
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
        window.showMyApiKeys();
      } catch (error) {
        setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
      } finally { restoreButton(); }
    };

    window.revokeMyApiKey = async function(keyId) {
      if (!confirm('Revoke this key?')) return;
      const restoreButton = setButtonBusy(null, 'Revoking...');
      setOperationStatus('Revoking...', 'warn');
      try {
        const resp = await fetch(`/api/auth/me/api-keys/${keyId}`, { method: 'DELETE' });
        if (!resp.ok) throw new Error((await resp.json()).detail || 'Failed');
        setOperationStatus('Revoked.', 'ok');
        window.showMyApiKeys();
      } catch (error) {
        setOperationStatus(`Failed: ${summarizeError(error)}`, 'bad');
      } finally { restoreButton(); }
    };

    setChartGroup(activeChartGroup);
    refresh().catch(renderDashboardError);

    // Render Profile once (not inside renderOverview to avoid SSE refresh destroying panels)
    (async () => {
      const panel = document.getElementById('profile-panel');
      if (!panel) return;
      try {
        const resp = await fetch('/api/auth/me');
        if (!resp.ok) { panel.innerHTML = ''; return; }
        const me = await resp.json();
        panel.innerHTML = `
          <div class="stat-row"><span class="label">Email</span><span class="value">${escapeHtml(me.email)}</span></div>
          <div class="stat-row"><span class="label">Name</span><span class="value">${escapeHtml(me.display_name)}</span></div>
          <div class="stat-row"><span class="label">Role</span><span class="value"><span class="pill ${me.role === 'admin' ? 'warn' : 'ok'}">${escapeHtml(me.role)}</span></span></div>
          <hr style="border-color:var(--line);margin:12px 0">
          <div style="display:flex;gap:8px;flex-wrap:wrap">
            <button class="btn btn-sm" onclick="showChangePasswordModal()">Change Password</button>
            <button class="btn btn-sm" onclick="showMyApiKeys()">Manage My API Keys</button>
          </div>
          <div id="my-api-keys-panel" style="margin-top:12px;display:none"></div>`;
      } catch(e) {
        panel.innerHTML = '';
      }
    })();

    // SSE with exponential backoff
    let sseRetryDelay = 1000;
    const SSE_MAX_DELAY = 30000;
    let sseTimer = null;

    function connectSSE() {
      const es = new EventSource('/api/events');
      es.onmessage = (event) => {
        sseRetryDelay = 1000; // reset on successful message
        try {
          renderOverview(JSON.parse(event.data));
        } catch(e) {
          renderDashboardError(e);
        }
      };
      es.onerror = () => {
        es.close();
        if (sseTimer) clearTimeout(sseTimer);
        sseTimer = setTimeout(connectSSE, sseRetryDelay);
        sseRetryDelay = Math.min(sseRetryDelay * 2, SSE_MAX_DELAY);
      };
    }
    connectSSE();
