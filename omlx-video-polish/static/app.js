(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const state = {
    clips: [],
    selectedId: null,
    suggestions: {},
    analysisComplete: false,
    busy: false,
    result: null,
  };
  const STORAGE_KEY = 'omlx-video-polish-project-v1';
  const settingIds = [
    'autoBalance', 'light', 'exposure', 'contrast', 'highlights', 'shadows', 'saturation', 'warmth',
    'voiceIsolation', 'noiseRemoval', 'dialogueEnhance', 'compression', 'normalizeAudio', 'targetLufs',
    'removeFillers', 'removeFalseStarts', 'useLlm', 'cutPadding', 'transitionStyle', 'transitionDuration',
    'outputName', 'resolution', 'codec', 'quality',
  ];

  const formatTime = (seconds) => {
    const value = Math.max(0, Number(seconds) || 0);
    const mins = Math.floor(value / 60);
    const secs = value - mins * 60;
    return `${mins}:${secs.toFixed(1).padStart(4, '0')}`;
  };

  const escapeHtml = (value) => String(value ?? '')
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;').replaceAll("'", '&#039;');

  function currentSettings() {
    return {
      video: {
        auto_balance: $('autoBalance').checked,
        light: Number($('light').value),
        exposure: Number($('exposure').value),
        contrast: Number($('contrast').value),
        highlights: Number($('highlights').value),
        shadows: Number($('shadows').value),
        saturation: Number($('saturation').value),
        warmth: Number($('warmth').value),
      },
      audio: {
        voice_isolation: Number($('voiceIsolation').value),
        noise_removal: Number($('noiseRemoval').value),
        dialogue_enhance: Number($('dialogueEnhance').value),
        compression: Number($('compression').value),
        normalize: $('normalizeAudio').checked,
        target_lufs: Number($('targetLufs').value),
        true_peak: -1.5,
      },
      edit: {
        remove_fillers: $('removeFillers').checked,
        remove_false_starts: $('removeFalseStarts').checked,
        use_llm: $('useLlm').checked,
        cut_padding_ms: Number($('cutPadding').value),
      },
      transition: {
        style: $('transitionStyle').value,
        duration: Number($('transitionDuration').value),
      },
      export: {
        filename: $('outputName').value.trim(),
        resolution: $('resolution').value,
        codec: $('codec').value,
        quality: $('quality').value,
      },
    };
  }

  function saveProject() {
    const settings = {};
    settingIds.forEach((id) => {
      const control = $(id);
      settings[id] = control.type === 'checkbox' ? control.checked : control.value;
    });
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      clips: state.clips,
      selectedId: state.selectedId,
      suggestions: state.suggestions,
      analysisComplete: state.analysisComplete,
      settings,
    }));
  }

  async function restoreProject() {
    let saved;
    try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null'); } catch (_) { return; }
    if (!saved) return;
    Object.entries(saved.settings || {}).forEach(([id, value]) => {
      const control = $(id);
      if (!control) return;
      if (control.type === 'checkbox') control.checked = Boolean(value);
      else control.value = value;
    });
    updateOutputs();
    const candidates = Array.isArray(saved.clips) ? saved.clips : [];
    const verified = await Promise.all(candidates.map(async (clip) => {
      try {
        const response = await fetch(`/api/uploads/${clip.id}`, { cache: 'no-store' });
        if (!response.ok) return null;
        const record = await response.json();
        return { ...clip, url: record.url, meta: record.media, name: record.filename };
      } catch (_) { return null; }
    }));
    state.clips = verified.filter(Boolean);
    state.selectedId = state.clips.some((clip) => clip.id === saved.selectedId)
      ? saved.selectedId : state.clips[0]?.id || null;
    state.suggestions = saved.suggestions || {};
    state.analysisComplete = Boolean(saved.analysisComplete) && state.clips.length > 0;
    renderAll();
  }

  async function checkHealth() {
    try {
      const response = await fetch('/api/health', { cache: 'no-store' });
      if (!response.ok) throw new Error('Service unavailable');
      const data = await response.json();
      $('health').className = 'health online';
      $('health').lastChild.textContent = 'Ready · local';
      $('outputDirectory').textContent = data.output_dir;
      const missing = Object.entries(data.capabilities || {}).filter(([, available]) => !available).map(([name]) => name);
      if (missing.length) {
        setStatus('Setup needs attention', `Missing: ${missing.join(', ')}`, 0);
      }
    } catch (_) {
      $('health').className = 'health error';
      $('health').lastChild.textContent = 'Service offline';
    }
  }

  function renderAll() {
    renderClipList();
    renderSelectedClip();
    renderSuggestions();
    $('clipCount').textContent = state.clips.length;
    $('analyseBtn').disabled = state.busy || state.clips.length === 0;
    $('renderBtn').disabled = state.busy || state.clips.length === 0;
    if (!state.busy && state.clips.length) {
      const seconds = state.clips.reduce((sum, clip) => sum + Math.max(0, clip.trimEnd - clip.trimStart), 0);
      setStatus('Ready to polish', `${state.clips.length} clip${state.clips.length === 1 ? '' : 's'} · about ${formatTime(seconds)}`, 0);
    }
    saveProject();
  }

  function renderClipList() {
    $('clipList').innerHTML = state.clips.map((clip, index) => `
      <div class="clip-card ${clip.id === state.selectedId ? 'selected' : ''}" draggable="true" data-id="${clip.id}">
        <span class="clip-order" title="Drag to reorder">${index + 1}</span>
        <div class="clip-copy" data-select="${clip.id}">
          <strong>${escapeHtml(clip.name)}</strong>
          <span>${formatTime(clip.trimEnd - clip.trimStart)} · ${clip.meta.width}×${clip.meta.height}${clip.meta.has_audio ? ' · audio' : ' · silent'}</span>
        </div>
        <button class="icon-button" type="button" data-remove="${clip.id}" title="Remove clip">×</button>
      </div>`).join('');

    $('clipList').querySelectorAll('[data-select]').forEach((element) => {
      element.addEventListener('click', () => selectClip(element.dataset.select));
    });
    $('clipList').querySelectorAll('[data-remove]').forEach((element) => {
      element.addEventListener('click', (event) => { event.stopPropagation(); removeClip(element.dataset.remove); });
    });
    let draggedId = null;
    $('clipList').querySelectorAll('.clip-card').forEach((card) => {
      card.addEventListener('dragstart', () => { draggedId = card.dataset.id; card.classList.add('dragging'); });
      card.addEventListener('dragend', () => card.classList.remove('dragging'));
      card.addEventListener('dragover', (event) => event.preventDefault());
      card.addEventListener('drop', (event) => {
        event.preventDefault();
        if (!draggedId || draggedId === card.dataset.id) return;
        const from = state.clips.findIndex((clip) => clip.id === draggedId);
        const to = state.clips.findIndex((clip) => clip.id === card.dataset.id);
        const [moved] = state.clips.splice(from, 1);
        state.clips.splice(to, 0, moved);
        renderAll();
      });
    });
  }

  function renderSelectedClip() {
    const clip = state.clips.find((item) => item.id === state.selectedId);
    if (!clip) {
      $('viewerPlaceholder').classList.remove('hidden');
      $('preview').classList.add('hidden');
      $('clipTools').classList.add('hidden');
      $('preview').removeAttribute('src');
      return;
    }
    $('viewerPlaceholder').classList.add('hidden');
    $('preview').classList.remove('hidden');
    $('clipTools').classList.remove('hidden');
    if ($('preview').dataset.clipId !== clip.id) {
      $('preview').src = clip.url;
      $('preview').dataset.clipId = clip.id;
      $('preview').load();
    }
    $('selectedName').textContent = clip.name;
    $('selectedMeta').textContent = `${clip.meta.width}×${clip.meta.height} · ${clip.meta.fps.toFixed(2)} fps · ${formatTime(clip.meta.duration)}`;
    $('trimStart').value = Number(clip.trimStart).toFixed(2);
    $('trimStart').max = Math.max(0, clip.trimEnd - 0.05).toFixed(2);
    $('trimEnd').value = Number(clip.trimEnd).toFixed(2);
    $('trimEnd').max = Number(clip.meta.duration).toFixed(2);
  }

  function renderSuggestions() {
    const groups = state.clips.map((clip) => ({ clip, suggestions: state.suggestions[clip.id] || [] }))
      .filter((group) => group.suggestions.length);
    if (!groups.length) {
      $('emptyReview').classList.remove('hidden');
      $('suggestionList').classList.add('hidden');
      $('suggestionList').innerHTML = '';
      return;
    }
    $('emptyReview').classList.add('hidden');
    $('suggestionList').classList.remove('hidden');
    $('suggestionList').innerHTML = groups.map(({ clip, suggestions }) => `
      <div class="suggestion-group">
        <strong>${escapeHtml(clip.name)} · ${suggestions.filter((item) => item.enabled).length} selected</strong>
        ${suggestions.map((item) => `
          <label class="suggestion" title="${escapeHtml(item.source || '')} · confidence ${Math.round((item.confidence || 0) * 100)}%">
            <input type="checkbox" data-clip="${clip.id}" data-suggestion="${item.id}" ${item.enabled ? 'checked' : ''}>
            <time>${formatTime(item.start)}</time>
            <span class="suggestion-text">${escapeHtml(item.text || '(pause)')}</span>
            <span class="suggestion-tag">${escapeHtml(item.reason)}</span>
          </label>`).join('')}
      </div>`).join('');
    $('suggestionList').querySelectorAll('input[type="checkbox"]').forEach((input) => {
      input.addEventListener('change', () => {
        const item = (state.suggestions[input.dataset.clip] || []).find((candidate) => candidate.id === input.dataset.suggestion);
        if (item) item.enabled = input.checked;
        renderSuggestions();
        saveProject();
      });
    });
  }

  function selectClip(id) {
    state.selectedId = id;
    renderClipList();
    renderSelectedClip();
    saveProject();
  }

  async function removeClip(id) {
    try { await fetch(`/api/uploads/${id}`, { method: 'DELETE' }); } catch (_) {}
    state.clips = state.clips.filter((clip) => clip.id !== id);
    delete state.suggestions[id];
    state.analysisComplete = false;
    if (state.selectedId === id) state.selectedId = state.clips[0]?.id || null;
    renderAll();
  }

  function updateTrim(which) {
    const clip = state.clips.find((item) => item.id === state.selectedId);
    if (!clip) return;
    if (which === 'start') {
      clip.trimStart = Math.max(0, Math.min(Number($('trimStart').value) || 0, clip.trimEnd - 0.05));
      $('preview').currentTime = clip.trimStart;
    } else {
      clip.trimEnd = Math.min(clip.meta.duration, Math.max(Number($('trimEnd').value) || clip.meta.duration, clip.trimStart + 0.05));
      $('preview').currentTime = Math.max(clip.trimStart, clip.trimEnd - 0.3);
    }
    renderAll();
  }

  async function uploadFiles(fileList) {
    const files = [...fileList].filter((file) => file.type.startsWith('video/') || /\.(mkv|mts|m2ts)$/i.test(file.name));
    if (!files.length) return;
    for (const file of files) {
      const id = crypto.randomUUID();
      $('uploadProgress').classList.remove('hidden');
      $('uploadName').textContent = file.name;
      try {
        const record = await uploadFile(id, file);
        state.clips.push({
          id: record.id,
          name: record.filename,
          url: record.url,
          meta: record.media,
          trimStart: 0,
          trimEnd: record.media.duration,
        });
        state.selectedId = record.id;
        state.analysisComplete = false;
        renderAll();
      } catch (error) {
        setStatus('Upload failed', error.message, 0);
      }
    }
    $('uploadProgress').classList.add('hidden');
    $('fileInput').value = '';
  }

  function uploadFile(id, file) {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      request.open('PUT', `/api/uploads/${id}`);
      request.setRequestHeader('X-Filename', encodeURIComponent(file.name));
      request.setRequestHeader('Content-Type', 'application/octet-stream');
      request.upload.onprogress = (event) => {
        if (!event.lengthComputable) return;
        const percent = Math.round(event.loaded / event.total * 100);
        $('uploadPercent').textContent = `${percent}%`;
        $('uploadBar').value = percent;
      };
      request.onerror = () => reject(new Error('The local service interrupted the upload'));
      request.onload = () => {
        let payload = {};
        try { payload = JSON.parse(request.responseText); } catch (_) {}
        if (request.status >= 200 && request.status < 300) resolve(payload);
        else reject(new Error(payload.error || `Upload failed (${request.status})`));
      };
      request.send(file);
    });
  }

  function setStatus(stage, message, progress) {
    $('jobStage').textContent = stage;
    $('jobMessage').textContent = message || '';
    $('jobProgress').value = Number(progress) || 0;
  }

  async function postJson(path, payload) {
    const response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
    return data;
  }

  async function pollJob(jobId) {
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const response = await fetch(`/api/jobs/${jobId}`, { cache: 'no-store' });
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || 'Job disappeared');
      const lastMessage = (job.messages || []).at(-1) || '';
      setStatus(job.stage || 'Working', lastMessage, job.progress || 0);
      if (job.status === 'complete') return job.result;
      if (job.status === 'error') throw new Error(job.error || 'Processing failed');
    }
  }

  async function analyseSpeech() {
    if (!state.clips.length || state.busy) return false;
    state.busy = true;
    renderAll();
    setStatus('Preparing speech analysis', 'Starting MLX Whisper…', 2);
    try {
      const settings = currentSettings();
      const created = await postJson('/api/analyze', {
        clips: state.clips.map((clip) => ({ upload_id: clip.id })),
        edit: settings.edit,
      });
      const result = await pollJob(created.job_id);
      state.suggestions = {};
      (result.clips || []).forEach((clipResult) => {
        state.suggestions[clipResult.upload_id] = clipResult.suggestions || [];
      });
      state.analysisComplete = true;
      const selected = Object.values(state.suggestions).flat().filter((item) => item.enabled);
      renderSuggestions();
      setStatus('Speech review ready', `${selected.length} suggested cut${selected.length === 1 ? '' : 's'} selected · review them below`, 100);
      saveProject();
      return true;
    } catch (error) {
      setStatus('Speech analysis failed', error.message, 0);
      return false;
    } finally {
      state.busy = false;
      $('analyseBtn').disabled = state.clips.length === 0;
      $('renderBtn').disabled = state.clips.length === 0;
    }
  }

  function renderPayload() {
    const settings = currentSettings();
    return {
      ...settings,
      clips: state.clips.map((clip) => ({
        upload_id: clip.id,
        trim_start: clip.trimStart,
        trim_end: clip.trimEnd,
        cuts: (state.suggestions[clip.id] || []).map((item) => ({
          start: item.start,
          end: item.end,
          enabled: item.enabled,
          reason: item.reason,
          text: item.text,
        })),
      })),
    };
  }

  async function renderProject() {
    if (!state.clips.length || state.busy) return;
    const edit = currentSettings().edit;
    if ((edit.remove_fillers || edit.remove_false_starts) && !state.analysisComplete) {
      const okay = await analyseSpeech();
      if (okay) setStatus('Review speech cuts', 'Check the suggestions, then press Polish & export again.', 100);
      return;
    }
    state.busy = true;
    renderAll();
    setStatus('Preparing export', 'Building the local processing plan…', 1);
    try {
      const created = await postJson('/api/render', renderPayload());
      const result = await pollJob(created.job_id);
      state.result = result;
      showResult(result);
      setStatus('Export complete', result.output, 100);
    } catch (error) {
      setStatus('Export failed', error.message, 0);
    } finally {
      state.busy = false;
      $('analyseBtn').disabled = state.clips.length === 0;
      $('renderBtn').disabled = state.clips.length === 0;
    }
  }

  function showResult(result) {
    $('resultSummary').textContent = `${formatTime(result.duration)} · ${result.width}×${result.height} · ${result.fps.toFixed(2)} fps`;
    $('resultVideo').src = result.url;
    $('downloadResult').href = result.url;
    $('downloadResult').download = result.filename;
    $('resultPath').textContent = result.output;
    $('resultModal').classList.remove('hidden');
  }

  function updateOutputs() {
    const signed = new Set(['light', 'contrast', 'highlights', 'shadows', 'saturation', 'warmth']);
    document.querySelectorAll('input[type="range"]').forEach((input) => {
      const output = document.querySelector(`output[for="${input.id}"]`);
      if (!output) return;
      const value = Number(input.value);
      if (input.id === 'transitionDuration') output.textContent = `${value.toFixed(2)} s`;
      else if (input.id === 'exposure') output.textContent = value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1);
      else output.textContent = signed.has(input.id) && value > 0 ? `+${value}` : String(value);
    });
  }

  function resetVideo() {
    const defaults = { autoBalance: true, light: 15, exposure: 0, contrast: 8, highlights: 0, shadows: 8, saturation: 8, warmth: 0 };
    Object.entries(defaults).forEach(([id, value]) => {
      if ($(id).type === 'checkbox') $(id).checked = value;
      else $(id).value = value;
    });
    updateOutputs(); saveProject();
  }

  function resetAudio() {
    const defaults = { voiceIsolation: 70, noiseRemoval: 45, dialogueEnhance: 40, compression: 35, normalizeAudio: true, targetLufs: -16 };
    Object.entries(defaults).forEach(([id, value]) => {
      if ($(id).type === 'checkbox') $(id).checked = value;
      else $(id).value = value;
    });
    updateOutputs(); saveProject();
  }

  async function clearProject() {
    if (state.clips.length && !confirm('Start a new project and remove the current imported copies?')) return;
    await Promise.all(state.clips.map((clip) => fetch(`/api/uploads/${clip.id}`, { method: 'DELETE' }).catch(() => {})));
    state.clips = [];
    state.selectedId = null;
    state.suggestions = {};
    state.analysisComplete = false;
    state.result = null;
    localStorage.removeItem(STORAGE_KEY);
    $('resultVideo').removeAttribute('src');
    renderAll();
  }

  function bindEvents() {
    $('heroImport').addEventListener('click', () => $('fileInput').click());
    $('fileInput').addEventListener('change', () => uploadFiles($('fileInput').files));
    const dropZone = $('dropZone');
    ['dragenter', 'dragover'].forEach((type) => dropZone.addEventListener(type, (event) => {
      event.preventDefault(); dropZone.classList.add('dragover');
    }));
    ['dragleave', 'drop'].forEach((type) => dropZone.addEventListener(type, (event) => {
      event.preventDefault(); dropZone.classList.remove('dragover');
    }));
    dropZone.addEventListener('drop', (event) => uploadFiles(event.dataTransfer.files));
    $('trimStart').addEventListener('change', () => updateTrim('start'));
    $('trimEnd').addEventListener('change', () => updateTrim('end'));
    $('analyseBtn').addEventListener('click', analyseSpeech);
    $('renderBtn').addEventListener('click', renderProject);
    $('resetVideo').addEventListener('click', resetVideo);
    $('resetAudio').addEventListener('click', resetAudio);
    $('clearProject').addEventListener('click', clearProject);
    $('closeResult').addEventListener('click', () => $('resultModal').classList.add('hidden'));
    $('resultModal').addEventListener('click', (event) => {
      if (event.target === $('resultModal')) $('resultModal').classList.add('hidden');
    });
    $('revealResult').addEventListener('click', async () => {
      if (!state.result) return;
      try { await postJson('/api/reveal', { path: state.result.output }); } catch (_) {}
    });
    document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((item) => item.classList.toggle('active', item === tab));
      document.querySelectorAll('.tab-panel').forEach((panel) => panel.classList.toggle('active', panel.dataset.panel === tab.dataset.tab));
    }));
    settingIds.forEach((id) => {
      const control = $(id);
      control.addEventListener('input', () => { updateOutputs(); saveProject(); });
      control.addEventListener('change', () => {
        if (['removeFillers', 'removeFalseStarts', 'useLlm', 'cutPadding'].includes(id)) state.analysisComplete = false;
        saveProject();
      });
    });
  }

  async function init() {
    bindEvents();
    updateOutputs();
    await checkHealth();
    await restoreProject();
    renderAll();
  }

  init();
})();
