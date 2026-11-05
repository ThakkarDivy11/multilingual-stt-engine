/**
 * Trilingual STT Studio — Frontend Application Engine
 * Handles microphone recording, waveform rendering, audio playback,
 * REST API communication with FastAPI backend, and Levenshtein accuracy telemetry.
 */

// State
let currentLanguage = "auto";
let activeAudioBlob = null;
let activeAudioUrl = null;
let audioElement = new Audio();
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let recordingInterval = null;
let recordingStartTime = null;

let audioContext = null;
let analyser = null;
let dataArray = null;
let animFrameId = null;

// DOM Elements
const micBtn = document.getElementById("micButton");
const micIcon = document.getElementById("micIcon");
const micStatusText = document.getElementById("micStatusText");
const recordingTimer = document.getElementById("recordingTimer");
const processBadge = document.getElementById("processBadge");

const dropZone = document.getElementById("dropZone");
const audioFileInput = document.getElementById("audioFileInput");
const langTabs = document.querySelectorAll(".lang-tab");
const samplePills = document.querySelectorAll(".sample-pill");

const playerContainer = document.getElementById("playerContainer");
const playPauseBtn = document.getElementById("playPauseBtn");
const playIcon = document.getElementById("playIcon");
const scrubBar = document.getElementById("scrubBar");
const scrubFill = document.getElementById("scrubFill");
const audioTime = document.getElementById("audioTime");
const canvas = document.getElementById("waveformCanvas");
const canvasCtx = canvas.getContext("2d");

const transcriptionText = document.getElementById("transcriptionText");
const copyBtn = document.getElementById("copyBtn");
const downloadBtn = document.getElementById("downloadBtn");

const telemetryLang = document.getElementById("telemetryLang");
const telemetryLatency = document.getElementById("telemetryLatency");
const telemetryRtf = document.getElementById("telemetryRtf");
const telemetryDuration = document.getElementById("telemetryDuration");

const refTextInput = document.getElementById("refTextInput");
const calcMetricBtn = document.getElementById("calcMetricBtn");
const metricCer = document.getElementById("metricCer");
const metricWer = document.getElementById("metricWer");
const metricCharDist = document.getElementById("metricCharDist");
const metricWordDist = document.getElementById("metricWordDist");

const modelPillBtn = document.getElementById("modelPillBtn");
const modelModal = document.getElementById("modelModal");
const closeModalBtn = document.getElementById("closeModalBtn");


// ==============================================================================
// 1. INITIALIZATION & LISTENERS
// ==============================================================================
document.addEventListener("DOMContentLoaded", () => {
  setupLanguageTabs();
  setupMicrophone();
  setupFileDrop();
  setupPresets();
  setupAudioPlayer();
  setupTools();
  setupModal();
  drawIdleWaveform();
});


// ==============================================================================
// 2. LANGUAGE SELECTOR TABS
// ==============================================================================
function setupLanguageTabs() {
  langTabs.forEach(tab => {
    tab.addEventListener("click", () => {
      langTabs.forEach(t => t.classList.remove("active"));
      tab.classList.add("active");
      currentLanguage = tab.dataset.lang;
      console.log(`Switched target language mode to: ${currentLanguage}`);
    });
  });
}


// ==============================================================================
// 3. MICROPHONE RECORDING & LIVE AUDIO CONTEXT
// ==============================================================================
function setupMicrophone() {
  micBtn.addEventListener("click", () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  });
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    isRecording = true;
    audioChunks = [];

    // Audio Context & Analyser
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    dataArray = new Uint8Array(analyser.frequencyBinCount);

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      if (audioContext && audioContext.state !== "closed") audioContext.close();
      cancelAnimationFrame(animFrameId);

      const mimeType = mediaRecorder.mimeType || "audio/webm";
      activeAudioBlob = new Blob(audioChunks, { type: mimeType });
      loadAudioIntoPlayer(activeAudioBlob, "microphone_recording.webm");
      submitAudioForTranscription(activeAudioBlob, "microphone_recording.webm");
    };

    mediaRecorder.start();

    // UI Updates
    micBtn.classList.add("active");
    processBadge.className = "pill-status recording";
    processBadge.textContent = "Recording";
    micStatusText.textContent = "Listening... Click to finish speaking";
    recordingStartTime = Date.now();
    recordingTimer.textContent = "00:00";
    recordingInterval = setInterval(updateRecordingTimer, 1000);

    visualizeLiveWaveform();
  } catch (err) {
    console.error("Microphone access error:", err);
    alert("Microphone permission denied or device not found: " + err.message);
  }
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;
  mediaRecorder.stop();
  clearInterval(recordingInterval);

  micBtn.classList.remove("active");
  processBadge.className = "pill-status processing";
  processBadge.textContent = "Processing";
  micStatusText.textContent = "Audio captured. Transcribing...";
}

function updateRecordingTimer() {
  const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
  const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const secs = String(elapsed % 60).padStart(2, "0");
  recordingTimer.textContent = `${mins}:${secs}`;
}

function visualizeLiveWaveform() {
  if (!isRecording) return;
  animFrameId = requestAnimationFrame(visualizeLiveWaveform);

  analyser.getByteFrequencyData(dataArray);
  canvasCtx.fillStyle = "rgba(9, 12, 21, 0.4)";
  canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

  const barWidth = (canvas.width / dataArray.length) * 2.2;
  let x = 0;

  for (let i = 0; i < dataArray.length; i++) {
    const barHeight = (dataArray[i] / 255) * canvas.height * 0.9;
    const gradient = canvasCtx.createLinearGradient(0, canvas.height, 0, 0);
    gradient.addColorStop(0, "#6366f1");
    gradient.addColorStop(0.5, "#8b5cf6");
    gradient.addColorStop(1, "#ec4899");

    canvasCtx.fillStyle = gradient;
    canvasCtx.fillRect(x, canvas.height - barHeight, barWidth - 1, barHeight);
    x += barWidth;
  }
}


// ==============================================================================
// 4. FILE UPLOAD & DRAG/DROP
// ==============================================================================
function setupFileDrop() {
  dropZone.addEventListener("click", () => audioFileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  audioFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });
}

function handleSelectedFile(file) {
  activeAudioBlob = file;
  loadAudioIntoPlayer(file, file.name);
  submitAudioForTranscription(file, file.name);
}


// ==============================================================================
// 5. SAMPLE PRESETS
// ==============================================================================
function setupPresets() {
  samplePills.forEach(pill => {
    pill.addEventListener("click", async () => {
      const lang = pill.dataset.sample;
      const sampleUrl = `samples/sample_${lang}.wav`;
      processBadge.className = "pill-status processing";
      processBadge.textContent = "Loading Sample";
      transcriptionText.textContent = `Loading ${lang.toUpperCase()} audio sample...`;

      try {
        const res = await fetch(sampleUrl);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        activeAudioBlob = blob;
        loadAudioIntoPlayer(blob, `sample_${lang}.wav`);

        // Automatically set reference text for verification if matching sample
        if (lang === "en") refTextInput.value = "please turn off the lights in the room";
        if (lang === "hi") refTextInput.value = "घर जाने का रास्ता दिखाओ";
        if (lang === "gu") refTextInput.value = "તેવીસમી તારીખે મારા કેલેન્ડરમાં શું છે";

        submitAudioForTranscription(blob, `sample_${lang}.wav`);
      } catch (err) {
        console.error("Error loading sample:", err);
        alert(`Could not load ${lang} voice sample: ${err.message}`);
      }
    });
  });
}


// ==============================================================================
// 6. AUDIO PLAYER & STATIC WAVEFORM
// ==============================================================================
function setupAudioPlayer() {
  playPauseBtn.addEventListener("click", toggleAudioPlayback);

  audioElement.addEventListener("timeupdate", () => {
    if (!audioElement.duration) return;
    const pct = (audioElement.currentTime / audioElement.duration) * 100;
    scrubFill.style.width = `${pct}%`;
    audioTime.textContent = `${formatTime(audioElement.currentTime)} / ${formatTime(audioElement.duration)}`;
  });

  audioElement.addEventListener("ended", () => {
    playIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
    scrubFill.style.width = "0%";
  });

  scrubBar.addEventListener("click", (e) => {
    if (!audioElement.duration) return;
    const rect = scrubBar.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const ratio = clickX / rect.width;
    audioElement.currentTime = ratio * audioElement.duration;
  });
}

function loadAudioIntoPlayer(blob, filename) {
  if (activeAudioUrl) URL.revokeObjectURL(activeAudioUrl);
  activeAudioUrl = URL.createObjectURL(blob);
  audioElement.src = activeAudioUrl;
  audioElement.load();

  audioElement.onloadedmetadata = () => {
    audioTime.textContent = `0:00 / ${formatTime(audioElement.duration)}`;
    drawStaticWaveformFromBlob(blob);
  };
}

function toggleAudioPlayback() {
  if (!audioElement.src) return;
  if (audioElement.paused) {
    audioElement.play();
    playIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
  } else {
    audioElement.pause();
    playIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
  }
}

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

async function drawStaticWaveformFromBlob(blob) {
  try {
    const arrayBuffer = await blob.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuf = await ctx.decodeAudioData(arrayBuffer);
    const rawData = audioBuf.getChannelData(0);

    const samples = 100;
    const blockSize = Math.floor(rawData.length / samples);
    const filteredData = [];
    for (let i = 0; i < samples; i++) {
      let blockStart = blockSize * i;
      let sum = 0;
      for (let j = 0; j < blockSize; j++) {
        sum += Math.abs(rawData[blockStart + j]);
      }
      filteredData.push(sum / blockSize);
    }

    const maxVal = Math.max(...filteredData) || 1;
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    const barWidth = canvas.width / samples;
    for (let i = 0; i < samples; i++) {
      const h = (filteredData[i] / maxVal) * (canvas.height * 0.85);
      const gradient = canvasCtx.createLinearGradient(0, canvas.height, 0, 0);
      gradient.addColorStop(0, "#6366f1");
      gradient.addColorStop(1, "#a855f7");
      canvasCtx.fillStyle = gradient;
      canvasCtx.fillRect(i * barWidth, (canvas.height - h) / 2, barWidth - 1, h);
    }
    ctx.close();
  } catch (err) {
    drawIdleWaveform();
  }
}

function drawIdleWaveform() {
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
  canvasCtx.fillStyle = "rgba(255, 255, 255, 0.08)";
  const bars = 80;
  const width = canvas.width / bars;
  for (let i = 0; i < bars; i++) {
    const h = 4 + Math.sin(i * 0.3) * 3;
    canvasCtx.fillRect(i * width, (canvas.height - h) / 2, width - 2, h);
  }
}


// ==============================================================================
// 7. TRANSCRIPTION INFERENCE BACKEND CALL
// ==============================================================================
async function submitAudioForTranscription(audioBlob, filename = "audio.wav") {
  processBadge.className = "pill-status processing";
  processBadge.textContent = "Transcribing";
  transcriptionText.innerHTML = '<span style="color: #94a3b8;">Processing acoustic audio with fine-tuned MMS-300M...</span>';

  const formData = new FormData();
  formData.append("file", audioBlob, filename);
  formData.append("language", currentLanguage);
  if (refTextInput.value.trim()) {
    formData.append("reference_text", refTextInput.value.trim());
  }

  const t0 = performance.now();

  try {
    const res = await fetch("/api/transcribe_upload", {
      method: "POST",
      body: formData
    });

    if (!res.ok) {
      throw new Error(`HTTP error ${res.status}: ${res.statusText}`);
    }

    const data = await res.json();
    const elapsed = Math.round(performance.now() - t0);

    // Render Transcription Output
    transcriptionText.textContent = data.predicted_text || "(No speech detected)";

    // Update Telemetry Badges
    const langNames = { gu: "Gujarati (ગુજરાતી)", hi: "Hindi (हिन्दी)", en: "English (Latin)", auto: "Auto" };
    telemetryLang.textContent = langNames[data.detected_language] || data.detected_language.toUpperCase();
    telemetryLatency.textContent = `${data.inference_time_ms || elapsed} ms`;
    telemetryRtf.textContent = `${data.real_time_factor || 0.15}x`;
    telemetryDuration.textContent = `${data.duration_sec || 0.0}s`;

    // Accuracy Telemetry
    if (data.benchmark_metrics) {
      renderAccuracyMetrics(data.benchmark_metrics);
    }

    processBadge.className = "pill-status";
    processBadge.textContent = "Done";
    micStatusText.textContent = "Transcription complete. Ready for next audio.";
  } catch (err) {
    console.error("Transcription API error:", err);
    processBadge.className = "pill-status";
    processBadge.style.color = "#ef4444";
    processBadge.textContent = "Error";
    transcriptionText.textContent = `Inference failed: ${err.message}. Ensure backend is running.`;
  }
}


// ==============================================================================
// 8. EXACT LEVENSHTEIN METRIC COMPUTATION
// ==============================================================================
calcMetricBtn.addEventListener("click", async () => {
  const ref = refTextInput.value.trim();
  const hyp = transcriptionText.textContent.trim();

  if (!ref || !hyp) {
    alert("Please provide both a ground-truth reference and transcription text to measure accuracy.");
    return;
  }

  try {
    const res = await fetch("/api/metrics/calculate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reference: ref, hypothesis: hyp })
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const metrics = await res.json();
    renderAccuracyMetrics(metrics);
  } catch (err) {
    console.error("Metric calculation error:", err);
  }
});

function renderAccuracyMetrics(m) {
  metricCer.textContent = `${m.cer}%`;
  metricWer.textContent = `${m.wer}%`;
  metricCharDist.textContent = `${m.char_errors} edits / ${m.total_chars} chars`;
  metricWordDist.textContent = `${m.word_errors} edits / ${m.total_words} words`;
}


// ==============================================================================
// 9. TOOLS: COPY & DOWNLOAD
// ==============================================================================
function setupTools() {
  copyBtn.addEventListener("click", () => {
    const text = transcriptionText.textContent.trim();
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      const orig = copyBtn.innerHTML;
      copyBtn.innerHTML = '<span>✓ Copied!</span>';
      setTimeout(() => copyBtn.innerHTML = orig, 1800);
    });
  });

  downloadBtn.addEventListener("click", () => {
    const text = transcriptionText.textContent.trim();
    if (!text) return;
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `stt_transcript_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  });
}


// ==============================================================================
// 10. MODEL INFORMATION MODAL
// ==============================================================================
function setupModal() {
  modelPillBtn.addEventListener("click", () => modelModal.classList.remove("hidden"));
  closeModalBtn.addEventListener("click", () => modelModal.classList.add("hidden"));
  modelModal.addEventListener("click", (e) => {
    if (e.target === modelModal) modelModal.classList.add("hidden");
  });
}
