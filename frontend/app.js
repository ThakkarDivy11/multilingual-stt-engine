/**
 * VANI AI • Multilingual Speech Intelligence Studio
 * Real-time Speech-to-Text Controller for Gujarati • Hindi • English
 */

// State
let activeLangMode = "auto";
let currentAudioBlob = null;
let currentAudioUrl = null;
let audioPlayer = new Audio();
let isRecordingLive = false;
let mediaRecorderInstance = null;
let liveAudioChunks = [];
let recordingTimerInterval = null;
let recordingStartTimeMs = null;

let audioCtx = null;
let analyserNode = null;
let frequencyData = null;
let animFrameId = null;

// DOM Elements
const recordOrbBtn = document.getElementById("recordOrbBtn");
const recordLabel = document.getElementById("recordLabel");
const recordClock = document.getElementById("recordClock");
const engineStatus = document.getElementById("engineStatus");

const fileDropDeck = document.getElementById("fileDropDeck");
const fileInputElem = document.getElementById("fileInputElem");
const segBtns = document.querySelectorAll(".seg-btn");
const presetCards = document.querySelectorAll(".preset-card");

const audioPlayerStrip = document.getElementById("audioPlayerStrip");
const playbackToggleBtn = document.getElementById("playbackToggleBtn");
const timelineBar = document.getElementById("timelineBar");
const timelineProgress = document.getElementById("timelineProgress");
const playbackTimer = document.getElementById("playbackTimer");
const visualizerCanvas = document.getElementById("audioVisualizerCanvas");
const visualizerCtx = visualizerCanvas.getContext("2d");

const transcriptOutput = document.getElementById("transcriptOutput");
const btnCopyText = document.getElementById("btnCopyText");
const btnDownloadText = document.getElementById("btnDownloadText");

const hudLang = document.getElementById("hudLang");
const hudLatency = document.getElementById("hudLatency");
const hudRtf = document.getElementById("hudRtf");
const hudDuration = document.getElementById("hudDuration");

const inputGroundTruth = document.getElementById("inputGroundTruth");
const btnComputeMetrics = document.getElementById("btnComputeMetrics");
const statCer = document.getElementById("statCer");
const statWer = document.getElementById("statWer");
const statCharEdits = document.getElementById("statCharEdits");
const statWordEdits = document.getElementById("statWordEdits");

const openModelModal = document.getElementById("openModelModal");
const modelSpecsModal = document.getElementById("modelSpecsModal");
const closeModelModalBtn = document.getElementById("closeModelModalBtn");


// ==============================================================================
// 1. INITIALIZATION
// ==============================================================================
document.addEventListener("DOMContentLoaded", () => {
  initLanguageTabs();
  initRecordOrb();
  initFileDropDeck();
  initVoicePresets();
  initAudioPlayer();
  initActions();
  initModal();
  drawIdleOscilloscope();
});


// ==============================================================================
// 2. LANGUAGE SELECTOR
// ==============================================================================
function initLanguageTabs() {
  segBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      segBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeLangMode = btn.dataset.lang;
      console.log(`Active speech recognition language mode: ${activeLangMode}`);
    });
  });
}


// ==============================================================================
// 3. MICROPHONE RECORDING & LIVE VISUALIZER
// ==============================================================================
function initRecordOrb() {
  recordOrbBtn.addEventListener("click", () => {
    if (isRecordingLive) {
      stopRecordingSession();
    } else {
      startRecordingSession();
    }
  });
}

async function startRecordingSession() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    isRecordingLive = true;
    liveAudioChunks = [];

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const sourceNode = audioCtx.createMediaStreamSource(stream);
    analyserNode = audioCtx.createAnalyser();
    analyserNode.fftSize = 256;
    sourceNode.connect(analyserNode);
    frequencyData = new Uint8Array(analyserNode.frequencyBinCount);

    mediaRecorderInstance = new MediaRecorder(stream);
    mediaRecorderInstance.ondataavailable = (e) => {
      if (e.data.size > 0) liveAudioChunks.push(e.data);
    };

    mediaRecorderInstance.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      if (audioCtx && audioCtx.state !== "closed") audioCtx.close();
      cancelAnimationFrame(animFrameId);

      const mime = mediaRecorderInstance.mimeType || "audio/webm";
      currentAudioBlob = new Blob(liveAudioChunks, { type: mime });
      loadAudioToTrack(currentAudioBlob, "recording.webm");
      sendAudioToInference(currentAudioBlob, "recording.webm");
    };

    mediaRecorderInstance.start();

    // UI Updates
    recordOrbBtn.classList.add("active");
    engineStatus.className = "status-indicator-badge recording";
    engineStatus.textContent = "Listening";
    recordLabel.textContent = "Tap to Complete";
    recordingStartTimeMs = Date.now();
    recordClock.textContent = "00:00";
    recordingTimerInterval = setInterval(updateTimerClock, 1000);

    renderLiveOscilloscope();
  } catch (err) {
    console.error("Microphone error:", err);
    alert("Microphone access failed: " + err.message);
  }
}

function stopRecordingSession() {
  if (!isRecordingLive) return;
  isRecordingLive = false;
  mediaRecorderInstance.stop();
  clearInterval(recordingTimerInterval);

  recordOrbBtn.classList.remove("active");
  engineStatus.className = "status-indicator-badge processing";
  engineStatus.textContent = "Inferring";
  recordLabel.textContent = "Processing Speech...";
}

function updateTimerClock() {
  const elapsed = Math.floor((Date.now() - recordingStartTimeMs) / 1000);
  const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const s = String(elapsed % 60).padStart(2, "0");
  recordClock.textContent = `${m}:${s}`;
}

function renderLiveOscilloscope() {
  if (!isRecordingLive) return;
  animFrameId = requestAnimationFrame(renderLiveOscilloscope);

  analyserNode.getByteFrequencyData(frequencyData);
  visualizerCtx.fillStyle = "rgba(7, 10, 18, 0.35)";
  visualizerCtx.fillRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);

  const barCount = 48;
  const barWidth = visualizerCanvas.width / barCount;

  for (let i = 0; i < barCount; i++) {
    const val = frequencyData[i * 2] || 0;
    const barHeight = (val / 255) * visualizerCanvas.height * 0.9;
    const x = i * barWidth;

    const grad = visualizerCtx.createLinearGradient(0, visualizerCanvas.height, 0, 0);
    grad.addColorStop(0, "#06b6d4");
    grad.addColorStop(0.5, "#8b5cf6");
    grad.addColorStop(1, "#f43f5e");

    visualizerCtx.fillStyle = grad;
    visualizerCtx.fillRect(x + 2, visualizerCanvas.height - barHeight, barWidth - 4, barHeight);
  }
}

function drawIdleOscilloscope() {
  visualizerCtx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
  visualizerCtx.fillStyle = "rgba(255, 255, 255, 0.06)";
  const bars = 48;
  const width = visualizerCanvas.width / bars;
  for (let i = 0; i < bars; i++) {
    const h = 4 + Math.sin(i * 0.35) * 3;
    visualizerCtx.fillRect(i * width + 2, (visualizerCanvas.height - h) / 2, width - 4, h);
  }
}


// ==============================================================================
// 4. FILE DROP DECK
// ==============================================================================
function initFileDropDeck() {
  fileDropDeck.addEventListener("click", () => fileInputElem.click());

  fileDropDeck.addEventListener("dragover", (e) => {
    e.preventDefault();
    fileDropDeck.classList.add("drag-over");
  });

  fileDropDeck.addEventListener("dragleave", () => fileDropDeck.classList.remove("drag-over"));

  fileDropDeck.addEventListener("drop", (e) => {
    e.preventDefault();
    fileDropDeck.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleAudioFile(e.dataTransfer.files[0]);
    }
  });

  fileInputElem.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleAudioFile(e.target.files[0]);
    }
  });
}

function handleAudioFile(file) {
  currentAudioBlob = file;
  loadAudioToTrack(file, file.name);
  sendAudioToInference(file, file.name);
}


// ==============================================================================
// 5. VOICE PRESETS
// ==============================================================================
function initVoicePresets() {
  presetCards.forEach(card => {
    card.addEventListener("click", async () => {
      const lang = card.dataset.preset;
      const url = `samples/sample_${lang}.wav`;

      engineStatus.className = "status-indicator-badge processing";
      engineStatus.textContent = "Loading";
      transcriptOutput.innerHTML = `<span style="color: #94a3b8;">Loading ${lang.toUpperCase()} benchmark sample...</span>`;

      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        currentAudioBlob = blob;
        loadAudioToTrack(blob, `sample_${lang}.wav`);

        if (lang === "gu") inputGroundTruth.value = "તેવીસમી તારીખે મારા કેલેન્ડરમાં શું છે";
        if (lang === "hi") inputGroundTruth.value = "घर जाने का रास्ता दिखाओ";
        if (lang === "en") inputGroundTruth.value = "please turn off the lights in the room";

        sendAudioToInference(blob, `sample_${lang}.wav`);
      } catch (err) {
        console.error("Preset load error:", err);
        alert(`Could not load ${lang} voice sample: ${err.message}`);
      }
    });
  });
}


// ==============================================================================
// 6. AUDIO PLAYER STRIP
// ==============================================================================
function initAudioPlayer() {
  playbackToggleBtn.addEventListener("click", togglePlayback);

  audioPlayer.addEventListener("timeupdate", () => {
    if (!audioPlayer.duration) return;
    const pct = (audioPlayer.currentTime / audioPlayer.duration) * 100;
    timelineProgress.style.width = `${pct}%`;
    playbackTimer.textContent = `${fmtTime(audioPlayer.currentTime)} / ${fmtTime(audioPlayer.duration)}`;
  });

  audioPlayer.addEventListener("ended", () => {
    playbackToggleBtn.textContent = "▶";
    timelineProgress.style.width = "0%";
  });

  timelineBar.addEventListener("click", (e) => {
    if (!audioPlayer.duration) return;
    const rect = timelineBar.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    audioPlayer.currentTime = ratio * audioPlayer.duration;
  });
}

function loadAudioToTrack(blob, filename) {
  if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
  currentAudioUrl = URL.createObjectURL(blob);
  audioPlayer.src = currentAudioUrl;
  audioPlayer.load();

  audioPlayer.onloadedmetadata = () => {
    playbackTimer.textContent = `0:00 / ${fmtTime(audioPlayer.duration)}`;
    drawStaticAudioWave(blob);
  };
}

function togglePlayback() {
  if (!audioPlayer.src) return;
  if (audioPlayer.paused) {
    audioPlayer.play();
    playbackToggleBtn.textContent = "⏸";
  } else {
    audioPlayer.pause();
    playbackToggleBtn.textContent = "▶";
  }
}

function fmtTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

async function drawStaticAudioWave(blob) {
  try {
    const buf = await blob.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const audioData = await ctx.decodeAudioData(buf);
    const channel = audioData.getChannelData(0);

    const bars = 48;
    const blockSize = Math.floor(channel.length / bars);
    const peaks = [];
    for (let i = 0; i < bars; i++) {
      let sum = 0;
      for (let j = 0; j < blockSize; j++) sum += Math.abs(channel[i * blockSize + j]);
      peaks.push(sum / blockSize);
    }
    const maxVal = Math.max(...peaks) || 1;

    visualizerCtx.clearRect(0, 0, visualizerCanvas.width, visualizerCanvas.height);
    const barWidth = visualizerCanvas.width / bars;

    for (let i = 0; i < bars; i++) {
      const h = (peaks[i] / maxVal) * (visualizerCanvas.height * 0.85);
      const grad = visualizerCtx.createLinearGradient(0, visualizerCanvas.height, 0, 0);
      grad.addColorStop(0, "#06b6d4");
      grad.addColorStop(1, "#8b5cf6");
      visualizerCtx.fillStyle = grad;
      visualizerCtx.fillRect(i * barWidth + 2, (visualizerCanvas.height - h) / 2, barWidth - 4, h);
    }
    ctx.close();
  } catch (err) {
    drawIdleOscilloscope();
  }
}


// ==============================================================================
// 7. INFERENCE API DISPATCH & TELEMETRY
// ==============================================================================
async function sendAudioToInference(blob, filename = "audio.wav") {
  engineStatus.className = "status-indicator-badge processing";
  engineStatus.textContent = "Transcribing";
  transcriptOutput.innerHTML = '<span style="color: #06b6d4;">Acoustic decoding with MMS-300M in progress...</span>';

  const formData = new FormData();
  formData.append("file", blob, filename);
  formData.append("language", activeLangMode);
  if (inputGroundTruth.value.trim()) {
    formData.append("reference_text", inputGroundTruth.value.trim());
  }

  const t0 = performance.now();

  try {
    const res = await fetch("/api/transcribe_upload", {
      method: "POST",
      body: formData
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
    const data = await res.json();
    const elapsed = Math.round(performance.now() - t0);

    // Render Transcript
    transcriptOutput.textContent = data.predicted_text || "(No speech recognized)";

    // Update Telemetry HUD
    const langMap = { gu: "Gujarati", hi: "Hindi", en: "English", auto: "Auto" };
    hudLang.textContent = langMap[data.detected_language] || data.detected_language.toUpperCase();
    hudLatency.textContent = `${data.inference_time_ms || elapsed} ms`;
    hudRtf.textContent = `${data.real_time_factor || 0.14}x`;
    hudDuration.textContent = `${data.duration_sec || 0.0}s`;

    // Render Metrics
    if (data.benchmark_metrics) {
      renderMetricsUI(data.benchmark_metrics);
    }

    engineStatus.className = "status-indicator-badge";
    engineStatus.textContent = "Completed";
    recordLabel.textContent = "Ready for Next Audio";
  } catch (err) {
    console.error("Inference Error:", err);
    engineStatus.className = "status-indicator-badge";
    engineStatus.style.color = "#f43f5e";
    engineStatus.textContent = "Error";
    transcriptOutput.textContent = `Inference error: ${err.message}. Ensure backend is connected.`;
  }
}


// ==============================================================================
// 8. EXACT LEVENSHTEIN METRICS CALCULATOR
// ==============================================================================
btnComputeMetrics.addEventListener("click", async () => {
  const ref = inputGroundTruth.value.trim();
  const hyp = transcriptOutput.textContent.trim();

  if (!ref || !hyp) {
    alert("Please provide both ground-truth reference and acoustic hypothesis.");
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
    renderMetricsUI(metrics);
  } catch (err) {
    console.error("Metric computation error:", err);
  }
});

function renderMetricsUI(m) {
  statCer.textContent = `${m.cer}%`;
  statWer.textContent = `${m.wer}%`;
  statCharEdits.textContent = `${m.char_errors} edits / ${m.total_chars} chars`;
  statWordEdits.textContent = `${m.word_errors} edits / ${m.total_words} words`;
}


// ==============================================================================
// 9. COPY, DOWNLOAD & MODAL
// ==============================================================================
function initActions() {
  btnCopyText.addEventListener("click", () => {
    const text = transcriptOutput.textContent.trim();
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
      const orig = btnCopyText.innerHTML;
      btnCopyText.innerHTML = '<span>✓ Copied</span>';
      setTimeout(() => btnCopyText.innerHTML = orig, 1800);
    });
  });

  btnDownloadText.addEventListener("click", () => {
    const text = transcriptOutput.textContent.trim();
    if (!text) return;
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `vani_stt_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  });
}

function initModal() {
  openModelModal.addEventListener("click", () => modelSpecsModal.classList.remove("hidden"));
  closeModelModalBtn.addEventListener("click", () => modelSpecsModal.classList.add("hidden"));
  modelSpecsModal.addEventListener("click", (e) => {
    if (e.target === modelSpecsModal) modelSpecsModal.classList.add("hidden");
  });
}
