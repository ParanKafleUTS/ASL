/**
 * ASL Hand Sign Detection - Web Application
 *
 * Real-time ASL sign recognition using TensorFlow.js.
 * Captures webcam frames, preprocesses images, and runs
 * inference using a converted TensorFlow.js model.
 */

'use strict';

// ============================================================
// Configuration
// ============================================================
const CONFIG = {
    imageSize: 28,
    numClasses: 24,
    inferenceInterval: 200, // ms between predictions
    confidenceThreshold: 0.2,
    modelPaths: {
        custom_cnn:    'models/custom_cnn/model.json',
        mobilenet:     'models/mobilenet/model.json',
        efficientnet:  'models/efficientnet/model.json',
        attention_cnn: 'models/attention_cnn/model.json',
        ensemble:      'models/ensemble/model.json',
    },
    // ASL labels (no J=9, Z=25)
    labels: (() => {
        const labels = [];
        for (let i = 0; i < 26; i++) {
            if (i !== 9 && i !== 25) {
                labels.push(String.fromCharCode(65 + i));
            }
        }
        return labels;
    })()
};

// ============================================================
// State
// ============================================================
const state = {
    model: null,
    modelName: 'custom_cnn',
    stream: null,
    inferenceTimer: null,
    isRunning: false,
    stats: {
        totalPredictions: 0,
        confidenceSum: 0,
        letterCounts: {},
        fps: 0,
        lastFrameTime: Date.now(),
        frameCount: 0,
    }
};

// ============================================================
// DOM References
// ============================================================
const video       = document.getElementById('webcam');
const canvas      = document.getElementById('overlay-canvas');
const ctx         = canvas ? canvas.getContext('2d') : null;
const captureCanvas = document.getElementById('captured-canvas');
const captureCtx  = captureCanvas ? captureCanvas.getContext('2d') : null;

// ============================================================
// Camera Management
// ============================================================

/**
 * Start the webcam stream.
 */
async function startCamera() {
    try {
        state.stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 640, height: 480 },
            audio: false
        });
        video.srcObject = state.stream;
        await video.play();

        // Sync canvas size
        video.addEventListener('loadedmetadata', () => {
            if (canvas) {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
            }
        });

        setButtonState(true);
        updateCameraStatus(true);
        showNotification('Camera started', 'success');

        // Begin inference loop if model is loaded
        if (state.model) {
            startInferenceLoop();
        }
    } catch (err) {
        console.error('Camera error:', err);
        showNotification(`Camera error: ${err.message}`, 'error');
    }
}

/**
 * Stop the webcam stream and inference loop.
 */
function stopCamera() {
    stopInferenceLoop();
    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        state.stream = null;
    }
    video.srcObject = null;
    setButtonState(false);
    updateCameraStatus(false);
    clearPredictions();
}

// ============================================================
// Model Management
// ============================================================

/**
 * Load the selected TensorFlow.js model.
 */
async function loadSelectedModel() {
    const select = document.getElementById('model-select');
    state.modelName = select ? select.value : 'custom_cnn';
    const modelPath = CONFIG.modelPaths[state.modelName];

    setModelStatus('Loading...');

    try {
        if (state.model) {
            state.model.dispose();
            state.model = null;
        }
        state.model = await tf.loadLayersModel(modelPath);
        setModelStatus('Loaded ✓');
        showNotification(`${state.modelName} loaded successfully`, 'success');

        // Warm up model
        const dummyInput = tf.zeros([1, CONFIG.imageSize, CONFIG.imageSize, 1]);
        const warmup = state.model.predict(dummyInput);
        warmup.dispose();
        dummyInput.dispose();

        if (state.stream) {
            startInferenceLoop();
        }
    } catch (err) {
        console.error('Model load error:', err);
        setModelStatus('Failed to load ✗');
        showNotification(
            `Model not found. Run model_converter.py first. (${err.message})`,
            'error'
        );
    }
}

// ============================================================
// Inference Loop
// ============================================================

/**
 * Start continuous inference on webcam frames.
 */
function startInferenceLoop() {
    if (state.inferenceTimer) return;
    state.isRunning = true;
    state.inferenceTimer = setInterval(runInference, CONFIG.inferenceInterval);
}

/**
 * Stop the inference loop.
 */
function stopInferenceLoop() {
    if (state.inferenceTimer) {
        clearInterval(state.inferenceTimer);
        state.inferenceTimer = null;
    }
    state.isRunning = false;
}

/**
 * Run a single inference pass on the current video frame.
 */
async function runInference() {
    if (!state.model || !video.videoWidth) return;

    const startTime = performance.now();

    const predictions = tf.tidy(() => {
        // Capture and preprocess frame
        const frameTensor = preprocessFrame(video);
        if (!frameTensor) return null;

        // Run inference
        return state.model.predict(frameTensor);
    });

    if (!predictions) return;

    const probs = await predictions.data();
    predictions.dispose();

    const inferenceMs = performance.now() - startTime;

    updatePredictions(probs, inferenceMs);
    updateFPS();
}

/**
 * Preprocess a video frame for model inference.
 * @param {HTMLVideoElement} videoEl - The video element.
 * @returns {tf.Tensor|null} Preprocessed tensor or null.
 */
function preprocessFrame(videoEl) {
    if (!videoEl.videoWidth) return null;

    return tf.tidy(() => {
        const frameTensor = tf.browser.fromPixels(videoEl, 1); // Grayscale
        const resized = tf.image.resizeBilinear(
            frameTensor, [CONFIG.imageSize, CONFIG.imageSize]
        );
        const normalized = resized.div(255.0);
        return normalized.expandDims(0); // Add batch dim: [1, 28, 28, 1]
    });
}

// ============================================================
// UI Updates
// ============================================================

/**
 * Update prediction display with new inference results.
 * @param {Float32Array} probs - Probability array.
 * @param {number} inferenceMs - Inference time in milliseconds.
 */
function updatePredictions(probs, inferenceMs) {
    // Get top-5 predictions
    const indexed = Array.from(probs).map((p, i) => ({
        label: CONFIG.labels[i] || `C${i}`,
        prob: p
    }));
    indexed.sort((a, b) => b.prob - a.prob);
    const top5 = indexed.slice(0, 5);

    // Update main prediction
    const top1 = top5[0];
    const predEl = document.getElementById('main-prediction');
    const confEl = document.getElementById('main-confidence');

    if (predEl) predEl.textContent = top1.label;
    if (confEl) {
        confEl.textContent = `${(top1.prob * 100).toFixed(1)}% confidence`;
        confEl.style.color = top1.prob > CONFIG.confidenceThreshold ? '#10b981' : '#ef4444';
    }

    // Highlight active letter in reference chart
    highlightLetter(top1.label);

    // Update top-5 list
    renderTop5(top5);

    // Update inference time
    const timeEl = document.getElementById('inference-time');
    if (timeEl) timeEl.textContent = `${inferenceMs.toFixed(0)}ms`;

    // Update statistics
    updateStats(top1.label, top1.prob);
}

/**
 * Render top-5 predictions as progress bars.
 * @param {Array} top5 - Array of {label, prob} objects.
 */
function renderTop5(top5) {
    const container = document.getElementById('top5-predictions');
    if (!container) return;

    container.innerHTML = top5.map((item, i) => `
        <div class="top5-item">
            <span class="top5-rank">${i + 1}</span>
            <span class="top5-letter">${item.label}</span>
            <div class="top5-bar-container">
                <div class="top5-bar" style="width: ${(item.prob * 100).toFixed(1)}%"></div>
            </div>
            <span class="top5-pct">${(item.prob * 100).toFixed(1)}%</span>
        </div>
    `).join('');
}

/**
 * Capture the current video frame for display.
 */
function captureFrame() {
    if (!video.videoWidth) return;

    const preview = document.getElementById('capture-preview');
    if (captureCanvas && captureCtx) {
        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
        captureCtx.save();
        captureCtx.scale(-1, 1);
        captureCtx.drawImage(video, -captureCanvas.width, 0,
                              captureCanvas.width, captureCanvas.height);
        captureCtx.restore();
    }
    if (preview) preview.style.display = 'block';
    showNotification('Frame captured!', 'info');
}

/**
 * Highlight the predicted letter in the reference chart.
 * @param {string} letter - The predicted letter.
 */
function highlightLetter(letter) {
    document.querySelectorAll('.letter-card').forEach(card => {
        card.classList.toggle('active', card.dataset.letter === letter);
    });
}

/**
 * Build the ASL alphabet reference chart.
 */
function buildAlphabetGrid() {
    const grid = document.getElementById('alphabet-grid');
    if (!grid) return;

    grid.innerHTML = CONFIG.labels.map(letter => `
        <div class="letter-card" data-letter="${letter}"
             title="ASL sign for ${letter}">
            ${letter}
        </div>
    `).join('');
}

// ============================================================
// Statistics
// ============================================================

/**
 * Update session statistics.
 * @param {string} letter - Predicted letter.
 * @param {number} confidence - Prediction confidence.
 */
function updateStats(letter, confidence) {
    const s = state.stats;
    s.totalPredictions++;
    s.confidenceSum += confidence;
    s.letterCounts[letter] = (s.letterCounts[letter] || 0) + 1;

    const avgConf = s.confidenceSum / s.totalPredictions;
    const mostDetected = Object.entries(s.letterCounts)
        .sort((a, b) => b[1] - a[1])[0]?.[0] || '—';

    const totalEl = document.getElementById('total-predictions');
    const confEl  = document.getElementById('avg-confidence');
    const mostEl  = document.getElementById('most-detected');

    if (totalEl) totalEl.textContent = s.totalPredictions;
    if (confEl)  confEl.textContent  = `${(avgConf * 100).toFixed(1)}%`;
    if (mostEl)  mostEl.textContent  = mostDetected;
}

/**
 * Update FPS counter.
 */
function updateFPS() {
    const s = state.stats;
    s.frameCount++;
    const now = Date.now();
    const elapsed = (now - s.lastFrameTime) / 1000;

    if (elapsed >= 1.0) {
        s.fps = Math.round(s.frameCount / elapsed);
        s.frameCount = 0;
        s.lastFrameTime = now;
        const fpsEl = document.getElementById('fps-counter');
        if (fpsEl) fpsEl.textContent = s.fps;
    }
}

/**
 * Reset session statistics.
 */
function resetStats() {
    state.stats = {
        totalPredictions: 0,
        confidenceSum: 0,
        letterCounts: {},
        fps: 0,
        lastFrameTime: Date.now(),
        frameCount: 0,
    };
    ['total-predictions', 'avg-confidence', 'most-detected', 'fps-counter']
        .forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = id === 'fps-counter' ? '0' : '—';
        });
    showNotification('Statistics reset', 'info');
}

// ============================================================
// UI Helpers
// ============================================================

function setButtonState(cameraOn) {
    const startBtn   = document.getElementById('start-btn');
    const stopBtn    = document.getElementById('stop-btn');
    const captureBtn = document.getElementById('capture-btn');
    if (startBtn)   startBtn.disabled   = cameraOn;
    if (stopBtn)    stopBtn.disabled    = !cameraOn;
    if (captureBtn) captureBtn.disabled = !cameraOn;
}

function updateCameraStatus(active) {
    const dot  = document.querySelector('.status-dot');
    const text = document.getElementById('status-text');
    if (dot)  dot.classList.toggle('active', active);
    if (text) text.textContent = active ? 'Camera live' : 'Camera off';
}

function setModelStatus(status) {
    const el = document.getElementById('model-status');
    if (el) el.textContent = status;
}

function clearPredictions() {
    const predEl = document.getElementById('main-prediction');
    const confEl = document.getElementById('main-confidence');
    const top5   = document.getElementById('top5-predictions');
    if (predEl) predEl.textContent = '?';
    if (confEl) confEl.textContent = 'Waiting...';
    if (top5)   top5.innerHTML = '';
}

/**
 * Show a temporary notification banner.
 * @param {string} message - Notification text.
 * @param {string} type - 'success' | 'error' | 'info'.
 */
function showNotification(message, type = 'info') {
    const existing = document.querySelector('.notification');
    if (existing) existing.remove();

    const notif = document.createElement('div');
    notif.className = `notification ${type}`;
    notif.textContent = message;
    document.body.appendChild(notif);

    setTimeout(() => notif.remove(), 3000);
}

// ============================================================
// Initialisation
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    buildAlphabetGrid();
    setModelStatus('Not loaded');
    console.log('ASL Detection App initialized');
    console.log('TF.js version:', tf.version.tfjs);
});
