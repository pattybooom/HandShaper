// =============================================================================
// HandShaperr - app.js
// Real-time hand shape classification frontend
// =============================================================================

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

const CONFIG = {
    wsUrl: 'ws://localhost:8000',
    targetFps: 30,                    // Higher FPS for more responsive updates
    reconnectDelay: 2000,             // ms before reconnect attempt
    maxReconnectAttempts: 10,
    mediaPipe: {
        maxNumHands: 2,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
    },
    lowConfidenceThreshold: 0.3,      // Lower threshold for easier detection
};

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------

const state = {
    ws: null,
    wsConnected: false,
    reconnectAttempts: 0,
    hands: null,                      // MediaPipe Hands instance
    camera: null,
    videoStream: null,
    isRunning: false,
    lastSendTime: 0,
    fps: 0,
    fpsFrames: 0,
    fpsLastTime: performance.now(),

    // Smoothing & hysteresis
    smoothingWindow: 5,               // Number of frames to average
    predictionHistory: [],
    displayedShape: null,             // Currently displayed shape (sticky)
    shapeConfirmCount: 0,             // How many frames the new shape has been predicted
    requiredConfirmFrames: 3,         // Frames needed to confirm a shape change

    // Current prediction
    currentLabel: null,
    currentProb: 0,
    topk: [],

    // Three.js
    scene: null,
    camera3d: null,
    renderer: null,
    currentMesh: null,
    targetMesh: null,
    meshes: {},
    transitionProgress: 1,
    hidingCurrent: false,  // True when hiding without showing new shape
};

// -----------------------------------------------------------------------------
// DOM Elements
// -----------------------------------------------------------------------------

const dom = {
    startOverlay: document.getElementById('startOverlay'),
    startButton: document.getElementById('startButton'),
    connectionStatus: document.getElementById('connectionStatus'),
    fpsDisplay: document.getElementById('fpsDisplay'),
    predictionLabel: document.getElementById('predictionLabel'),
    predictionProb: document.getElementById('predictionProb'),
    topkDisplay: document.getElementById('topkDisplay'),
    cameraSelect: document.getElementById('cameraSelect'),
    showLandmarks: document.getElementById('showLandmarks'),
    smoothingSlider: document.getElementById('smoothingSlider'),
    smoothingValue: document.getElementById('smoothingValue'),
    videoElement: document.getElementById('videoElement'),
    overlayCanvas: document.getElementById('overlayCanvas'),
    noHandsOverlay: document.getElementById('noHandsOverlay'),
    threeContainer: document.getElementById('threeContainer'),
    errorToast: document.getElementById('errorToast'),
};

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------

async function init() {
    // Event listeners
    dom.startButton.addEventListener('click', start);
    dom.cameraSelect.addEventListener('change', onCameraChange);
    dom.smoothingSlider.addEventListener('input', onSmoothingChange);

    // Populate camera list
    await populateCameraList();

    // Initialize Three.js scene
    initThreeJS();

    // Start render loop
    animate();
}

async function start() {
    try {
        // Hide start overlay
        dom.startOverlay.classList.add('hidden');

        // Connect WebSocket
        connectWebSocket();

        // Initialize MediaPipe Hands
        await initMediaPipe();

        // Start camera
        await startCamera();

        state.isRunning = true;
    } catch (error) {
        showError(`Failed to start: ${error.message}`);
        dom.startOverlay.classList.remove('hidden');
    }
}

// -----------------------------------------------------------------------------
// Camera
// -----------------------------------------------------------------------------

async function populateCameraList() {
    try {
        // Request permission first
        await navigator.mediaDevices.getUserMedia({ video: true });

        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');

        dom.cameraSelect.innerHTML = '';
        videoDevices.forEach((device, idx) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Camera ${idx + 1}`;
            dom.cameraSelect.appendChild(option);
        });
    } catch (error) {
        console.warn('Could not enumerate cameras:', error);
    }
}

async function startCamera() {
    const deviceId = dom.cameraSelect.value;

    const constraints = {
        video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user',
        }
    };

    if (deviceId) {
        constraints.video.deviceId = { exact: deviceId };
    }

    state.videoStream = await navigator.mediaDevices.getUserMedia(constraints);
    dom.videoElement.srcObject = state.videoStream;

    await new Promise(resolve => {
        dom.videoElement.onloadedmetadata = () => {
            dom.videoElement.play();
            resolve();
        };
    });

    // Size overlay canvas to match video
    resizeOverlayCanvas();
    window.addEventListener('resize', resizeOverlayCanvas);

    // Start MediaPipe camera
    state.camera = new Camera(dom.videoElement, {
        onFrame: async () => {
            if (state.hands) {
                await state.hands.send({ image: dom.videoElement });
            }
        },
        width: 1280,
        height: 720,
    });
    state.camera.start();
}

async function onCameraChange() {
    if (state.isRunning) {
        // Stop current stream
        if (state.videoStream) {
            state.videoStream.getTracks().forEach(t => t.stop());
        }
        if (state.camera) {
            state.camera.stop();
        }
        await startCamera();
    }
}

function resizeOverlayCanvas() {
    const video = dom.videoElement;
    const canvas = dom.overlayCanvas;

    // Match the actual displayed size of the video element
    const rect = video.getBoundingClientRect();
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;

    // Set display size to match video
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
}

// -----------------------------------------------------------------------------
// MediaPipe Hands
// -----------------------------------------------------------------------------

async function initMediaPipe() {
    state.hands = new Hands({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
    });

    state.hands.setOptions({
        maxNumHands: CONFIG.mediaPipe.maxNumHands,
        modelComplexity: 1,
        minDetectionConfidence: CONFIG.mediaPipe.minDetectionConfidence,
        minTrackingConfidence: CONFIG.mediaPipe.minTrackingConfidence,
    });

    state.hands.onResults(onHandsResults);
}

function onHandsResults(results) {
    // Update FPS
    updateFps();

    // Draw landmarks
    const canvas = dom.overlayCanvas;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const handsData = [];

    if (results.multiHandLandmarks && results.multiHandedness) {
        // Show/hide "no hands" overlay
        dom.noHandsOverlay.classList.add('hidden');

        for (let i = 0; i < results.multiHandLandmarks.length; i++) {
            const landmarks = results.multiHandLandmarks[i];
            const handedness = results.multiHandedness[i];

            // Draw landmarks if enabled
            if (dom.showLandmarks.checked) {
                drawConnectors(ctx, landmarks, HAND_CONNECTIONS, {
                    color: '#00FF00',
                    lineWidth: 2,
                });
                drawLandmarks(ctx, landmarks, {
                    color: '#FF0000',
                    lineWidth: 1,
                    radius: 3,
                });
            }

            // Package for WebSocket
            handsData.push({
                handedness: handedness.label,
                landmarks: landmarks.map(lm => ({
                    x: lm.x,
                    y: lm.y,
                    z: lm.z,
                })),
            });
        }
    } else {
        dom.noHandsOverlay.classList.remove('hidden');
    }

    // Send to backend (throttled)
    sendToBackend(handsData);
}

function updateFps() {
    state.fpsFrames++;
    const now = performance.now();
    const elapsed = now - state.fpsLastTime;

    if (elapsed >= 1000) {
        state.fps = Math.round(state.fpsFrames * 1000 / elapsed);
        dom.fpsDisplay.textContent = state.fps;
        state.fpsFrames = 0;
        state.fpsLastTime = now;
    }
}

// -----------------------------------------------------------------------------
// WebSocket
// -----------------------------------------------------------------------------

function connectWebSocket() {
    updateConnectionStatus('connecting');

    state.ws = new WebSocket(CONFIG.wsUrl);

    state.ws.onopen = () => {
        state.wsConnected = true;
        state.reconnectAttempts = 0;
        updateConnectionStatus('connected');
        console.log('[WS] Connected');
    };

    state.ws.onclose = () => {
        state.wsConnected = false;
        updateConnectionStatus('disconnected');
        console.log('[WS] Disconnected');

        // Attempt reconnect
        if (state.isRunning && state.reconnectAttempts < CONFIG.maxReconnectAttempts) {
            state.reconnectAttempts++;
            setTimeout(connectWebSocket, CONFIG.reconnectDelay);
        }
    };

    state.ws.onerror = (error) => {
        console.error('[WS] Error:', error);
    };

    state.ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handlePrediction(data);
        } catch (error) {
            console.error('[WS] Parse error:', error);
        }
    };
}

function updateConnectionStatus(status) {
    dom.connectionStatus.className = 'status-value ' + status;
    dom.connectionStatus.textContent = status.charAt(0).toUpperCase() + status.slice(1);
}

function sendToBackend(handsData) {
    const now = performance.now();
    const minInterval = 1000 / CONFIG.targetFps;

    if (now - state.lastSendTime < minInterval) {
        return; // Throttle
    }

    if (!state.wsConnected || !state.ws) {
        return;
    }

    const message = {
        t: Date.now(),
        hands: handsData,
    };

    try {
        state.ws.send(JSON.stringify(message));
        state.lastSendTime = now;
    } catch (error) {
        console.error('[WS] Send error:', error);
    }
}

// -----------------------------------------------------------------------------
// Prediction Handling & Smoothing
// -----------------------------------------------------------------------------

function handlePrediction(data) {
    // data: { t, label, prob, topk }

    if (!data.label || data.prob < CONFIG.lowConfidenceThreshold) {
        // Low confidence - count towards "no shape"
        state.predictionHistory.push({ label: null, prob: 0, topk: [] });
    } else {
        state.predictionHistory.push({
            label: data.label.toLowerCase(),
            prob: data.prob,
            topk: data.topk || [],
        });
    }

    // Keep only last N
    while (state.predictionHistory.length > state.smoothingWindow) {
        state.predictionHistory.shift();
    }

    // Get majority vote from recent predictions
    const majorityLabel = getMajorityLabel();
    const latestProb = state.predictionHistory[state.predictionHistory.length - 1].prob;
    const latestTopk = state.predictionHistory[state.predictionHistory.length - 1].topk;

    // Hysteresis: require multiple consistent frames to change shape
    if (majorityLabel !== state.displayedShape) {
        state.shapeConfirmCount++;

        // Only change if we've seen the new shape consistently
        if (state.shapeConfirmCount >= state.requiredConfirmFrames) {
            console.log('[Shape] Changing from', state.displayedShape, 'to', majorityLabel);
            state.displayedShape = majorityLabel;
            state.shapeConfirmCount = 0;
            updateShape(majorityLabel);
        }
    } else {
        // Same shape, reset confirm counter
        state.shapeConfirmCount = 0;
    }

    // Always update the UI text (shows what model predicts)
    updatePredictionUI(majorityLabel, latestProb, latestTopk);
}

function getMajorityLabel() {
    if (state.predictionHistory.length === 0) return null;

    const counts = {};
    let maxCount = 0;
    let maxLabel = null;

    for (const pred of state.predictionHistory) {
        const label = pred.label;
        counts[label] = (counts[label] || 0) + 1;
        if (counts[label] > maxCount) {
            maxCount = counts[label];
            maxLabel = label;
        }
    }

    return maxLabel;
}

function applySmoothing() {
    if (state.predictionHistory.length === 0) {
        return { label: null, prob: 0, topk: [] };
    }

    // Instant mode - just use the latest prediction directly
    if (state.smoothingWindow <= 1) {
        return state.predictionHistory[state.predictionHistory.length - 1];
    }

    // For small windows, use latest if high confidence
    if (state.smoothingWindow <= 3) {
        const latest = state.predictionHistory[state.predictionHistory.length - 1];
        if (latest.prob >= 0.5) {
            return latest;
        }
    }

    // Count label occurrences with recency weighting
    const counts = {};
    let maxCount = 0;
    let maxLabel = null;

    for (let i = 0; i < state.predictionHistory.length; i++) {
        const pred = state.predictionHistory[i];
        const weight = 1 + (i / state.predictionHistory.length);
        counts[pred.label] = (counts[pred.label] || 0) + weight;
        if (counts[pred.label] > maxCount) {
            maxCount = counts[pred.label];
            maxLabel = pred.label;
        }
    }

    const latestWithLabel = [...state.predictionHistory]
        .reverse()
        .find(p => p.label === maxLabel);

    return latestWithLabel || { label: maxLabel, prob: maxCount / state.predictionHistory.length, topk: [] };
}

function updatePredictionUI(label, prob, topk) {
    state.currentLabel = label;
    state.currentProb = prob;
    state.topk = topk;

    if (label) {
        dom.predictionLabel.textContent = label;
        dom.predictionProb.textContent = `(${(prob * 100).toFixed(0)}%)`;
    } else {
        dom.predictionLabel.textContent = 'No shape';
        dom.predictionProb.textContent = '';
    }

    // Update top-k display
    dom.topkDisplay.innerHTML = '';
    if (topk && topk.length > 0) {
        topk.slice(0, 3).forEach(item => {
            const span = document.createElement('span');
            span.className = 'topk-item';
            span.innerHTML = `<span class="topk-label">${item.label}</span>: <span class="topk-prob">${(item.prob * 100).toFixed(0)}%</span>`;
            dom.topkDisplay.appendChild(span);
        });
    }
}

function onSmoothingChange() {
    state.smoothingWindow = parseInt(dom.smoothingSlider.value);
    dom.smoothingValue.textContent = state.smoothingWindow;
}

// -----------------------------------------------------------------------------
// Three.js Scene
// -----------------------------------------------------------------------------

function initThreeJS() {
    const container = dom.threeContainer;
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    state.scene = new THREE.Scene();
    state.scene.background = new THREE.Color(0x0d0d14);

    // Camera
    state.camera3d = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    state.camera3d.position.set(0, 1, 5);
    state.camera3d.lookAt(0, 0, 0);

    // Renderer
    state.renderer = new THREE.WebGLRenderer({ antialias: true });
    state.renderer.setSize(width, height);
    state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(state.renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404060, 0.5);
    state.scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 10, 7);
    state.scene.add(directionalLight);

    const pointLight = new THREE.PointLight(0x6366f1, 0.5);
    pointLight.position.set(-5, 5, 5);
    state.scene.add(pointLight);

    // Floor grid
    const gridHelper = new THREE.GridHelper(10, 20, 0x303040, 0x202030);
    gridHelper.position.y = -1.5;
    state.scene.add(gridHelper);

    // Create shape meshes
    createShapeMeshes();

    // Start with no shape visible
    state.currentMesh = null;

    // Handle resize
    window.addEventListener('resize', onThreeResize);
}

function createShapeMeshes() {
    // Neon colors for each shape
    const neonColors = {
        square: 0x00ffff,   // Cyan
        triangle: 0x00ff88, // Green
        circle: 0xff00ff,   // Magenta
        heart: 0xff3366,    // Pink-red
    };

    // Helper to create neon wireframe with glow
    function createNeonShape(geometry, color) {
        const group = new THREE.Group();

        // Get edges from geometry
        const edges = new THREE.EdgesGeometry(geometry, 15);

        // Core bright line
        const coreMaterial = new THREE.LineBasicMaterial({
            color: 0xffffff,
            linewidth: 3,
        });
        const core = new THREE.LineSegments(edges, coreMaterial);
        group.add(core);

        // Main neon color line
        const neonMaterial = new THREE.LineBasicMaterial({
            color: color,
            linewidth: 2,
            transparent: true,
            opacity: 0.9,
        });
        const neon = new THREE.LineSegments(edges.clone(), neonMaterial);
        neon.scale.set(1.01, 1.01, 1.01);
        group.add(neon);

        // Outer glow (larger, more transparent)
        const glowMaterial = new THREE.LineBasicMaterial({
            color: color,
            linewidth: 1,
            transparent: true,
            opacity: 0.4,
        });
        const glow = new THREE.LineSegments(edges.clone(), glowMaterial);
        glow.scale.set(1.05, 1.05, 1.05);
        group.add(glow);

        // Inner fill with emissive glow
        const fillMaterial = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.15,
            side: THREE.DoubleSide,
        });
        const fill = new THREE.Mesh(geometry.clone(), fillMaterial);
        fill.scale.set(0.99, 0.99, 0.99);
        group.add(fill);

        return group;
    }

    // SQUARE: Neon wireframe cube
    const squareGeometry = new THREE.BoxGeometry(1.5, 1.5, 1.5);
    state.meshes.square = createNeonShape(squareGeometry, neonColors.square);

    // TRIANGLE: Neon wireframe tetrahedron
    const triangleGeometry = new THREE.ConeGeometry(1.2, 1.8, 3);
    state.meshes.triangle = createNeonShape(triangleGeometry, neonColors.triangle);

    // CIRCLE: Neon wireframe torus
    const circleGeometry = new THREE.TorusGeometry(0.9, 0.15, 8, 32);
    state.meshes.circle = createNeonShape(circleGeometry, neonColors.circle);

    // HEART: Neon wireframe heart
    state.meshes.heart = createNeonHeartMesh(neonColors.heart);

    // No idle mesh - "no shape" shows nothing
    state.meshes.idle = null;

    // Initially hide all
    Object.values(state.meshes).forEach(mesh => {
        if (mesh) {
            mesh.visible = false;
            mesh.scale.set(0.001, 0.001, 0.001);
            state.scene.add(mesh);
        }
    });
}

function createNeonHeartMesh(color) {
    const group = new THREE.Group();

    const heartShape = new THREE.Shape();
    const x = 0, y = 0;

    heartShape.moveTo(x + 0.5, y + 0.5);
    heartShape.bezierCurveTo(x + 0.5, y + 0.5, x + 0.4, y, x, y);
    heartShape.bezierCurveTo(x - 0.6, y, x - 0.6, y + 0.7, x - 0.6, y + 0.7);
    heartShape.bezierCurveTo(x - 0.6, y + 1.1, x - 0.3, y + 1.54, x + 0.5, y + 1.9);
    heartShape.bezierCurveTo(x + 1.2, y + 1.54, x + 1.6, y + 1.1, x + 1.6, y + 0.7);
    heartShape.bezierCurveTo(x + 1.6, y + 0.7, x + 1.6, y, x + 1, y);
    heartShape.bezierCurveTo(x + 0.7, y, x + 0.5, y + 0.5, x + 0.5, y + 0.5);

    const extrudeSettings = {
        depth: 0.3,
        bevelEnabled: false,
    };

    const geometry = new THREE.ExtrudeGeometry(heartShape, extrudeSettings);
    geometry.center();
    geometry.rotateX(Math.PI);

    // Edges
    const edges = new THREE.EdgesGeometry(geometry, 15);

    // Core
    const coreMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 3 });
    const core = new THREE.LineSegments(edges, coreMaterial);
    group.add(core);

    // Neon
    const neonMaterial = new THREE.LineBasicMaterial({
        color: color, linewidth: 2, transparent: true, opacity: 0.9
    });
    const neon = new THREE.LineSegments(edges.clone(), neonMaterial);
    neon.scale.set(1.01, 1.01, 1.01);
    group.add(neon);

    // Glow
    const glowMaterial = new THREE.LineBasicMaterial({
        color: color, linewidth: 1, transparent: true, opacity: 0.4
    });
    const glow = new THREE.LineSegments(edges.clone(), glowMaterial);
    glow.scale.set(1.05, 1.05, 1.05);
    group.add(glow);

    // Fill
    const fillMaterial = new THREE.MeshBasicMaterial({
        color: color, transparent: true, opacity: 0.15, side: THREE.DoubleSide
    });
    const fill = new THREE.Mesh(geometry.clone(), fillMaterial);
    fill.scale.set(0.99, 0.99, 0.99);
    group.add(fill);

    return group;
}

function updateShape(label) {
    const shapeKey = label ? label.toLowerCase() : null;
    const targetMesh = shapeKey ? state.meshes[shapeKey] : null;

    // Same state - no change needed
    if (state.currentMesh === targetMesh) {
        return;
    }

    // If no shape, just hide current
    if (!targetMesh) {
        if (state.currentMesh) {
            console.log('[Shape] Hiding current shape');
            state.targetMesh = null;
            state.transitionProgress = 0;
            state.hidingCurrent = true;
        }
        return;
    }

    console.log('[Shape] Transitioning to:', shapeKey);
    state.hidingCurrent = false;
    state.targetMesh = targetMesh;
    state.transitionProgress = 0;
}

function setActiveShape(shapeKey) {
    Object.entries(state.meshes).forEach(([key, mesh]) => {
        if (!mesh) return;  // Skip null meshes (idle)
        if (key === shapeKey) {
            mesh.visible = true;
            mesh.scale.set(1, 1, 1);
            state.currentMesh = mesh;
        } else {
            mesh.visible = false;
            mesh.scale.set(0.001, 0.001, 0.001);
        }
    });
}

function onThreeResize() {
    const container = dom.threeContainer;
    const width = container.clientWidth;
    const height = container.clientHeight;

    state.camera3d.aspect = width / height;
    state.camera3d.updateProjectionMatrix();
    state.renderer.setSize(width, height);
}

// -----------------------------------------------------------------------------
// Animation Loop
// -----------------------------------------------------------------------------

// Easing function for smooth transitions
function easeInOutCubic(t) {
    return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function easeOutElastic(t) {
    const c4 = (2 * Math.PI) / 3;
    return t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
}

function lerp(start, end, t) {
    return start + (end - start) * t;
}

function animate() {
    requestAnimationFrame(animate);

    const time = performance.now() * 0.001;

    // Handle hiding current shape (no new shape)
    if (state.hidingCurrent && state.transitionProgress < 1) {
        state.transitionProgress += 0.1;
        state.transitionProgress = Math.min(state.transitionProgress, 1);

        const easedProgress = easeInOutCubic(state.transitionProgress);

        if (state.currentMesh) {
            const outScale = Math.max(0.001, 1 - easedProgress);
            state.currentMesh.scale.set(outScale, outScale, outScale);

            if (state.transitionProgress >= 1) {
                state.currentMesh.visible = false;
                state.currentMesh = null;
                state.hidingCurrent = false;
            }
        }
    }
    // Two-phase transition to new shape
    else if (state.transitionProgress < 1 && state.targetMesh) {
        // Transition speed (slower = smoother)
        state.transitionProgress += 0.05;
        state.transitionProgress = Math.min(state.transitionProgress, 1);

        const progress = state.transitionProgress;

        // PHASE 1: Shrink old shape (progress 0 to 0.5)
        if (progress <= 0.5) {
            const shrinkProgress = progress * 2;
            const easedShrink = easeInOutCubic(shrinkProgress);

            if (state.currentMesh && state.currentMesh !== state.targetMesh) {
                const outScale = Math.max(0.001, 1 - easedShrink);
                state.currentMesh.scale.set(outScale, outScale, outScale);
            }
            if (state.targetMesh) {
                state.targetMesh.visible = false;
                state.targetMesh.scale.set(0.001, 0.001, 0.001);
            }
        }
        // PHASE 2: Grow new shape (progress 0.5 to 1.0)
        else {
            const growProgress = (progress - 0.5) * 2;
            const easedGrow = easeOutElastic(growProgress);

            if (state.currentMesh && state.currentMesh !== state.targetMesh) {
                state.currentMesh.visible = false;
                state.currentMesh.scale.set(0.001, 0.001, 0.001);
            }
            if (state.targetMesh) {
                state.targetMesh.visible = true;
                const inScale = Math.max(0.001, easedGrow);
                state.targetMesh.scale.set(inScale, inScale, inScale);
            }
        }

        // Complete transition
        if (state.transitionProgress >= 1) {
            if (state.currentMesh && state.currentMesh !== state.targetMesh) {
                state.currentMesh.visible = false;
            }
            state.currentMesh = state.targetMesh;
            if (state.currentMesh) {
                state.currentMesh.scale.set(1, 1, 1);
            }
            state.targetMesh = null;
        }
    }

    // Smooth rotation for current mesh
    if (state.currentMesh && state.currentMesh.visible) {
        const targetRotY = time * 0.4;
        const targetRotX = Math.sin(time * 0.3) * 0.1;

        state.currentMesh.rotation.y = lerp(state.currentMesh.rotation.y, targetRotY, 0.08);
        state.currentMesh.rotation.x = lerp(state.currentMesh.rotation.x, targetRotX, 0.05);

        state.currentMesh.position.y = Math.sin(time * 0.5) * 0.1;
    }

    // Render
    if (state.renderer && state.scene && state.camera3d) {
        state.renderer.render(state.scene, state.camera3d);
    }
}

// -----------------------------------------------------------------------------
// Error Handling
// -----------------------------------------------------------------------------

function showError(message) {
    dom.errorToast.textContent = message;
    dom.errorToast.classList.remove('hidden');

    setTimeout(() => {
        dom.errorToast.classList.add('hidden');
    }, 5000);
}

// -----------------------------------------------------------------------------
// Start
// -----------------------------------------------------------------------------

init();
