/**
 * app.js - Main Application Controller for Little Lungs Feasibility Test
 */

(function () {
  let detector = null;
  let config = null;
  let videoStream = null;
  let isRunning = false;
  let isProcessingFrame = false;
  let currentFacingMode = "environment"; // default rear camera

  // Performance tracking
  let lastFrameTime = performance.now();
  let frameCount = 0;
  let fps = 0;
  let lastFpsUpdate = performance.now();

  // DOM Elements
  const modelStatusBadge = document.getElementById("modelStatusBadge");
  const fpsDisplay = document.getElementById("fpsDisplay");
  const inferenceTimeDisplay = document.getElementById("inferenceTimeDisplay");
  const totalTimeDisplay = document.getElementById("totalTimeDisplay");
  const detectionCountDisplay = document.getElementById("detectionCountDisplay");

  const viewportWrapper = document.getElementById("viewportWrapper");
  const videoElement = document.getElementById("videoElement");
  const staticImageElement = document.getElementById("staticImageElement");
  const overlayCanvas = document.getElementById("overlayCanvas");
  const overlayCtx = overlayCanvas.getContext("2d");
  const emptyStateMessage = document.getElementById("emptyStateMessage");

  const btnStartCamera = document.getElementById("btnStartCamera");
  const btnStopCamera = document.getElementById("btnStopCamera");
  const btnSwitchCamera = document.getElementById("btnSwitchCamera");
  const btnRunTestImage = document.getElementById("btnRunTestImage");
  const imageFileInput = document.getElementById("imageFileInput");
  const detectionsTableBody = document.getElementById("detectionsTableBody");
  const offlineStatus = document.getElementById("offlineStatus");

  // Distinct palette for classes
  const CLASS_COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#9333ea",
    "#06b6d4", "#ea580c", "#84cc16", "#ec4899", "#14b8a6"
  ];

  function getColorForClass(classId) {
    return CLASS_COLORS[classId % CLASS_COLORS.length];
  }

  /**
   * Monitor online/offline state
   */
  function updateNetworkStatus() {
    if (navigator.onLine) {
      offlineStatus.textContent = "Online";
      offlineStatus.style.color = "#16a34a";
    } else {
      offlineStatus.textContent = "Offline (Cached)";
      offlineStatus.style.color = "#2563eb";
    }
  }
  window.addEventListener("online", updateNetworkStatus);
  window.addEventListener("offline", updateNetworkStatus);
  updateNetworkStatus();

  /**
   * Initialize Configuration and Load YOLO Model
   */
  async function init() {
    try {
      modelStatusBadge.textContent = "Loading config...";
      const cfgResp = await fetch("config.json");
      config = await cfgResp.json();

      modelStatusBadge.textContent = "Loading YOLOv8...";
      detector = new YOLOv8Detector(config);
      await detector.loadModel(config.modelPath);

      modelStatusBadge.textContent = "Model Ready";
      modelStatusBadge.classList.add("ready");

      btnStartCamera.disabled = false;
      btnRunTestImage.disabled = false;
      console.log("[App] Initialization complete.");
    } catch (err) {
      console.error("[App] Initialization failed:", err);
      modelStatusBadge.textContent = "Model Load Failed";
      modelStatusBadge.classList.add("error");
      emptyStateMessage.innerHTML = `<p style="color: #ef4444;">Failed to load model.</p><p class="hint">${err.message}</p>`;
    }
  }

  /**
   * Resize overlay canvas to match displayed element aspect ratio
   */
  function syncCanvasSize(sourceEl) {
    const srcW = sourceEl.videoWidth || sourceEl.naturalWidth || sourceEl.width;
    const srcH = sourceEl.videoHeight || sourceEl.naturalHeight || sourceEl.height;

    if (!srcW || !srcH) return;

    if (overlayCanvas.width !== srcW || overlayCanvas.height !== srcH) {
      overlayCanvas.width = srcW;
      overlayCanvas.height = srcH;
    }
  }

  /**
   * Draw detected bounding boxes, class names, and scores
   */
  function renderDetections(detections, sourceEl) {
    syncCanvasSize(sourceEl);
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    for (const det of detections) {
      const color = getColorForClass(det.classId);

      // Draw box border
      overlayCtx.lineWidth = Math.max(3, Math.round(overlayCanvas.width / 250));
      overlayCtx.strokeStyle = color;
      overlayCtx.strokeRect(det.x1, det.y1, det.width, det.height);

      // Draw label background and text
      const scoreText = `${det.className} ${(det.score * 100).toFixed(1)}%`;
      const fontSize = Math.max(14, Math.round(overlayCanvas.width / 35));
      overlayCtx.font = `bold ${fontSize}px sans-serif`;
      const textWidth = overlayCtx.measureText(scoreText).width;
      const tagHeight = fontSize + 6;

      const tagY = det.y1 >= tagHeight ? det.y1 - tagHeight : det.y1;

      overlayCtx.fillStyle = color;
      overlayCtx.fillRect(det.x1, tagY, textWidth + 8, tagHeight);

      overlayCtx.fillStyle = "#ffffff";
      overlayCtx.fillText(scoreText, det.x1 + 4, tagY + fontSize - 2);
    }

    // Update Detections Table
    updateDetectionsTable(detections);
  }

  /**
   * Update verification table in the UI
   */
  function updateDetectionsTable(detections) {
    detectionCountDisplay.textContent = detections.length;

    if (detections.length === 0) {
      detectionsTableBody.innerHTML = `<tr><td colspan="3" class="text-muted">No objects detected.</td></tr>`;
      return;
    }

    let rowsHtml = "";
    for (const d of detections) {
      const boxStr = `[${d.x1.toFixed(1)}, ${d.y1.toFixed(1)}, ${d.x2.toFixed(1)}, ${d.y2.toFixed(1)}]`;
      rowsHtml += `
        <tr>
          <td><strong>${d.className}</strong> (id: ${d.classId})</td>
          <td>${(d.score * 100).toFixed(1)}%</td>
          <td><code>${boxStr}</code></td>
        </tr>
      `;
    }
    detectionsTableBody.innerHTML = rowsHtml;
  }

  /**
   * Continuous Camera Detection Loop
   */
  async function runCameraLoop() {
    if (!isRunning) return;

    if (!isProcessingFrame && videoElement.readyState >= 2) {
      isProcessingFrame = true;
      try {
        const result = await detector.detect(videoElement);
        renderDetections(result.detections, videoElement);

        inferenceTimeDisplay.textContent = `${Math.round(result.inferenceTimeMs)} ms`;
        totalTimeDisplay.textContent = `${Math.round(result.totalTimeMs)} ms`;

        // Calculate running FPS
        frameCount++;
        const now = performance.now();
        const elapsed = now - lastFpsUpdate;
        if (elapsed >= 500) {
          fps = Math.round((frameCount * 1000) / elapsed);
          fpsDisplay.textContent = fps;
          frameCount = 0;
          lastFpsUpdate = now;
        }
      } catch (err) {
        console.error("[Camera Loop Error]", err);
      } finally {
        isProcessingFrame = false;
      }
    }

    if (isRunning) {
      requestAnimationFrame(runCameraLoop);
    }
  }

  /**
   * Start Camera Stream
   */
  async function startCamera(facingMode = "environment") {
    stopCurrentFeed();
    emptyStateMessage.style.display = "none";
    staticImageElement.style.display = "none";
    videoElement.style.display = "block";

    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: facingMode },
        width: { ideal: 640 },
        height: { ideal: 480 }
      }
    };

    try {
      videoStream = await navigator.mediaDevices.getUserMedia(constraints);
      videoElement.srcObject = videoStream;
      currentFacingMode = facingMode;

      await new Promise((resolve) => {
        videoElement.onloadedmetadata = () => {
          videoElement.play();
          resolve();
        };
      });

      isRunning = true;
      btnStartCamera.style.display = "none";
      btnStopCamera.style.display = "inline-flex";
      btnSwitchCamera.style.display = "inline-flex";

      lastFpsUpdate = performance.now();
      frameCount = 0;
      requestAnimationFrame(runCameraLoop);
    } catch (err) {
      console.error("[Camera Error]", err);
      emptyStateMessage.style.display = "block";
      emptyStateMessage.innerHTML = `
        <p style="color: #ef4444; font-weight: bold;">Camera Access Failed</p>
        <p class="hint">${err.name}: ${err.message}</p>
        <p class="hint">Ensure camera permissions are granted and served via HTTPS or localhost.</p>
      `;
    }
  }

  /**
   * Stop Camera Stream
   */
  function stopCurrentFeed() {
    isRunning = false;
    if (videoStream) {
      videoStream.getTracks().forEach((track) => track.stop());
      videoStream = null;
    }
    videoElement.srcObject = null;
    videoElement.style.display = "none";
    staticImageElement.style.display = "none";
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    btnStartCamera.style.display = "inline-flex";
    btnStopCamera.style.display = "none";
    btnSwitchCamera.style.display = "none";
    emptyStateMessage.style.display = "block";
    fpsDisplay.textContent = "--";
    inferenceTimeDisplay.textContent = "-- ms";
    totalTimeDisplay.textContent = "-- ms";
  }

  /**
   * Run inference on a static image (e.g. bus.jpg or uploaded file)
   */
  async function runStaticImage(imgUrl) {
    stopCurrentFeed();
    emptyStateMessage.style.display = "none";
    staticImageElement.style.display = "block";
    staticImageElement.src = imgUrl;

    await new Promise((resolve) => {
      staticImageElement.onload = () => resolve();
    });

    try {
      const result = await detector.detect(staticImageElement);
      renderDetections(result.detections, staticImageElement);

      fpsDisplay.textContent = "N/A";
      inferenceTimeDisplay.textContent = `${Math.round(result.inferenceTimeMs)} ms`;
      totalTimeDisplay.textContent = `${Math.round(result.totalTimeMs)} ms`;
    } catch (err) {
      console.error("[Static Image Inference Error]", err);
      alert("Inference failed: " + err.message);
    }
  }

  // Event Listeners
  btnStartCamera.addEventListener("click", () => startCamera("environment"));
  btnStopCamera.addEventListener("click", stopCurrentFeed);

  btnSwitchCamera.addEventListener("click", () => {
    const nextMode = currentFacingMode === "environment" ? "user" : "environment";
    startCamera(nextMode);
  });

  btnRunTestImage.addEventListener("click", () => {
    runStaticImage("bus.jpg");
  });

  imageFileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      runStaticImage(url);
    }
  });

  // Start initialization
  init();
})();
