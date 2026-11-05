/**
 * yolo.js - YOLOv8 ONNX Detector for Web Browsers
 * 
 * Features:
 * - Config-driven (reads input size, thresholds, class names from config.json)
 * - Letterbox preprocessing (preserves aspect ratio with gray padding, normalizes to [0, 1] RGB, NCHW layout)
 * - YOLOv8 output tensor parsing: shape [1, 4 + num_classes, num_anchors]
 * - Coordinate un-letterboxing back to original source dimensions
 * - Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
 * - Configured for single-threaded WebAssembly to guarantee compatibility without COOP/COEP headers
 */

class YOLOv8Detector {
  constructor(config) {
    this.config = config || {};
    this.session = null;
    this.inputWidth = this.config.inputWidth || 320;
    this.inputHeight = this.config.inputHeight || 320;
    this.confThreshold = this.config.confThreshold || 0.25;
    this.iouThreshold = this.config.iouThreshold || 0.45;
    this.classNames = this.config.classNames || [];
    this.numClasses = this.config.numClasses || (this.classNames.length > 0 ? this.classNames.length : 80);

    // Reusable offscreen canvas for letterboxing preprocessing
    this.offscreenCanvas = document.createElement("canvas");
    this.offscreenCanvas.width = this.inputWidth;
    this.offscreenCanvas.height = this.inputHeight;
    this.offscreenCtx = this.offscreenCanvas.getContext("2d", { willReadFrequently: true });
  }

  /**
   * Initializes ONNX Runtime Web and loads the model weights.
   * Configures single-thread execution so it functions cleanly on any mobile browser
   * without requiring SharedArrayBuffer COOP/COEP isolation headers.
   */
  async loadModel(modelPath) {
    const path = modelPath || this.config.modelPath || "models/yolov8n.onnx";
    
    // Configure ONNX Runtime environment
    if (window.ort) {
      // Point WASM path to local vendor directory so no network CDN is used
      ort.env.wasm.wasmPaths = "libs/";
      ort.env.wasm.numThreads = 1; // Works without Cross-Origin-Opener-Policy headers
      ort.env.wasm.simd = true;    // Hardware vector acceleration supported on modern tablets
    } else {
      throw new Error("ONNX Runtime Web library (ort.min.js) not found. Check libs/ folder.");
    }

    const sessionOptions = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    };

    console.log(`[YOLO] Loading ONNX model from: ${path}`);
    this.session = await ort.InferenceSession.create(path, sessionOptions);
    console.log("[YOLO] Model loaded successfully. Input names:", this.session.inputNames, "Output names:", this.session.outputNames);
  }

  /**
   * Performs letterbox resizing on source image/video element.
   * Maintains aspect ratio by fitting within inputWidth x inputHeight and padding with gray (114, 114, 114).
   * Converts RGBA pixels to normalized Float32Array in NCHW layout: [1, 3, H, W].
   */
  preprocess(sourceElement) {
    const srcW = sourceElement.videoWidth || sourceElement.naturalWidth || sourceElement.width;
    const srcH = sourceElement.videoHeight || sourceElement.naturalHeight || sourceElement.height;

    if (!srcW || !srcH) {
      throw new Error("Invalid source dimensions. Video or image is not ready.");
    }

    const dstW = this.inputWidth;
    const dstH = this.inputHeight;

    // Compute aspect-ratio-preserving scale and padding offsets
    const scale = Math.min(dstW / srcW, dstH / srcH);
    const scaledW = Math.round(srcW * scale);
    const scaledH = Math.round(srcH * scale);
    const padX = (dstW - scaledW) / 2;
    const padY = (dstH - scaledH) / 2;

    // Fill background with standard YOLO padding color (114, 114, 114)
    this.offscreenCtx.fillStyle = "#727272";
    this.offscreenCtx.fillRect(0, 0, dstW, dstH);

    // Draw scaled source image centered in letterbox canvas
    this.offscreenCtx.drawImage(sourceElement, 0, 0, srcW, srcH, padX, padY, scaledW, scaledH);

    // Extract RGBA pixel buffer
    const imgData = this.offscreenCtx.getImageData(0, 0, dstW, dstH);
    const pixels = imgData.data;
    const totalPixels = dstW * dstH;

    // Allocate NCHW Float32 buffer: [1, 3, dstH, dstW]
    const float32Data = new Float32Array(3 * totalPixels);
    const rOffset = 0;
    const gOffset = totalPixels;
    const bOffset = totalPixels * 2;

    for (let i = 0; i < totalPixels; i++) {
      const p = i * 4;
      // Normalize pixel values from [0, 255] to [0.0, 1.0]
      float32Data[rOffset + i] = pixels[p] / 255.0;
      float32Data[gOffset + i] = pixels[p + 1] / 255.0;
      float32Data[bOffset + i] = pixels[p + 2] / 255.0;
    }

    const tensor = new ort.Tensor("float32", float32Data, [1, 3, dstH, dstW]);

    return {
      tensor,
      scale,
      padX,
      padY,
      srcW,
      srcH
    };
  }

  /**
   * Decodes the raw YOLOv8 output tensor:
   * Output shape: [1, 4 + num_classes, num_anchors]
   * Row 0: Center X (cx) in 320x320 space
   * Row 1: Center Y (cy) in 320x320 space
   * Row 2: Width (w) in 320x320 space
   * Row 3: Height (h) in 320x320 space
   * Rows 4..(4 + num_classes - 1): Class confidence probabilities
   */
  decodeOutput(outputTensor, prepInfo) {
    const dims = outputTensor.dims; // e.g. [1, 84, 2100]
    const channels = dims[1];       // 4 + numClasses
    const numAnchors = dims[2];     // e.g. 2100 for 320x320
    const data = outputTensor.data;

    const { scale, padX, padY, srcW, srcH } = prepInfo;
    const candidates = [];

    for (let a = 0; a < numAnchors; a++) {
      // Find class with highest score for this anchor
      let maxScore = -Infinity;
      let bestClassId = -1;

      for (let c = 0; c < this.numClasses; c++) {
        // Index in row-major order: (4 + c) * numAnchors + a
        const score = data[(4 + c) * numAnchors + a];
        if (score > maxScore) {
          maxScore = score;
          bestClassId = c;
        }
      }

      // Filter by confidence threshold
      if (maxScore < this.confThreshold) {
        continue;
      }

      // Read bounding box in letterboxed coordinates [cx, cy, w, h]
      const cx = data[0 * numAnchors + a];
      const cy = data[1 * numAnchors + a];
      const w  = data[2 * numAnchors + a];
      const h  = data[3 * numAnchors + a];

      // Convert center-based box to corner coordinates [bx1, by1, bx2, by2]
      const bx1 = cx - w / 2;
      const by1 = cy - h / 2;
      const bx2 = cx + w / 2;
      const by2 = cy + h / 2;

      // Inverse letterbox: remove padding and scale back to original source dimensions
      let x1 = (bx1 - padX) / scale;
      let y1 = (by1 - padY) / scale;
      let x2 = (bx2 - padX) / scale;
      let y2 = (by2 - padY) / scale;

      // Clamp coordinates to image boundaries
      x1 = Math.max(0, Math.min(srcW, x1));
      y1 = Math.max(0, Math.min(srcH, y1));
      x2 = Math.max(0, Math.min(srcW, x2));
      y2 = Math.max(0, Math.min(srcH, y2));

      const boxWidth = x2 - x1;
      const boxHeight = y2 - y1;

      if (boxWidth <= 1 || boxHeight <= 1) {
        continue;
      }

      candidates.push({
        classId: bestClassId,
        className: this.classNames[bestClassId] || `class_${bestClassId}`,
        score: maxScore,
        x1,
        y1,
        x2,
        y2,
        width: boxWidth,
        height: boxHeight
      });
    }

    // Apply Non-Maximum Suppression (NMS)
    return this.applyNMS(candidates);
  }

  /**
   * Intersection-over-Union (IoU) calculation between two boxes.
   */
  calculateIoU(boxA, boxB) {
    const x1 = Math.max(boxA.x1, boxB.x1);
    const y1 = Math.max(boxA.y1, boxB.y1);
    const x2 = Math.min(boxA.x2, boxB.x2);
    const y2 = Math.min(boxA.y2, boxB.y2);

    const interW = Math.max(0, x2 - x1);
    const interH = Math.max(0, y2 - y1);
    const interArea = interW * interH;

    const areaA = boxA.width * boxA.height;
    const areaB = boxB.width * boxB.height;
    const unionArea = areaA + areaB - interArea;

    return unionArea <= 0 ? 0 : interArea / unionArea;
  }

  /**
   * Multi-class Non-Maximum Suppression.
   * Performs suppression separately per class to preserve co-occurring objects.
   */
  applyNMS(candidates) {
    if (candidates.length === 0) return [];

    // Group candidates by classId
    const byClass = new Map();
    for (const box of candidates) {
      if (!byClass.has(box.classId)) {
        byClass.set(box.classId, []);
      }
      byClass.get(box.classId).push(box);
    }

    const finalBoxes = [];

    for (const [, classBoxes] of byClass.entries()) {
      // Sort descending by confidence score
      classBoxes.sort((a, b) => b.score - a.score);

      const picked = [];
      const suppressed = new Array(classBoxes.length).fill(false);

      for (let i = 0; i < classBoxes.length; i++) {
        if (suppressed[i]) continue;

        const current = classBoxes[i];
        picked.push(current);

        for (let j = i + 1; j < classBoxes.length; j++) {
          if (suppressed[j]) continue;
          const iou = this.calculateIoU(current, classBoxes[j]);
          if (iou > this.iouThreshold) {
            suppressed[j] = true;
          }
        }
      }

      finalBoxes.push(...picked);
    }

    // Sort final detections by confidence descending
    finalBoxes.sort((a, b) => b.score - a.score);
    return finalBoxes;
  }

  /**
   * End-to-end detection on a single image or video frame.
   * Returns: { detections: [...], inferenceTimeMs: number, prepTimeMs: number }
   */
  async detect(sourceElement) {
    if (!this.session) {
      throw new Error("Model is not loaded. Call loadModel() first.");
    }

    const t0 = performance.now();
    const prepInfo = this.preprocess(sourceElement);
    const t1 = performance.now();

    const inputName = this.session.inputNames[0];
    const feeds = { [inputName]: prepInfo.tensor };

    const runResult = await this.session.run(feeds);
    const t2 = performance.now();

    const outputName = this.session.outputNames[0];
    const outputTensor = runResult[outputName];

    const detections = this.decodeOutput(outputTensor, prepInfo);
    const t3 = performance.now();

    return {
      detections,
      prepTimeMs: t1 - t0,
      inferenceTimeMs: t2 - t1,
      decodeTimeMs: t3 - t2,
      totalTimeMs: t3 - t0
    };
  }
}

// Export for browser global scope
window.YOLOv8Detector = YOLOv8Detector;
