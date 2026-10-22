
"use strict";

const API_URL = "http://127.0.0.1:5000";

const camera = document.getElementById("camera");
const processedCamera = document.getElementById("processedCamera");
const cameraFrame = document.getElementById("cameraFrame");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const cameraStatus = document.getElementById("cameraStatus");
const cameraError = document.getElementById("cameraError");
const resolution = document.getElementById("resolution");
const previewStatus = document.getElementById("previewStatus");
const previewDot = document.getElementById("previewDot");
const sessionBadge = document.querySelector(".session-badge");

const startButton = document.getElementById("startCamera");
const stopButton = document.getElementById("stopCamera");
const fullscreenButton = document.getElementById("fullscreenCamera");

const brightness = document.getElementById("brightness");
const contrast = document.getElementById("contrast");
const brightnessValue = document.getElementById("brightnessValue");
const contrastValue = document.getElementById("contrastValue");
const resetButton = document.getElementById("resetAdjustments");

const PROCESSING_MAX_WIDTH = 640;
const PROCESSING_INTERVAL_MS = 80;
const JPEG_QUALITY = 0.7;

let performanceSamples = [];
const PERFORMANCE_SAMPLE_SIZE = 30;

let cameraStream = null;
let processing = false;
let processingTimer = null;
let stopped = true;
let backendFailed = false;
let lastRequestTime = 0;

const effects = {
    none: "",
    mono: "grayscale(1)",
    warm: "sepia(0.35) saturate(1.35)",
    dream: "saturate(1.4) hue-rotate(12deg) contrast(0.92)"
};

let selectedEffect = "none";

const captureCanvas = document.createElement("canvas");
const captureContext = captureCanvas.getContext("2d");

function updatePreviewFilter() {
    const brightnessAmount = Number(brightness.value);
    const contrastAmount = Number(contrast.value);

    const filter = [
        effects[selectedEffect],
        `brightness(${brightnessAmount}%)`,
        `contrast(${contrastAmount}%)`
    ].filter(Boolean).join(" ");

    camera.style.filter = filter;
    processedCamera.style.filter = filter;

    brightnessValue.value = `${brightnessAmount}%`;
    contrastValue.value = `${contrastAmount}%`;
}

function showCameraError(message) {
    cameraError.textContent = message;
    cameraError.hidden = false;
}

function clearCameraError() {
    cameraError.textContent = "";
    cameraError.hidden = true;
}

function showProcessedPreview(enabled) {
    processedCamera.hidden = !enabled;
    camera.style.visibility = enabled ? "hidden" : "visible";
}

function stopProcessing() {
    stopped = true;

    if (processingTimer !== null) {
        clearTimeout(processingTimer);
        processingTimer = null;
    }

    processing = false;
}

function scheduleProcessing(delay = 100) {
    if (stopped || backendFailed || !cameraStream) return;

    if (processingTimer !== null) {
        clearTimeout(processingTimer);
    }

    processingTimer = setTimeout(processFrame, delay);
}

async function processFrame() {
    processingTimer = null;

    if (stopped || backendFailed || !cameraStream || processing) return;
    if (camera.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
        scheduleProcessing(100);
        return;
    }

    processing = true;

    try {
        const scale = Math.min(
    1,
    PROCESSING_MAX_WIDTH / camera.videoWidth
);

const targetWidth = Math.round(camera.videoWidth * scale);
const targetHeight = Math.round(camera.videoHeight * scale);

if (
    captureCanvas.width !== targetWidth ||
    captureCanvas.height !== targetHeight
) {
    captureCanvas.width = targetWidth;
    captureCanvas.height = targetHeight;
}

        captureContext.drawImage(
            camera,
            0,
            0,
            captureCanvas.width,
            captureCanvas.height
        );

        const encodingStarted = performance.now();

const image = captureCanvas.toDataURL(
    "image/jpeg",
    JPEG_QUALITY
);

const browserEncodingMs = performance.now() - encodingStarted;
const roundTripStarted = performance.now();

        const response = await fetch(`${API_URL}/api/process-frame`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image })
        });

        if (!response.ok) {
            throw new Error(`Python backend returned HTTP ${response.status}`);
        }

        const data = await response.json();
        const roundTripMs = performance.now() - roundTripStarted;

        if (data.timings_ms) {
    performanceSamples.push({
        browserEncoding: browserEncodingMs,
        roundTrip: roundTripMs,
        pythonDecode: data.timings_ms.decode,
        mediapipe: data.timings_ms.mediapipe,
        portal: data.timings_ms.portal,
        outputEncode: data.timings_ms.output_encode,
        backendTotal: data.timings_ms.total,
    });

    if (performanceSamples.length >= PERFORMANCE_SAMPLE_SIZE) {
        const keys = Object.keys(performanceSamples[0]);
        const averages = {};

        for (const key of keys) {
            const total = performanceSamples.reduce(
                (sum, sample) => sum + sample[key],
                0
            );

            averages[key + "_ms"] = Number(
                (total / performanceSamples.length).toFixed(2)
            );
        }

        console.table(averages);
        performanceSamples = [];
    }
}

        if (stopped || !cameraStream) return;

        processedCamera.src = data.image;
        showProcessedPreview(true);

        cameraStatus.textContent = data.hands_detected
            ? `Portal active · Filter ${data.filter_index + 1}`
            : "Python processing active";

        previewStatus.textContent = "Python processing connected";
        previewDot.style.background = "var(--accent)";
        resolution.textContent =
            `${camera.videoWidth} × ${camera.videoHeight}`;
    } catch (error) {
        console.error("Python processing failed:", error);

        backendFailed = true;
        showProcessedPreview(false);
        previewStatus.textContent = "Python backend unavailable";

        showCameraError(
            "Python processing could not continue. Check that server.py is running, then stop and restart the camera."
        );
    } finally {
        processing = false;

        if (!stopped && !backendFailed && cameraStream) {
            // One request at a time; leave room for the backend to respond.
            scheduleProcessing(PROCESSING_INTERVAL_MS);
        }
    }
}

async function startCamera() {
    if (cameraStream) return;

    clearCameraError();
    stopped = true;
    backendFailed = false;

    if (!navigator.mediaDevices?.getUserMedia) {
        showCameraError(
            "Camera access is unavailable. Open this page on localhost or through HTTPS."
        );
        return;
    }

    startButton.disabled = true;
    startButton.textContent = "Requesting permission…";

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: {
                facingMode: "user",
                width: { ideal: 1280 },
                height: { ideal: 720 }
            }
        });

        camera.srcObject = cameraStream;
        await camera.play();

        stopped = false;
        camera.classList.add("is-visible");
        cameraPlaceholder.hidden = true;
        cameraFrame.classList.add("is-live");
        sessionBadge?.classList.add("is-live");
        previewDot.style.background = "var(--accent)";

        cameraStatus.textContent = "Connecting to Python…";
        previewStatus.textContent = "Starting Python processing";
        resolution.textContent =
            `${camera.videoWidth} × ${camera.videoHeight}`;

        showProcessedPreview(false);
        stopButton.disabled = false;
        fullscreenButton.disabled = false;

        scheduleProcessing(0);
    } catch (error) {
        console.error("Camera startup failed:", error);

        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }

        camera.srcObject = null;

        let message = "Unable to start the camera. Please try again.";

        if (error.name === "NotAllowedError") {
            message = "Camera permission was denied. Allow camera access and try again.";
        } else if (error.name === "NotFoundError") {
            message = "No camera was found. Connect a camera and try again.";
        } else if (error.name === "NotReadableError") {
            message = "The camera may already be in use by another application.";
        }

        showCameraError(message);
    } finally {
        startButton.disabled = false;
        startButton.textContent = "Enable camera";
    }
}

function stopCamera() {
    stopProcessing();

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    camera.srcObject = null;
    processedCamera.removeAttribute("src");
    showProcessedPreview(false);

    camera.classList.remove("is-visible");
    camera.style.visibility = "visible";
    cameraPlaceholder.hidden = false;
    cameraFrame.classList.remove("is-live");
    sessionBadge?.classList.remove("is-live");

    cameraStatus.textContent = "Camera off";
    previewStatus.textContent = "Preview not started";
    resolution.textContent = "WAITING FOR CAMERA";
    previewDot.style.background = "var(--subtle)";

    stopButton.disabled = true;
    fullscreenButton.disabled = true;
    clearCameraError();
}

async function toggleFullscreen() {
    try {
        if (document.fullscreenElement === cameraFrame) {
            await document.exitFullscreen();
        } else if (cameraFrame.requestFullscreen) {
            await cameraFrame.requestFullscreen();
        } else {
            showCameraError("Fullscreen is not supported by this browser.");
        }
    } catch {
        showCameraError("Unable to enter fullscreen.");
    }
}

function updateFullscreenButton() {
    const isFullscreen = document.fullscreenElement === cameraFrame;
    fullscreenButton.textContent =
        isFullscreen ? "Exit fullscreen" : "Enter fullscreen";
    fullscreenButton.setAttribute("aria-pressed", String(isFullscreen));
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
fullscreenButton.addEventListener("click", toggleFullscreen);
document.addEventListener("fullscreenchange", updateFullscreenButton);

document.querySelectorAll(".effect-option").forEach(button => {
    button.addEventListener("click", () => {
        selectedEffect = button.dataset.effect;

        document.querySelectorAll(".effect-option").forEach(option => {
            const selected = option === button;
            option.classList.toggle("active", selected);
            option.setAttribute("aria-pressed", String(selected));
        });

        updatePreviewFilter();
    });

    button.setAttribute(
        "aria-pressed",
        String(button.classList.contains("active"))
    );
});

brightness.addEventListener("input", updatePreviewFilter);
contrast.addEventListener("input", updatePreviewFilter);

resetButton.addEventListener("click", () => {
    brightness.value = 100;
    contrast.value = 100;
    selectedEffect = "none";

    document.querySelectorAll(".effect-option").forEach(button => {
        const selected = button.dataset.effect === "none";
        button.classList.toggle("active", selected);
        button.setAttribute("aria-pressed", String(selected));
    });

    updatePreviewFilter();
});

window.addEventListener("pagehide", stopCamera);

updatePreviewFilter();
