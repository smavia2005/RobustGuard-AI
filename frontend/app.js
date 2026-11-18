
"use strict";

// =====================================
// ROBUSTGUARD AI - MULTI-PAGE APP
// =====================================

const API_URL = "http://127.0.0.1:8000";
const MAX_SIZE = 20 * 1024 * 1024;
const HISTORY_KEY = "rg_scan_history_v1";
const RESULT_KEY = "rg_latest_result_v1";

const page = document.body.dataset.page;
const app = document.getElementById("app");

const mediaConfig = {
  image: {
    title: "Image Detection",
    hint: "JPG, PNG or WebP",
    accept: "image/jpeg,image/png,image/webp",
    mime: ["image/jpeg", "image/png", "image/webp"]
  },
  video: {
    title: "Video Detection",
    hint: "MP4 or WebM",
    accept: "video/mp4,video/webm",
    mime: ["video/mp4", "video/webm"]
  },
  audio: {
    title: "Audio Detection",
    hint: "MP3, WAV, M4A, FLAC or OGG",
    accept: ".mp3,.wav,.m4a,.flac,.ogg,.opus,.aac,.webm",
    mime: [
      "audio/mpeg",
      "audio/mp3",
      "audio/wav",
      "audio/x-wav",
      "audio/wave",
      "audio/mp4",
      "audio/x-m4a",
      "audio/flac",
      "audio/x-flac",
      "audio/ogg",
      "audio/opus",
      "audio/aac",
      "audio/webm",
      "video/webm"
    ]
  },
  text: {
    title: "Text Detection",
    hint: "Paste text"
  }
};

let currentType = "image";
let selectedFile = null;
let previewURL = null;

// =====================================
// SAFE TEXT OUTPUT
// =====================================

function escapeHTML(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    char => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    })[char]
  );
}

// =====================================
// COMMON HEADER AND SIDEBAR
// =====================================

function layout(content) {
  const links = [
    ["index.html", "⌂ Dashboard", "home"],
    ["detect.html", "◈ Media Analysis", "detect"],
    ["history.html", "◷ Analysis History", "history"],
    ["result.html", "▤ Latest Report", "result"],
    ["guardbot.html", "✦ GuardBot AI", "guardbot"]
  ];

  const sidebarLinks = links.map(([url, label, key]) => `
    <a href="${url}"
       class="${page === key ? "active" : ""}">
       ${label}
    </a>
  `).join("");

  app.innerHTML = `
    <header class="header">
      <a class="logo" href="index.html">
        <span class="logo-icon">✦</span>
        <span>
          <strong>RobustGuard AI</strong>
          <small>Media Authenticity Platform</small>
        </span>
      </a>

      
<nav class="topnav">
  <a href="index.html">Home</a>
  <a href="detect.html">Workspace</a>
  <a href="history.html">History</a>
  <a href="guardbot.html">GuardBot AI</a>
  <a href="index.html#learn">Learn</a>
  <a href="index.html#models">Models</a>
</nav>


      <span class="header-tag">Research Prototype</span>

      <button class="menu-btn" id="menuBtn"
        aria-label="Open navigation"
        aria-expanded="false">
        ☰
      </button>
    </header>

    <div class="layout">
      <aside class="sidebar" id="sidebar">
        <div class="side-intro">
          <div class="symbol">✧</div>
          <h2>RobustGuard AI</h2>
          <p>
            Verify media. Fight manipulation.
            Build a more trusted digital world.
          </p>
        </div>

        <nav class="side-nav">
          ${sidebarLinks}

          <a href="index.html#learn">
            ♧ Learn & Explore
          </a>

          <a href="index.html#models">
            ◇ Datasets & Models
          </a>

          <a href="index.html#trust">
            ♡ Trust & Safety
          </a>
        </nav>

        <div class="side-card">
          <h3>✧ Research Suite</h3>
          <p>
            Explore the science behind digital
            media authentication and deepfake detection.
          </p>
          <a href="index.html#models"
             class="btn btn-secondary">
            Explore Research
          </a>
        </div>
      </aside>

      <main class="content">
        ${content}

        <footer class="footer">
          <strong>✦ RobustGuard AI</strong>
          <span>AI-Powered Media Authenticity Research</span>
          <span>© 2026 RobustGuard AI</span>
        </footer>
      </main>
    </div>
  `;

  document.getElementById("menuBtn")
    .addEventListener("click", () => {
      const sidebar = document.getElementById("sidebar");
      const open = sidebar.classList.toggle("open");

      document.getElementById("menuBtn")
        .setAttribute("aria-expanded", String(open));
    });
}

// =====================================
// HOME DASHBOARD
// =====================================

function renderHome() {
  layout(`
    <section class="hero">
      <div>
        <span class="eyebrow">
          DEEPFAKE DETECTION · DIGITAL TRUST
        </span>

        <h1>
          AI-Powered Authenticity Analysis
          <em>for a Safer Digital World</em>
        </h1>

        <p>
          RobustGuard AI is a multimodal deepfake
          detection research platform designed to
          analyse images, videos, audio and text
          for potentially AI-generated or
          digitally manipulated content.
        </p>

        <a href="detect.html" class="btn btn-primary">
          Start Detection →
        </a>

        <a href="#how" class="btn btn-secondary">
          How It Works
        </a>
      </div>

      <div class="hero-art">
        <div class="shield">✦ <span>AI</span></div>
      </div>
    </section>

    <section class="panel">
      <div class="panel-heading">
        <div>
          <span class="eyebrow">OUR FEATURES</span>
          <h2>Multimodal Detection Platform</h2>
          <p>Explore four digital content categories.</p>
        </div>
      </div>

      <div class="features-grid">

        <article class="feature-card">
          <span class="feature-icon">▧</span>
          <h3>Image Detection</h3>
          <p>
            Examine photographs and synthetic
            images for potential manipulation.
          </p>
          <a href="detect.html?type=image">
            Explore Image Detection →
          </a>
        </article>

        <article class="feature-card">
          <span class="feature-icon">▷</span>
          <h3>Video Detection</h3>
          <p>
            Analyse sampled video frames using face
            detection and a trained video classifier.
          </p>
          <a href="detect.html?type=video">
            Explore Video Detection →
          </a>
        </article>

        <article class="feature-card">
          <span class="feature-icon">≋</span>
          <h3>Audio Detection</h3>
          <p>
            Assess voice recordings for
            potential synthetic speech.
          </p>
          <a href="detect.html?type=audio">
            Explore Audio Detection →
          </a>
        </article>

        <article class="feature-card">
          <span class="feature-icon">T</span>
          <h3>Text Detection</h3>
          <p>
            Examine writing for patterns
            associated with AI generation.
          </p>
          <a href="detect.html?type=text">
            Explore Text Detection →
          </a>
        </article>

        <article class="feature-card">
          <span class="feature-icon">◷</span>
          <h3>Analysis History</h3>
          <p>
            View local submission records
            and their processing status.
          </p>
          <a href="history.html">
            View History →
          </a>
        </article>

        <article class="feature-card">
          <span class="feature-icon">▤</span>
          <h3>Detailed Reports</h3>
          <p>
            Review model predictions,
            analysis scores and explanations.
          </p>
          <a href="result.html">
            View Latest Report →
          </a>
        </article>

      </div>
    </section>

    <section id="how" class="panel">
      <span class="eyebrow">OUR WORKFLOW</span>
      <h2>How RobustGuard AI Works</h2>
      <p>Four steps to understanding digital media.</p>

      <div class="steps-grid" style="margin-top:20px">

        <article class="step-card">
          <span class="step-number">STEP 01</span>
          <span class="step-icon">⇧</span>
          <h3>Upload Content</h3>
          <p>
            Select an image, video, audio or text sample.
          </p>
        </article>

        <article class="step-card">
          <span class="step-number">STEP 02</span>
          <span class="step-icon">✦</span>
          <h3>AI Processing</h3>
          <p>
            A connected model analyses
            the selected media.
          </p>
        </article>

        <article class="step-card">
          <span class="step-number">STEP 03</span>
          <span class="step-icon">⌕</span>
          <h3>View Results</h3>
          <p>
            Review predictions and model scores.
          </p>
        </article>

        <article class="step-card">
          <span class="step-number">STEP 04</span>
          <span class="step-icon">▤</span>
          <h3>Understand Findings</h3>
          <p>
            Read explanations and detection limitations.
          </p>
        </article>

      </div>
    </section>

    <section id="learn" class="panel">
      <span class="eyebrow">LEARNING CENTRE</span>
      <h2>Understanding Deepfakes</h2>

      <p>
        Deepfakes are synthetic or manipulated media
        created using artificial intelligence techniques.
        They can affect how people judge the authenticity
        of information.
      </p>

      <div class="features-grid" style="margin-top:20px">

        <article class="feature-card">
          <h3>What Are Deepfakes?</h3>
          <p>
            Artificially generated or altered media
            that can imitate realistic content.
          </p>
        </article>

        <article class="feature-card">
          <h3>Why Detection Matters</h3>
          <p>
            Detection tools can support investigations
            into misinformation and impersonation.
          </p>
        </article>

        <article class="feature-card">
          <h3>Responsible Interpretation</h3>
          <p>
            Model predictions have limitations
            and should be reviewed carefully.
          </p>
        </article>

      </div>
    </section>

    // =====================================
    // DATASETS AND MODEL INFORMATION
    // =====================================

    <section id="models" class="panel">
      <span class="eyebrow">RESEARCH</span>

      <h2>Datasets & Model Information</h2>

      <p>
        RobustGuard AI integrates four working deepfake
        detection models for images, video, audio and text.
        Below are the implemented models, datasets,
        methodologies and experimental evaluation results.
      </p>

      <div class="guidelines">

        <div class="guide-item">
          <strong>Image Deepfake Detection</strong>

          <span>
            <b>Model:</b>
            prithivMLmods/Mirage-Photo-Classifier
            (SigLIP2).<br>

            <b>Approach:</b>
            Pretrained vision-based image classification
            for authentic and AI-generated content.<br>

            <b>Evaluation:</b>
            71.10% accuracy on a mixed benchmark
            of 2,000 images.<br>

            <b>Note:</b>
            Performance varies across image datasets.
          </span>
        </div>

        <div class="guide-item">
          <strong>Video Deepfake Detection</strong>

          <span>
            <b>Model:</b>
            Face ViT feature extractor,
            PCA and Logistic Regression.<br>

            <b>Approach:</b>
            Video-frame sampling, YuNet face detection
            and face-feature classification.<br>

            <b>Dataset:</b>
            SDFVD.<br>

            <b>Evaluation:</b>
            65.38% accuracy on 26 held-out videos.<br>

            <b>Note:</b>
            Small evaluation set; performance on
            unseen video sources may differ.
          </span>
        </div>

        <div class="guide-item">
          <strong>Audio Deepfake Detection</strong>

          <span>
            <b>Model:</b>
            garystafford/wav2vec2-deepfake-voice-detector
            (Wav2Vec2).<br>

            <b>Approach:</b>
            Pretrained speech-classification model
            with a 90% Fake-score decision threshold.<br>

            <b>Dataset:</b>
            ASVspoof 2019 Logical Access (LA).<br>

            <b>Evaluation:</b>
            92.00% accuracy on 100 previously unused
            audio samples.<br>

            <b>Fake Recall:</b> 88.00%.<br>
            <b>Real Recall:</b> 96.00%.<br>

            <b>Note:</b>
            Exploratory same-dataset validation.
            Model scores are not calibrated probabilities.
          </span>
        </div>

        <div class="guide-item">
          <strong>Text AI Detection</strong>

          <span>
            <b>Model:</b>
            TF-IDF + Logistic Regression.<br>

            <b>Approach:</b>
            Custom-trained machine learning classifier
            for distinguishing human-written text
            from AI-generated text.<br>

            <b>Dataset:</b>
            HC3 (Human ChatGPT Comparison Corpus).<br>

            <b>Evaluation:</b>
            97.84% test accuracy on the evaluated split.
          </span>
        </div>

      </div>

      <p style="margin-top:20px">
        <strong>Research Note:</strong>
        Evaluation percentages were obtained using
        different datasets and experimental protocols.
        These figures should not be compared as
        equivalent benchmarks or treated as guaranteed
        accuracy on real-world content.
      </p>

    </section>

    <section id="trust" class="panel">
      <span class="eyebrow">TRUST & SAFETY</span>
      <h2>Responsible AI and Digital Trust</h2>

      <p>
        RobustGuard AI is an educational research
        prototype. AI results should be treated
        as decision-support information,
        not definitive evidence.
      </p>
    </section>
  `);

  // Scroll to anchored section after dynamic rendering.
  if (location.hash) {
    requestAnimationFrame(() => {
      document.querySelector(location.hash)?.scrollIntoView();
    });
  }
}

// =====================================
// DETECTION PAGE
// =====================================

function renderDetect() {
  layout(`
    <section class="panel">
      <span class="eyebrow">
        AI-POWERED MEDIA AUTHENTICITY
      </span>

      <h2>Detection Workspace</h2>

      <p>
        Select the type of content you want to examine.
        Uploaded content will be previewed
        before submission.
      </p>

      <div class="tabs">
        <button class="tab" data-type="image">
          ▧ Image
        </button>
        <button class="tab" data-type="video">
          ▷ Video
        </button>
        <button class="tab" data-type="audio">
          ≋ Audio
        </button>
        <button class="tab" data-type="text">
          T Text
        </button>
      </div>
    </section>

    <div class="detect-layout">

      <section class="panel">
        <span class="eyebrow">MEDIA UPLOAD</span>
        <h2 id="moduleHeading">
          Image Detection
        </h2>

        <p id="moduleDescription">
          Upload your media to begin.
        </p>

        <div class="dropzone" id="dropZone">
          <div class="upload-icon">⇧</div>
          <h3>Drag & Drop Your Media</h3>
          <p id="formatHint">JPG, PNG or WebP</p>

          <input
            id="fileInput"
            type="file"
            aria-label="Choose media file">
        </div>

        <textarea
          id="textInput"
          placeholder="Paste or type your text here..."
          rows="8"
          hidden></textarea>

        <div id="previewArea"
             class="preview-box"
             hidden>
          <h3>Uploaded Content Preview</h3>
          <div id="previewContent"></div>
          <p id="fileMeta" class="file-meta"></p>
        </div>

        <button
          id="analyseButton"
          class="btn btn-primary"
          style="width:100%;margin-top:20px">
          ✦ Analyse Content
        </button>

        <p id="statusMessage"
           class="status-message"
           role="status"></p>
      </section>

      <aside class="panel">
        <span class="eyebrow">
          UPLOAD GUIDELINES
        </span>

        <h2>Prepare Your Media</h2>

        <p>
          Follow these recommendations for a
          better media assessment experience.
        </p>

        <div class="guidelines">

          <div class="guide-item">
            <strong>✓ Original Media</strong>
            <span>
              Prefer original, unedited files.
            </span>
          </div>

          <div class="guide-item">
            <strong>✓ Image Formats</strong>
            <span>JPG, PNG, WebP</span>
          </div>

          <div class="guide-item">
            <strong>✓ Video Formats</strong>
            <span>MP4, WebM</span>
          </div>

          <div class="guide-item">
            <strong>✓ Audio Formats</strong>
            <span>
              MP3, WAV, M4A, FLAC, OGG,
              OPUS, AAC and WebM
            </span>
          </div>

          <div class="guide-item">
            <strong>✓ File Size</strong>
            <span>Maximum 20 MB</span>
          </div>

          <div class="guide-item">
            <strong>✓ Evaluation</strong>
            <span>
              AI assessment does not guarantee
              authenticity or manipulation.
            </span>
          </div>

        </div>
      </aside>

    </div>
  `);

  setupDetection();
}

// =====================================
// LOCAL BROWSER FILE STORAGE
// =====================================

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(
      "RobustGuardLocal",
      1
    );

    request.onupgradeneeded = () => {
      if (
        !request.result.objectStoreNames.contains("media")
      ) {
        request.result.createObjectStore("media");
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function saveMedia(file) {
  const db = await openDatabase();

  return new Promise((resolve, reject) => {
    const tx = db.transaction("media", "readwrite");

    tx.objectStore("media").put(file, "latest");

    tx.oncomplete = () => {
      db.close();
      resolve();
    };

    tx.onerror = () => {
      db.close();
      reject(tx.error);
    };
  });
}

async function getMedia() {
  const db = await openDatabase();

  return new Promise((resolve, reject) => {
    const tx = db.transaction("media", "readonly");

    const request = tx.objectStore("media").get("latest");

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);

    tx.oncomplete = () => db.close();
  });
}

// =====================================
// DETECTION PAGE FUNCTIONS
// =====================================

function setupDetection() {
  const tabs = document.querySelectorAll(".tab");
  const input = document.getElementById("fileInput");
  const textInput = document.getElementById("textInput");
  const dropZone = document.getElementById("dropZone");
  const previewArea = document.getElementById("previewArea");
  const previewContent = document.getElementById("previewContent");
  const fileMeta = document.getElementById("fileMeta");
  const status = document.getElementById("statusMessage");

  const analyseButton =
    document.getElementById("analyseButton");

  function clearPreview() {
    if (previewURL) {
      URL.revokeObjectURL(previewURL);
      previewURL = null;
    }

    previewContent.replaceChildren();
    previewArea.hidden = true;
    fileMeta.textContent = "";
    status.textContent = "";
  }

  function setType(type) {
    if (
      !mediaConfig[type] ||
      analyseButton.disabled
    ) {
      return;
    }

    currentType = type;
    selectedFile = null;
    input.value = "";
    textInput.value = "";

    clearPreview();

    tabs.forEach(tab => {
      tab.classList.toggle(
        "active",
        tab.dataset.type === type
      );
    });

    document.getElementById(
      "moduleHeading"
    ).textContent = mediaConfig[type].title;

    if (type === "text") {
      dropZone.hidden = true;
      textInput.hidden = false;
    } else {
      dropZone.hidden = false;
      textInput.hidden = true;
      input.accept = mediaConfig[type].accept;

      document.getElementById(
        "formatHint"
      ).textContent = mediaConfig[type].hint;
    }
  }

  function handleFile(file) {
    clearPreview();
    selectedFile = null;

    if (!file) return;

    const extension = (
      file.name.split(".").pop() || ""
    ).toLowerCase();

    const allowedExtensions = {
      image: ["jpg", "jpeg", "png", "webp"],
      video: ["mp4", "webm"],
      audio: [
        "mp3", "wav", "m4a", "flac",
        "ogg", "opus", "aac", "webm"
      ]
    };

    const extensionAllowed =
      allowedExtensions[currentType]?.includes(extension);

    const mimeAllowed =
      mediaConfig[currentType].mime.includes(file.type);

    if (!extensionAllowed || (
      file.type && !mimeAllowed
    )) {
      status.textContent =
        "Unsupported file format.";
      return;
    }

    if (
      file.size === 0 ||
      file.size > MAX_SIZE
    ) {
      status.textContent =
        "File must be non-empty and no larger than 20 MB.";
      return;
    }

    selectedFile = file;
    previewURL = URL.createObjectURL(file);

    let element;

    if (currentType === "image") {
      element = document.createElement("img");
      element.alt = "Uploaded image preview";
    } else if (currentType === "video") {
      element = document.createElement("video");
      element.controls = true;
    } else {
      element = document.createElement("audio");
      element.controls = true;
    }

    element.src = previewURL;
    previewContent.replaceChildren(element);

    fileMeta.textContent =
      file.name + " · " +
      (file.size / 1024 / 1024).toFixed(2) +
      " MB";

    previewArea.hidden = false;
  }

  tabs.forEach(tab => {
    tab.addEventListener("click", () => {
      setType(tab.dataset.type);
    });
  });

  input.addEventListener("change", () => {
    handleFile(input.files[0]);
  });

  ["dragover", "dragenter"].forEach(name => {
    dropZone.addEventListener(name, event => {
      event.preventDefault();
      dropZone.classList.add("over");
    });
  });

  ["dragleave", "drop"].forEach(name => {
    dropZone.addEventListener(name, event => {
      event.preventDefault();
      dropZone.classList.remove("over");
    });
  });

  dropZone.addEventListener("drop", event => {
    const file = event.dataTransfer.files[0];
    handleFile(file);
  });

  analyseButton.addEventListener(
    "click",
    async () => {
      const form = new FormData();

      let endpoint;
      let name;
      let size = 0;

      if (currentType === "text") {
        const value = textInput.value.trim();

        if (!value) {
          status.textContent =
            "Please enter text first.";
          return;
        }

        name = "Text input";
        size = value.length;

        form.append("text", value);
        endpoint = "/detect/text";

      } else {
        if (!selectedFile) {
          status.textContent =
            "Please select a valid file.";
          return;
        }

        name = selectedFile.name;
        size = selectedFile.size;

        form.append(
          "detection_type",
          currentType
        );

        form.append("file", selectedFile);
        endpoint = "/detect/file";
      }

      analyseButton.disabled = true;
      analyseButton.textContent = "Processing...";

      status.textContent =
        "Connecting to Python backend...";

      let state = "Backend Unavailable";
      let prediction = null;
      let confidence = null;
      let explanation = "";

      const controller = new AbortController();

      const timeout = setTimeout(
        () => controller.abort(),
        120000
      );

      try {
        if (currentType !== "text") {
          await saveMedia(selectedFile);
        }

        const response = await fetch(
          API_URL + endpoint,
          {
            method: "POST",
            body: form,
            signal: controller.signal
          }
        );

        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            typeof data.detail === "string"
              ? data.detail
              : "Request rejected."
          );
        }

        const modelReady =
          data.model_status === "ready";

        if (
          modelReady &&
          data.prediction != null
        ) {
          prediction = String(data.prediction);
          confidence = data.confidence ?? null;

          explanation =
            typeof data.explanation === "string"
              ? data.explanation
              : "";

          state = "Model Analysis Completed";

        } else {
          state = "Input Received — Model Pending";
        }

      } catch (error) {
        state = error.name === "AbortError"
          ? "Backend Timed Out"
          : "Backend Unavailable";

        console.error(error);

      } finally {
        clearTimeout(timeout);
      }

      const record = {
        id: Date.now(),
        type: currentType,
        name,
        size,
        state,
        prediction,
        confidence,
        explanation,
        date: new Date().toISOString()
      };

      sessionStorage.setItem(
        RESULT_KEY,
        JSON.stringify(record)
      );

      try {
        const history = JSON.parse(
          localStorage.getItem(HISTORY_KEY) || "[]"
        );

        history.unshift(record);

        localStorage.setItem(
          HISTORY_KEY,
          JSON.stringify(history.slice(0, 20))
        );

      } catch (error) {
        console.warn(
          "History could not be saved:",
          error
        );
      }

      window.location.href = "result.html";
    }
  );

  const requestedType = new URLSearchParams(
    location.search
  ).get("type");

  setType(
    mediaConfig[requestedType]
      ? requestedType
      : "image"
  );

  window.addEventListener(
    "pagehide",
    clearPreview
  );
}

// =====================================
// RESULTS PAGE
// =====================================

async function renderResult() {
  let record = null;

  try {
    record = JSON.parse(
      sessionStorage.getItem(RESULT_KEY)
    );
  } catch (error) {
    console.warn(error);
  }

  if (!record) {
    layout(`
      <section class="panel empty-state">
        <h2>No Analysis Report Available</h2>

        <p>
          Upload a file or enter text in the
          detection workspace to begin.
        </p>

        <p style="margin-top:20px">
          <a class="btn btn-primary"
             href="detect.html">
            Start Detection
          </a>
        </p>
      </section>
    `);

    return;
  }

  const hasResult =
    record.prediction !== null;

  layout(`
    <section class="panel">
      <span class="eyebrow">
        AUTHENTICITY ASSESSMENT
      </span>

      <h2>Detection Analysis Report</h2>

      <p>
        Review the submitted content,
        processing status and available
        analysis explanation.
      </p>
    </section>

    <div class="result-layout">

      <section class="panel">
        <h2>Uploaded Content</h2>
        <p>Media information and preview.</p>

        <div id="resultPreview"
             class="result-preview">
          <p>Loading preview...</p>
        </div>

        <div class="result-row">
          <span>Content Name</span>
          <strong>
            ${escapeHTML(record.name)}
          </strong>
        </div>

        <div class="result-row">
          <span>Media Type</span>
          <strong>
            ${escapeHTML(
              record.type.toUpperCase()
            )}
          </strong>
        </div>

        <div class="result-row">
          <span>Submitted At</span>
          <strong>
            ${escapeHTML(
              new Date(record.date).toLocaleString()
            )}
          </strong>
        </div>
      </section>

      <section class="panel">
        <span class="eyebrow">
          ANALYSIS RESULT
        </span>

        <h2>Authenticity Assessment</h2>

        <div class="result-row">
          <span>Processing Status</span>
          <strong>
            ${escapeHTML(record.state)}
          </strong>
        </div>

        <div class="result-row">
          <span>AI Prediction</span>
          <strong>
            ${hasResult
              ? escapeHTML(record.prediction)
              : "Not Available"}
          </strong>
        </div>

        <div class="result-row">
          <span>Model Score</span>
          <strong>
            ${record.confidence != null
              ? escapeHTML(record.confidence)
              : "N/A"}
          </strong>
        </div>

        <div class="explanation">
          <h3>AI Detection Explanation</h3>

          <p id="explanationText">
            ${hasResult && record.explanation
              ? escapeHTML(record.explanation)
              : "A model explanation is not available."}
          </p>

          <p>
            Deepfake detectors can produce
            incorrect results. Model scores
            should not be interpreted as
            guaranteed probabilities.
          </p>
        </div>

        <div class="notice">
          ${hasResult
            ? "This result is an AI assessment, not definitive proof."
            : "No AI model prediction has been produced. Check the backend connection."}
        </div>

        <div style="margin-top:20px">
          <a class="btn btn-primary"
             href="detect.html">
            New Detection
          </a>

          <a class="btn btn-secondary"
             href="history.html">
            View History
          </a>
        </div>
      </section>

    </div>
  `);

  const container =
    document.getElementById("resultPreview");

  if (record.type === "text") {
    container.textContent =
      "Text input submitted (" +
      record.size +
      " characters).";
    return;
  }

  try {
    const file = await getMedia();

    if (
      !file ||
      file.name !== record.name ||
      file.size !== record.size
    ) {
      container.textContent =
        "Preview unavailable. Upload a new file.";
      return;
    }

    const url = URL.createObjectURL(file);
    let element;

    if (record.type === "image") {
      element = document.createElement("img");
      element.alt = "Submitted image";

    } else if (record.type === "video") {
      element = document.createElement("video");
      element.controls = true;

    } else {
      element = document.createElement("audio");
      element.controls = true;
    }

    element.src = url;
    container.replaceChildren(element);

    window.addEventListener(
      "pagehide",
      () => {
        URL.revokeObjectURL(url);
      },
      { once: true }
    );

  } catch (error) {
    container.textContent =
      "Local preview could not be loaded.";
  }
}

// =====================================
// HISTORY PAGE
// =====================================

function renderHistory() {
  let history = [];

  try {
    history = JSON.parse(
      localStorage.getItem(HISTORY_KEY) || "[]"
    );

    if (!Array.isArray(history)) {
      history = [];
    }

  } catch (error) {
    history = [];
  }

  const records = history.map(record => `
    <div class="history-item">
      <div>
        <strong>
          ${escapeHTML(record.name)}
        </strong>

        <small>
          ${escapeHTML(record.type)} ·
          ${escapeHTML(record.state)} ·
          ${escapeHTML(
            new Date(record.date).toLocaleString()
          )}
        </small>
      </div>

      <span class="tag">
        ${record.prediction === null
          ? "No Prediction"
          : escapeHTML(record.prediction)}
      </span>
    </div>
  `).join("");

  layout(`
    <section class="panel">
      <div class="panel-heading">
        <div>
          <span class="eyebrow">
            SUBMISSION RECORDS
          </span>

          <h2>Analysis History</h2>

          <p>
            View your most recent local submissions
            and backend processing statuses.
          </p>
        </div>

        <button class="btn btn-danger"
                id="clearHistory">
          Clear History
        </button>
      </div>

      <div class="history-list">
        ${records || `
          <div class="empty-state">
            <h3>No History Yet</h3>

            <p>
              Your submitted analyses will appear here.
            </p>

            <a class="btn btn-primary"
               href="detect.html"
               style="margin-top:20px">
              Start Detection
            </a>
          </div>
        `}
      </div>
    </section>
  `);

  document.getElementById("clearHistory")
    .addEventListener("click", async () => {

      if (
        !confirm("Clear your local analysis history?")
      ) {
        return;
      }

      localStorage.removeItem(HISTORY_KEY);
      sessionStorage.removeItem(RESULT_KEY);

      try {
        const db = await openDatabase();

        const tx = db.transaction(
          "media",
          "readwrite"
        );

        tx.objectStore("media").delete("latest");

        await new Promise((resolve, reject) => {
          tx.oncomplete = resolve;
          tx.onerror = () => reject(tx.error);
        });

        db.close();

      } catch (error) {
        console.warn(
          "Local media cleanup failed:",
          error
        );
      }

      window.location.reload();
    });
}


// =====================================
// GUARDBOT AI - FIFTH MODULE
// Uses the shared website layout
// =====================================

function renderGuardBot() {

  layout(`

    <div class="rg-guardbot-page">

      <!-- PAGE INTRODUCTION -->

      <section class="panel rg-guardbot-intro">

        <div class="rg-guardbot-intro-text">

          <span class="eyebrow">
            INTELLIGENT AI ASSISTANCE
          </span>

          <h1>GuardBot AI</h1>

          <p>
            Your intelligent digital safety assistant
            for understanding detection results,
            exploring deepfake technologies,
            and learning about RobustGuard AI.
          </p>

        </div>

        <div class="rg-guardbot-status">

          <div class="rg-guardbot-status-icon">
            ✦
          </div>

          <div>
            <strong>AI Assistant</strong>
            <small>Ready to Chat</small>
          </div>

        </div>

      </section>


      <!-- CHAT INTERFACE -->

      <section class="panel guardbot-chat-section">

        <div class="guardbot-chat-header">

          <div>

            <span class="eyebrow">
              INTERACTIVE ASSISTANT
            </span>

            <h2>Chat with GuardBot</h2>

            <p>
              Ask questions, explore system features,
              or understand your latest detection result.
            </p>

          </div>

          <button
            id="clearGuardBotChat"
            class="btn btn-secondary"
            type="button"
          >
            New Chat
          </button>

        </div>


        <!-- QUICK QUESTIONS -->

        <div class="guardbot-quick-actions">

          <button
            class="guardbot-chip"
            type="button"
            data-prompt="Give me an overview of RobustGuard AI and explain its features."
          >
            Project Overview
          </button>

          <button
            class="guardbot-chip"
            type="button"
            data-prompt="Explain the differences between image, video, audio and text detection in RobustGuard AI."
          >
            Detection Modules
          </button>

          <button
            class="guardbot-chip"
            type="button"
            data-prompt="Explain my latest detection result in simple terms. What does its prediction and model score mean?"
          >
            Explain My Result
          </button>

          <button
            class="guardbot-chip"
            type="button"
            data-prompt="Explain the AI models and technologies used in RobustGuard AI."
          >
            AI Technologies
          </button>

        </div>


        <!-- CHAT MESSAGES -->

        <div
          id="guardbotMessages"
          class="guardbot-messages"
          role="log"
          aria-live="polite"
          aria-label="GuardBot conversation"
        ></div>


        <!-- MESSAGE INPUT -->

        <form
          id="guardbotForm"
          class="guardbot-input-area"
        >

          <textarea
            id="guardbotInput"
            rows="2"
            maxlength="2000"
            placeholder="Ask GuardBot anything..."
            aria-label="Your message"
            required
          ></textarea>

          <div class="guardbot-input-actions">

            <span class="guardbot-tip">
              Press Enter to send.
              Shift + Enter for a new line.
            </span>

            <button
              type="submit"
              class="btn btn-primary guardbot-send-btn"
            >
              Send Message
            </button>

          </div>

        </form>

      </section>


      <!-- SAFETY INFORMATION -->

      <div class="rg-guardbot-note">

        <span>✧</span>

        <p>
          GuardBot provides AI-assisted explanations.
          Detection predictions are not definitive
          proof of authenticity or manipulation.
        </p>

      </div>

    </div>

  `);

}


// =====================================
// LOAD THE CORRECT PAGE
// =====================================

if (page === "home") {
  renderHome();

} else if (page === "detect") {
  renderDetect();

} else if (page === "result") {
  renderResult();


} else if (page === "history") {
  renderHistory();

} else if (page === "guardbot") {
  renderGuardBot();
}

console.log(
  "RobustGuard AI page loaded:",
  page
);
