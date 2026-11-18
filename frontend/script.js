
"use strict";

// 1. Get HTML elements

const homeSection = document.getElementById("homeSection");

const detectionSection = document.getElementById("detectionSection");

const startBtn = document.getElementById("startBtn");

const backBtn = document.getElementById("backBtn");

const detectionButtons = document.querySelectorAll(".detect-btn");

const uploadArea = document.getElementById("uploadArea");

const selectedType = document.getElementById("selectedType");

const fileInput = document.getElementById("fileInput");

const textInput = document.getElementById("textInput");

const fileName = document.getElementById("fileName");

const analyseBtn = document.getElementById("analyseBtn");

const resultMessage = document.getElementById("resultMessage");

// 2. Store selected detection type

let currentType = "";

// 3. Open detection workspace

startBtn.addEventListener("click", function () {

    homeSection.hidden = true;
    detectionSection.hidden = false;

    window.scrollTo(0, 0);

});

// 4. Select detection module

detectionButtons.forEach(function (button) {

    button.addEventListener("click", function () {

        currentType = button.dataset.type;

        uploadArea.hidden = false;

        selectedType.textContent =
            currentType.toUpperCase() + " Detection";

        fileInput.value = "";
        textInput.value = "";

        fileName.textContent = "";
        resultMessage.textContent = "";

        if (currentType === "text") {

            fileInput.hidden = true;
            textInput.hidden = false;

        } else {

            fileInput.hidden = false;
            textInput.hidden = true;

            if (currentType === "image") {
                fileInput.accept = "image/jpeg,image/png,image/webp";
            }

            if (currentType === "video") {
                fileInput.accept = "video/mp4,video/webm";
            }

            if (currentType === "audio") {
                fileInput.accept = "audio/mpeg,audio/wav,audio/ogg";
            }
        }

    });

});

// 5. Show selected file name

fileInput.addEventListener("change", function () {

    const selectedFile = fileInput.files[0];

    if (selectedFile) {

        fileName.textContent =
            "Selected File: " + selectedFile.name;

    } else {

        fileName.textContent = "";

    }

});

// 6. Analyse button (frontend test only)

analyseBtn.addEventListener("click", function () {

    if (currentType === "") {
        resultMessage.textContent =
            "Please select a detection module.";
        return;
    }

    if (currentType === "text") {

        const enteredText = textInput.value.trim();

        if (enteredText === "") {

            resultMessage.textContent =
                "Please enter some text first.";

            return;
        }

    } else {

        if (!fileInput.files.length) {

            resultMessage.textContent =
                "Please select a file first.";

            return;
        }
    }

    resultMessage.textContent =
        "Input received successfully! " +
        "AI detection will be connected in the backend phase.";

});

// 7. Return to home page

backBtn.addEventListener("click", function () {

    detectionSection.hidden = true;
    homeSection.hidden = false;

    uploadArea.hidden = true;

    currentType = "";

    fileInput.value = "";
    textInput.value = "";
    fileName.textContent = "";
    resultMessage.textContent = "";

    window.scrollTo(0, 0);

});

// Image Preview Feature

const imagePreviewArea =
    document.getElementById("imagePreviewArea");

const imagePreview =
    document.getElementById("imagePreview");

let imagePreviewURL = null;

// Reset old preview
function clearImagePreview() {

    if (imagePreviewURL) {
        URL.revokeObjectURL(imagePreviewURL);
        imagePreviewURL = null;
    }

    imagePreview.removeAttribute("src");
    imagePreviewArea.hidden = true;
}

// Show image when user selects a file
fileInput.addEventListener("change", function () {

    clearImagePreview();

    const file = fileInput.files[0];

    if (
        currentType === "image" &&
        file &&
        ["image/jpeg", "image/png", "image/webp"].includes(file.type)
    ) {

        imagePreviewURL = URL.createObjectURL(file);

        imagePreview.src = imagePreviewURL;

        imagePreviewArea.hidden = false;
    }

});

// Clear preview when changing module
detectionButtons.forEach(function (button) {

    button.addEventListener("click", function () {
        clearImagePreview();
    });

});

// Clear preview when going home
backBtn.addEventListener("click", function () {
    clearImagePreview();
});

// ====================================
// VIDEO AND AUDIO PREVIEW
// ====================================

// Get HTML elements
const videoPreviewArea =
    document.getElementById("videoPreviewArea");

const videoPreview =
    document.getElementById("videoPreview");

const audioPreviewArea =
    document.getElementById("audioPreviewArea");

const audioPreview =
    document.getElementById("audioPreview");

// Store temporary media URLs
let videoPreviewURL = null;
let audioPreviewURL = null;

// Clear video preview
function clearVideoPreview() {

    videoPreview.pause();

    videoPreview.removeAttribute("src");
    videoPreview.load();

    if (videoPreviewURL) {
        URL.revokeObjectURL(videoPreviewURL);
        videoPreviewURL = null;
    }

    videoPreviewArea.hidden = true;
}

// Clear audio preview
function clearAudioPreview() {

    audioPreview.pause();

    audioPreview.removeAttribute("src");
    audioPreview.load();

    if (audioPreviewURL) {
        URL.revokeObjectURL(audioPreviewURL);
        audioPreviewURL = null;
    }

    audioPreviewArea.hidden = true;
}

// Show video or audio after file selection
fileInput.addEventListener("change", function () {

    clearVideoPreview();
    clearAudioPreview();

    const file = fileInput.files[0];

    if (!file) {
        return;
    }

    // Video preview
    if (
        currentType === "video" &&
        ["video/mp4", "video/webm"].includes(file.type)
    ) {

        videoPreviewURL = URL.createObjectURL(file);

        videoPreview.src = videoPreviewURL;

        videoPreviewArea.hidden = false;
    }

    // Audio preview
    if (
        currentType === "audio" &&
        [
            "audio/mpeg",
            "audio/wav",
            "audio/x-wav",
            "audio/ogg"
        ].includes(file.type)
    ) {

        audioPreviewURL = URL.createObjectURL(file);

        audioPreview.src = audioPreviewURL;

        audioPreviewArea.hidden = false;
    }

});

// Clear previews when changing detection module
detectionButtons.forEach(function (button) {

    button.addEventListener("click", function () {

        clearVideoPreview();
        clearAudioPreview();

    });

});

// Clear previews when going back home
backBtn.addEventListener("click", function () {

    clearVideoPreview();
    clearAudioPreview();

});

// ====================================
// RESULT DASHBOARD
// ====================================

const resultDashboard =
    document.getElementById("resultDashboard");

const resultType =
    document.getElementById("resultType");

const resultFile =
    document.getElementById("resultFile");

const analysisStatus =
    document.getElementById("analysisStatus");

const aiPrediction =
    document.getElementById("aiPrediction");

const confidenceScore =
    document.getElementById("confidenceScore");

// Reset dashboard
function resetResultDashboard() {
    resultDashboard.hidden = true;

    resultType.textContent = "Not Selected";
    resultFile.textContent = "Not Available";
    analysisStatus.textContent = "Pending";
    aiPrediction.textContent = "Not Available";
    confidenceScore.textContent = "N/A";
}

// Show report when analyse button is clicked
analyseBtn.addEventListener("click", function () {

    if (!currentType) {
        resetResultDashboard();
        return;
    }

    let inputName = "";

    if (currentType === "text") {
        const text = textInput.value.trim();

        if (!text) {
            resetResultDashboard();
            return;
        }

        inputName = "Text (" + text.length + " characters)";

    } else {
        const file = fileInput.files[0];

        if (!file) {
            resetResultDashboard();
            return;
        }

        inputName = file.name;
    }

    resultType.textContent = currentType.toUpperCase();
    resultFile.textContent = inputName;
    analysisStatus.textContent = "AI Model Not Connected";
    aiPrediction.textContent = "Pending Integration";
    confidenceScore.textContent = "N/A";

    resultDashboard.hidden = false;
});

// Clear dashboard when selecting another module
detectionButtons.forEach(function (button) {
    button.addEventListener("click", resetResultDashboard);
});

// Clear dashboard when changing file
fileInput.addEventListener("change", resetResultDashboard);

// Clear dashboard when editing text
textInput.addEventListener("input", resetResultDashboard);

// Clear dashboard when returning home
backBtn.addEventListener("click", resetResultDashboard);
