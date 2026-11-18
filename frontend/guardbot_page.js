
"use strict";

// ==========================================
// ROBUSTGUARD AI - FULL GUARDBOT PAGE
// ==========================================

const GUARDBOT_API = "http://127.0.0.1:8000";

const GB_SESSION_KEY = "rg_guardbot_page_session_v1";
const GB_MESSAGES_KEY = "rg_guardbot_page_messages_v1";
const GB_RESULT_KEY = "rg_latest_result_v1";

const guardbotMessages =
  document.getElementById("guardbotMessages");

const guardbotForm =
  document.getElementById("guardbotForm");

const guardbotInput =
  document.getElementById("guardbotInput");

const clearGuardBotChat =
  document.getElementById("clearGuardBotChat");

const guardbotChips =
  document.querySelectorAll(".guardbot-chip");

const sendButton = guardbotForm.querySelector(
  'button[type="submit"]'
);

let guardbotSessionId =
  sessionStorage.getItem(GB_SESSION_KEY);

let guardbotBusy = false;


// ==========================================
// CHAT WELCOME
// ==========================================

const WELCOME_MESSAGE =
  "Hello! I'm GuardBot AI, your intelligent " +
  "digital safety assistant.\n\n" +
  "I can help you understand detection results, " +
  "explore AI technologies and learn about " +
  "RobustGuard AI.\n\n" +
  "How can I assist you today?";


// ==========================================
// CHAT MESSAGE DISPLAY
// ==========================================

function appendGuardBotMessage(
  role,
  text,
  save = true
) {

  const wrapper = document.createElement("div");

  wrapper.className =
    "guardbot-message " + role;

  const avatar = document.createElement("div");

  avatar.className = "guardbot-avatar";

  avatar.textContent =
    role === "user" ? "You" : "AI";

  const bubble = document.createElement("div");

  bubble.className = "guardbot-bubble";

  // Safe text, no HTML injection
  bubble.textContent = text;

  wrapper.appendChild(avatar);
  wrapper.appendChild(bubble);

  guardbotMessages.appendChild(wrapper);

  guardbotMessages.scrollTop =
    guardbotMessages.scrollHeight;

  if (save) {
    saveGuardBotMessage(role, text);
  }

  return wrapper;
}


// ==========================================
// SAVE CHAT IN CURRENT BROWSER TAB
// ==========================================

function saveGuardBotMessage(role, text) {

  try {

    const history = JSON.parse(
      sessionStorage.getItem(
        GB_MESSAGES_KEY
      ) || "[]"
    );

    const safeHistory =
      Array.isArray(history) ? history : [];

    safeHistory.push({
      role: role,
      text: text
    });

    // Store only last 40 messages
    sessionStorage.setItem(
      GB_MESSAGES_KEY,
      JSON.stringify(
        safeHistory.slice(-40)
      )
    );

  } catch (error) {
    console.warn(
      "Unable to save GuardBot messages.",
      error
    );
  }
}


// ==========================================
// RESTORE CONVERSATION AFTER REFRESH
// ==========================================

function restoreGuardBotMessages() {

  guardbotMessages.replaceChildren();

  try {

    const history = JSON.parse(
      sessionStorage.getItem(
        GB_MESSAGES_KEY
      ) || "[]"
    );

    if (
      Array.isArray(history) &&
      history.length > 0
    ) {

      history.forEach(item => {

        if (
          ["user", "bot"].includes(item.role) &&
          typeof item.text === "string"
        ) {

          appendGuardBotMessage(
            item.role,
            item.text,
            false
          );

        }

      });

      return;
    }

  } catch (error) {
    console.warn(
      "Unable to restore messages.",
      error
    );
  }

  appendGuardBotMessage(
    "bot",
    WELCOME_MESSAGE
  );
}


// ==========================================
// READ LATEST DETECTION RESULT
// ==========================================

function getLatestDetectionContext() {

  try {

    const raw = sessionStorage.getItem(
      GB_RESULT_KEY
    );

    if (!raw) return null;

    const result = JSON.parse(raw);

    if (
      result.state !==
        "Model Analysis Completed" ||
      !["Real", "Fake"].includes(
        result.prediction
      ) ||
      !["image", "video", "audio", "text"]
        .includes(result.type)
    ) {
      return null;
    }

    return {
      type: result.type,
      prediction: result.prediction,
      model_score:
        result.confidence ?? null,
      explanation: String(
        result.explanation || ""
      ).slice(0, 1200)
    };

  } catch (error) {

    console.warn(
      "Detection context unavailable.",
      error
    );

    return null;
  }
}


// ==========================================
// DETECT RESULT-RELATED QUESTIONS
// ==========================================

function isResultQuestion(message) {

  return (
    /result|prediction|score|confidence|my detection|my analysis|explain my|meri (audio|video|image|text|report)|mera (result|analysis)/i
      .test(message)
  );
}


// ==========================================
// ENABLE / DISABLE INPUT
// ==========================================

function setGuardBotBusy(value) {

  guardbotBusy = value;

  guardbotInput.disabled = value;
  sendButton.disabled = value;
  clearGuardBotChat.disabled = value;

  guardbotChips.forEach(chip => {
    chip.disabled = value;
  });

  sendButton.textContent =
    value ? "Thinking..." : "Send";
}


// ==========================================
// SEND MESSAGE TO BACKEND
// ==========================================

async function sendGuardBotMessage(question) {

  const message = question.trim();

  if (!message || guardbotBusy) {
    return;
  }

  const needsResult =
    isResultQuestion(message);

  const detectionContext =
    needsResult
      ? getLatestDetectionContext()
      : null;

  appendGuardBotMessage(
    "user",
    message
  );

  // If user asks about their result,
  // ensure an actual result exists.
  if (
    needsResult &&
    !detectionContext
  ) {

    appendGuardBotMessage(
      "bot",
      "I couldn't find a completed detection " +
      "result in this browser tab. " +
      "Please run a detection first, " +
      "then return to GuardBot AI."
    );

    return;
  }

  setGuardBotBusy(true);

  const typing = appendGuardBotMessage(
    "bot",
    "GuardBot is thinking...",
    false
  );

  try {

    const response = await fetch(
      GUARDBOT_API + "/guardbot/chat",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          message: message,
          session_id: guardbotSessionId,
          detection_context: detectionContext
        })
      }
    );

    const data = await response.json();

    if (!response.ok) {

      throw new Error(
        data.detail ||
        "GuardBot API error."
      );

    }

    if (
      data.status !== "success" ||
      typeof data.reply !== "string"
    ) {

      throw new Error(
        "Invalid GuardBot response."
      );

    }

    // Save server-side conversation ID
    guardbotSessionId = data.session_id;

    sessionStorage.setItem(
      GB_SESSION_KEY,
      guardbotSessionId
    );

    typing.remove();

    appendGuardBotMessage(
      "bot",
      data.reply
    );

  } catch (error) {

    typing.remove();

    console.error(
      "GuardBot Error:",
      error
    );

    appendGuardBotMessage(
      "bot",
      "Sorry, GuardBot couldn't respond. " +
      "Please check your backend connection " +
      "and try again."
    );

  } finally {

    setGuardBotBusy(false);

    guardbotInput.focus();

    guardbotMessages.scrollTop =
      guardbotMessages.scrollHeight;
  }
}


// ==========================================
// SUBMIT USER MESSAGE
// ==========================================

guardbotForm.addEventListener(
  "submit",
  async event => {

    event.preventDefault();

    if (guardbotBusy) return;

    const question =
      guardbotInput.value.trim();

    if (!question) return;

    guardbotInput.value = "";

    await sendGuardBotMessage(question);

  }
);


// ==========================================
// ENTER TO SEND / SHIFT+ENTER NEW LINE
// ==========================================

guardbotInput.addEventListener(
  "keydown",
  event => {

    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.isComposing
    ) {

      event.preventDefault();

      if (!guardbotBusy) {
        guardbotForm.requestSubmit();
      }

    }

  }
);


// ==========================================
// QUICK ACTION BUTTONS
// ==========================================

guardbotChips.forEach(chip => {

  chip.addEventListener(
    "click",
    async () => {

      if (guardbotBusy) return;

      const prompt =
        chip.dataset.prompt;

      if (prompt) {
        await sendGuardBotMessage(prompt);
      }

    }
  );

});


// ==========================================
// RESET CHAT
// ==========================================

clearGuardBotChat.addEventListener(
  "click",
  async () => {

    if (guardbotBusy) return;

    const oldSession =
      guardbotSessionId;

    guardbotSessionId = null;

    sessionStorage.removeItem(
      GB_SESSION_KEY
    );

    sessionStorage.removeItem(
      GB_MESSAGES_KEY
    );

    guardbotMessages.replaceChildren();

    appendGuardBotMessage(
      "bot",
      WELCOME_MESSAGE
    );

    guardbotInput.value = "";

    // Tell backend to clear old conversation
    if (oldSession) {

      try {

        await fetch(
          GUARDBOT_API + "/guardbot/clear",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              session_id: oldSession
            })
          }
        );

      } catch (error) {

        console.warn(
          "Previous session cleanup failed.",
          error
        );
      }
    }

    guardbotInput.focus();

  }
);


// ==========================================
// START GUARDBOT PAGE
// ==========================================

restoreGuardBotMessages();

console.log(
  "GuardBot full page initialized."
);
