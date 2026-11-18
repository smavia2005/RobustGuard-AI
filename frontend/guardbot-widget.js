
"use strict";

// ==========================================
// ROBUSTGUARD AI - GUARDBOT CHAT WIDGET
// Uses existing FastAPI GuardBot endpoints
// ==========================================

(() => {
  const API = "http://127.0.0.1:8000";
  const SESSION_KEY = "rg_guardbot_session_v1";
  const RESULT_KEY = "rg_latest_result_v1";

  let sessionId = sessionStorage.getItem(SESSION_KEY);
  let isBusy = false;

  // ----------------------------------------
  // STYLES - MATCH EXISTING PINK THEME
  // ----------------------------------------

  const style = document.createElement("style");

  style.textContent = `
    .gb-launcher {
      position: fixed;
      right: 24px;
      bottom: 24px;
      width: 64px;
      height: 64px;
      border: none;
      border-radius: 50%;
      background: linear-gradient(
        135deg, #d36aa5, #9171c4
      );
      color: white;
      font-size: 27px;
      box-shadow: 0 10px 30px
        rgba(125, 74, 143, 0.28);
      cursor: pointer;
      z-index: 9998;
      transition: transform 0.2s;
    }

    .gb-launcher:hover {
      transform: translateY(-3px);
    }

    .gb-panel {
      position: fixed;
      right: 24px;
      bottom: 100px;
      width: 390px;
      height: 530px;
      max-height: 75vh;
      background: #fff;
      border: 1px solid #eee0ed;
      border-radius: 22px;
      box-shadow: 0 18px 55px
        rgba(75, 40, 85, 0.18);
      z-index: 9999;
      display: none;
      flex-direction: column;
      overflow: hidden;
      font-family: "Segoe UI", Arial, sans-serif;
    }

    .gb-panel.gb-open {
      display: flex;
    }

    .gb-header {
      padding: 18px;
      background: linear-gradient(
        110deg, #c75f9c, #9d7ac9
      );
      color: white;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .gb-brand {
      font-size: 17px;
      font-weight: 700;
    }

    .gb-subtitle {
      font-size: 11px;
      opacity: 0.92;
      margin-top: 3px;
    }

    .gb-header button {
      border: none;
      background: rgba(255,255,255,0.2);
      color: white;
      border-radius: 9px;
      padding: 7px 10px;
      cursor: pointer;
      font-size: 13px;
    }

    .gb-messages {
      flex: 1;
      overflow-y: auto;
      padding: 18px 14px;
      background: #fcf9fd;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .gb-message {
      max-width: 90%;
      padding: 11px 13px;
      border-radius: 15px;
      font-size: 13px;
      line-height: 1.6;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .gb-bot {
      align-self: flex-start;
      background: white;
      border: 1px solid #eadfee;
      color: #45384d;
      border-bottom-left-radius: 4px;
    }

    .gb-user {
      align-self: flex-end;
      background: #b45c99;
      color: white;
      border-bottom-right-radius: 4px;
    }

    .gb-input-area {
      display: flex;
      gap: 9px;
      padding: 13px;
      background: white;
      border-top: 1px solid #eee5ef;
    }

    .gb-input {
      flex: 1;
      min-width: 0;
      border: 1px solid #e4d7e8;
      border-radius: 12px;
      padding: 11px;
      font-size: 13px;
      outline: none;
      color: #403548;
      background: white;
    }

    .gb-input:focus {
      border-color: #ba6ca5;
      box-shadow: 0 0 0 3px
        rgba(186,108,165,0.12);
    }

    .gb-send {
      border: none;
      padding: 0 15px;
      border-radius: 11px;
      background: #b45c99;
      color: white;
      font-weight: 600;
      cursor: pointer;
    }

    .gb-send:disabled {
      opacity: 0.5;
      cursor: wait;
    }

    .gb-footer {
      text-align: center;
      font-size: 10px;
      padding: 7px;
      color: #827689;
      background: white;
    }

    @media (max-width: 480px) {
      .gb-panel {
        width: calc(100vw - 24px);
        height: min(560px, 78vh);
        right: 12px;
        bottom: 86px;
      }

      .gb-launcher {
        right: 16px;
        bottom: 16px;
        width: 56px;
        height: 56px;
      }
    }
  `;

  document.head.appendChild(style);

  // ----------------------------------------
  // CREATE CHAT INTERFACE
  // ----------------------------------------

  const launcher = document.createElement("button");
  launcher.className = "gb-launcher";
  launcher.textContent = "✦";
  launcher.type = "button";
  launcher.title = "Open GuardBot AI";
  launcher.setAttribute("aria-label", "Open GuardBot AI");

  const panel = document.createElement("section");
  panel.className = "gb-panel";
  panel.setAttribute("aria-label", "GuardBot AI chat");

  panel.innerHTML = `
    <div class="gb-header">
      <div>
        <div class="gb-brand">✦ GuardBot AI</div>
        <div class="gb-subtitle">
          Your Intelligent Digital Safety Assistant
        </div>
      </div>
      <div>
        <button type="button" id="gbReset"
          title="Start a new conversation">
          New
        </button>
        <button type="button" id="gbClose"
          aria-label="Close chat">
          ✕
        </button>
      </div>
    </div>

    <div class="gb-messages"
         id="gbMessages"
         role="log"
         aria-live="polite"></div>

    <form class="gb-input-area" id="gbForm">
      <input
        class="gb-input"
        id="gbInput"
        type="text"
        maxlength="2000"
        autocomplete="off"
        placeholder="Ask GuardBot anything..."
        aria-label="Your message"
        required>

      <button class="gb-send"
              id="gbSend"
              type="submit">
        Send
      </button>
    </form>

    <div class="gb-footer">
      AI predictions are not definitive proof.
    </div>
  `;

  document.body.appendChild(launcher);
  document.body.appendChild(panel);

  const messages = panel.querySelector("#gbMessages");
  const input = panel.querySelector("#gbInput");
  const sendButton = panel.querySelector("#gbSend");
  const form = panel.querySelector("#gbForm");

  // ----------------------------------------
  // DISPLAY MESSAGES SAFELY
  // ----------------------------------------

  function addMessage(text, sender) {
    const bubble = document.createElement("div");

    bubble.className =
      "gb-message " +
      (sender === "user" ? "gb-user" : "gb-bot");

    // textContent prevents HTML injection.
    bubble.textContent = text;

    messages.appendChild(bubble);
    messages.scrollTop = messages.scrollHeight;

    return bubble;
  }

  function showWelcome() {
    messages.replaceChildren();

    addMessage(
      "Assalam-o-Alaikum! I'm GuardBot AI.\n\n" +
      "I can answer your questions about AI, " +
      "deepfakes, digital safety and RobustGuard AI.\n\n" +
      "On the Result page, you can also ask me " +
      "to explain your latest detection.",
      "bot"
    );
  }

  // ----------------------------------------
  // READ LATEST DETECTION RESULT
  // ----------------------------------------

  function getDetectionContext() {
    // Only attach a result on result.html.
    if (document.body.dataset.page !== "result") {
      return null;
    }

    try {
      const raw = sessionStorage.getItem(RESULT_KEY);

      if (!raw) return null;

      const result = JSON.parse(raw);

      if (
        result.state !== "Model Analysis Completed" ||
        !["Real", "Fake"].includes(result.prediction) ||
        !["image", "video", "audio", "text"]
          .includes(result.type)
      ) {
        return null;
      }

      return {
        type: result.type,
        prediction: result.prediction,
        model_score: result.confidence ?? null,
        explanation: String(
          result.explanation || ""
        ).slice(0, 1200)
      };

    } catch (error) {
      console.warn(
        "GuardBot could not read detection context.",
        error
      );
      return null;
    }
  }

  // ----------------------------------------
  // SEND MESSAGE TO FASTAPI
  // ----------------------------------------

  async function sendMessage(question) {
    if (isBusy) return;

    const message = question.trim();
    if (!message) return;

    isBusy = true;
    sendButton.disabled = true;
    input.disabled = true;

    addMessage(message, "user");

    const loading = addMessage(
      "GuardBot is thinking...",
      "bot"
    );

    try {
      const response = await fetch(
        API + "/guardbot/chat",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            message: message,
            session_id: sessionId,
            detection_context: getDetectionContext()
          })
        }
      );

      if (!response.ok) {
        throw new Error(
          "GuardBot API returned " + response.status
        );
      }

      const data = await response.json();

      if (
        data.status !== "success" ||
        typeof data.reply !== "string"
      ) {
        throw new Error("Invalid chatbot response.");
      }

      sessionId = data.session_id;

      sessionStorage.setItem(
        SESSION_KEY,
        sessionId
      );

      loading.textContent = data.reply;

    } catch (error) {
      console.error("GuardBot error:", error);

      loading.textContent =
        "Sorry, I couldn't connect right now. " +
        "Please check that the backend is running " +
        "and try again.";
    } finally {
      isBusy = false;
      sendButton.disabled = false;
      input.disabled = false;
      input.focus();

      messages.scrollTop = messages.scrollHeight;
    }
  }

  // ----------------------------------------
  // BUTTON EVENTS
  // ----------------------------------------

  launcher.addEventListener("click", () => {
    panel.classList.toggle("gb-open");

    if (panel.classList.contains("gb-open")) {
      input.focus();
    }
  });

  panel.querySelector("#gbClose")
    .addEventListener("click", () => {
      panel.classList.remove("gb-open");
    });

  form.addEventListener("submit", event => {
    event.preventDefault();

    const question = input.value.trim();
    if (!question) return;

    input.value = "";
    sendMessage(question);
  });

  panel.querySelector("#gbReset")
    .addEventListener("click", async () => {
      if (isBusy) return;

      const oldSession = sessionId;

      sessionId = null;
      sessionStorage.removeItem(SESSION_KEY);
      showWelcome();

      if (oldSession) {
        try {
          await fetch(API + "/guardbot/clear", {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({
              session_id: oldSession
            })
          });
        } catch (error) {
          console.warn(
            "Old server session could not be cleared.",
            error
          );
        }
      }
    });

  showWelcome();

})();
