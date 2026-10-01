const tabs = document.querySelectorAll(".tab");
const panels = document.querySelectorAll(".panel");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((item) => item.classList.toggle("active", item === tab));
    panels.forEach((panel) => {
      panel.classList.toggle("active", panel.id === tab.dataset.panel);
    });
  });
});

function displayNumber(value, digits = 2) {
  return typeof value === "number" ? value.toFixed(digits) : "--";
}

async function updateStatus() {
  const statusText = document.querySelector("#status-text");
  const statusDot = document.querySelector("#status-dot");
  try {
    const response = await fetch("/api/assessment", { cache: "no-store" });
    const state = await response.json();
    statusText.textContent = state.camera_connected
      ? state.detector_ready ? "Camera + detector connected" : "Camera connected"
      : "Camera unavailable";
    statusDot.className = state.camera_connected ? "online" : "error";
    document.querySelector("#fps").textContent = `${state.fps.toFixed(1)} FPS`;
    document.querySelector("#camera-error").textContent = state.error ?? "";
    document.querySelector("#risk-score").textContent = displayNumber(state.risk_score);
    document.querySelector("#covariance").textContent = displayNumber(state.covariance, 3);
    Object.entries(state.entity_counts).forEach(([name, count]) => {
      document.querySelector(`#${name}`).textContent = count;
    });
  } catch (error) {
    statusText.textContent = "Application disconnected";
    statusDot.className = "error";
    document.querySelector("#camera-error").textContent = error.message;
  }
}

updateStatus();
window.setInterval(updateStatus, 1000);
