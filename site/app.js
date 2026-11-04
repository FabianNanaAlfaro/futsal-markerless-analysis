const stages = [
  {
    id: "S01", label: "Pose routes", title: "Keep each observation route identifiable",
    summary: "Use manual DVIDEOW tracks as the reference and keep them separate from Sports2D, MediaPipe Pose and DeepLabCut predictions.",
    input: "Synchronized camera videos",
    output: "Pixel-space 2D landmarks",
    detail: "MediaPipe side selection is explicit. The five DLC checkpoints share the thesis's eight-point lower-limb schema.",
    doc: "docs/methods.md"
  },
  {
    id: "S02", label: "Homologation", title: "Map route-specific labels to four functional points",
    summary: "Normalize hip, knee, ankle and foot names while preserving frame index, camera, confidence and missing coordinates.",
    input: "Estimator-specific labels",
    output: "Common 2D CSV",
    detail: "Side-bearing labels are filtered using the selected kicking side. The pipeline keeps original pixel coordinates.",
    doc: "docs/code.md"
  },
  {
    id: "S03", label: "2D curation", title: "Process automated tracks before reconstruction",
    summary: "Apply the documented Hampel velocity mask, short-gap interpolation and light Savitzky–Golay smoothing to markerless tracks.",
    input: "Automated 2D observations",
    output: "Curated 2D tracks",
    detail: "Manual DVIDEOW reference coordinates bypass this automated 2D curation stage.",
    doc: "docs/methods.md"
  },
  {
    id: "S04", label: "3D DLT", title: "Reconstruct with researcher-supplied calibration",
    summary: "Triangulate homologous points from at least two camera views using 3×4 projection matrices or 11 DLT coefficients.",
    input: "Paired pixels + calibration JSON",
    output: "3D coordinates + reprojection errors",
    detail: "The calibration units define the reconstructed coordinate units.",
    doc: "docs/pipeline.md"
  },
  {
    id: "S05", label: "Kinematics", title: "Apply the common 3D temporal processing",
    summary: "Filter finite 3D segments and derive positions, vector speeds, sagittal knee and ankle angles, and angular velocities.",
    input: "Reconstructed 3D trajectories",
    output: "Per-frame kinematics",
    detail: "Axes, angle signs, sampling interval and the 18 Hz filter are documented in the methods page.",
    doc: "docs/methods.md"
  },
  {
    id: "S06", label: "Phase + bilateral", title: "Normalize the declared interval and compare sides",
    summary: "Map pre-impact to impact into 101 phase points and calculate signed dominant/non-dominant speed and angle metrics.",
    input: "Per-frame kinematics + frame range",
    output: "Normalized curves + bilateral metrics",
    detail: "The signed speed index follows the thesis equation 200 × (D − ND) / (D + ND).",
    doc: "docs/reproducibility-notes.md"
  },
  {
    id: "S07", label: "Evaluation", title: "Match each metric to its scientific question",
    summary: "Use coordinate agreement, trajectory association, peak errors, participant-cluster Bland–Altman intervals, p-value adjustment and optional SPM1D analysis.",
    input: "Paired reference and model outputs",
    output: "Metric summaries",
    detail: "The DeepLabCut selection partition is not an independent test set.",
    doc: "docs/limitations.md"
  }
];

const list = document.querySelector("#stage-list");
const detail = document.querySelector("#stage-detail");

function renderDetail(stage) {
  detail.innerHTML = `
    <div class="detail-kicker">${stage.id} · ${stage.label.toUpperCase()}</div>
    <h3 class="detail-title">${stage.title}</h3>
    <p class="detail-summary">${stage.summary}</p>
    <div class="detail-grid">
      <div class="detail-cell"><span>Input</span><p>${stage.input}</p></div>
      <div class="detail-cell"><span>Output</span><p>${stage.output}</p></div>
      <div class="detail-cell"><span>Method</span><p>${stage.detail}</p></div>
    </div>
    <a class="detail-link" href="https://github.com/FabianNanaAlfaro/futsal-markerless-analysis/blob/main/${stage.doc}">Read the documentation ↗</a>
  `;
}

function setActive(index) {
  const stage = stages[index];
  document.querySelectorAll(".stage-button").forEach((button, buttonIndex) => {
    button.setAttribute("aria-pressed", String(buttonIndex === index));
  });
  renderDetail(stage);
}

stages.forEach((stage, index) => {
  const button = document.createElement("button");
  button.className = "stage-button";
  button.type = "button";
  button.setAttribute("aria-pressed", index === 0 ? "true" : "false");
  button.innerHTML = `<span class="stage-id">${stage.id}</span><span class="stage-name">${stage.label}</span><span class="stage-arrow" aria-hidden="true">›</span>`;
  button.addEventListener("click", () => setActive(index));
  list.appendChild(button);
});

renderDetail(stages[0]);
