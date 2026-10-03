// Builds the Manufacturing Analytics project report (A4, single column).
// Numbers come from PAPER_RESULTS_PACKAGE.md, results/*.json|csv and the executed notebook
// (traceability: paper/CLAIM_EVIDENCE_MAP.md). Word fills the TOC / lists (see finalize_report.ps1).
const fs = require("fs");
const path = require("path");
const docx = require(process.env.DOCX_MODULE || "docx");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType, WidthType,
  BorderStyle, ShadingType, SectionType, VerticalAlign, Footer, PageNumber, NumberFormat, LevelFormat,
  TableOfContents, StyleLevel, HeadingLevel, TabStopType, PageBreak,
} = docx;

const ROOT = path.resolve(__dirname, "..");
const PFIG = (f) => path.join(ROOT, "paper", "figures", f);
const RFIG = (f) => path.join(__dirname, "figures", f);
const NBFIG = (f) => path.join(ROOT, "results", "figures", f);
const OMML = JSON.parse(fs.readFileSync(path.join(__dirname, "equations.omml.json"), "utf8"));

// ---------------------------------------------------------------- A4 geometry
const PAGE_W = 11906, PAGE_H = 16838;
const M_LEFT = 1800, M_RIGHT = 1440, M_TOP = 1440, M_BOTTOM = 1440;   // 1.25 in binding margin
const TEXT_W = PAGE_W - M_LEFT - M_RIGHT;                              // 8666 DXA = 6.02 in
const FONT = "Times New Roman";
const SZ = 24;   // 12 pt body

function pngSize(file) { const b = fs.readFileSync(file); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }

// inline markup: *italic*, **bold**, _{sub}, ^{sup}, [[placeholder]] (yellow highlight)
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\}|\[\[[^\]]+\]\])/g;
  let last = 0, m;
  const push = (t, extra = {}) => out.push(new TextRun({ text: t, font: FONT, ...base, ...extra }));
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) push(text.slice(last, m.index));
    const t = m[0];
    if (t.startsWith("**")) push(t.slice(2, -2), { bold: true });
    else if (t.startsWith("[[")) push(t.slice(2, -2), { highlight: "yellow" });
    else if (t.startsWith("*")) push(t.slice(1, -1), { italics: true });
    else if (t.startsWith("_{")) push(t.slice(2, -1), { subScript: true });
    else push(t.slice(2, -1), { superScript: true });
    last = m.index + t.length;
  }
  if (last < text.length) push(text.slice(last));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  children: runs(text, { size: SZ }), alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: 100, line: 276 }, keepNext: opts.keepNext,
});
const B = (text) => new Paragraph({
  children: runs(text, { size: SZ }), numbering: { reference: "bullets", level: 0 },
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 40, line: 276 },
});
const blank = (after = 120) => new Paragraph({ spacing: { after }, children: [] });

// ---------------------------------------------------------------- headings (chapter-numbered)
let ch = 0, sec = 0, figN = 0, tabN = 0, eqN = 0;
function H1(title, { numbered = true, first = false } = {}) {
  if (numbered) { ch += 1; sec = 0; figN = 0; tabN = 0; eqN = 0; }
  return new Paragraph({
    heading: HeadingLevel.HEADING_1, pageBreakBefore: !first, alignment: AlignmentType.CENTER,
    spacing: { after: 300 },
    children: [new TextRun({ text: (numbered ? `${ch}. ` : "") + title.toUpperCase(), font: FONT, bold: true, size: 28 })],
  });
}
function H2(title) {
  sec += 1;
  return new Paragraph({
    heading: HeadingLevel.HEADING_2, keepNext: true, spacing: { before: 200, after: 100 },
    children: [new TextRun({ text: `${ch}.${sec} ${title}`, font: FONT, bold: true, size: 24 })],
  });
}

// ---------------------------------------------------------------- figures / tables / equations
function Fig(file, caption, widthIn) {
  figN += 1;
  const { w, h } = pngSize(file);
  const dw = widthIn * 96, dh = dw * h / w;
  const svg = file.replace(/\.png$/, ".svg");
  const tr = { width: Math.round(dw), height: Math.round(dh) };
  const img = fs.existsSync(svg)
    ? new ImageRun({ type: "svg", data: fs.readFileSync(svg), fallback: { type: "png", data: fs.readFileSync(file) }, transformation: tr })
    : new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: tr });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 }, children: [img] }),
    new Paragraph({ style: "FigureCaption", children: [new TextRun({ text: `Figure ${ch}.${figN}: `, bold: true }), ...runs(caption)] }),
  ];
}

const line = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
function Tbl(caption, header, rows, colW, opts = {}) {
  tabN += 1;
  const total = colW.reduce((a, b) => a + b, 0);
  const cell = (t, head, i, j) => new TableCell({
    width: { size: colW[j], type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    shading: head ? { type: ShadingType.CLEAR, fill: "D9E2F3", color: "auto" } : undefined,
    borders: { top: line, bottom: line, left: line, right: line },
    children: [new Paragraph({
      keepNext: rows.length <= 8 && i < rows.length - 1,
      alignment: (opts.leftCols || [0]).includes(j) ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(t), { size: opts.size || 20, bold: head || (opts.boldRow && opts.boldRow(i)) }),
    })],
  });
  const trs = [new TableRow({ tableHeader: true, children: header.map((t, j) => cell(t, true, -1, j)) })];
  rows.forEach((r, i) => trs.push(new TableRow({ cantSplit: true, children: r.map((t, j) => cell(t, false, i, j)) })));
  const out = [
    new Paragraph({ style: "TableCaption", children: [new TextRun({ text: `Table ${ch}.${tabN}: `, bold: true }), ...runs(caption)] }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: colW, rows: trs, alignment: AlignmentType.CENTER }),
  ];
  out.push(opts.note
    ? new Paragraph({ spacing: { before: 60, after: 200 }, children: runs(opts.note, { size: 18, italics: true }) })
    : blank(200));
  return out;
}

const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const NUM_W = 900;
function Eq(name) {
  if (!OMML[name]) throw new Error("missing equation " + name);
  eqN += 1;
  const nb = { top: none, bottom: none, left: none, right: none };
  return [new Table({
    width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W - NUM_W, NUM_W],
    rows: [new TableRow({ cantSplit: true, children: [
      new TableCell({ width: { size: TEXT_W - NUM_W, type: WidthType.DXA }, borders: nb, verticalAlign: VerticalAlign.CENTER,
        margins: { top: 60, bottom: 60, left: 0, right: 0 },
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: `@@EQ:${name}@@`, font: FONT })] })] }),
      new TableCell({ width: { size: NUM_W, type: WidthType.DXA }, borders: nb, verticalAlign: VerticalAlign.CENTER,
        margins: { top: 60, bottom: 60, left: 0, right: 0 },
        children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: `(${ch}.${eqN})`, font: FONT, size: SZ })] })] }),
    ] })],
  }), blank(60)];
}

function Algorithm(title, lines) {
  const c = new TableCell({
    width: { size: TEXT_W, type: WidthType.DXA }, margins: { top: 100, bottom: 100, left: 160, right: 160 },
    borders: { top: line, bottom: line, left: line, right: line },
    children: [
      new Paragraph({ spacing: { after: 80 }, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000", space: 2 } },
        children: runs(title, { size: 21 }) }),
      ...lines.map(([ind, t]) => new Paragraph({ indent: { left: 300 * ind }, spacing: { after: 20 }, children: runs(t, { size: 20 }) })),
    ],
  });
  return [new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W], rows: [new TableRow({ cantSplit: true, children: [c] })] }), blank(200)];
}

// ================================================================ PRELIMINARY PAGES
const TITLE = "EXPLAINABLE DUAL-HEAD CNN–BiLSTM–ATTENTION PROGNOSTICS FOR INDUSTRIAL BEARING HEALTH MONITORING";
const TEAM = [["MOHAMMED FAISAL SM", "[[Register No.]]"], ["MOHAMED FAISAL A", "[[Register No.]]"], ["VISWESWAR GOPAL REDDY", "[[Register No.]]"]];
const C = (text, size = SZ, opts = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: opts.after ?? 60, line: 276 },
  children: runs(text, { size, bold: opts.bold, italics: opts.italics }) });

const titlePage = [
  blank(600),
  C("**" + TITLE + "**", 32, { after: 480 }),
  C("*A PROJECT REPORT*", 24, { after: 120 }),
  C("*Submitted by*", 24, { after: 200 }),
  ...TEAM.map(([n, r]) => C(`**${n}**   (${r})`, 24, { after: 80 })),
  blank(360),
  C("*in partial fulfilment of the requirements for the course*", 24, { after: 120 }),
  C("[[Course Code]] **– MANUFACTURING ANALYTICS**", 24, { after: 360 }),
  C("*of*", 24, { after: 120 }),
  C("**BACHELOR OF TECHNOLOGY**", 26, { after: 60 }),
  C("*in*", 24, { after: 60 }),
  C("**ARTIFICIAL INTELLIGENCE AND DATA SCIENCE**", 26, { after: 480 }),
  C("*Under the guidance of*", 24, { after: 80 }),
  C("**Selvarani K**", 24, { after: 40 }),
  C("Assistant Professor, Department of Artificial Intelligence and Data Science", 22, { after: 600 }),
  C("**DEPARTMENT OF ARTIFICIAL INTELLIGENCE AND DATA SCIENCE**", 24, { after: 60 }),
  C("**RAJALAKSHMI ENGINEERING COLLEGE**", 28, { after: 60 }),
  C("Rajalakshmi Nagar, Thandalam, Chennai – 602 105, Tamil Nadu, India", 22, { after: 200 }),
  C("[[Month Year]]", 24),
];

const noBorders = { top: none, bottom: none, left: none, right: none };
const sigCell = (lines) => new TableCell({ width: { size: TEXT_W / 2, type: WidthType.DXA }, borders: noBorders,
  margins: { left: 80, right: 80 },
  children: lines.map((t, i) => new Paragraph({ spacing: { after: i === 0 ? 700 : 40, line: 276 }, children: runs(t, { size: 22 }) })) });
const sigTable = (a, b) => new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W / 2, TEXT_W / 2],
  rows: [new TableRow({ children: [sigCell(a), sigCell(b)] })] });

const bonafide = [
  new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, spacing: { after: 360 },
    children: [new TextRun({ text: "BONAFIDE CERTIFICATE", font: FONT, bold: true, size: 28 })] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 480, line: 480 }, children: runs(
    `Certified that this project report titled **“Explainable Dual-Head CNN–BiLSTM–Attention Prognostics for Industrial Bearing Health Monitoring”** is the bonafide work of **Mohammed Faisal SM** ([[Register No.]]), **Mohamed Faisal A** ([[Register No.]]) and **Visweswar Gopal Reddy** ([[Register No.]]), who carried out this project work for the course [[Course Code]] **– Manufacturing Analytics** under my supervision.`,
    { size: SZ }) }),
  sigTable(
    ["**SIGNATURE**", "**Selvarani K**", "SUPERVISOR", "Assistant Professor", "Department of Artificial Intelligence and Data Science", "Rajalakshmi Engineering College", "Chennai – 602 105"],
    ["**SIGNATURE**", "[[Name of HoD]]", "HEAD OF THE DEPARTMENT", "Department of Artificial Intelligence and Data Science", "Rajalakshmi Engineering College", "Chennai – 602 105"]),
  blank(600),
  new Paragraph({ spacing: { after: 900 }, children: runs("Submitted for the project viva-voce examination held on [[Date]].", { size: SZ }) }),
  new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W / 2, TEXT_W / 2], rows: [new TableRow({ children: [
    new TableCell({ width: { size: TEXT_W / 2, type: WidthType.DXA }, borders: noBorders, children: [new Paragraph({ children: runs("**INTERNAL EXAMINER**", { size: SZ }) })] }),
    new TableCell({ width: { size: TEXT_W / 2, type: WidthType.DXA }, borders: noBorders, children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: runs("**EXTERNAL EXAMINER**", { size: SZ }) })] }),
  ] })] }),
];

const prelimHead = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, alignment: AlignmentType.CENTER,
  spacing: { after: 300 }, children: [new TextRun({ text: t, font: FONT, bold: true, size: 28 })] });

const acknowledgement = [
  prelimHead("ACKNOWLEDGEMENT"),
  P("We express our sincere thanks to the management of Rajalakshmi Engineering College and to our Principal for providing the facilities and the academic environment that made this project possible."),
  P("We thank the Head of the Department of Artificial Intelligence and Data Science for the encouragement and support given to us throughout the course."),
  P("We are deeply grateful to our project guide, **Selvarani K**, Assistant Professor, Department of Artificial Intelligence and Data Science, for her guidance, patience and valuable suggestions at every stage of this work. Her feedback helped us keep the evaluation honest and the report clear."),
  P("We also thank the FEMTO-ST Institute, France, for making the PRONOSTIA bearing run-to-failure data publicly available through the IEEE PHM 2012 Prognostic Challenge. This project would not have been possible without that open dataset."),
  P("Finally, we thank our families and friends for their constant support and motivation."),
  blank(600),
  ...TEAM.map(([n]) => new Paragraph({ alignment: AlignmentType.RIGHT, spacing: { after: 40 }, children: runs(`**${n}**`, { size: SZ }) })),
];

const abstract = [
  prelimHead("ABSTRACT"),
  P("Rolling-element bearings are among the most failure-prone parts of rotating machines, and an unexpected bearing failure can stop a production line. Predictive maintenance aims to replace a bearing shortly before it fails, which requires an estimate of its **remaining useful life (RUL)**. In this project we built and evaluated a deep-learning model that estimates the RUL of a bearing from its vibration signals and, at the same time, assigns it to one of three risk stages: Normal, Warning or Critical."),
  P("We used the public PRONOSTIA / IEEE PHM 2012 dataset, which contains 17 run-to-failure bearing experiments recorded under three operating conditions. From every 0.1-second vibration snapshot we extracted 30 time- and frequency-domain features (15 per accelerometer axis). The model reads a window of 16 consecutive snapshots (160 s of machine operation). It has three parts: a one-dimensional convolutional (CNN) block that learns local patterns, a bidirectional LSTM (BiLSTM) that models how those patterns change over time, and a temporal-attention layer that shows which time steps the model relies on most. Two output heads share this backbone: one predicts the RUL and the other predicts the risk stage. The RUL target is capped at 16,812 s, and both this cap and the feature scaling are computed only from the training bearings. As a result, no information from the test bearings is used during prediction."),
  P("The model has 113,924 parameters. It was trained on three bearings and tested, without any retraining, on all 11 test bearings (17,190 prediction windows). On these bearings it achieved an RUL mean absolute error (MAE) of 93.40 min, a root mean square error (RMSE) of 110.49 min, a normalized MAE of 0.3333 and a trajectory-averaged PHM score (T-Score) of 0.1667. The auxiliary risk-stage head reached 52.82% accuracy and a 40.22% macro F1-score. Under the official single-inspection protocol of the 2012 challenge, the score is 0.0234 with the published ground truth (0.0718 in a sensitivity check). These results show that the pipeline works end to end without data leakage. They also show clear limitations: RUL is under-estimated for long-lived bearings and over-estimated for short-lived ones, and the Warning stage is rarely recognized. The project is therefore a transparent academic baseline, not a system ready for industrial deployment."),
  new Paragraph({ spacing: { before: 120 }, children: runs("**Keywords:** predictive maintenance, bearing prognostics, remaining useful life, CNN–BiLSTM, temporal attention, multi-task learning, PRONOSTIA, manufacturing analytics.", { size: SZ }) }),
];

const tocPage = [
  new Paragraph({ pageBreakBefore: true, alignment: AlignmentType.CENTER, spacing: { after: 160 },
    children: [new TextRun({ text: "TABLE OF CONTENTS", font: FONT, bold: true, size: 28 })] }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
];
const listsPage = [
  prelimHead("LIST OF FIGURES"),
  new TableOfContents("List of Figures", { hyperlink: true, stylesWithLevels: [new StyleLevel("FigureCaption", 1)] }),
  new Paragraph({ heading: HeadingLevel.HEADING_1, alignment: AlignmentType.CENTER, spacing: { before: 480, after: 300 },
    children: [new TextRun({ text: "LIST OF TABLES", font: FONT, bold: true, size: 28 })] }),
  new TableOfContents("List of Tables", { hyperlink: true, stylesWithLevels: [new StyleLevel("TableCaption", 1)] }),
];

// ================================================================ MAIN CHAPTERS
const body = [];
const add = (...xs) => xs.forEach((x) => (Array.isArray(x) ? body.push(...x) : body.push(x)));

// ---------------------------------------------------------------- 1. INTRODUCTION
add(H1("Introduction", { first: true }));
add(H2("Background"));
add(P("Almost every manufacturing plant depends on rotating machines such as motors, pumps, fans, compressors, conveyors and machine-tool spindles. Inside each of these machines, rolling-element bearings carry the load of the rotating shaft while allowing it to turn with very little friction. Because bearings work continuously under load, their raceways and rolling elements slowly wear. Small surface cracks grow into pits and spalls, the bearing starts to vibrate more, and eventually it can no longer do its job. Bearing faults are widely recognized as one of the most common causes of breakdowns in rotating machinery [1], [2]."));
add(H2("Why Bearing Failures Matter"));
add(P("When a bearing fails without warning, the damage is rarely limited to the bearing itself. The machine stops, the production line waits, spare parts must be found in a hurry, and the shaft, housing or gears nearby may also be damaged. In a manufacturing setting this means lost output, missed delivery schedules and higher maintenance costs. On the other hand, replacing bearings too early wastes components that still have useful life left, along with the labour and downtime needed to change them. Getting the timing right therefore has direct economic value."));
add(H2("From Reactive to Predictive Maintenance"));
add(P("Maintenance strategies are usually grouped into three types [2]:"));
add(B("**Reactive (run-to-failure) maintenance:** the machine is repaired only after it breaks. It is simple, but every failure is unplanned."));
add(B("**Preventive (time-based) maintenance:** parts are replaced on a fixed schedule. This reduces surprises but ignores the actual condition of the part, so healthy parts are often thrown away."));
add(B("**Predictive (condition-based) maintenance:** the condition of the machine is monitored continuously, and maintenance is planned when the data show that failure is approaching."));
add(P("Predictive maintenance is a core application of manufacturing analytics, because it turns sensor data into maintenance decisions. Its value depends on two questions: *is the component degrading?* and *how much time is left before it fails?* This project focuses on the second question."));
add(H2("Vibration-Based Condition Monitoring"));
add(P("Vibration is the most widely used signal for monitoring bearing health [3]. An accelerometer mounted on the bearing housing records how strongly the machine vibrates. A healthy bearing produces low, fairly random vibration. As defects develop, every time a rolling element passes over a damaged spot it produces a small impact, so the signal becomes more impulsive. Later the overall vibration energy rises sharply. Simple statistics of the signal, such as the root mean square (RMS), the peak value and the kurtosis, together with frequency-domain measures from the Fourier transform, are therefore useful health indicators."));
add(H2("Remaining Useful Life Prediction"));
add(P("The **remaining useful life (RUL)** of a component is the time it can still operate before it reaches a defined failure condition. Predicting RUL is called *prognostics*. Older approaches rely on physical degradation models or reliability formulas, which need detailed knowledge of the failure mechanism [2], [4]. Data-driven approaches instead learn the relationship between sensor data and remaining life directly from recorded run-to-failure histories. Deep-learning models such as convolutional neural networks (CNNs) and long short-term memory (LSTM) networks have become popular for this task because they can learn useful patterns from sequences of sensor data [5]–[8]."));
add(H2("Motivation and Objectives"));
add(P("Three practical problems motivated this project. First, public bearing datasets are small, and bearings tested under the same load can last very different lengths of time, which makes RUL prediction hard. Second, it is easy to evaluate a model in a way that accidentally uses information about the test bearing's failure time, for example by normalizing with the test bearing's own lifetime. This makes results look much better than they really are [9]. Third, maintenance engineers are reluctant to trust a single number produced by an opaque model. The objectives of the project were therefore:"));
add(B("to build a complete pipeline from raw vibration files to RUL predictions on the PRONOSTIA dataset;"));
add(B("to design a dual-head CNN–BiLSTM–attention model that predicts both the RUL and a coarse risk stage;"));
add(B("to keep the whole pipeline free of test-set information (no data leakage);"));
add(B("to evaluate the frozen model on all 11 test bearings, both over complete trajectories and at the official challenge inspection points;"));
add(B("to examine the attention weights as a simple form of explanation, and to report the weaknesses of the model honestly."));
add(H2("Organization of the Report"));
add(P("Chapter 2 states the problem. Chapter 3 describes the dataset and the preprocessing steps. Chapter 4 explains the model and the methodology. Chapter 5 defines the metrics, and Chapter 6 reviews the results. Chapter 7 discusses the findings, limitations and future work, and lists the project resources needed to reproduce the work."));

// ---------------------------------------------------------------- 2. PROBLEM STATEMENT
add(H1("Problem Statement"));
add(H2("Problem Definition"));
add(P("Given the recent vibration history of a rolling-element bearing, the task is to estimate how much longer the bearing can run before it fails, and to indicate how urgent maintenance is. The estimate must be made from vibration data alone and must not use any information about when the test bearing actually failed. In the PRONOSTIA experiments, failure is defined as the point at which the vibration amplitude exceeds 20 g, where the test was stopped [10], [11]."));
add(P("Formally, for a bearing observed up to snapshot *j*, the model receives a window **X**_{j} of the last 16 feature vectors and produces two outputs:"));
add(B("**Primary output – RUL regression:** a continuous estimate of the remaining useful life, reported in seconds and minutes."));
add(B("**Auxiliary output – risk-stage classification:** one of three stages, *Normal*, *Warning* or *Critical*, defined by how much (capped) remaining life is left. The stage acts as a simple, readable alert level and also helps the model learn through multi-task training."));
add(...Tbl("Inputs and outputs of the prognostic system",
  ["Item", "Description"],
  [
    ["Raw input", "Horizontal and vertical acceleration, 2,560 samples per axis per snapshot (25.6 kHz, 0.1 s), one snapshot every 10 s"],
    ["Model input", "Window of 16 consecutive snapshots × 30 standardized features (160 s of operation)"],
    ["Output 1", "Normalized RUL in [0, 1], converted to seconds by multiplying with RUL_{cap} = 16,812 s"],
    ["Output 2", "Probabilities of the Normal, Warning and Critical risk stages (arg max = predicted stage)"],
    ["Explanation", "16 temporal attention weights showing how much each time step in the window contributed"],
  ],
  [1900, 6700], { size: 20 }));
add(H2("Scope and Constraints"));
add(B("Only the vibration signals are used; temperature data are not used."));
add(B("The training set contains just three bearings (one per operating condition), and three further bearings are used for validation."));
add(B("All preprocessing constants (feature scaler and RUL cap) must come from the training bearings only."));
add(B("The trained model is evaluated as a frozen model on 11 unseen test bearings, without any retraining or tuning on test data."));
add(B("The project is an offline, laboratory-data study. It is not intended to replace industrial safety or protection systems."));

// ---------------------------------------------------------------- 3. DATA SET
add(H1("Data Set Description & Preprocessing"));
add(H2("The PRONOSTIA / IEEE PHM 2012 Dataset"));
add(P("The data come from PRONOSTIA, an accelerated bearing-degradation test rig built at the FEMTO-ST Institute in France [10]. The dataset was released for the IEEE PHM 2012 Prognostic Challenge [11]. In each experiment, a new ball bearing was run at constant speed under a constant radial load until it failed. Two accelerometers mounted at 90° to each other on the bearing housing recorded vibration in the horizontal and vertical directions. Every 10 s the system stored a 0.1 s burst of 2,560 samples per axis at a sampling frequency of 25.6 kHz. Each burst is saved as one CSV file (a *snapshot*) holding a time stamp and the horizontal and vertical acceleration."));
add(...Tbl("Operating conditions of the PRONOSTIA experiments",
  ["Condition", "Speed (rpm)", "Radial load (N)", "Learning bearings", "Test bearings"],
  [
    ["Condition 1", "1800", "4000", "Bearing1_1, Bearing1_2", "Bearing1_3 – Bearing1_7"],
    ["Condition 2", "1650", "4200", "Bearing2_1, Bearing2_2", "Bearing2_3 – Bearing2_7"],
    ["Condition 3", "1500", "5000", "Bearing3_1, Bearing3_2", "Bearing3_3"],
  ],
  [1400, 1200, 1500, 2300, 2200], { size: 20 }));
add(H2("Bearing Data Organization and Dataset Split"));
add(P("The dataset folder contains three sub-folders: *Learning_set* (6 complete run-to-failure histories), *Test_set* (the 11 test histories truncated before failure, as given to challenge participants) and *Full_Test_Set* (the same 11 histories run until failure). Altogether the 17 bearings contain 24,889 snapshots. We split the data **by bearing**, not by random windows, so that no bearing appears in more than one split. Table 3.2 lists every bearing and its role. The lifetimes vary from 0.64 h to 7.78 h, even within the same operating condition."));
add(...Tbl("Bearing inventory, lifetimes and roles in this project",
  ["Bearing", "Cond.", "Snapshots", "Lifetime (s)", "Lifetime (h)", "Role"],
  [
    ["Bearing1_1", "1", "2,803", "28,020", "7.78", "Training"],
    ["Bearing2_1", "2", "911", "9,100", "2.53", "Training"],
    ["Bearing3_1", "3", "515", "5,140", "1.43", "Training"],
    ["Bearing1_2", "1", "871", "8,700", "2.42", "Validation"],
    ["Bearing2_2", "2", "797", "7,960", "2.21", "Validation"],
    ["Bearing3_2", "3", "1,637", "16,360", "4.54", "Validation"],
    ["Bearing1_3 / 1_4 / 1_5", "1", "2,375 / 1,428 / 2,463", "23,740 / 14,270 / 24,620", "6.59 / 3.96 / 6.84", "Test"],
    ["Bearing1_6 / 1_7", "1", "2,448 / 2,259", "24,470 / 22,580", "6.80 / 6.27", "Test"],
    ["Bearing2_3 / 2_4 / 2_5", "2", "1,955 / 751 / 2,311", "19,540 / 7,500 / 23,100", "5.43 / 2.08 / 6.42", "Test"],
    ["Bearing2_6 / 2_7", "2", "701 / 230", "7,000 / 2,290", "1.94 / 0.64", "Test"],
    ["Bearing3_3", "3", "434", "4,330", "1.20", "Test"],
  ],
  [2250, 650, 1800, 1850, 1350, 750], { size: 19, note: "Lifetime = (number of snapshots − 1) × 10 s. Test lifetimes are taken from Full_Test_Set." }));
add(P("Figure 3.1 shows two raw snapshots of the training bearing Bearing1_1. Early in life the vibration peaks at about 1.7 g. At the end of the test it exceeds the 20 g stopping limit (31.17 g horizontal and 37.19 g vertical)."));
add(...Fig(PFIG("fig_raw.png"), "Raw horizontal and vertical vibration of Bearing1_1 at snapshot 10 (early life) and snapshot 2800 (end of test)", 5.0));
add(H2("Feature Extraction"));
add(P("A raw snapshot has 5,120 values, which is too many to feed directly to the model for every time step. Instead, each snapshot is summarized by 15 features per axis, or **30 features in total**. The features, listed in Table 3.3, were chosen because they are standard indicators of bearing condition [3]. Time-domain features describe the energy and the impulsiveness of the signal, and frequency-domain features, computed from the FFT of the mean-removed signal, describe how the vibration energy is distributed over frequency (0–12.8 kHz). Kurtosis is computed as excess kurtosis (0 for a Gaussian signal), and a constant of 10^{−8} guards against division by zero."));
add(...Tbl("The 15 features computed for each axis (30 features per snapshot)",
  ["No.", "Feature", "Domain", "Definition / meaning"],
  [
    ["1", "RMS", "Time", "√(mean of x²) – overall vibration energy"],
    ["2", "Peak", "Time", "max |x| – largest impact"],
    ["3", "Peak-to-peak", "Time", "max x − min x – total signal range"],
    ["4", "Standard deviation", "Time", "spread of the signal around its mean"],
    ["5", "Kurtosis", "Time", "4th standardized moment − 3 – impulsiveness"],
    ["6", "Skewness", "Time", "3rd standardized moment – asymmetry"],
    ["7", "Crest factor", "Time", "Peak / RMS"],
    ["8", "Shape factor", "Time", "RMS / mean |x|"],
    ["9", "Margin factor", "Time", "Peak / (mean √|x|)²"],
    ["10", "Impulse factor", "Time", "Peak / mean |x|"],
    ["11", "Spectral energy", "Frequency", "Σ s_{k}² – total spectral power"],
    ["12", "Spectral centroid", "Frequency", "Σ f_{k}s_{k} / Σ s_{k} – mean frequency"],
    ["13", "Spectral spread", "Frequency", "spread of the spectrum around the centroid"],
    ["14", "Peak frequency", "Frequency", "frequency of the largest spectral component"],
    ["15", "RMS frequency", "Frequency", "√(Σ f_{k}²s_{k} / Σ s_{k})"],
  ],
  [650, 2100, 1300, 4550], { size: 19, leftCols: [1, 3] }));
add(P("Figure 3.2 shows three of these features over the life of Bearing1_1. RMS stays almost flat for several hours, rises slowly after about 4 h and jumps sharply at the very end. Kurtosis shows isolated spikes as early as the first 2–3 h, and the spectral centroid moves up and down without a simple trend. No single feature tracks degradation cleanly, so the model combines all 30 features over time."));
add(...Fig(PFIG("fig_features_b11.png"), "Evolution of (a) RMS, (b) horizontal kurtosis and (c) horizontal spectral centroid over the life of Bearing1_1", 5.0));
add(H2("Feature Scaling Using Training Data Only"));
add(P("The 30 features have very different ranges; spectral energy, for example, is many orders of magnitude larger than skewness. They are therefore standardized with a StandardScaler. Its mean *μ* and standard deviation *σ* for each feature are estimated **only from the 4,229 snapshots of the three training bearings**, saved as *models/scaler.pkl*, and applied unchanged to the validation and test bearings:"));
add(...Eq("eq_std"));
add(H2("RUL Target Generation and the Training-Derived Cap"));
add(P("For a snapshot recorded at operating time *t*, the true remaining life is *T*_{EOL} − *t*, where *T*_{EOL} is the time of the bearing's last snapshot. Early in life, however, the vibration carries almost no information about when failure will happen, so asking the model to predict very large RUL values from healthy-looking signals is not meaningful. We therefore use a **piecewise-capped** target, a common practice in prognostics [7]. The cap is 60% of the longest training lifetime (Bearing1_1, 28,020 s):"));
add(...Eq("eq_cap"));
add(...Eq("eq_target"));
add(...Eq("eq_ynorm"));
add(P("The normalized target *y*_{rul} lies between 0 and 1. Because the cap comes from training bearings only, it is a fixed constant that is valid for unseen bearings. Figure 3.3 shows the capped target for two training bearings."));
add(...Fig(PFIG("fig_target.png"), "Piecewise-capped RUL target for two training bearings, with the risk-stage thresholds", 2.6));
add(H2("Risk-Stage Labels"));
add(P("The auxiliary risk stage is obtained by thresholding the normalized target:", { keepNext: true }));
add(...Eq("eq_stage"));
add(P("The boundaries correspond to 112.1 min and 42.0 min of remaining life, so the stages are *RUL horizons*, not ISO vibration-severity zones."));
add(H2("Temporal Window Construction"));
add(P("For each bearing separately, sliding windows of **16 consecutive snapshots** (160 s) are built with a stride of 1:"));
add(...Eq("eq_window"));
add(P("A window takes the RUL target and risk stage of its last snapshot. Windows never mix two bearings, and a bearing with *N* snapshots gives *N* − 15 windows."));
add(H2("Outcome of Preprocessing"));
add(P("Table 3.4 summarizes the data that leave the preprocessing stage (each window has shape 16 × 30; the 11 test bearings contain 17,355 snapshots). Normal windows dominate, so the classes are imbalanced. No validation window is capped (every validation bearing lived less than 16,812 s), whereas 26.4% of the training windows (all from Bearing1_1) and 19.8% of the test-subset windows are capped."));
add(...Tbl("Windows produced by preprocessing and their risk-stage distribution",
  ["Split", "Bearings", "Windows", "Normal", "Warning", "Critical"],
  [
    ["Training", "3", "4,184", "2,338", "1,087", "759"],
    ["Validation", "3", "3,260", "1,241", "1,260", "759"],
    ["Test (subset: 1_3, 2_3, 3_3)", "3", "4,719", "2,954", "1,006", "759"],
    ["Test (all)", "11", "17,190", "10,499", "3,946", "2,745"],
  ],
  [2900, 1000, 1150, 1150, 1150, 1150], { size: 20 }));

// ---------------------------------------------------------------- 4. MODEL DEVELOPMENT & METHODOLOGY
add(H1("Model Development & Methodology"));
add(H2("Overall System and Workflow"));
add(P("Figure 4.1 gives a high-level block diagram of the system, and Figure 4.2 shows the complete workflow followed in the project. The left column of the flowchart covers data preparation (Chapter 3). The right column covers training, frozen inference, conversion of predictions to physical time, and the three kinds of evaluation. The only quantities carried from training into testing are the scaler statistics, the trained weights and the constant *RUL*_{cap}."));
add(...Fig(RFIG("fig_system_block.png"), "High-level block diagram of the bearing prognostics system", 6.0));
add(...Fig(PFIG("fig_flowchart.png"), "Complete project workflow: data preparation (left) and modelling and evaluation (right)", 5.0));
add(H2("Model Selection and Justification"));
add(P("The input to the model is a short multivariate time series: 16 time steps, each with 30 features. We chose a hybrid architecture because each component addresses a specific need of this input:"));
add(B("**1-D convolution (CNN):** a convolution with kernel size 3 combines each snapshot with its immediate neighbours across all 30 features. This lets the model detect short, local patterns, such as a sudden rise in kurtosis together with the peak value, while sharing weights along the time axis [5], [6]."));
add(B("**Bidirectional LSTM:** LSTM cells [12] keep a memory across time steps and are well suited to modelling how degradation develops. The bidirectional version [13] reads the window both forwards and backwards. This is allowed here because the whole 160 s window has already been recorded when a prediction is made."));
add(B("**Temporal attention:** instead of using only the last LSTM state, an attention layer [14], [15] learns a weight for every time step and forms a weighted summary. The weights also give a simple view of which part of the window the model relied on."));
add(B("**Two output heads (multi-task learning):** predicting the risk stage together with the RUL encourages the shared layers to learn features that are useful for both tasks, which can act as a regularizer when data are scarce [16]. It also gives the user an easy-to-read alert level."));
add(P("Because only three training bearings were available, we deliberately kept the model small (113,924 parameters) and used dropout and batch normalization. We did not experimentally compare this design with other architectures in this project. The choices above are therefore design decisions supported by the literature, not results proven by our own experiments."));
add(H2("CNN Feature Extractor"));
add(P("The standardized window (16 × 30) first passes through two identical convolutional blocks. Each block has a Conv1D layer with 64 filters, kernel size 3, 'same' padding and ReLU activation, followed by batch normalization [17] and dropout with rate 0.2 [18]. 'Same' padding keeps all 16 time steps, so the output of each block is a 16 × 64 feature map."));
add(H2("Bidirectional LSTM"));
add(P("The feature map is passed to a bidirectional LSTM with 64 units in each direction and *return_sequences = True*. At every time step *t* the forward and backward hidden states are concatenated into a 128-dimensional vector **h**_{t}, giving a 16 × 128 output, followed by dropout of 0.2."));
add(H2("Temporal Attention Layer"));
add(P("The custom *TemporalAttention* layer scores each hidden state with a small learned network, turns the scores into weights that add up to 1 with a softmax over time, and computes a weighted sum called the context vector **c**:"));
add(...Eq("eq_att"));
add(P("Here **W**_{a} (128 × 128), **b**_{a} and **u** (128) are learned parameters. The layer returns both **c** (128 values) and the 16 weights *α*_{t}. The weights are what we inspect for explainability in Chapter 6."));
add(H2("Dual-Head Architecture"));
add(P("The context vector feeds a shared dense layer of 64 ReLU units with dropout 0.2. The network then splits into two heads. Figure 4.3 shows the complete architecture exactly as implemented, and Table 4.1 lists every layer with its output shape and number of parameters."));
add(...Fig(PFIG("fig_architecture.png"), "Architecture of the implemented dual-head CNN–BiLSTM–attention model (dashed boxes: attention output and post-processing outside the network)", 5.8));
add(...Tbl("Layer-wise configuration of the model (output shapes exclude the batch dimension)",
  ["Layer (Keras name)", "Configuration", "Output shape", "Parameters"],
  [
    ["Input (sequence_input)", "16 time steps × 30 features", "(16, 30)", "0"],
    ["conv1 + bn1 + drop1", "Conv1D 64, k = 3, ReLU; BatchNorm; Dropout 0.2", "(16, 64)", "5,824 + 256"],
    ["conv2 + bn2 + drop2", "Conv1D 64, k = 3, ReLU; BatchNorm; Dropout 0.2", "(16, 64)", "12,352 + 256"],
    ["bilstm + drop_bilstm", "Bidirectional LSTM, 64 units per direction; Dropout 0.2", "(16, 128)", "66,048"],
    ["temporal_attention", "W_{a} 128 × 128, b_{a}, u", "(128) and (16, 1)", "16,640"],
    ["shared_dense + drop_shared", "Dense 64, ReLU; Dropout 0.2", "(64)", "8,256"],
    ["rul_dense → rul_output", "Dense 32, ReLU → Dense 1, linear", "(1)", "2,080 + 33"],
    ["risk_dense → risk_output", "Dense 32, ReLU → Dense 3, softmax", "(3)", "2,080 + 99"],
    ["**Total**", "113,668 trainable + 256 non-trainable", "", "**113,924**"],
  ],
  [2350, 3350, 1450, 1450], { size: 19, leftCols: [0, 1] }));
add(H2("RUL Regression Head"));
add(P("The RUL head is a dense layer of 32 ReLU units followed by a single linear output unit, *ŷ*_{rul}. During training this output is compared with the normalized target *y*_{rul}. At prediction time the output is clipped to the range [0, 1] and converted back to seconds with the training constant:"));
add(...Eq("eq_denorm"));
add(P("The lifetime of the bearing being tested never appears in this conversion. It is used only afterwards, to compute the true labels for the error metrics."));
add(H2("Auxiliary Risk-Stage Head"));
add(P("The risk head is a dense layer of 32 ReLU units followed by a 3-unit softmax layer that outputs the probabilities of the Normal, Warning and Critical stages. The predicted stage is the class with the highest probability. Because the stage labels are derived from the RUL target, this head does not add new information about the physical condition of the bearing. Its purposes are to regularize the shared layers and to give a readable alert level."));
add(H2("Loss Functions"));
add(P("The two heads are trained together by minimizing a weighted sum of two losses over a mini-batch of *B* windows:"));
add(...Eq("eq_loss"));
add(...Eq("eq_huber"));
add(P("The first term is the **Huber loss** [19] on the normalized RUL. It behaves like the squared error for small errors and like the absolute error for large ones, so the sudden jumps in late-life vibration do not dominate training. The second term is the **sparse categorical cross-entropy** of the risk head. The weights 1.0 and 0.5 make RUL prediction the main task. They were fixed by hand and not tuned."));
add(H2("Training Process"));
add(P("The model was implemented in Python 3.12.6 with TensorFlow 2.20.0 / Keras 3, NumPy 2.2.6, pandas 2.2.2 and scikit-learn, with all random seeds set to 42. Table 4.2 lists the training settings. The validation bearings were used only for early stopping and for reducing the learning rate."));
add(...Tbl("Training configuration",
  ["Setting", "Value"],
  [
    ["Optimizer / initial learning rate", "Adam [20] / 1 × 10^{−3}"],
    ["Batch size / maximum epochs", "64 / 35"],
    ["Loss weights", "RUL (Huber) 1.0, risk stage (cross-entropy) 0.5"],
    ["Early stopping", "monitor validation loss, patience 7, restore best weights"],
    ["Learning-rate schedule", "ReduceLROnPlateau: × 0.5 after 3 epochs without improvement, minimum 10^{−5}"],
    ["Regularization", "Dropout 0.2 after every block; batch normalization after each Conv1D"],
    ["Training time", "13.45 s wall-clock (reported in the notebook output)"],
  ],
  [3300, 5300], { size: 20, leftCols: [0, 1] }));
add(P("Figure 4.4 shows what happened during training. The training loss fell steadily, from 0.2339 to 0.0202, and training stage accuracy rose to 98.97%. The validation loss, however, was lowest after the first epoch (0.7702) and rose afterwards. Early stopping therefore ended training after epoch 8 and restored the epoch-1 weights, which form the frozen model evaluated in Chapter 6 (validation stage accuracy stayed between 34.97% and 39.20% throughout). This gap between training and validation shows that the model fits the three training bearings much better than it transfers to new bearings. We think a large part of the reason is that bearings under the same condition live very different lengths of time (for example, 7.78 h for Bearing1_1 versus 2.42 h for Bearing1_2)."));
add(...Fig(PFIG("fig_training.png"), "Training convergence: (a) total loss, (b) normalized RUL MAE, (c) stage accuracy for training and validation bearings", 5.8));
add(H2("Algorithm"));
add(P("Algorithm 4.1 summarizes the complete procedure, from feature extraction to evaluation."));
add(...Algorithm("**Algorithm 4.1:** Leakage-free training and evaluation of the dual-head model", [
  [0, "**Input:** training bearings (1_1, 2_1, 3_1), validation bearings (1_2, 2_2, 3_2), 11 test bearings; window length W = 16"],
  [0, "**Output:** frozen model; predicted RUL and risk stage for every test window; evaluation metrics"],
  [0, "1.  **for** every snapshot of every bearing **do** compute the 30 features (Table 3.3)"],
  [0, "2.  T_{max} ← longest training lifetime (28,020 s);  RUL_{cap} ← 0.6 × T_{max} = 16,812 s"],
  [0, "3.  compute the capped, normalized RUL target and the risk stage of every snapshot"],
  [0, "4.  fit the StandardScaler on training snapshots only; apply it to all bearings"],
  [0, "5.  **for** each bearing separately **do** build sliding windows of 16 snapshots (stride 1)"],
  [0, "6.  build the dual-head CNN–BiLSTM–attention model (Table 4.1); Adam, learning rate 10^{−3}"],
  [0, "7.  **for** epoch = 1 … 35 **do**"],
  [1, "8.   train on training windows (batch 64) with loss = 1.0 · Huber + 0.5 · cross-entropy"],
  [1, "9.   compute validation loss; halve learning rate after 3 epochs without improvement"],
  [1, "10.  **if** no improvement for 7 epochs **then** stop"],
  [0, "11. restore the weights with the lowest validation loss and freeze the model"],
  [0, "12. **for** every window of every test bearing **do**"],
  [1, "13.  (ŷ_{rul}, p̂, α) ← model(window)"],
  [1, "14.  predicted RUL ← clip(ŷ_{rul}, 0, 1) × RUL_{cap};  predicted stage ← arg max p̂"],
  [0, "15. compute trajectory metrics, single-inspection challenge score and attention plots"],
]));

// ---------------------------------------------------------------- 5. METRICS
add(H1("Metrics to Measure"));
add(H2("Choice of Metrics"));
add(P("The project has two outputs, so it needs two families of metrics. RUL predictions are judged by how far they are from the true remaining life, both in physical time (minutes) and in normalized units. The official PHM scoring function is also used, because an over-estimate of RUL is more dangerous in maintenance than an under-estimate. Risk-stage predictions are judged with standard classification metrics. All RUL errors are measured against the **capped** ground truth *RUL*_{c}, over *M* evaluated windows."));
add(H2("Mean Absolute Error and Root Mean Square Error"));
add(...Eq("eq_mae"));
add(...Eq("eq_rmse"));
add(P("**MAE** is the average size of the error, in the same unit as RUL (we report minutes). It is easy to interpret for maintenance planning. **RMSE** squares the errors before averaging, so it gives more weight to large mistakes. RMSE is always at least as large as MAE."));
add(H2("Normalized MAE"));
add(P("The **normalized MAE** is the same average error computed on the normalized values *y*_{rul} and *ŷ*_{rul}, i.e. as a fraction of the RUL cap (both lie in [0, 1]). A normalized MAE of 0.33 means the average error is one third of the 280.2-minute capped range. This makes it easier to compare bearings with very different lifetimes."));
add(H2("PHM Scoring Function and Trajectory T-Score"));
add(P("The IEEE PHM 2012 challenge defined an asymmetric accuracy score for each prediction [11]. The percent error and the score are:"));
add(...Eq("eq_er"));
add(...Eq("eq_ai"));
add(P("A perfect prediction gives *A*_{i} = 1. A negative %*Er* means the model predicted *more* life than actually remained (an over-estimate, which could lead to a failure before maintenance). This case is penalized much more strongly (divisor 5) than an under-estimate (divisor 20). For example, a 20% under-estimate still scores 0.5, while a 10% over-estimate scores only 0.25. Averaging *A*_{i} gives two different summary scores:"));
add(...Eq("eq_score"));
add(P("The **T-Score** (trajectory-averaged PHM score) is our own diagnostic. It averages *A*_{i} over **all M sliding windows** of the test trajectories, using the capped RUL as the true value. It shows how reliable the predictions are across the whole life of the bearings. (The notebook's three-bearing run skips windows whose capped RUL is zero, while the 11-bearing script keeps them, so the two T-Scores are not strictly comparable.)"));
add(H2("Single-Inspection Challenge Score"));
add(P("The **single-inspection challenge score** follows the official 2012 protocol exactly: for each of the 11 test bearings, **one** prediction is made at the last snapshot of its truncated *Test_set* record, and the 11 *A*_{i} values are averaged. This is the only number that can be compared directly with the original challenge. It must not be confused with the T-Score: it rests on only 11 predictions, each made at one late point in a bearing's life."));
add(H2("Risk-Stage Classification Metrics"));
add(...Eq("eq_prf"));
add(P("**Accuracy** is the fraction of all windows whose stage is predicted correctly. For each stage *c*, **precision** is the fraction of windows predicted as *c* that really are *c*, **recall** is the fraction of true *c* windows that were found, and **F1** is their harmonic mean. The **weighted** average weights each stage by its number of windows, while the **macro** average treats the three stages equally. Macro F1 is important here because the stages are imbalanced: a model that predicts only Normal would score well on accuracy but very badly on macro F1."));
add(...Tbl("Summary of the metrics used in the project",
  ["Metric", "Output", "Better", "Why it was selected"],
  [
    ["MAE (min)", "RUL", "Lower", "Average error in physical time; easy to interpret for planning"],
    ["RMSE (min)", "RUL", "Lower", "Highlights large errors"],
    ["Normalized MAE", "RUL", "Lower", "Error relative to the capped range; comparable across bearings"],
    ["T-Score", "RUL", "Higher (max 1)", "Challenge scoring function averaged over the whole trajectory"],
    ["Single-inspection score", "RUL", "Higher (max 1)", "Official 2012 challenge protocol, one prediction per bearing"],
    ["Accuracy", "Risk stage", "Higher", "Overall fraction of correct stage predictions"],
    ["Weighted P / R / F1", "Risk stage", "Higher", "Performance weighted by class size"],
    ["Macro F1", "Risk stage", "Higher", "Treats Normal, Warning and Critical equally (class imbalance)"],
  ],
  [2300, 1250, 1450, 3600], { size: 19, leftCols: [0, 3] }));

// ---------------------------------------------------------------- 6. RESULT REVIEW
add(H1("Result Review"));
add(H2("Aggregate Results on the Test Bearings"));
add(P("The frozen model was first evaluated on three test bearings (one per condition) and then on all 11 test bearings. Table 6.1 shows both evaluations. On all 11 bearings the model reaches an **RUL MAE of 93.40 min**, an **RMSE of 110.49 min**, a **normalized MAE of 0.3333** and a **T-Score of 0.1667**. The risk-stage head reaches **52.82% accuracy**, a weighted F1 of 51.61% and a macro F1 of 40.22%."));
add(...Tbl("Aggregate results of the frozen model (no retraining between evaluations)",
  ["Metric", "3-bearing subset (1_3, 2_3, 3_3)", "All 11 test bearings"],
  [
    ["Snapshots / windows evaluated", "4,764 / 4,719", "17,355 / 17,190"],
    ["RUL MAE", "104.41 min (6,264.38 s)", "93.40 min (5,603.87 s)"],
    ["RUL RMSE", "122.10 min (7,326.13 s)", "110.49 min (6,629.43 s)"],
    ["Normalized RUL MAE", "0.3726", "0.3333"],
    ["Trajectory T-Score", "0.1244", "0.1667"],
    ["Risk-stage accuracy", "50.16%", "52.82%"],
    ["Weighted precision / recall", "47.44% / 50.16%", "50.73% / 52.82%"],
    ["Weighted F1 / macro F1", "48.46% / 33.12%", "51.61% / 40.22%"],
  ],
  [3100, 2750, 2750], { size: 20 }));
add(P("The 11-bearing numbers are better than the 3-bearing ones. This does not mean the model generalizes better when more data are added, because the same frozen model was simply scored on a different mix of bearings. Several of the additional bearings are short-lived, and short trajectories naturally produce smaller absolute errors in minutes. An average error of about one and a half hours is large compared with bearing lifetimes of 0.6–6.8 h, so the model should be regarded as a baseline."));
add(H2("Per-Bearing Results"));
add(P("Table 6.2 and Figure 6.1 break the result down by bearing. The lowest RUL errors belong to Bearing2_4 (54.12 min) and Bearing1_4 (56.23 min), and the highest to Bearing1_6 (129.52 min) and Bearing1_3 (114.43 min). Bearing1_4 also has the best T-Score (0.2668). The unweighted mean per-bearing MAE is 96.96 min for Condition 1, 73.50 min for Condition 2 and 61.40 min for Condition 3 (a single bearing). The lower errors for Conditions 2 and 3 mostly reflect their shorter lifetimes."));
add(...Tbl("Per-bearing results on the 11 test bearings",
  ["Bearing", "Life (h)", "MAE (min)", "RMSE (min)", "Norm. MAE", "T-Score", "Acc. (%)", "W-F1 (%)", "M-F1 (%)"],
  [
    ["Bearing1_3", "6.59", "114.43", "128.51", "0.4084", "0.1150", "77.80", "74.64", "49.16"],
    ["Bearing1_4", "3.96", "56.23", "67.08", "0.2007", "0.2668", "70.28", "58.03", "55.59"],
    ["Bearing1_5", "6.84", "89.57", "101.31", "0.3197", "0.2092", "75.04", "65.93", "41.56"],
    ["Bearing1_6", "6.80", "129.52", "146.49", "0.4622", "0.0951", "23.35", "19.73", "31.57"],
    ["Bearing1_7", "6.27", "95.03", "105.63", "0.3392", "0.1716", "58.42", "59.78", "33.47"],
    ["Bearing2_3", "5.43", "101.50", "123.41", "0.3623", "0.1615", "25.52", "27.27", "14.90"],
    ["Bearing2_4", "2.08", "54.12", "64.94", "0.1931", "0.1887", "42.39", "44.90", "26.67"],
    ["Bearing2_5", "6.42", "82.85", "100.21", "0.2957", "0.2306", "69.95", "62.18", "39.93"],
    ["Bearing2_6", "1.94", "59.27", "68.35", "0.2115", "0.1529", "8.60", "12.56", "10.42"],
    ["Bearing2_7", "0.64", "69.78", "80.96", "0.2491", "0.0000", "12.09", "21.58", "7.19"],
    ["Bearing3_3", "1.20", "61.40", "66.15", "0.2191", "0.0051", "8.59", "15.04", "8.30"],
    ["**All 11**", "–", "**93.40**", "**110.49**", "**0.3333**", "**0.1667**", "**52.82**", "**51.61**", "**40.22**"],
  ],
  [1400, 800, 950, 1000, 1000, 950, 850, 850, 800], { size: 19, note: "W-F1 / M-F1: weighted / macro F1 of the risk-stage head within each bearing." }));
add(...Fig(PFIG("fig_per_bearing.png"), "Per-bearing (a) RUL MAE and (b) risk-stage accuracy, coloured by operating condition", 5.6));
add(H2("RUL Prediction Behaviour"));
add(P("Figure 6.2 compares the predicted and true (capped) RUL over the life of four test bearings. Averaged over each bearing's windows, the signed error (predicted − true) shows a clear pattern. The model **under-estimates** the RUL of long-lived bearings (for example −103.77 min for Bearing1_3 and −112.70 min for Bearing1_6) and **over-estimates** it for the shortest ones (+69.78 min for Bearing2_7 and +60.15 min for Bearing3_3). Over all windows the mean signed error is −51.73 min. In other words, the predictions stay in a middle range of roughly one to two and a half hours, whatever the actual lifetime of the bearing. The model does react to changes in the vibration: on Bearing1_3, for example, the prediction falls sharply near the end of life. However, its estimate of the remaining time is not well calibrated across bearings."));
add(...Fig(PFIG("fig_trajectories.png"), "Predicted versus capped true RUL for four test bearings", 5.2));
add(H2("Auxiliary Risk-Stage Results"));
add(P("Figure 6.3 and Table 6.3 show the risk-stage results on all 17,190 test windows. The Normal stage is recognized reasonably well (71.19% recall), and the Critical stage partly (39.20% recall). The Warning stage, however, is almost never recognized correctly (13.41% recall): Warning windows are more often predicted as Normal (2,150) or Critical (1,267) than as Warning (529). For reference, always predicting Normal would already give 61.08% accuracy (10,499 of 17,190 windows) but a macro F1 of only about 25%. The model's accuracy of 52.82% is therefore **below** this trivial baseline, while its macro F1 of 40.22% is above it, because it detects some Critical windows. The risk head is useful only as a coarse indicator."));
add(...Tbl("Per-stage metrics of the risk head on all 11 test bearings",
  ["Stage", "Windows", "Precision (%)", "Recall (%)", "F1 (%)"],
  [
    ["Normal", "10,499", "67.46", "71.19", "69.27"],
    ["Warning", "3,946", "18.56", "13.41", "15.57"],
    ["Critical", "2,745", "33.00", "39.20", "35.83"],
    ["Macro average", "17,190", "39.67", "41.26", "40.22"],
    ["Weighted average", "17,190", "50.73", "52.82", "51.61"],
  ],
  [2300, 1400, 1650, 1600, 1650], { size: 20 }));
add(...Fig(PFIG("fig_confusion_11.png"), "Confusion matrix of the risk-stage head (counts and row percentages)", 3.0));
add(H2("IEEE PHM 2012 Single-Inspection Evaluation"));
add(P("Table 6.4 and Figure 6.4 show the single prediction made for each bearing at the last snapshot of its truncated Test_set record. The challenge document lists the true RUL of Bearing1_4 as 339 s, whereas the released files give (1,428 − 1,139) × 10 = 2,890 s. We cannot tell which value is intended, so both are reported. With the published values the challenge score is **0.0234**. Using 2,890 s for Bearing1_4 gives **0.0718**, which we treat only as a sensitivity check. Bearing1_4 is the best case (8.74 min error, *A*_{i} = 0.5333 with 2,890 s). In eight of the eleven cases, however, the model over-estimated the remaining life, and seven of these scored zero. The single-inspection MAE is 68.60 min (RMSE 80.61 min), and over the three-bearing subset the score is 0.0375. These values show that the model is not competitive with the leading 2012 challenge entries."));
add(...Tbl("Single-inspection results at the last Test_set snapshot",
  ["Bearing", "Actual RUL (s)", "Predicted RUL (s)", "|Error| (min)", "%Er", "A_{i}"],
  [
    ["Bearing1_3", "5,730", "0.0", "95.50", "+100.00", "0.0313"],
    ["Bearing1_4", "2,890 (339*)", "2,365.7", "8.74", "+18.14", "0.5333 (0.0000*)"],
    ["Bearing1_5", "1,610", "7,485.7", "97.93", "−364.95", "0.0000"],
    ["Bearing1_6", "1,460", "3,612.2", "35.87", "−147.41", "0.0000"],
    ["Bearing1_7", "7,570", "3,345.9", "70.40", "+55.80", "0.1446"],
    ["Bearing2_3", "7,530", "8,894.7", "22.74", "−18.12", "0.0811"],
    ["Bearing2_4", "1,390", "3,597.4", "36.79", "−158.81", "0.0000"],
    ["Bearing2_5", "3,090", "6,111.7", "50.36", "−97.79", "0.0000"],
    ["Bearing2_6", "1,290", "8,661.3", "122.86", "−571.42", "0.0000"],
    ["Bearing2_7", "580", "9,723.7", "152.39", "−1,576.49", "0.0000"],
    ["Bearing3_3", "820", "4,481.5", "61.02", "−446.52", "0.0000"],
    ["**Score**", "", "", "**68.60** (MAE)", "", "**0.0234*** / 0.0718"],
  ],
  [1500, 1600, 1600, 1300, 1200, 1400], { size: 19, note: "* Published ground truth for Bearing1_4 (challenge Table 3). Official score = 0.0234; 0.0718 uses the file-derived 2,890 s." }));
add(...Fig(PFIG("fig_single_inspection.png"), "Actual and predicted RUL at the single inspection point of each test bearing, with the per-bearing score A_{i}", 5.6));
add(H2("Attention and Explainability"));
add(P("Figure 6.5 shows the attention weights of the frozen model for two windows of test bearing Bearing1_3: an early window starting at snapshot 50, and the final window of the record. If the model paid equal attention to every step, each weight would be 1/16 = 0.0625. In the early window the weights are spread fairly evenly (0.0367 to 0.0813). In the final window the three most recent steps receive the largest weights (up to 0.1120 at step 16). Near failure the model therefore relies more on the latest observations, but the effect is moderate. These are two example windows from one bearing, not a statistical analysis. The weights explain *which time steps* were used, not *which of the 30 features* mattered, and they do not prove cause and effect [21]."));
add(...Fig(PFIG("fig_attention.png"), "Temporal attention weights for Bearing1_3: (a) early window, (b) final window of the record", 6.0));
// ---------------------------------------------------------------- 7. DISCUSSION / CONCLUSION
add(H1("Discussion, Conclusion and Future Work"));
add(H2("Discussion"));
add(P("The most useful outcome of the project is a clean, leakage-free baseline. An earlier internal version of our pipeline converted predictions to seconds using each test bearing's own lifetime, which quietly gave the model information it should not have had and made the errors look smaller. After this was corrected, the numbers in this report became the honest measure of performance. This experience shows why the evaluation protocol matters as much as the model."));
add(P("The main weakness is that the predictions do not scale with the lifetime of the bearing. Two features of our setup probably contribute to this, although we did not test them separately. First, only three bearings with very different lifetimes (1.43–7.78 h) were used for training, and only Bearing1_1 contributes capped windows. Second, the weights from the very first epoch were kept, because validation loss rose afterwards, so the model is only lightly trained and tends to predict an average value. The low Warning recall follows from the same problem: a model that cannot tell 60 minutes of remaining life from 150 minutes cannot reliably separate the Warning band from its neighbours."));
add(P("The trajectory T-Score (0.1667) and the single-inspection score (0.0234) differ by a factor of about seven even though they use the same scoring function. The trajectory view is closer to how a monitoring system would be used in a plant, while the single-inspection score is the only fair comparison with the 2012 challenge. Both should be reported together."));
add(H2("Limitations"));
add(B("All results come from a single training run (seed 42), and no baseline models or ablation studies were run."));
add(B("Only three bearings were used for training, and only one test bearing represents Condition 3."));
add(B("The RUL cap ratio (0.60) and stage thresholds (0.40 and 0.15) were chosen by hand; the stages are based on RUL, not on measured vibration severity."));
add(B("The ground truth for Bearing1_4 is ambiguous (339 s versus 2,890 s)."));
add(B("The attention analysis covers only two windows of one bearing and explains time steps, not features."));
add(B("The model gives point estimates without uncertainty. It was tested offline on one laboratory dataset and has not been validated on real plant machinery, so it must not be used as a safety or protection system."));
add(H2("Conclusion"));
add(P("We designed, trained and evaluated an explainable dual-head CNN–BiLSTM–attention model for bearing RUL prediction on the PRONOSTIA / IEEE PHM 2012 data. The pipeline extracts 30 vibration features per snapshot, uses 16-snapshot windows, caps the RUL target at a training-derived 16,812 s and keeps all preprocessing free of test information. On all 11 test bearings the frozen model achieved an RUL MAE of 93.40 min, an RMSE of 110.49 min, a normalized MAE of 0.3333 and a T-Score of 0.1667, with 52.82% risk-stage accuracy. The official single-inspection score is 0.0234 (0.0718 in the Bearing1_4 sensitivity check). The project shows the complete manufacturing-analytics workflow, from raw sensor data to maintenance-oriented predictions. It also shows, honestly, that much more work is needed before such a model could support real maintenance decisions."));
add(H2("Future Work"));
add(B("Train with several random seeds and compare against simpler baselines and ablated versions of the model (without attention, without the risk head)."));
add(B("Use all six learning bearings with leave-one-bearing-out validation."));
add(B("Replace the fixed cap with automatic detection of degradation onset or a learned health indicator."));
add(B("Add uncertainty estimates so that each RUL prediction comes with a confidence range."));
add(B("Add feature-level explanations (for example SHAP [22]) and compare them with the attention weights."));
add(B("Test the approach on another bearing dataset, such as XJTU-SY [23], and on streaming data."));

// ---------------------------------------------------------------- 8. RESOURCES
add(H2("Project Resources and Reproducibility"));
add(P("All project files are kept in one repository so that every number in this report can be reproduced or checked. The public GitHub repository is **github.com/MOHAMMEDFAISALSM/bearing-prognostics-cnn-bilstm-attention**. The raw dataset is not stored in the repository because of its size; it must be downloaded separately and placed in the project folder."));
add(...Tbl("Main project files",
  ["File / folder", "Purpose"],
  [
    ["explainable_dual_head_bearing_prognostics.ipynb", "Complete executed notebook: data loading, features, training, evaluation, figures"],
    ["build_notebook.py", "Script that generates the notebook"],
    ["evaluate_all_11_test_bearings.py", "Evaluates the frozen model on all 11 test bearings"],
    ["evaluate_challenge_single_inspection.py", "Computes the IEEE PHM 2012 single-inspection score"],
    ["models/", "Trained model (.keras), fitted scaler (scaler.pkl), feature names (30)"],
    ["results/", "Metrics (JSON), per-bearing and per-window predictions (CSV), notebook figures"],
    ["PAPER_RESULTS_PACKAGE.md", "Consolidated, frozen results used in this report"],
    ["paper/, report/", "Research manuscript, this report, figure and document build scripts"],
  ],
  [3700, 4900], { size: 19, leftCols: [0, 1] }));
add(P("To reproduce the evaluation, install Python with NumPy, pandas, scikit-learn, TensorFlow and Matplotlib, place the dataset folder *ieee-phm-2012-data-challenge-dataset-master* in the project root, and run *evaluate_all_11_test_bearings.py* followed by *evaluate_challenge_single_inspection.py*. The notebook can be executed from start to finish to retrain the model and regenerate all results."));

// ---------------------------------------------------------------- REFERENCES
const refs = [
  "Y. Lei, N. Li, L. Guo, N. Li, T. Yan, and J. Lin, “Machinery health prognostics: A systematic review from data acquisition to RUL prediction,” *Mechanical Systems and Signal Processing*, vol. 104, pp. 799–834, 2018.",
  "A. K. S. Jardine, D. Lin, and D. Banjevic, “A review on machinery diagnostics and prognostics implementing condition-based maintenance,” *Mechanical Systems and Signal Processing*, vol. 20, no. 7, pp. 1483–1510, 2006.",
  "R. B. Randall and J. Antoni, “Rolling element bearing diagnostics—A tutorial,” *Mechanical Systems and Signal Processing*, vol. 25, no. 2, pp. 485–520, 2011.",
  "X.-S. Si, W. Wang, C.-H. Hu, and D.-H. Zhou, “Remaining useful life estimation – A review on the statistical data driven approaches,” *European Journal of Operational Research*, vol. 213, no. 1, pp. 1–14, 2011.",
  "G. S. Babu, P. Zhao, and X.-L. Li, “Deep convolutional neural network based regression approach for estimation of remaining useful life,” in *Database Systems for Advanced Applications (DASFAA 2016)*, Lecture Notes in Computer Science. Cham, Switzerland: Springer, 2016, pp. 214–228.",
  "X. Li, Q. Ding, and J.-Q. Sun, “Remaining useful life estimation in prognostics using deep convolution neural networks,” *Reliability Engineering & System Safety*, vol. 172, pp. 1–11, 2018.",
  "F. O. Heimes, “Recurrent neural networks for remaining useful life estimation,” in *Proc. Int. Conf. Prognostics and Health Management (PHM)*, 2008, pp. 1–6.",
  "S. Zheng, K. Ristovski, A. Farahat, and C. Gupta, “Long short-term memory network for remaining useful life estimation,” in *Proc. IEEE Int. Conf. Prognostics and Health Management (ICPHM)*, 2017, pp. 88–95.",
  "S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, “Leakage in data mining: Formulation, detection, and avoidance,” *ACM Transactions on Knowledge Discovery from Data*, vol. 6, no. 4, pp. 1–21, 2012.",
  "P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Chebel-Morello, N. Zerhouni, and C. Varnier, “PRONOSTIA: An experimental platform for bearings accelerated degradation tests,” in *Proc. IEEE Int. Conf. Prognostics and Health Management (PHM)*, Denver, CO, USA, 2012, pp. 1–8.",
  "FEMTO-ST Institute, “IEEE PHM 2012 Prognostic Challenge: Outline, experiments, scoring of results, winners,” challenge documentation distributed with the PRONOSTIA dataset, 2012.",
  "S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.",
  "M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” *IEEE Transactions on Signal Processing*, vol. 45, no. 11, pp. 2673–2681, 1997.",
  "D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by jointly learning to align and translate,” in *Proc. Int. Conf. Learning Representations (ICLR)*, 2015.",
  "Z. Yang, D. Yang, C. Dyer, X. He, A. Smola, and E. Hovy, “Hierarchical attention networks for document classification,” in *Proc. NAACL-HLT*, 2016, pp. 1480–1489.",
  "R. Caruana, “Multitask learning,” *Machine Learning*, vol. 28, no. 1, pp. 41–75, 1997.",
  "S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in *Proc. 32nd Int. Conf. Machine Learning (ICML)*, PMLR vol. 37, 2015, pp. 448–456.",
  "N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A simple way to prevent neural networks from overfitting,” *Journal of Machine Learning Research*, vol. 15, pp. 1929–1958, 2014.",
  "P. J. Huber, “Robust estimation of a location parameter,” *Annals of Mathematical Statistics*, vol. 35, no. 1, pp. 73–101, 1964.",
  "D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in *Proc. 3rd Int. Conf. Learning Representations (ICLR)*, San Diego, CA, USA, 2015.",
  "S. Jain and B. C. Wallace, “Attention is not explanation,” in *Proc. NAACL-HLT*, 2019, pp. 3543–3556.",
  "S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 2017.",
  "B. Wang, Y. Lei, N. Li, and N. Li, “A hybrid prognostics approach for estimating remaining useful life of rolling element bearings,” *IEEE Transactions on Reliability*, vol. 69, no. 1, pp. 401–412, 2020.",
];
body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, alignment: AlignmentType.CENTER, spacing: { after: 300 },
  children: [new TextRun({ text: "REFERENCES", font: FONT, bold: true, size: 28 })] }));
refs.forEach((r, i) => body.push(new Paragraph({
  tabStops: [{ type: TabStopType.LEFT, position: 560 }], indent: { left: 560, hanging: 560 },
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 30, line: 240 },
  children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 20 }), ...runs(r, { size: 20 })],
})));

// ================================================================ ASSEMBLE
const pageProps = { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: M_TOP, bottom: M_BOTTOM, left: M_LEFT, right: M_RIGHT, footer: 700 } } };
const pageFooter = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 22 })] })] });

const doc = new Document({
  creator: "Mohammed Faisal SM, Mohamed Faisal A, Visweswar Gopal Reddy",
  title: "Explainable Dual-Head CNN–BiLSTM–Attention Prognostics for Industrial Bearing Health Monitoring",
  styles: {
    default: { document: { run: { font: FONT, size: SZ } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 28, bold: true, color: "000000" }, paragraph: { spacing: { after: 300 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 24, bold: true, color: "000000" }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
      { id: "TOC1", name: "toc 1", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 20, bold: true }, paragraph: { spacing: { before: 0, after: 0, line: 228, lineRule: "exact" } } },
      { id: "TOC2", name: "toc 2", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 20 }, paragraph: { spacing: { after: 0, line: 228, lineRule: "exact" }, indent: { left: 360 } } },
      { id: "FigureCaption", name: "FigureCaption", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 21 }, paragraph: { alignment: AlignmentType.CENTER, spacing: { after: 240 } } },
      { id: "TableCaption", name: "TableCaption", basedOn: "Normal", next: "Normal",
        run: { font: FONT, size: 21 }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 120, after: 100 }, keepNext: true } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 460, hanging: 280 } } } }] }] },
  sections: [
    { properties: { ...pageProps }, children: titlePage },
    { properties: { ...pageProps, type: SectionType.NEXT_PAGE, page: { ...pageProps.page, pageNumbers: { start: 2, formatType: NumberFormat.LOWER_ROMAN } } },
      footers: { default: pageFooter }, children: [...bonafide, ...acknowledgement, ...abstract, ...tocPage, ...listsPage] },
    { properties: { ...pageProps, type: SectionType.NEXT_PAGE, page: { ...pageProps.page, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } },
      footers: { default: pageFooter }, children: body },
  ],
});

const outPath = path.join(__dirname, "Manufacturing_Analytics_Project_Report_Bearing_Prognostics.docx");
const JSZip = require(require.resolve("jszip", { paths: [path.dirname(require.resolve(process.env.DOCX_MODULE || "docx"))] }));
Packer.toBuffer(doc).then(async (buf) => {
  const zip = await JSZip.loadAsync(buf);
  let xml = await zip.file("word/document.xml").async("string");
  for (const [name, omml] of Object.entries(OMML)) {
    const re = new RegExp(String.raw`<w:p>(?:(?!<w:p>)[\s\S])*?@@EQ:${name}@@[\s\S]*?</w:p>`, "g");
    const spaced = omml
      .replace(/<m:plcHide m:val="1" \/><m:mcs>(<m:mc><m:mcPr><m:mcJc m:val="right" \/>)/g, '<m:plcHide m:val="1" /><m:rSpRule m:val="1"/><m:cGpRule m:val="3"/><m:cGp m:val="60"/><m:mcs>$1')
      .replace(/<m:plcHide m:val="1" \/><m:mcs>(?!<m:mc><m:mcPr><m:mcJc m:val="right")/g, '<m:plcHide m:val="1" /><m:rSpRule m:val="1"/><m:cGpRule m:val="2"/><m:mcs>');
    xml = xml.replace(re, `<w:p><w:pPr><w:spacing w:before="0" w:after="0"/><w:jc w:val="center"/></w:pPr>${spaced}</w:p>`);
  }
  if (xml.includes("@@EQ:")) throw new Error("unreplaced equation placeholder");
  zip.file("word/document.xml", xml);
  fs.writeFileSync(outPath, await zip.generateAsync({ type: "nodebuffer" }));
  console.log("written", outPath);
});
