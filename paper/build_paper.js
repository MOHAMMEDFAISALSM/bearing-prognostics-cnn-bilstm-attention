// Builds the IEEE-style manuscript (.docx). Every number below is taken from
// PAPER_RESULTS_PACKAGE.md, results/*.json|csv, or the executed notebook outputs
// (see CLAIM_EVIDENCE_MAP.md for traceability).
const fs = require("fs");
const path = require("path");
const docx = require(process.env.DOCX_MODULE || "docx");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  WidthType, BorderStyle, ShadingType, SectionType, TabStopType, VerticalAlign, Footer, PageNumber,
  LevelFormat, HeightRule,
} = docx;

const ROOT = path.resolve(__dirname, "..");
const FIG = (f) => path.join(__dirname, "figures", f);
const OLDFIG = (f) => path.join(ROOT, "results", "figures", f);

// ---------------------------------------------------------------- page geometry (US Letter, IEEE-like)
const PAGE_W = 12240, PAGE_H = 15840, MARGIN_LR = 1080, MARGIN_T = 1080, MARGIN_B = 1080;
const TEXT_W = PAGE_W - 2 * MARGIN_LR;          // 10080 DXA = 7.0 in
const COL_GAP = 360;
const COL_W = (TEXT_W - COL_GAP) / 2;           // 4860 DXA = 3.375 in
const FONT = "Times New Roman";

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

// ---------------------------------------------------------------- text helpers
// Inline markup: *italic*, **bold**, _{sub}, ^{sup}
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("*")) out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(2, -1), superScript: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({
  children: runs(text, { size: 20 }),
  alignment: AlignmentType.JUSTIFIED,
  indent: opts.noIndent ? undefined : { firstLine: 202 },
  spacing: { after: 40, line: 240 },
  ...opts.para,
});

let secNo = 0, subNo = 0;
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV"];
function H1(text) {
  subNo = 0;
  const label = `${ROMAN[secNo++]}. `;
  return new Paragraph({
    children: [new TextRun({ text: label + text.toUpperCase(), font: FONT, size: 20, smallCaps: false })],
    alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
  });
}
function H1u(text) { // unnumbered (Acknowledgment, References)
  return new Paragraph({
    children: [new TextRun({ text: text.toUpperCase(), font: FONT, size: 20 })],
    alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
  });
}
function H2(text) {
  const label = String.fromCharCode(65 + subNo++) + ". ";
  return new Paragraph({
    children: [new TextRun({ text: label + text, italics: true, font: FONT, size: 20 })],
    spacing: { before: 120, after: 60 }, keepNext: true,
  });
}

const bulletRef = "bullets";
const B = (text) => new Paragraph({
  children: runs(text, { size: 20 }), numbering: { reference: bulletRef, level: 0 },
  alignment: AlignmentType.JUSTIFIED, spacing: { after: 30 },
});

// ---------------------------------------------------------------- equations
let eqNo = 0;
// Equations are native Word math (OMML, from equations.tex.json via pandoc), placed in a
// borderless two-cell table so the number sits right-aligned and can never wrap.
const OMML = JSON.parse(fs.readFileSync(path.join(__dirname, "equations.omml.json"), "utf8"));
const NUM_W = 560;
function Eq(name, width = COL_W) {
  if (!OMML[name]) throw new Error("missing equation " + name);
  eqNo += 1;
  const nb = { top: none, bottom: none, left: none, right: none };
  return [new Table({
    width: { size: width, type: WidthType.DXA }, columnWidths: [width - NUM_W, NUM_W],
    rows: [new TableRow({ cantSplit: true, children: [
      new TableCell({ width: { size: width - NUM_W, type: WidthType.DXA }, borders: nb, verticalAlign: VerticalAlign.CENTER,
        margins: { top: 50, bottom: 50, left: 0, right: 0 },
        children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: `@@EQ:${name}@@`, font: FONT })] })] }),
      new TableCell({ width: { size: NUM_W, type: WidthType.DXA }, borders: nb, verticalAlign: VerticalAlign.CENTER,
        margins: { top: 50, bottom: 50, left: 0, right: 0 },
        children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: `(${eqNo})`, font: FONT, size: 20 })] })] }),
    ] })],
  }), new Paragraph({ spacing: { after: 20 }, children: [] })];
}

// ---------------------------------------------------------------- figures
let figNo = 0;
const figIds = {};
function Fig(file, caption, width, id) {
  const { w, h } = pngSize(file);
  const svg = file.replace(/\.png$/, ".svg");
  const img = (tr) => fs.existsSync(svg)
    ? new ImageRun({ type: "svg", data: fs.readFileSync(svg), fallback: { type: "png", data: fs.readFileSync(file) }, transformation: tr })
    : new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: tr });
  const dw = width / 1440 * 96;
  const dh = dw * h / w;
  figNo += 1;
  if (id) figIds[id] = figNo;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
      children: [img({ width: Math.round(dw), height: Math.round(dh) })],
    }),
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED, spacing: { after: 140 },
      children: [new TextRun({ text: `Fig. ${figNo}.  `, font: FONT, size: 16 }), ...runs(caption, { size: 16 })],
    }),
  ];
}

// ---------------------------------------------------------------- tables
let tabNo = 0;
const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function Tbl(caption, header, rows, colW, opts = {}) {
  tabNo += 1;
  const total = colW.reduce((a, b) => a + b, 0);
  const fs_ = opts.size || 15;
  const mk = (txt, isHead, i, j) => new TableCell({
    width: { size: colW[j], type: WidthType.DXA },
    margins: { top: 25, bottom: 25, left: 50, right: 50 },
    verticalAlign: VerticalAlign.CENTER,
    shading: isHead ? { type: ShadingType.CLEAR, fill: "E7E6E6", color: "auto" }
      : (opts.highlight && opts.highlight(i) ? { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" } : undefined),
    borders: {
      top: isHead ? border : none,
      bottom: isHead || i === rows.length - 1 ? border : none,
      left: none, right: none,
    },
    children: [new Paragraph({
      keepNext: i < rows.length - 1,
      alignment: j === 0 && !opts.centerFirst ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(txt), { size: fs_, bold: isHead || (opts.boldRow && opts.boldRow(i)) }),
    })],
  });
  const trs = [new TableRow({ tableHeader: true, children: header.map((t, j) => mk(t, true, -1, j)) })];
  rows.forEach((r, i) => trs.push(new TableRow({ cantSplit: true, children: r.map((t, j) => mk(t, false, i, j)) })));
  const out = [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 140, after: 20 }, keepNext: true,
      children: [new TextRun({ text: `TABLE ${ROMAN[tabNo - 1]}`, font: FONT, size: 16 })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true,
      children: runs(caption.toUpperCase ? caption : caption, { size: 16, smallCaps: true }),
    }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: colW, rows: trs, alignment: AlignmentType.CENTER }),
  ];
  if (opts.note) out.push(new Paragraph({ spacing: { before: 30, after: 140 }, children: runs(opts.note, { size: 14, italics: false }) }));
  else out.push(new Paragraph({ spacing: { after: 100 }, children: [] }));
  return out;
}

// ---------------------------------------------------------------- algorithm box
function Algorithm(title, lines) {
  const cell = new TableCell({
    width: { size: COL_W, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    borders: { top: border, bottom: border, left: none, right: none },
    children: [
      new Paragraph({ spacing: { after: 60 }, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000", space: 2 } },
        children: runs(title, { size: 17 }) }),
      ...lines.map(([indent, text]) => new Paragraph({
        indent: { left: 180 * indent, hanging: 0 }, spacing: { after: 10 },
        children: runs(text, { size: 16 }),
      })),
    ],
  });
  return [new Table({ width: { size: COL_W, type: WidthType.DXA }, columnWidths: [COL_W], rows: [new TableRow({ children: [cell] })] }),
    new Paragraph({ spacing: { after: 100 }, children: [] })];
}

const wide = (items) => ({ wide: items });

// ================================================================ CONTENT
const title = "Explainable Dual-Head CNN–BiLSTM–Attention Prognostics for Industrial Bearing Health Monitoring";

const authors = [
  ["Mohammed Faisal SM*", "Artificial Intelligence and Data Science", "Rajalakshmi Engineering College", "Chennai, India", "ORCID: 0009-0007-3423-3389", "mf5330766@gmail.com"],
  ["Mohamed Faisal A", "Artificial Intelligence and Data Science", "Rajalakshmi Engineering College", "Chennai, India", "ORCID: 0009-0007-4250-9349", "mfmf98236@gmail.com"],
  ["Visweswar Gopal Reddy", "Artificial Intelligence and Data Science", "Rajalakshmi Engineering College", "Chennai, India", "ORCID: 0009-0005-7852-8571", "reddyvisweswar4@gmail.com"],
  ["A. Aswin Jeba Mahir", "Assistant Professor, Artificial Intelligence and Data Science", "Rajalakshmi Engineering College", "Chennai, India", "ORCID: 0009-0002-9772-8403", "jebamahir@gmail.com"],
];
function authorCell(a) {
  return new TableCell({
    width: { size: TEXT_W / 2, type: WidthType.DXA },
    borders: { top: none, bottom: none, left: none, right: none },
    margins: { top: 40, bottom: 120, left: 60, right: 60 },
    children: a.map((line, i) => new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 0 },
      children: [new TextRun({ text: line, font: FONT, size: i === 0 ? 22 : 18, italics: i === 5 })],
    })),
  });
}
const authorTable = new Table({
  width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: [TEXT_W / 2, TEXT_W / 2],
  rows: [new TableRow({ children: [authorCell(authors[0]), authorCell(authors[1])] }),
    new TableRow({ children: [authorCell(authors[2]), authorCell(authors[3])] })],
});

const front = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: title, font: FONT, size: 44 })] }),
  authorTable,
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 20 },
    children: [new TextRun({ text: "*Corresponding author: Mohammed Faisal SM (mf5330766@gmail.com)", font: FONT, size: 18, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "Rajalakshmi Engineering College, Rajalakshmi Nagar, Vellore – Chennai Road, Thandalam, Chennai – 602 105, Tamil Nadu, India.", font: FONT, size: 18, italics: true })] }),
];

const abstractText =
  "Remaining useful life (RUL) prediction for rolling-element bearings is difficult when only a few run-to-failure histories are available and when bearings run under the same nominal load can last very different lengths of time. This paper presents a dual-head deep prognostic model that combines a one-dimensional convolutional front end, a bidirectional long short-term memory (BiLSTM) layer and an additive temporal-attention layer. The model reads windows of 16 consecutive vibration snapshots (160 s), each described by 30 time- and frequency-domain features. It has two outputs: a regression head that estimates a piecewise-capped, normalized RUL and an auxiliary three-class head that places the bearing in a Normal, Warning or Critical RUL stage. The whole pipeline was built to avoid information leakage. Feature scaling is fitted on the training bearings only, and the RUL cap (16,812 s) is derived from the longest training lifetime. Predictions are converted back to seconds with this fixed constant and never with the lifetime of a test bearing. The model has 113,924 parameters. It was trained on three PRONOSTIA / IEEE PHM 2012 bearings and evaluated, frozen, on all 11 test bearings (17,190 windows). Across these bearings it reaches an RUL mean absolute error (MAE) of 93.40 min, an RMSE of 110.49 min, a normalized MAE of 0.3333 and a trajectory-averaged PHM score of 0.1667; the auxiliary stage head reaches 52.82% accuracy and a 40.22% macro-F1. Under the official single-inspection protocol of the 2012 challenge, the score is 0.0234 with the published ground truth, or 0.0718 if the Bearing1_4 value is recomputed from the released full-test files. In the example examined, the attention weights shift toward the most recent time steps near failure, although the effect is modest. We present these results as a transparent, leakage-free reference point rather than as a deployable monitoring system, and we discuss the large error that remains.";

const keywords = "Bearing prognostics, remaining useful life, CNN–BiLSTM, temporal attention, multi-task learning, PRONOSTIA, IEEE PHM 2012, data leakage, explainability.";

const abstractBlock = [
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 80 },
    children: [new TextRun({ text: "Abstract", bold: true, italics: true, font: FONT, size: 18 }),
      new TextRun({ text: "—" + abstractText, bold: true, font: FONT, size: 18 })] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 },
    children: [new TextRun({ text: "Index Terms", bold: true, italics: true, font: FONT, size: 18 }),
      new TextRun({ text: "—" + keywords, bold: true, font: FONT, size: 18 })] }),
];

// ---------------------------------------------------------------- body (two-column)
const body = [];
const add = (...xs) => xs.forEach((x) => (Array.isArray(x) ? body.push(...x) : body.push(x)));  // wide() objects pass through

// I. INTRODUCTION
add(H1("Introduction"));
add(P("Rolling-element bearings sit in almost every rotating machine, from pumps and fans to spindles and traction drives, and a failed bearing can stop a production line or damage the parts around it. Condition-based maintenance tries to replace components when their condition calls for it rather than on a fixed calendar, and this depends on two things: detecting that degradation has begun and estimating how much useful life is left [1]–[3]. The second task, remaining useful life (RUL) prediction, is the subject of this paper."));
add(P("Data-driven prognostics has moved steadily toward deep learning. Convolutional networks learn local patterns from sensor data, recurrent networks model how those patterns evolve over time, and attention mechanisms let a network weight the parts of its input history that matter most [4]–[9]. Bearing prognostics, however, remains hard for reasons that model capacity alone does not solve. Public run-to-failure datasets are small. The PRONOSTIA data used in the IEEE PHM 2012 challenge provide only six training histories across three operating conditions, and the organizers point out that experiment durations range from about 1 h to 7 h [10], [11]. Bearings run under identical nominal conditions can therefore degrade on very different time scales. It is also easy, when a model is evaluated on complete run-to-failure records, to let information about the test bearing's end of life slip into preprocessing or into the conversion of normalized predictions back to physical time. Such leakage makes reported errors look better than they would be in any real prospective use [12]. Finally, operators and reliability engineers are reluctant to act on a single number produced by an opaque model."));
add(P("This paper describes a compact prognostic model built with these issues in mind and reports its performance as it is, including where it falls short. Its main elements are:"));
add(B("**A leakage-free preprocessing and evaluation protocol.** The bearings are split at bearing level, the feature scaler is fitted on the training bearings only, and the RUL cap and normalization constant (16,812 s) are derived from the training bearings alone. Predictions are converted to seconds with this fixed constant, so no test-bearing lifetime is used at inference time."));
add(B("**A dual-head CNN–BiLSTM–attention network** (113,924 parameters) that jointly regresses a piecewise-capped, normalized RUL and classifies an auxiliary three-stage RUL horizon, using a weighted Huber plus cross-entropy loss."));
add(B("**Two complementary evaluation views on all 11 PRONOSTIA test bearings:** trajectory-level metrics over 17,190 sliding windows and the official single-inspection score of the IEEE PHM 2012 challenge. We report both the published ground truth and a sensitivity value for Bearing1_4."));
add(B("**Temporal-attention analysis** that shows which of the 16 time steps in a window the network relies on, together with an explicit statement of what such weights can and cannot explain."));
add(P("The results are modest. Averaged over the 11 test bearings, the RUL error is roughly one and a half hours. The auxiliary stage head's overall accuracy is lower than that of always predicting the majority stage, and it rarely recognizes the Warning stage. We think that stating this plainly, with per-bearing detail, is more useful to the community than a single optimistic headline number. Section II reviews related work, Sections III–VI describe the data, problem formulation, features and model, Section VII gives the experimental setup, Section VIII reports the results, and Sections IX–XI discuss them, list the limitations and conclude."));

// II. RELATED WORK
add(H1("Related Work"));
add(H2("Model-Based and Data-Driven Prognostics"));
add(P("Classical prognostics builds on physics-of-failure models, reliability laws such as L10 life, or statistical degradation models whose parameters are updated as data arrive [1], [3]. These approaches are interpretable, but they need an accurate degradation model, and the PHM 2012 challenge document itself notes that theoretical bearing-life formulas and characteristic defect frequencies did not match the PRONOSTIA observations well [11]. Data-driven methods instead learn the mapping from condition-monitoring signals to health state or RUL directly. Surveys by Lei *et al.* [1] and Zhao *et al.* [4] trace the move from hand-crafted health indicators and shallow regressors to end-to-end deep networks."));
add(H2("Deep Sequence Models for RUL Estimation"));
add(P("Babu *et al.* [5] and Li *et al.* [6] showed that convolutional networks can regress RUL from multichannel sensor windows, while Heimes [7] and Zheng *et al.* [8] used recurrent and LSTM networks to capture temporal dependencies. Heimes also popularized the piecewise-linear RUL target, in which RUL is held constant early in life, when degradation cannot be observed [7]. For bearings in particular, Guo *et al.* [9] built an RNN-based health indicator on PRONOSTIA, and Wang *et al.* [13] proposed a hybrid prognostics approach for rolling-element bearings. Later work combined convolutional or recurrent encoders with attention mechanisms to weight the most informative parts of the input before regression [14], [15]. Bidirectional recurrence [16] lets every time step within a window use context from both earlier and later steps, which is admissible here because the whole window has already been observed when a prediction is made."));
add(H2("Attention and Multi-Task Learning"));
add(P("Additive attention [17] and its sentence-level variant with a learned context vector [18] weight a sequence of hidden states and summarize it as a single context vector. Multi-task learning [19] trains related objectives together through a shared representation, which can act as a regularizer when data are scarce. In this work, an auxiliary classification head that predicts coarse RUL stages shares its backbone with the regression head."));
add(H2("Evaluation Practice and Leakage"));
add(P("Leakage in predictive modelling occurs when information that would not be available at prediction time influences training or evaluation [12]. In prognostics with complete run-to-failure data, typical sources are random window-level splits, where overlapping windows from the same bearing end up on both sides of the split; scalers fitted on all data; and de-normalizing predicted RUL fractions with the true lifetime of the test unit. The PHM 2012 challenge itself scored a single RUL estimate per test bearing, made at the last available snapshot of a truncated record, with an asymmetric penalty that punishes over-estimation more heavily [11]. Many later studies instead report errors averaged over complete trajectories. The two protocols answer different questions, so we report both and keep them clearly apart."));
add(H2("Explainability"));
add(P("Attention weights are often presented as explanations. Jain and Wallace [20] showed, however, that attention distributions do not always agree with other importance measures, and Wiegreffe and Pinter [21] argued that whether attention counts as an explanation depends on how the claim is framed and tested. We therefore treat the attention weights here only as a description of how the model distributes weight over time steps within its input window. They are not a causal or feature-level attribution."));

// III. DATASET
add(H1("Dataset"));
add(H2("PRONOSTIA Platform and Operating Conditions"));
add(P("We use the PRONOSTIA accelerated-degradation dataset released by the FEMTO-ST Institute for the IEEE PHM 2012 Prognostic Challenge [10], [11]. A radial load is applied to a test ball bearing mounted on a motor-driven shaft, and two accelerometers placed at 90° to each other (horizontal and vertical) record its vibration. Each vibration snapshot contains 2,560 samples per axis, sampled at 25.6 kHz (0.1 s), and a snapshot is recorded every 10 s. To protect the test rig, each experiment was stopped once the vibration amplitude exceeded 20 g [11]. Three operating conditions are used: 1800 rpm and 4000 N (Condition 1), 1650 rpm and 4200 N (Condition 2), and 1500 rpm and 5000 N (Condition 3)."));
add(P("The learning set has six complete run-to-failure histories (two per condition). The test set has 11 bearings: five for Condition 1, five for Condition 2 and one for Condition 3. For the challenge, the test records were truncated before failure (the Test_set folder). The complete histories were released later (the Full_Test_Set folder). In total, the 17 bearings contain 24,889 snapshots. Table I lists every bearing and the role it plays in this study. Lifetimes are computed as (number of snapshots − 1) × 10 s."));
add(wide(Fig(FIG("fig_raw.png"), "Raw horizontal and vertical acceleration of training bearing Bearing1_1 at snapshot 10 (early life) and at snapshot 2800 (end of test). In the notebook output, the early snapshot peaks at 1.74 g (horizontal) and 1.64 g (vertical), and the late snapshot at 31.17 g and 37.19 g, above the 20 g stopping criterion (dashed lines).", TEXT_W * 0.8, "raw")));
add(...Tbl("PRONOSTIA Bearings, Lifetimes and Roles in This Study",
  ["Bearing", "Cond.", "Snapshots", "Lifetime (s)", "Lifetime (h)", "Role"],
  [
    ["Bearing1_1", "1", "2,803", "28,020", "7.78", "Train"],
    ["Bearing2_1", "2", "911", "9,100", "2.53", "Train"],
    ["Bearing3_1", "3", "515", "5,140", "1.43", "Train"],
    ["Bearing1_2", "1", "871", "8,700", "2.42", "Validation"],
    ["Bearing2_2", "2", "797", "7,960", "2.21", "Validation"],
    ["Bearing3_2", "3", "1,637", "16,360", "4.54", "Validation"],
    ["Bearing1_3", "1", "2,375", "23,740", "6.59", "Test*"],
    ["Bearing1_4", "1", "1,428", "14,270", "3.96", "Test"],
    ["Bearing1_5", "1", "2,463", "24,620", "6.84", "Test"],
    ["Bearing1_6", "1", "2,448", "24,470", "6.80", "Test"],
    ["Bearing1_7", "1", "2,259", "22,580", "6.27", "Test"],
    ["Bearing2_3", "2", "1,955", "19,540", "5.43", "Test*"],
    ["Bearing2_4", "2", "751", "7,500", "2.08", "Test"],
    ["Bearing2_5", "2", "2,311", "23,100", "6.42", "Test"],
    ["Bearing2_6", "2", "701", "7,000", "1.94", "Test"],
    ["Bearing2_7", "2", "230", "2,290", "0.64", "Test"],
    ["Bearing3_3", "3", "434", "4,330", "1.20", "Test*"],
  ],
  [1150, 520, 830, 900, 800, 660], { size: 14, note: "* Member of the three-bearing test subset (one bearing per condition) that was evaluated first in the notebook. Test lifetimes are taken from Full_Test_Set." }));
add(H2("Data Handling Notes"));
add(P("Each snapshot file has six columns: hour, minute, second, microsecond, horizontal acceleration and vertical acceleration. Only the two acceleration columns are used. In the Full_Test_Set copy used here, all 1,428 files of Bearing1_4 use a semicolon as the field delimiter while the other bearings use commas, so the evaluation script selects the delimiter per bearing. Temperature files are not used. The learning set is split at bearing level into three training bearings (Bearing1_1, 2_1, 3_1) and three validation bearings (Bearing1_2, 2_2, 3_2), one per condition in each case. The validation bearings are used only for early stopping and learning-rate scheduling."));

// IV. PROBLEM FORMULATION
add(H1("Problem Formulation"));
add(P("For a bearing *b* with *N*_{b} snapshots, snapshot *k* ∈ {0, …, *N*_{b} − 1} is recorded at operating time *t*_{k} = 10*k* s, and the end of life is *T*_{EOL}(*b*) = 10(*N*_{b} − 1) s. The raw RUL at *t*_{k} is *T*_{EOL} − *t*_{k}. Early in life a bearing's vibration usually carries little information about when it will fail, so, following the piecewise-linear convention [7], we cap the target. The cap is 60% of the longest training lifetime, and only training bearings enter this calculation:"));
add(...Eq("eq_cap"));
add(P("The capped RUL and the normalized regression target are"));
add(...Eq("eq_target"));
add(...Eq("eq_ynorm"));
add(P("An auxiliary stage label is obtained by thresholding the normalized target:"));
add(...Eq("eq_stage"));
add(P("With *RUL*_{cap} = 16,812 s, the stage boundaries fall at 6,724.8 s (112.1 min) and 2,521.8 s (42.0 min). These stages are **RUL horizons**, not vibration-severity zones in the sense of ISO standards. They are defined from the same quantity the regression head predicts and are included mainly to regularize the shared representation. The task is to learn a mapping *f*_{θ} from a window of 16 standardized feature vectors to the pair (*ŷ*_{rul}, **p̂**), where **p̂** is a probability vector over the three stages. The model is trained on the training bearings, selected with the validation bearings, and then applied, frozen, to every window of every test bearing. Fig. 2 illustrates the target for two training bearings, and Table II gives the resulting class balance."));
add(...Fig(FIG("fig_target.png"), "Piecewise-capped RUL target for two training bearings. Bearing1_1 (28,020 s) is capped for its first 40% of snapshots; Bearing2_1 (9,100 s) never reaches the cap. Dotted lines mark the stage thresholds.", COL_W, "target"));
add(...Tbl("Stage Distribution of Sliding Windows (W = 16)",
  ["Split", "Windows", "Normal", "Warning", "Critical"],
  [
    ["Train (3 bearings)", "4,184", "2,338", "1,087", "759"],
    ["Validation (3)", "3,260", "1,241", "1,260", "759"],
    ["Test subset (3)", "4,719", "2,954", "1,006", "759"],
    ["Test, all 11", "17,190", "10,499", "3,946", "2,745"],
  ],
  [1500, 850, 830, 830, 850], { note: "Capped windows (y_{rul} = 1): 26.4% of training windows, none of the validation windows (every validation lifetime is below the cap), and 19.8% of the three-bearing test windows." }));

// V. FEATURE ENGINEERING
add(H1("Feature Engineering"));
add(P("Each snapshot is summarized, separately for each axis, by the 15 indicators in Table III, giving a 30-dimensional feature vector per snapshot. The time-domain indicators describe energy, spread and impulsiveness. For a signal *x*_{1}, …, *x*_{N} with mean *μ*, standard deviation *σ* and peak *x*_{pk} = max|*x*_{i}|, they are"));
add(...Eq("eq_rms"));
add(...Eq("eq_factors"));
add(P("where CF, SF, IF and MF are the crest, shape, impulse and margin factors and the overbar denotes the mean of |*x*_{i}|. The frequency-domain indicators use the single-sided magnitude spectrum *s*_{k} = |*X*(*f*_{k})| of the mean-removed signal (*f*_{k} from 0 to 12.8 kHz):"));
add(...Eq("eq_spectral"));
add(P("together with the spectral energy Σ*s*_{k}^{2} and the peak frequency arg max *s*_{k}. In the implementation, a small constant (10^{−8}) is added to denominators for numerical stability, and kurtosis is reported as excess kurtosis. These indicators are standard in bearing diagnostics [22]. We did not select features and did not rank their importance."));
add(...Tbl("The 15 Per-Axis Features (× 2 Axes = 30)",
  ["#", "Feature", "Domain", "Definition"],
  [
    ["1", "RMS", "Time", "√(mean *x*^{2})"],
    ["2", "Peak", "Time", "max |*x*|"],
    ["3", "Peak-to-peak", "Time", "max *x* − min *x*"],
    ["4", "Std. deviation", "Time", "*σ*"],
    ["5", "Kurtosis", "Time", "4th std. moment − 3"],
    ["6", "Skewness", "Time", "3rd std. moment"],
    ["7", "Crest factor", "Time", "Peak / RMS"],
    ["8", "Shape factor", "Time", "RMS / mean|*x*|"],
    ["9", "Margin factor", "Time", "Peak / (mean √|*x*|)^{2}"],
    ["10", "Impulse factor", "Time", "Peak / mean|*x*|"],
    ["11", "Spectral energy", "Freq.", "Σ *s*_{k}^{2}"],
    ["12", "Spectral centroid", "Freq.", "Σ *f*_{k}*s*_{k} / Σ *s*_{k}"],
    ["13", "Spectral spread", "Freq.", "Eq. (7)"],
    ["14", "Peak frequency", "Freq.", "arg max *s*_{k}"],
    ["15", "RMS frequency", "Freq.", "Eq. (7)"],
  ],
  [350, 1450, 700, 2300], { size: 14 }));
add(P("Fig. 3 shows three of these features over the life of training bearing Bearing1_1. RMS stays nearly flat for several hours, rises gradually after about 4 h and climbs steeply in the final minutes. Kurtosis shows isolated spikes as early as the first 2–3 h and fluctuates more strongly late in life. The spectral centroid wanders over the whole run with no simple monotonic trend. This behaviour is what motivates giving the model a short history of features rather than a single snapshot. It also shows that no single indicator tracks degradation cleanly for this bearing."));
add(P("Features are standardized with a StandardScaler whose mean and standard deviation (**μ**_{train}, **σ**_{train}) are estimated on the 4,229 training snapshots only. A model input is then the window of 16 consecutive standardized vectors ending at snapshot *j*:"));
add(...Eq("eq_window"));
add(P("The targets of a window are those of its last snapshot. Windows are built separately for each bearing with stride 1, so no window spans two bearings, and a bearing with *N*_{b} snapshots yields *N*_{b} − 15 windows."));
add(wide(Fig(FIG("fig_features_b11.png"), "Feature evolution for Bearing1_1 (training bearing): (a) horizontal and vertical RMS, (b) horizontal excess kurtosis, (c) horizontal spectral centroid. Values are read from results/extracted_bearing_features.csv.", TEXT_W * 0.85, "feat")));

// VI. PROPOSED METHODOLOGY
add(H1("Proposed Methodology"));
add(H2("Overview"));
add(P("Fig. 4 summarizes the complete workflow. The left column prepares the data: feature extraction, label generation, the bearing-level split, a scaler fitted on training data only, and windowing. The right column trains the network, runs frozen inference, converts predictions to seconds with the training constant, and evaluates the model in three ways: over trajectories, at the challenge inspection points and through attention weights."));
add(wide(Fig(FIG("fig_flowchart.png"), "Methodology flowchart. Stage A (left, top to bottom) prepares leakage-free inputs and targets; Stage B (right, bottom to top) trains, applies and evaluates the frozen model. The only quantities carried from training into inference are the scaler statistics, the network weights and RUL_{cap}.", TEXT_W * 6.4 / 7.0, "flow")));
add(H2("Network Architecture"));
add(P("The network is shown in Fig. 5 and detailed layer by layer in Table IV. Two 1-D convolutional blocks (64 filters, kernel size 3, 'same' padding, ReLU), each followed by batch normalization [23] and dropout of 0.2 [24], mix neighbouring snapshots and features into local patterns while keeping all 16 time steps:"));
add(...Eq("eq_conv"));
add(wide(Fig(FIG("fig_architecture.png"), "Architecture of the proposed dual-head CNN–BiLSTM–attention network as implemented (Keras layers conv1, bn1, drop1, conv2, bn2, drop2, bilstm, drop_bilstm, temporal_attention, shared_dense, drop_shared, rul_dense/rul_output, risk_dense/risk_output). Output shapes (right) exclude the batch dimension. Solid boxes are network layers; dashed boxes show the exposed attention weights and the post-processing steps applied outside the network.", TEXT_W, "arch")));
add(P("A bidirectional LSTM [16], [25] with 64 units per direction then returns a 128-dimensional hidden state for every time step:"));
add(...Eq("eq_bilstm"));
add(P("The temporal-attention layer is additive attention with a learned context vector [17], [18]. Its parameters are **W**_{a} ∈ ℝ^{128×128}, **b**_{a} ∈ ℝ^{128} and **u** ∈ ℝ^{128}:"));
add(...Eq("eq_att"));
add(P("The context vector **c** feeds a shared 64-unit dense layer. After dropout, this layer branches into two heads, each with its own 32-unit ReLU layer *ϕ*:"));
add(...Eq("eq_heads"));
add(...Eq("eq_heads2"));
add(P("The attention layer returns the weights *α*_{1}, …, *α*_{16} alongside **c**, so they can be read out for any input without changing the model. The network has 113,924 parameters, of which 113,668 are trainable. The 256 non-trainable parameters are the moving statistics of the two batch-normalization layers."));
add(H2("Multi-Task Objective"));
add(P("The two heads are trained jointly by minimizing a weighted sum of a Huber loss on the normalized RUL and a sparse categorical cross-entropy on the stage label, over a mini-batch of size *B*:"));
add(...Eq("eq_loss"));
add(...Eq("eq_huber"));
add(P("The Huber loss [26] behaves quadratically for small residuals and linearly for large ones, which limits the influence of the abrupt jumps typical of late-life vibration. The weights 1.0 and 0.5 make RUL regression the primary task. They were fixed by hand and not tuned."));
add(H2("Inference and De-Normalization"));
add(P("At test time the regression output is clipped to [0, 1] and multiplied by the training constant:"));
add(...Eq("eq_denorm"));
add(P("The lifetime of the bearing being evaluated appears nowhere in this step. That lifetime is used only afterwards, to compute the ground-truth labels against which errors are measured. The predicted stage is the arg max of **p̂**. Algorithm 1 summarizes the procedure."));
add(...Algorithm("**Algorithm 1:** Leakage-free training and inference", [
  [0, "**Input:** train set 𝓑_{tr}, validation set 𝓑_{va}, test set 𝓑_{te}; *W* = 16"],
  [0, "**Output:** frozen model *f*_{θ}; predicted RUL and stage per test window"],
  [0, "1: **for** each bearing *b* and snapshot *k* **do**"],
  [1, "2: **f**_{k} ← 30 features of both axes (Table III)"],
  [0, "3: *T*_{max} ← max_{b∈𝓑tr} *T*_{EOL}(*b*); *RUL*_{cap} ← 0.6 *T*_{max}"],
  [0, "4: compute *y*_{rul} and *s* for every snapshot (Eqs. 2–4)"],
  [0, "5: fit scaler (**μ**, **σ**) on 𝓑_{tr} snapshots only"],
  [0, "6: **for** each bearing separately **do**"],
  [1, "7: build windows **X**_{j} of length *W*, stride 1 (Eq. 8)"],
  [0, "8: initialize *f*_{θ} (Fig. 5); Adam, lr = 10^{−3}"],
  [0, "9: **for** epoch = 1 … 35 **do**"],
  [1, "10: minimize Eq. (14) on 𝓑_{tr} windows, batch 64"],
  [1, "11: evaluate val. loss on 𝓑_{va}; halve lr after 3 stale epochs"],
  [1, "12: **if** 7 epochs without improvement **then** stop"],
  [0, "13: restore weights with lowest validation loss; freeze"],
  [0, "14: **for** each window of each *b* ∈ 𝓑_{te} **do**"],
  [1, "15: (*ŷ*_{rul}, **p̂**, **α**) ← *f*_{θ}(**X**_{j})"],
  [1, "16: predicted RUL ← clip(*ŷ*_{rul}, 0, 1) × *RUL*_{cap}; *ŝ* ← arg max **p̂**"],
  [0, "17: compute metrics against *RUL*_{c} and *s* (Section VII)"],
]));

// VII. EXPERIMENTAL SETUP
const body2 = [];
const add2 = (...xs) => xs.forEach((x) => (Array.isArray(x) ? body2.push(...x) : body2.push(x)));

add2(...Tbl("Layer-Wise Configuration of the Proposed Network",
  ["Layer", "Configuration", "Output shape", "Params"],
  [
    ["Input", "16 steps × 30 features", "(16, 30)", "0"],
    ["Conv1D-1 + BN", "64 filters, k = 3, ReLU; dropout 0.2", "(16, 64)", "5,824 + 256"],
    ["Conv1D-2 + BN", "64 filters, k = 3, ReLU; dropout 0.2", "(16, 64)", "12,352 + 256"],
    ["BiLSTM", "64 units × 2 directions; dropout 0.2", "(16, 128)", "66,048"],
    ["Temporal attention", "W_{a} 128×128, b_{a}, u", "(128) + (16, 1)", "16,640"],
    ["Shared dense", "64, ReLU; dropout 0.2", "(64)", "8,256"],
    ["RUL head", "Dense 32 ReLU → Dense 1 linear", "(1)", "2,080 + 33"],
    ["Stage head", "Dense 32 ReLU → Dense 3 softmax", "(3)", "2,080 + 99"],
    ["**Total**", "", "", "**113,924**"],
  ],
  [1100, 2000, 950, 810], { size: 14 }));

add2(H1("Experimental Setup"));
add2(H2("Implementation and Training"));
add2(P("The pipeline was implemented in Python 3.12.6 with TensorFlow 2.20.0 / Keras 3, NumPy 2.2.6, pandas 2.2.2 and scikit-learn [27], with all random seeds set to 42. Table V lists the training settings. Early stopping monitored the validation loss with a patience of 7 epochs and restored the best weights; the learning rate was halved after 3 epochs without improvement. The training history is reported in Section VIII-A. Training took 13.45 s of wall-clock time, as reported in the notebook output."));
add2(...Tbl("Training Configuration",
  ["Setting", "Value"],
  [
    ["Window length / stride", "16 snapshots (160 s) / 1"],
    ["Optimizer, initial learning rate", "Adam [28], 1 × 10^{−3}"],
    ["Batch size / max. epochs", "64 / 35"],
    ["Loss weights (RUL, stage)", "1.0 (Huber), 0.5 (cross-entropy)"],
    ["Early stopping", "val. loss, patience 7, restore best"],
    ["LR schedule", "ReduceLROnPlateau ×0.5, patience 3, min 10^{−5}"],
    ["Dropout", "0.2 after every block"],
    ["RUL cap / stage thresholds", "16,812 s / 0.40 and 0.15 of cap"],
  ],
  [2300, 2560], { size: 15 }));
add2(H2("Evaluation Protocols"));
add2(P("**Trajectory evaluation.** The frozen model scores every window of every test bearing: 4,719 windows for the three-bearing subset and 17,190 windows (from 17,355 snapshots) for all 11 test bearings. Errors are measured against the capped ground truth *RUL*_{c}, both in physical units,"));
add2(...Eq("eq_mae"));
add2(...Eq("eq_rmse"));
add2(P("and in normalized units (the MAE between *y*_{rul} and the clipped *ŷ*_{rul}, which we call the normalized MAE)."));
add2(P("**PHM scoring function.** For a prediction, the percent error and the challenge accuracy score are [11]"));
add2(...Eq("eq_er"));
add2(...Eq("eq_ai"));
add2(P("A negative %*Er* means that the model over-estimates the remaining life, which the challenge penalizes more heavily (exponent 5) than an under-estimate (exponent 20). Averaging *A*_{i} gives two different summary scores:"));
add2(...Eq("eq_score"));
add2(P("The challenge **Score** averages one *A*_{i} per test bearing, computed at the last snapshot of that bearing's truncated Test_set record. The trajectory-averaged **T-Score** averages *A*_{i} over all *M* windows of a trajectory, using *RUL*_{c} as the actual value. The T-Score is our own diagnostic and is not part of the challenge protocol. Its implementations also differ slightly. The notebook (three-bearing subset) skips windows whose capped RUL is zero and clips *A*_{i} to [0, 1], whereas the 11-bearing script keeps those windows, with a small constant in the denominator. The two T-Scores are therefore close but not strictly comparable."));
add2(P("**Stage classification.** We report accuracy, weighted precision, recall and F1, and the macro F1, all computed with scikit-learn."));

// VIII. RESULTS
add2(H1("Results"));
add2(H2("Training Dynamics"));
add2(P("Table VI and Fig. 6 give the per-epoch history. The training loss falls steadily, from 0.2339 to 0.0202, and training stage accuracy climbs to 98.97%. The validation loss, however, is lowest after the first epoch (0.7702) and rises afterwards, reaching 1.8565 at epoch 8. Validation stage accuracy stays between 34.97% and 39.20%. Early stopping therefore ended training after epoch 8 and restored the epoch-1 weights. These epoch-1 weights are the model evaluated in the rest of the paper. The widening gap shows that the network fits the three training trajectories far better than it transfers to the validation bearings. In our view this reflects the different lifetimes of training and validation bearings under the same conditions (e.g., 7.78 h for Bearing1_1 versus 2.42 h for Bearing1_2), although the experiments here cannot separate that cause from ordinary over-fitting to three bearings."));
add2(...Tbl("Per-Epoch Training History (Epoch Averages)",
  ["Ep.", "Train loss", "Val. loss", "Train MAE*", "Val. MAE*", "Train acc.", "Val. acc.", "LR"],
  [
    ["1", "0.2339", "**0.7702**", "0.1894", "0.2326", "82.89%", "37.45%", "1e-3"],
    ["2", "0.0874", "1.1371", "0.1246", "0.2002", "94.12%", "34.97%", "1e-3"],
    ["3", "0.0636", "1.4890", "0.1039", "0.2373", "95.84%", "38.56%", "1e-3"],
    ["4", "0.0518", "1.6492", "0.0922", "0.3053", "96.44%", "35.49%", "1e-3"],
    ["5", "0.0317", "1.8276", "0.0851", "0.3027", "98.18%", "35.46%", "5e-4"],
    ["6", "0.0268", "1.8701", "0.0813", "0.2614", "98.42%", "39.20%", "5e-4"],
    ["7", "0.0265", "1.7724", "0.0794", "0.2645", "98.47%", "37.82%", "5e-4"],
    ["8", "0.0202", "1.8565", "0.0754", "0.2486", "98.97%", "37.73%", "2.5e-4"],
  ],
  [360, 640, 640, 640, 640, 650, 650, 640], { size: 14, note: "* Normalized RUL MAE. Bold: best validation loss; the epoch-1 weights were restored and used for all test results." }));
add2(wide(Fig(FIG("fig_training.png"), "Training convergence, redrawn from the per-epoch values logged in the executed notebook (Table VI): (a) total multi-task loss, (b) normalized RUL MAE, (c) stage accuracy, for training (solid) and validation (dashed) bearings. The dotted line marks epoch 1, whose weights were restored.", TEXT_W, "conv")));

add2(H2("Aggregate Test Performance"));
add2(P("Table VII compares the three-bearing subset with the full 11-bearing test set. On all 11 bearings the model reaches an RUL MAE of 93.40 min (5,603.87 s), an RMSE of 110.49 min and a normalized MAE of 0.3333, which is one third of the capped range. The T-Score is 0.1667. The stage head reaches 52.82% accuracy, a weighted F1 of 51.61% and a macro F1 of 40.22%. The 11-bearing figures are better than those of the three-bearing subset (104.41 min MAE, 0.1244 T-Score). This should not be read as the model generalizing better to more data, because the same frozen model is scored on a larger and different set of trajectories. As Section VIII-C shows, several of the added bearings are short-lived, and short trajectories contribute small absolute errors in minutes."));
add2(...Tbl("Aggregate Results on Held-Out Test Bearings",
  ["Metric", "3-bearing subset", "All 11 bearings"],
  [
    ["Test bearings / snapshots", "3 / 4,764", "11 / 17,355"],
    ["Evaluated windows (W = 16)", "4,719", "17,190"],
    ["RUL MAE (min)", "104.41", "93.40"],
    ["RUL MAE (s)", "6,264.38", "5,603.87"],
    ["RUL RMSE (min)", "122.10", "110.49"],
    ["RUL RMSE (s)", "7,326.13", "6,629.43"],
    ["Normalized RUL MAE", "0.3726", "0.3333"],
    ["Trajectory T-Score", "0.1244", "0.1667"],
    ["Stage accuracy (%)", "50.16", "52.82"],
    ["Weighted precision (%)", "47.44", "50.73"],
    ["Weighted recall (%)", "50.16", "52.82"],
    ["Weighted F1 (%)", "48.46", "51.61"],
    ["Macro F1 (%)", "33.12", "40.22"],
  ],
  [2100, 1350, 1350], { size: 15, note: "Subset: Bearing1_3, 2_3, 3_3. Macro precision / recall on all 11 bearings: 39.67% / 41.26%." }));

add2(H2("Per-Bearing Analysis"));
add2(P("Table VIII and Fig. 7 break the 11-bearing result down by bearing. The unweighted mean of the per-bearing MAEs is 96.96 min for Condition 1, 73.50 min for Condition 2 and 61.40 min for Condition 3 (normalized: 0.3460, 0.2623, 0.2191). The two lowest errors belong to Bearing2_4 (54.12 min) and Bearing1_4 (56.23 min). The highest belong to Bearing1_6 (129.52 min) and Bearing1_3 (114.43 min). Bearing1_4 also has the highest T-Score (0.2668)."));
add2(P("Absolute errors must be read against the length of each trajectory. Condition 3 has a single test bearing, and the three shortest-lived test bearings (Bearing2_6, Bearing2_7 and Bearing3_3, all under 2 h) belong to Conditions 2 and 3, so the lower averages for these conditions partly reflect shorter targets rather than better tracking. The T-Score shows the other side of this. It is essentially zero for Bearing2_7 (0.0000) and Bearing3_3 (0.0051), because on these short trajectories the model predicts more remaining life than is actually left, and the challenge metric punishes over-estimates severely."));
add2(P("The signed error confirms this pattern. Taking the mean of (predicted − true) over each bearing's windows, computed from the per-window predictions file, the model under-estimates RUL on all long-lived Condition 1 bearings (e.g., −103.77 min for Bearing1_3 and −112.70 min for Bearing1_6; −3.97 min for Bearing1_4) and on Bearing2_3 and Bearing2_5. It over-estimates on the four shortest bearings (Bearing2_4 +25.56, Bearing2_6 +48.19, Bearing2_7 +69.78 and Bearing3_3 +60.15 min). Over all windows the mean signed error is −51.73 min. The predictions are thus pulled toward a middle range of roughly one to two and a half hours, whatever the lifetime of the bearing. Fig. 8 shows this directly: early in the life of long bearings the prediction stays well below the 280-min cap, and on Bearing3_3 it stays above the true RUL for most of the run. On Bearing1_3 the prediction rises sharply at about 270 min, drops to zero at about 290 min, roughly 100 min before the actual end, stays near zero with intermittent spikes, and rises again to about 130 min in the final minutes. The model therefore reacts to changes in the vibration features, but its sense of how much time is left is not calibrated across bearings."));
add2(P("The stage head performs best on Bearing1_3 (77.80% accuracy), Bearing1_5 (75.04%), Bearing1_4 (70.28%) and Bearing2_5 (69.95%). It performs very poorly on the short bearings: 8.60% on Bearing2_6, 12.09% on Bearing2_7 and 8.59% on Bearing3_3. Because their lifetimes are short, most windows of these bearings are labelled Warning or Critical, while the model mostly predicts Normal or mixes up the stages. Their high weighted precision (e.g., 100.00% on Bearing2_7) arises because the few windows predicted as a present class happen to be correct. It is not evidence of good performance."));
add2(wide(Tbl("Per-Bearing Results on All 11 Test Bearings (Frozen Model)",
  ["Bearing", "Cond.", "Snap-shots", "Win-dows", "Life (h)", "MAE (min)", "RMSE (min)", "Norm. MAE", "T-Score", "Acc. (%)", "Prec. (%)", "Rec. (%)", "W-F1 (%)", "M-F1 (%)"],
  [
    ["Bearing1_3", "1", "2,375", "2,360", "6.59", "114.43", "128.51", "0.4084", "0.1150", "77.80", "75.12", "77.80", "74.64", "49.16"],
    ["Bearing1_4", "1", "1,428", "1,413", "3.96", "56.23", "67.08", "0.2007", "0.2668", "70.28", "49.43", "70.28", "58.03", "55.59"],
    ["Bearing1_5", "1", "2,463", "2,448", "6.84", "89.57", "101.31", "0.3197", "0.2092", "75.04", "64.28", "75.04", "65.93", "41.56"],
    ["Bearing1_6", "1", "2,448", "2,433", "6.80", "129.52", "146.49", "0.4622", "0.0951", "23.35", "79.17", "23.35", "19.73", "31.57"],
    ["Bearing1_7", "1", "2,259", "2,244", "6.27", "95.03", "105.63", "0.3392", "0.1716", "58.42", "63.50", "58.42", "59.78", "33.47"],
    ["Bearing2_3", "2", "1,955", "1,940", "5.43", "101.50", "123.41", "0.3623", "0.1615", "25.52", "29.50", "25.52", "27.27", "14.90"],
    ["Bearing2_4", "2", "751", "736", "2.08", "54.12", "64.94", "0.1931", "0.1887", "42.39", "49.89", "42.39", "44.90", "26.67"],
    ["Bearing2_5", "2", "2,311", "2,296", "6.42", "82.85", "100.21", "0.2957", "0.2306", "69.95", "58.63", "69.95", "62.18", "39.93"],
    ["Bearing2_6", "2", "701", "686", "1.94", "59.27", "68.35", "0.2115", "0.1529", "8.60", "98.14", "8.60", "12.56", "10.42"],
    ["Bearing2_7", "2", "230", "215", "0.64", "69.78", "80.96", "0.2491", "0.0000", "12.09", "100.00", "12.09", "21.58", "7.19"],
    ["Bearing3_3", "3", "434", "419", "1.20", "61.40", "66.15", "0.2191", "0.0051", "8.59", "60.38", "8.59", "15.04", "8.30"],
    ["**All 11**", "", "**17,355**", "**17,190**", "", "**93.40**", "**110.49**", "**0.3333**", "**0.1667**", "**52.82**", "**50.73**", "**52.82**", "**51.61**", "**40.22**"],
  ],
  [1060, 460, 700, 700, 560, 700, 720, 720, 700, 660, 660, 660, 700, 700],
  { size: 14, note: "Precision, recall and F1 of the stage head are support-weighted (W-F1) or macro-averaged (M-F1) within each bearing. Life = operating time of the last snapshot." })));
add2(wide(Fig(FIG("fig_per_bearing.png"), "Per-bearing results on the 11 test bearings: (a) RUL MAE and (b) auxiliary stage accuracy, coloured by operating condition. Dashed lines show the aggregate values over all 17,190 windows.", TEXT_W * 0.9, "perb")));
add2(wide(Fig(FIG("fig_trajectories.png"), "Predicted versus capped ground-truth RUL for four test bearings, redrawn from results/test_predictions_all_11_bearings.csv. The capped target starts below 280.2 min whenever the bearing's lifetime is shorter than RUL_{cap} (Bearing1_4, Bearing3_3).", TEXT_W * 0.9, "traj")));

add2(H2("Auxiliary Stage Classification"));
add2(P("The confusion matrix in Fig. 9 and the per-class metrics in Table IX show where the stage head succeeds and fails. The Normal stage, the majority class with 10,499 of 17,190 windows, is recognized with 71.19% recall and 67.46% precision. The Critical stage reaches 39.20% recall. The Warning stage is almost never recognized correctly: 13.41% recall and an F1 of 15.57%. Warning windows are more often classified as Normal (2,150) or Critical (1,267) than as Warning (529). For reference, always predicting Normal would give 61.08% accuracy (10,499 / 17,190) but a macro F1 of only about 25%. The model's 52.82% accuracy is below this trivial baseline, while its 40.22% macro F1 is above it, because it does detect some Critical windows. The stage head is therefore useful only as a coarse indicator. As a warning system it would miss most intermediate-horizon cases."));
add2(...Tbl("Per-Class Stage Metrics on All 11 Test Bearings",
  ["Stage", "Support", "Precision (%)", "Recall (%)", "F1 (%)"],
  [
    ["Normal (0)", "10,499", "67.46", "71.19", "69.27"],
    ["Warning (1)", "3,946", "18.56", "13.41", "15.57"],
    ["Critical (2)", "2,745", "33.00", "39.20", "35.83"],
    ["Macro avg.", "17,190", "39.67", "41.26", "40.22"],
    ["Weighted avg.", "17,190", "50.73", "52.82", "51.61"],
  ],
  [1250, 850, 950, 900, 850], { size: 15, note: "Per-class values are computed from results/test_predictions_all_11_bearings.csv; the averages match the frozen metrics JSON." }));
add2(...Fig(FIG("fig_confusion_11.png"), "Confusion matrix of the auxiliary stage head over all 17,190 test windows (counts, with row-normalized percentages in parentheses).", COL_W, "cm"));

add2(H2("IEEE PHM 2012 Single-Inspection Evaluation"));
add2(P("To compare against the original challenge protocol, we take the model's prediction for the window that ends at the last snapshot of each bearing's truncated Test_set record. The Test_set file counts are 1,802, 1,139, 2,302, 2,302, 1,502, 1,202, 612, 2,002, 572, 172 and 352 for Bearing1_3 through Bearing3_3, respectively. Each such window lies entirely within the released test data, and the features of a snapshot depend only on that snapshot, so these predictions use no information from after the inspection point. Table X lists the results and Fig. 10 compares actual and predicted RUL at each inspection point."));
add2(P("The actual RUL values in Table 3 of the challenge document agree with the file counts of Full_Test_Set for ten of the eleven bearings. For Bearing1_4 the document gives 339 s, whereas (1,428 − 1,139) × 10 s = 2,890 s follows from the released files. We cannot tell from the available material which value is intended, so we report both. With the published values the challenge score is **0.0234**. Replacing only the Bearing1_4 value with 2,890 s gives **0.0718**, which we present as a sensitivity analysis. The difference comes entirely from Bearing1_4, whose prediction of 2,365.7 s is 8.74 min below the file-derived RUL (%*Er* = +18.14%, *A*_{i} = 0.5333) but far above the published 339 s (*A*_{i} ≈ 0). Over the three-bearing subset the score is 0.0375."));
add2(P("Apart from Bearing1_4, only Bearing1_7 (0.1446), Bearing2_3 (0.0811) and Bearing1_3 (0.0313) receive a non-zero score. Eight of the eleven predictions over-estimate the remaining life. In seven of them the over-estimate ranges from 97.79% to 1,576.49%, and the score rounds to zero. For Bearing1_3 the prediction is clipped to 0 s while 5,730 s actually remain. The single-inspection MAE is 68.60 min and the RMSE is 80.61 min. These numbers mean the model is not competitive with the leading challenge entries. We did not compute published comparison scores ourselves and therefore do not tabulate them."));
add2(wide(Tbl("Single-Inspection Results at the Last Test_set Snapshot",
  ["Bearing", "Insp. file", "*T*_{trunc} (h)", "Actual RUL (s)", "Pred. RUL (s)", "|Err| (min)", "%*Er*", "*A*_{i}", "Pred. stage"],
  [
    ["Bearing1_3", "acc_01802", "5.00", "5,730", "0.0", "95.50", "+100.00", "0.0313", "Critical"],
    ["Bearing1_4", "acc_01139", "3.16", "2,890 (339†)", "2,365.7", "8.74", "+18.14", "0.5333 (0.0000†)", "Critical"],
    ["Bearing1_5", "acc_02302", "6.39", "1,610", "7,485.7", "97.93", "−364.95", "0.0000", "Normal"],
    ["Bearing1_6", "acc_02302", "6.39", "1,460", "3,612.2", "35.87", "−147.41", "0.0000", "Critical"],
    ["Bearing1_7", "acc_01502", "4.17", "7,570", "3,345.9", "70.40", "+55.80", "0.1446", "Critical"],
    ["Bearing2_3", "acc_01202", "3.34", "7,530", "8,894.7", "22.74", "−18.12", "0.0811", "Normal"],
    ["Bearing2_4", "acc_00612", "1.70", "1,390", "3,597.4", "36.79", "−158.81", "0.0000", "Warning"],
    ["Bearing2_5", "acc_02002", "5.56", "3,090", "6,111.7", "50.36", "−97.79", "0.0000", "Normal"],
    ["Bearing2_6", "acc_00572", "1.59", "1,290", "8,661.3", "122.86", "−571.42", "0.0000", "Normal"],
    ["Bearing2_7", "acc_00172", "0.47", "580", "9,723.7", "152.39", "−1576.49", "0.0000", "Critical"],
    ["Bearing3_3", "acc_00352", "0.97", "820", "4,481.5", "61.02", "−446.52", "0.0000", "Warning"],
    ["**Score**", "", "", "", "", "**68.60** (MAE)", "", "**0.0234**† / 0.0718", ""],
  ],
  [1150, 1000, 800, 1300, 1000, 950, 1000, 1500, 950],
  { size: 14, note: "† Published ground truth (challenge Table 3) for Bearing1_4. All other columns use the file-derived value of 2,890 s. The official score using the published values is 0.0234; 0.0718 is the Bearing1_4 sensitivity value. Single-inspection RMSE = 80.61 min." })));
add2(wide(Fig(FIG("fig_single_inspection.png"), "Actual (file-derived) and predicted RUL at the single inspection point of each test bearing, with the per-bearing challenge score A_{i} (file-derived ground truth; with the published 339 s, Bearing1_4 scores 0.0000).", TEXT_W * 0.9, "si")));

add2(H2("Temporal Attention Analysis"));
add2(P("Fig. 11 shows the attention weights *α*_{1}, …, *α*_{16} for two windows of test bearing Bearing1_3. One is an early window starting at snapshot 50, when the bearing is healthy; the other is the final window of the record, ending at the last snapshot. With uniform weights every step would receive 1/16 = 0.0625. In the early window the weights are spread fairly evenly, ranging from 0.0367 to a maximum of 0.0813 at step 12. In the final window the three most recent steps (14–16) receive the largest weights, with a maximum of 0.1120 at step 16, and the earliest steps receive less than 0.05. So near failure the model does weight its most recent observations more heavily. The concentration is moderate, though: even the largest weight is below 1.8 times the uniform value, and most of the weight is still spread over the whole window."));
add2(P("These two windows come from a single bearing and are illustrations, not a statistical analysis. The weights describe how the model combines time steps. They do not show which of the 30 features drove a prediction, and they do not establish that the recent steps cause the prediction [20], [21]. Feature-level attribution was not performed in this study."));
add2(wide(Fig(FIG("fig_attention.png"), "Temporal attention weights of the frozen model for test bearing Bearing1_3 (identical to the values printed in the executed notebook): (a) early window starting at snapshot 50, peak 0.0813 at step 12; (b) final window of the record, peak 0.1120 at step 16. Step 16 is the most recent snapshot; the dashed line marks uniform weighting (1/16).", TEXT_W, "att")));

// IX. DISCUSSION
add2(H1("Discussion"));
add2(H2("Evidence-Based Interpretation"));
add2(P("Under a protocol that keeps all test-bearing information out of preprocessing and inference, a CNN–BiLSTM–attention model trained on three bearings predicts capped RUL on 11 unseen bearings with an MAE of 93.40 min, about one third of the 280.2-min cap. Its single-inspection score on the challenge is 0.0234 with the published ground truth. These numbers, and the per-bearing tables behind them, are the paper's main empirical content."));
add2(H2("Interpretation of the Error Pattern"));
add2(P("The signed-error analysis and the trajectories point to one dominant failure mode: the predictions are compressed toward a middle range and do not scale with the lifetime of the bearing. We think two features of the setup contribute to this. First, the three training bearings span lifetimes from 1.43 h to 7.78 h, and only Bearing1_1 contributes capped windows, so the model sees few examples of long, flat early life. Second, the weights were taken from the first epoch, the point of lowest validation loss, when the network has seen the training data only once. A model selected this early is likely to predict a smooth average. Both explanations are plausible but untested, because we did not run ablations or repeated trainings."));
add2(H2("Significance of the Evaluation Protocol"));
add2(P("An earlier internal version of the pipeline converted normalized predictions to seconds by multiplying by each test bearing's own lifetime. That version was corrected before the results reported here were produced, and it gave visibly more optimistic errors, because multiplying by the true lifetime rescales every prediction to the correct horizon. We mention this only to underline a point made in the literature [12]: in prognostics, how the normalizing constant is chosen can matter as much as the choice of architecture."));
add2(H2("Trajectory-Level Versus Single-Inspection Evaluation"));
add2(P("The T-Score (0.1667) and the challenge score (0.0234) differ by a factor of about seven, even though they use the same scoring function *A*_{i}. The trajectory view rewards predictions that are reasonable over long stretches, while the single-inspection view depends on one prediction made near the end of each truncated record, where this model often over-estimates. For maintenance planning the trajectory view is closer to how a monitoring system would be used. The challenge score, on the other hand, is the only number that allows a like-for-like comparison with the 2012 results. We therefore report both and advise against quoting either one alone."));
add2(H2("Role of the Auxiliary Stage Head"));
add2(P("Because the stages are thresholds on the regression target, the stage head adds no new supervision about the bearing's physical condition. Its low Warning recall is consistent with the regression errors: a model that cannot tell 60 min of remaining life from 150 min cannot separate the Warning band from its neighbours. We did not test whether the auxiliary task helps or hurts the regression head."));

// X. LIMITATIONS
add2(H1("Limitations"));
add2(B("**Single training run without baselines or ablations.** All results come from a single training run with seed 42. We did not measure variance across seeds or compare against simpler baselines (e.g., a single-head model, the model without attention, or a regressor on hand-crafted features), so no component can be credited with a specific gain."));
add2(B("**Limited training data.** Only three bearings were used for training and three for validation. The six learning-set bearings were not pooled, and no cross-validation over bearings was done."));
add2(B("**Early model selection.** The evaluated weights come from epoch 1. Validation loss increased steadily after that, so the model is lightly trained, and the validation bearings strongly influenced which weights were kept."));
add2(B("**Hand-set label design.** The RUL cap ratio (0.60) and the stage thresholds (0.40, 0.15) were set by hand. The stages come from RUL, not from measured vibration severity."));
add2(B("**Ground-truth ambiguity.** The Bearing1_4 value in the challenge document (339 s) does not match the released files (2,890 s), which changes the challenge score from 0.0234 to 0.0718."));
add2(B("**Metric definitions.** The T-Score is non-standard, and its two implementations differ in how zero-RUL windows are handled, so the three-bearing and 11-bearing T-Scores are not strictly comparable."));
add2(B("**Limited coverage of Condition 3.** Condition 3 has one test bearing. Results for Condition 3 (and, to a lesser extent, the per-condition averages) should not be generalized."));
add2(B("**Limited scope of explainability.** The attention weights are temporal only, were inspected for two windows of one bearing, and were not validated against any other attribution method."));
add2(B("**No uncertainty quantification or deployment evaluation.** The model produces point estimates without confidence intervals. It was evaluated offline on one laboratory dataset and was not tested on other rigs, variable operating conditions, streaming data or embedded hardware. The notebook also renders an illustrative operator dashboard; it was not evaluated with users and is not part of the claims in this paper."));

// XI. CONCLUSION
add2(H1("Conclusion and Future Work"));
add2(P("We presented an explainable dual-head CNN–BiLSTM–attention model for bearing RUL prediction and evaluated it on the PRONOSTIA / IEEE PHM 2012 data under a protocol that uses no test-bearing lifetime information in preprocessing or inference. On all 11 test bearings the model reaches an RUL MAE of 93.40 min, an RMSE of 110.49 min, a normalized MAE of 0.3333 and a trajectory-averaged PHM score of 0.1667. Its auxiliary stage head reaches 52.82% accuracy, below a majority-class baseline, and recognizes the Warning stage poorly. At the official single-inspection points the challenge score is 0.0234, or 0.0718 when the Bearing1_4 ground truth is taken from the released files. The per-bearing analysis shows that predictions are compressed toward a middle range: RUL is under-estimated on long-lived bearings and over-estimated on short-lived ones. The attention weights shift moderately toward the most recent observations near failure."));
add2(P("These results establish a transparent, reproducible reference point rather than a finished monitoring system. The next steps follow directly from the limitations: (i) repeated training with several seeds, together with baselines and ablations of the attention layer and the auxiliary head; (ii) training on all six learning bearings with leave-one-bearing-out validation; (iii) degradation-onset detection or health-indicator-based targets in place of a fixed cap; (iv) uncertainty estimates for each prediction; (v) feature-level attribution (e.g., SHAP [29]) checked against the attention weights; and (vi) evaluation on a second dataset such as XJTU-SY [13] to test cross-dataset transfer."));

add2(H1u("Reproducibility Statement"));
add2(P("Every number in this paper can be traced to files in the project repository: the executed notebook (explainable_dual_head_bearing_prognostics.ipynb), the frozen model and scaler (models/), the evaluation scripts (evaluate_all_11_test_bearings.py, evaluate_challenge_single_inspection.py), and the results in results/ (metrics JSON files, per-bearing and per-window CSV files). Figures not taken from the notebook were redrawn from these CSV files with paper/make_figures.py."));

// REFERENCES
const refs = [
  "Y. Lei, N. Li, L. Guo, N. Li, T. Yan, and J. Lin, “Machinery health prognostics: A systematic review from data acquisition to RUL prediction,” *Mech. Syst. Signal Process.*, vol. 104, pp. 799–834, 2018.",
  "A. K. S. Jardine, D. Lin, and D. Banjevic, “A review on machinery diagnostics and prognostics implementing condition-based maintenance,” *Mech. Syst. Signal Process.*, vol. 20, no. 7, pp. 1483–1510, 2006.",
  "X.-S. Si, W. Wang, C.-H. Hu, and D.-H. Zhou, “Remaining useful life estimation – A review on the statistical data driven approaches,” *Eur. J. Oper. Res.*, vol. 213, no. 1, pp. 1–14, 2011.",
  "R. Zhao, R. Yan, Z. Chen, K. Mao, P. Wang, and R. X. Gao, “Deep learning and its applications to machine health monitoring,” *Mech. Syst. Signal Process.*, vol. 115, pp. 213–237, 2019.",
  "G. S. Babu, P. Zhao, and X.-L. Li, “Deep convolutional neural network based regression approach for estimation of remaining useful life,” in *Database Systems for Advanced Applications (DASFAA 2016)*, Lecture Notes in Computer Science. Cham, Switzerland: Springer, 2016, pp. 214–228.",
  "X. Li, Q. Ding, and J.-Q. Sun, “Remaining useful life estimation in prognostics using deep convolution neural networks,” *Reliab. Eng. Syst. Saf.*, vol. 172, pp. 1–11, 2018.",
  "F. O. Heimes, “Recurrent neural networks for remaining useful life estimation,” in *Proc. Int. Conf. Prognostics and Health Management (PHM)*, 2008, pp. 1–6.",
  "S. Zheng, K. Ristovski, A. Farahat, and C. Gupta, “Long short-term memory network for remaining useful life estimation,” in *Proc. IEEE Int. Conf. Prognostics and Health Management (ICPHM)*, 2017, pp. 88–95.",
  "L. Guo, N. Li, F. Jia, Y. Lei, and J. Lin, “A recurrent neural network based health indicator for remaining useful life prediction of bearings,” *Neurocomputing*, vol. 240, pp. 98–109, 2017.",
  "P. Nectoux, R. Gouriveau, K. Medjaher, E. Ramasso, B. Chebel-Morello, N. Zerhouni, and C. Varnier, “PRONOSTIA: An experimental platform for bearings accelerated degradation tests,” in *Proc. IEEE Int. Conf. Prognostics and Health Management (PHM)*, Denver, CO, USA, 2012, pp. 1–8.",
  "FEMTO-ST Institute, “IEEE PHM 2012 Prognostic Challenge: Outline, experiments, scoring of results, winners,” challenge documentation distributed with the PRONOSTIA dataset, 2012.",
  "S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, “Leakage in data mining: Formulation, detection, and avoidance,” *ACM Trans. Knowl. Discov. Data*, vol. 6, no. 4, pp. 1–21, Dec. 2012.",
  "B. Wang, Y. Lei, N. Li, and N. Li, “A hybrid prognostics approach for estimating remaining useful life of rolling element bearings,” *IEEE Trans. Rel.*, vol. 69, no. 1, pp. 401–412, 2020.",
  "Z. Chen, M. Wu, R. Zhao, F. Guretno, R. Yan, and X. Li, “Machine remaining useful life prediction via an attention-based deep learning approach,” *IEEE Trans. Ind. Electron.*, vol. 68, no. 3, pp. 2521–2531, 2021.",
  "Y. Chen, G. Peng, Z. Zhu, and S. Li, “A novel deep learning method based on attention mechanism for bearing remaining useful life prediction,” *Appl. Soft Comput.*, vol. 86, Art. no. 105919, 2020.",
  "M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” *IEEE Trans. Signal Process.*, vol. 45, no. 11, pp. 2673–2681, 1997.",
  "D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine translation by jointly learning to align and translate,” in *Proc. Int. Conf. Learning Representations (ICLR)*, 2015.",
  "Z. Yang, D. Yang, C. Dyer, X. He, A. Smola, and E. Hovy, “Hierarchical attention networks for document classification,” in *Proc. NAACL-HLT*, 2016, pp. 1480–1489.",
  "R. Caruana, “Multitask learning,” *Mach. Learn.*, vol. 28, no. 1, pp. 41–75, 1997.",
  "S. Jain and B. C. Wallace, “Attention is not explanation,” in *Proc. NAACL-HLT*, 2019, pp. 3543–3556.",
  "S. Wiegreffe and Y. Pinter, “Attention is not not explanation,” in *Proc. EMNLP-IJCNLP*, 2019, pp. 11–20.",
  "R. B. Randall and J. Antoni, “Rolling element bearing diagnostics—A tutorial,” *Mech. Syst. Signal Process.*, vol. 25, no. 2, pp. 485–520, 2011.",
  "S. Ioffe and C. Szegedy, “Batch normalization: Accelerating deep network training by reducing internal covariate shift,” in *Proc. 32nd Int. Conf. Machine Learning (ICML)*, PMLR vol. 37, 2015, pp. 448–456.",
  "N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: A simple way to prevent neural networks from overfitting,” *J. Mach. Learn. Res.*, vol. 15, pp. 1929–1958, 2014.",
  "S. Hochreiter and J. Schmidhuber, “Long short-term memory,” *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997.",
  "P. J. Huber, “Robust estimation of a location parameter,” *Ann. Math. Statist.*, vol. 35, no. 1, pp. 73–101, 1964.",
  "F. Pedregosa *et al.*, “Scikit-learn: Machine learning in Python,” *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.",
  "D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in *Proc. 3rd Int. Conf. Learning Representations (ICLR)*, San Diego, CA, USA, 2015.",
  "S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Advances in Neural Information Processing Systems 30 (NIPS 2017)*, 2017.",
];
add2(H1u("References"));
refs.forEach((r, i) => body2.push(new Paragraph({
  children: [new TextRun({ text: `[${i + 1}]`, font: FONT, size: 16 }), new TextRun({ text: "\t", font: FONT, size: 16 }), ...runs(r, { size: 16 })],
  tabStops: [{ type: TabStopType.LEFT, position: 400 }],
  indent: { left: 400, hanging: 400 }, alignment: AlignmentType.JUSTIFIED, spacing: { after: 30 },
})));

// ---------------------------------------------------------------- assemble sections
const pageProps = { page: { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MARGIN_T, bottom: MARGIN_B, left: MARGIN_LR, right: MARGIN_LR } } };
const footer = { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16 })] })] }) };
const twoCol = { column: { count: 2, space: COL_GAP, equalWidth: true } };

// split the body into two-column runs and full-width blocks
const bodySections = [];
let run = [...abstractBlock];
for (const x of [...body, ...body2]) {
  if (x && x.wide) {
    bodySections.push({ properties: { ...pageProps, type: SectionType.CONTINUOUS, ...twoCol }, footers: footer, children: run });
    bodySections.push({ properties: { ...pageProps, type: SectionType.CONTINUOUS }, footers: footer, children: x.wide });
    run = [];
  } else run.push(x);
}
bodySections.push({ properties: { ...pageProps, type: SectionType.CONTINUOUS, ...twoCol }, footers: footer, children: run });

const doc = new Document({
  creator: "Mohammed Faisal SM",
  title,
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  numbering: { config: [{ reference: bulletRef, levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 280, hanging: 200 } } } }] }] },
  sections: [
    { properties: { ...pageProps }, footers: footer, children: front },
    ...bodySections,
  ],
});

const outPath = path.join(__dirname, "Explainable_DualHead_CNN_BiLSTM_Attention_Bearing_Prognostics.docx");
const JSZip = require(require.resolve("jszip", { paths: [path.dirname(require.resolve(process.env.DOCX_MODULE || "docx"))] }));
Packer.toBuffer(doc).then(async (buf) => {
  const zip = await JSZip.loadAsync(buf);
  let xml = await zip.file("word/document.xml").async("string");
  for (const [name, omml] of Object.entries(OMML)) {
    const re = new RegExp(String.raw`<w:p>(?:(?!<w:p>)[\s\S])*?@@EQ:${name}@@[\s\S]*?</w:p>`);
    if (!re.test(xml)) continue;
    // piecewise (cases) matrices: 1.5x row spacing and a double-em column gap for readability
    // (pandoc also emits "aligned" as a right|left matrix; those get no extra column gap so "=" stays next to its symbol)
    const spaced = omml
      .replace(/<m:plcHide m:val="1" \/><m:mcs>(<m:mc><m:mcPr><m:mcJc m:val="right" \/>)/g, '<m:plcHide m:val="1" /><m:rSpRule m:val="1"/><m:cGpRule m:val="3"/><m:cGp m:val="60"/><m:mcs>$1')
      .replace(/<m:plcHide m:val="1" \/><m:mcs>(?!<m:mc><m:mcPr><m:mcJc m:val="right")/g, '<m:plcHide m:val="1" /><m:rSpRule m:val="1"/><m:cGpRule m:val="2"/><m:mcs>');
    xml = xml.replace(re, `<w:p><w:pPr><w:spacing w:before="0" w:after="0"/><w:jc w:val="center"/></w:pPr>${spaced}</w:p>`);
  }
  if (xml.includes("@@EQ:")) throw new Error("unreplaced equation placeholder");
  zip.file("word/document.xml", xml);
  buf = await zip.generateAsync({ type: "nodebuffer" });
  fs.writeFileSync(outPath, buf);
  console.log("written", outPath, "equations:", eqNo, "tables:", tabNo, "figures:", figNo);
});
