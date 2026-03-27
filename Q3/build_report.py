"""Generate q3_report.pdf using reportlab."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, PageBreak)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

W, H = letter

def build_report(path: str):
    doc = SimpleDocTemplate(
        path, pagesize=letter,
        leftMargin=0.85*inch, rightMargin=0.85*inch,
        topMargin=0.9*inch,  bottomMargin=0.9*inch,
    )
    styles = getSampleStyleSheet()

    # Custom styles
    title_st = ParagraphStyle("Title2", parent=styles["Title"],
                               fontSize=17, spaceAfter=4, textColor=colors.HexColor("#1a1a2e"))
    sub_st   = ParagraphStyle("Sub", parent=styles["Normal"],
                               fontSize=11, textColor=colors.HexColor("#4a4a6a"),
                               spaceAfter=14, alignment=TA_CENTER)
    h1_st    = ParagraphStyle("H1", parent=styles["Heading1"],
                               fontSize=13, spaceBefore=14, spaceAfter=4,
                               textColor=colors.HexColor("#1a1a2e"),
                               borderPad=2)
    h2_st    = ParagraphStyle("H2", parent=styles["Heading2"],
                               fontSize=11, spaceBefore=8, spaceAfter=3,
                               textColor=colors.HexColor("#2e4a7e"))
    body_st  = ParagraphStyle("Body", parent=styles["Normal"],
                               fontSize=9.5, leading=14, spaceAfter=7,
                               alignment=TA_JUSTIFY)
    mono_st  = ParagraphStyle("Mono", parent=styles["Code"],
                               fontSize=8.5, leading=12, spaceAfter=5,
                               backColor=colors.HexColor("#f4f4f8"),
                               borderColor=colors.HexColor("#ccccdd"),
                               borderWidth=0.5, borderPad=4)
    bullet_st = ParagraphStyle("Bullet", parent=body_st,
                                leftIndent=14, bulletIndent=4,
                                spaceAfter=3)

    def hr():
        return HRFlowable(width="100%", thickness=0.5,
                          color=colors.HexColor("#ccccdd"), spaceAfter=6)

    def h1(t): return Paragraph(t, h1_st)
    def h2(t): return Paragraph(t, h2_st)
    def p(t):  return Paragraph(t, body_st)
    def b(t):  return Paragraph(f"• {t}", bullet_st)
    def sp(n=6): return Spacer(1, n)

    story = []

    # ── Title Page ────────────────────────────────────────────────────────
    story += [
        sp(20),
        Paragraph("Q3: Ethical Auditing &amp; Documentation Debt Mitigation", title_st),
        Paragraph("Sound Check Audit · Privacy-Preserving AI · Fair ASR Training",
                  sub_st),
        hr(),
        sp(4),
    ]

    # Meta table
    meta = [
        ["Dataset",      "Mozilla Common Voice (demo: synthetic 5,000 utterances)"],
        ["Tools",        "Python 3.10 · PyTorch 2.x · Hugging Face Transformers"],
        ["Evaluation",   "FAD (Fréchet Audio Distance) · DNSMOS proxy · STOI proxy"],
        ["Deliverables", "audit.py · privacymodule.py · pp_demo.py · train_fair.py · "
                         "evaluation_scripts/ · audit_plots.pdf"],
    ]
    meta_table = Table(meta, colWidths=[1.4*inch, 5.1*inch])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE",       (0,0), (-1,-1), 9),
        ("FONTNAME",       (0,0), (0,-1), "Helvetica-Bold"),
        ("VALIGN",         (0,0), (-1,-1), "TOP"),
        ("ROWBACKGROUNDS", (0,0), (-1,-1),
         [colors.HexColor("#f0f0f8"), colors.white]),
        ("GRID",           (0,0), (-1,-1), 0.3, colors.HexColor("#ccccdd")),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 5),
        ("TOPPADDING",     (0,0), (-1,-1), 5),
        ("LEFTPADDING",    (0,0), (-1,-1), 6),
    ]))
    story += [meta_table, sp(10)]

    # ── Section 1: Bias Audit ─────────────────────────────────────────────
    story += [h1("1 · Bias Identification &amp; Documentation Debt Audit"), hr()]

    story += [
        h2("1.1 Dataset Overview"),
        p("We audited Mozilla Common Voice English (simulated via 5,000 synthetic "
          "utterances mirroring real Common Voice statistics). Three demographic "
          "axes were examined: <b>Gender</b>, <b>Age Group</b>, and "
          "<b>Accent/Dialect</b>."),
        sp(),
    ]

    story += [h2("1.2 Documentation Debt")]
    doc_table = Table(
        [["Field", "Missing %", "Assessment"],
         ["Gender",  "5.7%",   "Acceptable – minor gap"],
         ["Age",    "11.8%",   "Moderate – targeted re-labelling recommended"],
         ["Accent", "14.5%",   "High – significant documentation debt"],
        ],
        colWidths=[1.6*inch, 1.2*inch, 3.7*inch])
    doc_table.setStyle(TableStyle([
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#f4f4f8"), colors.white]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#aaaacc")),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ]))
    story += [doc_table, sp(8)]

    story += [h2("1.3 Representation Bias Results")]
    bias_table = Table(
        [["Metric",          "Gender", "Age",  "Accent"],
         ["Normalised Entropy", "0.611", "0.914", "0.886"],
         ["Gini Coefficient",   "0.542", "0.319", "0.357"],
         ["χ² p-value",         "<0.001","<0.001","<0.001"],
         ["Verdict",          "BIASED","BIASED","BIASED"],
        ],
        colWidths=[2.0*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    bias_table.setStyle(TableStyle([
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("BACKGROUND",    (0,0), (-1,0), colors.HexColor("#2e4a7e")),
        ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.HexColor("#f0f4ff"), colors.white]),
        ("GRID",          (0,0), (-1,-1), 0.3, colors.HexColor("#aaaacc")),
        ("FONTNAME",      (0,4), (-1,4), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0,4), (0,4), colors.HexColor("#c00000")),
        ("TEXTCOLOR",     (1,4), (-1,4), colors.HexColor("#c00000")),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 6),
    ]))
    story += [bias_table, sp(8)]

    story += [
        h2("1.4 Key Findings"),
        b("Gender: Male speakers (68.7%) dominate; female (23.4%) and other/non-binary "
          "voices are severely underrepresented. Normalised entropy of 0.611 vs ideal 1.0 "
          "indicates strong skew."),
        b("Age: Speakers in their 30s–40s dominate (46.7% combined); teens, elderly "
          "(60+), and children have &lt;10% combined representation. Models trained on "
          "this data will likely degrade for these populations."),
        b("Accent/Dialect: US accent (37.7%) and British English (17.9%) dominate. "
          "Non-Western accents (Indian, African, Asian) are critically underrepresented "
          "despite being spoken by billions of people globally."),
        b("Intersectional Gap: The male × thirties cell alone occupies 19.1% of total "
          "data. Female × elderly combinations each represent &lt;0.5%."),
        sp(8),
    ]

    # ── Section 2: Privacy Module ─────────────────────────────────────────
    story += [h1("2 · Privacy-Preserving Voice Transformation"), hr()]

    story += [
        h2("2.1 Architecture"),
        p("The <b>PrivacyModule</b> (privacymodule.py) implements a CycleGAN-inspired "
          "voice conversion system with disentangled representations:"),
        b("<b>ContentEncoder</b>: Strided 1D convolutions + residual blocks strip "
          "speaker identity, retaining only phonetic content (z_content)."),
        b("<b>AttributeEncoder</b>: Global average pooling over a parallel conv branch "
          "produces a speaker attribute vector (z_attr)."),
        b("<b>TargetAttributeEmbedder</b>: Maps discrete gender/age labels to a learned "
          "embedding space, allowing arbitrary target attribute specification."),
        b("<b>Decoder</b>: AdaIN-based residual decoder injects target style into content "
          "to generate the transformed mel spectrogram."),
        b("<b>Vocoder</b>: Griffin-Lim inversion (training mode) or learned HiFi-GAN "
          "(production). Outputs raw waveform."),
        sp(4),
    ]

    story += [
        h2("2.2 Training Objectives"),
        p("Four losses are combined to balance transformation quality and content fidelity:"),
    ]
    loss_table = Table(
        [["Loss Term",       "Formula",                  "Weight",  "Purpose"],
         ["Reconstruction",  "L1(mel_recon, mel_src)",   "λ=10",   "Cycle: src→tgt→src"],
         ["Content-Cycle",   "MSE(z_cont_cyc, z_cont)", "λ=5",    "Preserve phonemes"],
         ["Adversarial",     "MSE(D(mel_fake), 1)",      "λ=1",    "Realism"],
         ["Patch Discrim.",  "MSE(D(real),1)+MSE(D(fake),0)", "×0.5", "GAN stability"],
        ],
        colWidths=[1.5*inch, 2.1*inch, 0.9*inch, 2.0*inch])
    loss_table.setStyle(TableStyle([
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTSIZE",      (0,0),(-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.HexColor("#f4f4f8"), colors.white]),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#aaaacc")),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
    ]))
    story += [loss_table, sp(8)]

    story += [
        h2("2.3 Supported Transformations"),
        p("The module supports any combination of supported gender and age attributes. "
          "Example transformation pairs (demonstrated in pp_demo.py):"),
        b("male/old → female/young  (primary use case)"),
        b("female/young → male/old"),
        b("male/old → female/old    (gender-only)"),
        b("male/young → female/middle"),
        sp(4),
        p("<i>Note: With random weight initialisation (no training), transformations are "
          "structurally valid but acoustically random. Real-world quality requires "
          "training on ≥50 hours of multi-speaker data per demographic group.</i>"),
        sp(8),
    ]

    # ── Section 3: Fair Training ──────────────────────────────────────────
    story += [PageBreak()]
    story += [h1("3 · Fairness Loss &amp; Fair ASR Training"), hr()]

    story += [
        h2("3.1 Model Architecture"),
        p("The <b>FairASRModel</b> (train_fair.py) is a CTC-based ASR system consisting "
          "of a strided convolutional feature extractor (5 conv layers, total stride ×80) "
          "followed by 6 Conformer blocks (d=256, 4 heads) and a linear CTC projection."),
        sp(4),
    ]

    story += [
        h2("3.2 Fairness Loss Design"),
        p("Standard CTC training minimises the mean loss across all utterances, which "
          "allows the model to perform well on majority groups while failing silently on "
          "minorities. Our <b>FairnessLoss</b> adds three terms:"),
        sp(4),
    ]

    fl_eq = Paragraph(
        "<b>L<sub>total</sub> = L<sub>CTC</sub> + λ<sub>fair</sub>·(worst_g − best_g) "
        "+ λ<sub>reg</sub>·Σ<sub>g</sub> max(0, L<sub>g</sub>−τ)<super>2</super></b>",
        ParagraphStyle("Eq", parent=styles["Normal"], fontSize=10,
                       alignment=TA_CENTER, spaceBefore=4, spaceAfter=6,
                       backColor=colors.HexColor("#fff8e8"),
                       borderColor=colors.HexColor("#e8c800"),
                       borderWidth=0.8, borderPad=8))
    story.append(fl_eq)
    story += [sp(6)]

    story += [
        b("L<sub>CTC</sub>: Standard mean CTC loss across the batch."),
        b("Gap penalty (λ<sub>fair</sub>=0.5): Penalises the difference between the "
          "worst-performing and best-performing group in each batch. Forces the model "
          "to improve weak groups."),
        b("Threshold regulariser (λ<sub>reg</sub>=0.1, τ=0.05): Quadratic penalty for "
          "any group whose loss exceeds the acceptable threshold τ."),
        b("Group-DRO EMA weights: Exponential moving averages track per-group losses "
          "across batches to provide stable gradient signals even when groups are "
          "sparsely sampled."),
        sp(8),
    ]

    story += [
        h2("3.3 Demographic Groups"),
        p("Groups are defined by the cross-product of coarse gender × age buckets "
          "derived from the audit:"),
    ]
    groups_data = [
        ["Group",           "Description",                    "% in Demo Data"],
        ["male_young",      "Male, teens–twenties",           "~18%"],
        ["male_middle",     "Male, thirties–forties",         "~33%"],
        ["male_old",        "Male, fifties+",                 "~12%"],
        ["female_young",    "Female, teens–twenties",         "~6%"],
        ["female_middle",   "Female, thirties–forties",       "~10%"],
        ["female_old",      "Female, fifties+",               "~3%"],
        ["other",           "Non-binary / unlabelled gender", "~18%"],
    ]
    gt = Table(groups_data, colWidths=[1.5*inch, 3.0*inch, 1.5*inch])
    gt.setStyle(TableStyle([
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#2e4a7e")),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.HexColor("#f0f4ff"), colors.white]),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#aaaacc")),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("LEFTPADDING",   (0,0),(-1,-1), 5),
    ]))
    story += [gt, sp(10)]

    # ── Section 4: Validation ─────────────────────────────────────────────
    story += [h1("4 · Validation: FAD &amp; DNSMOS Evaluation"), hr()]

    story += [
        h2("4.1 Fréchet Audio Distance (FAD)"),
        p("FAD measures the distributional distance between reference audio and "
          "privacy-transformed audio in the embedding space of a VGGish-style neural "
          "network (128-dim proxy embeddings). It generalises FID to audio."),
    ]

    fad_table = Table(
        [["Scenario",                  "FAD",   "Interpretation"],
         ["Identical distribution",    "~0.0",  "Perfect (baseline)"],
         ["Small shift (high quality)","~2-5",  "Good – minor artefacts"],
         ["Medium shift",              "~10-25","Acceptable"],
         ["Large shift / mode collapse","100+", "Fail – retrain needed"],
         ["Our privacy transform",     "<15",   "Target (no toxicity traps)"],
        ],
        colWidths=[2.5*inch, 0.8*inch, 3.2*inch])
    fad_table.setStyle(TableStyle([
        ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
        ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",     (0,0),(-1,0), colors.white),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.HexColor("#f4f4f8"), colors.white]),
        ("BACKGROUND",    (0,5),(-1,5), colors.HexColor("#e8f4e8")),
        ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#aaaacc")),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
    ]))
    story += [fad_table, sp(8)]

    story += [
        h2("4.2 DNSMOS Proxy"),
        p("DNSMOS estimates MOS (Mean Opinion Score, 1–5) non-intrusively using a GRU "
          "network on STFT magnitude features. Three sub-scores are reported:"),
        b("<b>SIG</b> (Signal Quality): Measures speech clarity and absence of "
          "signal-level distortion. Target: ≥ 3.5."),
        b("<b>BAK</b> (Background): Measures background noise level and absence of "
          "artifacts introduced by the transformation. Target: ≥ 3.5."),
        b("<b>OVR</b> (Overall MOS): Weighted combination of SIG and BAK. "
          "Target: ≥ 3.0 to avoid 'Toxicity Traps'."),
        sp(6),
        p("A 'Toxicity Trap' is defined as a transformation that passes FAD "
          "(&lt;30) but introduces linguistic distortions or acoustic artefacts "
          "that degrade ASR accuracy or listener experience. Both FAD and DNSMOS "
          "are necessary – FAD alone can miss localised artefacts while DNSMOS "
          "alone can miss distributional drift."),
        sp(8),
    ]

    story += [
        h2("4.3 STOI Proxy (Intelligibility)"),
        p("Short-Time Objective Intelligibility (STOI) is computed as the average "
          "frame-level correlation between reference and transformed magnitude spectra. "
          "Values near 1.0 indicate preserved intelligibility. We require STOI &gt; 0.85 "
          "for the transformation to be considered linguistically safe."),
        sp(8),
    ]

    # ── Section 5: Ethics ─────────────────────────────────────────────────
    story += [h1("5 · Ethical Considerations"), hr()]

    story += [
        h2("5.1 Dual-Use Risk"),
        p("Voice transformation technology can be misused for: deepfakes, "
          "impersonation, non-consensual identity manipulation, and evidence "
          "fabrication. Our implementation includes:"),
        b("Watermarking: Production deployments should embed imperceptible "
          "watermarks in transformed audio (not implemented in this prototype)."),
        b("Consent framework: Transformations should only be applied with explicit "
          "speaker consent for research/anonymisation purposes."),
        b("Audit trails: All transformation operations should be logged with "
          "purpose, source attributes, and authorisation."),
        sp(4),
    ]

    story += [
        h2("5.2 Fairness Loss Limitations"),
        b("Group definition risk: Coarse gender/age buckets may reinforce binary "
          "categorisation. Future work should use continuous embeddings."),
        b("Proxy labels: Using self-reported demographic labels (as in Common Voice) "
          "introduces noise and may not reflect perceived demographic attributes."),
        b("Intersectional blindness: The current fairness loss operates on single-axis "
          "groups; intersectional groups (e.g. female × old × non-native accent) may "
          "still be overlooked without explicit multi-axis group definitions."),
        b("Fairness–accuracy trade-off: A fairness penalty of λ=0.5 may reduce overall "
          "WER slightly (&lt;5% relative) while closing the worst-group gap. This "
          "trade-off should be explicitly reported in production deployments."),
        sp(4),
    ]

    story += [
        h2("5.3 Recommendations"),
        b("Re-collect data targeting under-represented groups "
          "(female, elderly, non-English-accented speakers) with community partnerships."),
        b("Adopt participatory data governance: involve speaker communities in "
          "deciding how their voices are used and transformed."),
        b("Publish fairness evaluation results alongside WER metrics in all model "
          "cards and technical reports."),
        b("Adopt continuous re-auditing pipelines; run audit.py on each new dataset "
          "release as part of CI/CD."),
        sp(12),
    ]

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(hr())
    story.append(Paragraph(
        "<i>Generated by q3/ submission pipeline. All code is original and "
        "implemented using Python, PyTorch, and Hugging Face. "
        "Dataset: Mozilla Common Voice (demo mode). "
        "Tools: audit.py · privacymodule.py · pp_demo.py · train_fair.py · "
        "evaluation_scripts/fad_eval.py · evaluation_scripts/dnsmos_eval.py</i>",
        ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7.5,
                       textColor=colors.HexColor("#888888"),
                       alignment=TA_CENTER)))

    doc.build(story)
    print(f"  Report saved → {path}")


if __name__ == "__main__":
    build_report("/home/claude/q3/q3_report.pdf")
