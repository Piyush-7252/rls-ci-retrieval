"""
Clinical Trial Protocol — Domain Dictionary
============================================
High-precision, finite vocabulary for clinical trial protocol documents.
Used as a fast dictionary pre-pass before the GLiNER model so well-known
protocol entities are always correctly labeled (score=1.0, deterministic).

Covers the terms that generic biomedical NER models routinely miss:
  • Clinical endpoints (PFS, OS, MRD, ORR, …)
  • Study populations (ITT, Safety Set, FAS, …)
  • Questionnaire instruments (EORTC, EQ-5D-5L, PRO-CTCAE, …)
  • Statistical methods (Kaplan-Meier, Log-rank, Cox, …)
  • Drug names and common abbreviations
  • Biomarkers / molecular targets
  • Regulatory and administrative identifiers
  • Adverse-event terminology
  • Sampling timepoints, specimen types, lab assays (Schedule of Activities)
  • PK/PD/immunogenicity assessments
  • Clinical response criteria (CR, sCR, VGPR, PR, MR, SD, PD — IMWG)
  • Study visits and treatment phases / study periods

Entity schema output by match_entities()
-----------------------------------------
{
  "text":         str,   original text span
  "label":        str,   entity type (CLINICAL_ENDPOINT, CLINICAL_RESPONSE, …)
  "sub_type":     str,   finer sub-category (TIME_TO_EVENT, CRITERIA, …)
  "canonical":    str,   short canonical form  (e.g. "PFS")
  "normalized":   str,   full normalized name  (e.g. "Progression-Free Survival")
  "abbreviation": str,   abbreviation if one exists, else ""
  "family":       str,   label hierarchy bucket (ASSESSMENT, RESPONSE, TIMEPOINT, …)
  "start":        int,   char offset in input text
  "end":          int,
  "score":        float  always 1.0 for dictionary hits
}
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Term table
#   Each entry: (term_lower, label, sub_type, canonical_form)
#   Ordered longest-first inside each category so greedy matching works.
# ─────────────────────────────────────────────────────────────────────────────

CLINICAL_TERMS: list[tuple[str, str, str, str]] = [

    # ── Drugs / Interventions ──────────────────────────────────────────────
    ("talquetamab",              "MEDICATION", "DRUG", "Talquetamab"),
    ("jnj-64407564",             "MEDICATION", "DRUG", "Talquetamab (JNJ-64407564)"),
    ("daratumumab",              "MEDICATION", "DRUG", "Daratumumab"),
    ("daratumumab sc",           "MEDICATION", "DRUG", "Daratumumab SC"),
    ("pomalidomide",             "MEDICATION", "DRUG", "Pomalidomide"),
    ("bortezomib",               "MEDICATION", "DRUG", "Bortezomib"),
    ("lenalidomide",             "MEDICATION", "DRUG", "Lenalidomide"),
    ("dexamethasone",            "MEDICATION", "DRUG", "Dexamethasone"),
    ("cyclophosphamide",         "MEDICATION", "DRUG", "Cyclophosphamide"),
    ("carfilzomib",              "MEDICATION", "DRUG", "Carfilzomib"),
    ("ixazomib",                 "MEDICATION", "DRUG", "Ixazomib"),
    ("melphalan",                "MEDICATION", "DRUG", "Melphalan"),
    ("isatuximab",               "MEDICATION", "DRUG", "Isatuximab"),
    ("elotuzumab",               "MEDICATION", "DRUG", "Elotuzumab"),
    ("selinexor",                "MEDICATION", "DRUG", "Selinexor"),
    ("belantamab",               "MEDICATION", "DRUG", "Belantamab"),
    ("teclistamab",              "MEDICATION", "DRUG", "Teclistamab"),
    ("cevostamab",               "MEDICATION", "DRUG", "Cevostamab"),
    ("elranatamab",              "MEDICATION", "DRUG", "Elranatamab"),
    ("subcutaneous",             "MEDICATION", "ROUTE", "Subcutaneous"),
    ("intravenous",               "MEDICATION", "ROUTE", "Intravenous"),
    ("intramuscular",             "MEDICATION", "ROUTE", "Intramuscular"),
    # Route abbreviations — uppercase only (see _UPPERCASE_ONLY); longest first
    # so "daratumumab sc" (above) is claimed before bare "SC" at the same position.
    ("subq",            "MEDICATION", "ROUTE", "Subcutaneous"),
    ("sc",              "MEDICATION", "ROUTE", "Subcutaneous"),
    ("po",              "MEDICATION", "ROUTE", "Oral"),
    ("im",              "MEDICATION", "ROUTE", "Intramuscular"),

    # ── Clinical Endpoints (long forms first) ──────────────────────────────
    ("progression-free survival", "CLINICAL_ENDPOINT", "TIME_TO_EVENT", "PFS"),
    ("progression free survival", "CLINICAL_ENDPOINT", "TIME_TO_EVENT", "PFS"),
    ("overall survival",          "CLINICAL_ENDPOINT", "TIME_TO_EVENT", "OS"),
    ("overall response rate",     "CLINICAL_ENDPOINT", "RESPONSE",      "ORR"),
    ("mrd negativity status",     "CLINICAL_ENDPOINT", "RESPONSE",      "MRD Negativity Status"),
    ("mrd negativity",            "CLINICAL_ENDPOINT", "RESPONSE",      "MRD Negativity"),
    ("minimal residual disease",  "CLINICAL_ENDPOINT", "RESPONSE",      "MRD"),
    ("duration of response",      "CLINICAL_ENDPOINT", "DURATION",      "DOR"),
    ("clinical benefit rate",     "CLINICAL_ENDPOINT", "RESPONSE",      "CBR"),
    ("time to response",          "CLINICAL_ENDPOINT", "TIME_TO_EVENT", "TTR"),
    ("time to next treatment",    "CLINICAL_ENDPOINT", "TIME_TO_EVENT", "TTNT"),
    # ── Clinical Response Criteria (IMWG) ─────────────────────────────────
    ("stringent complete response", "CLINICAL_RESPONSE", "CRITERIA",    "sCR"),
    ("complete response",         "CLINICAL_RESPONSE", "CRITERIA",      "CR"),
    ("partial response",          "CLINICAL_RESPONSE", "CRITERIA",      "PR"),
    ("very good partial response","CLINICAL_RESPONSE", "CRITERIA",      "VGPR"),
    ("minor response",            "CLINICAL_RESPONSE", "CRITERIA",      "MR"),
    ("stable disease",            "CLINICAL_RESPONSE", "CRITERIA",      "SD"),
    ("progressive disease",       "CLINICAL_RESPONSE", "CRITERIA",      "PD"),
    ("mrd-negative",              "CLINICAL_RESPONSE", "CRITERIA",      "MRD-Negative"),
    ("mrd negative",              "CLINICAL_RESPONSE", "CRITERIA",      "MRD-Negative"),
    # Response criteria acronyms (uppercase / specific-case — see _UPPERCASE_ONLY, _SPECIFIC_CASE)
    ("vgpr",   "CLINICAL_RESPONSE", "CRITERIA",     "VGPR"),
    ("scr",    "CLINICAL_RESPONSE", "CRITERIA",     "sCR"),    # matched as "sCR" via _SPECIFIC_CASE
    ("cr",     "CLINICAL_RESPONSE", "CRITERIA",     "CR"),
    ("pr",     "CLINICAL_RESPONSE", "CRITERIA",     "PR"),
    # Clinical outcome acronyms (uppercase required — see _UPPERCASE_ONLY set)
    ("pfs2",   "CLINICAL_ENDPOINT", "TIME_TO_EVENT", "PFS2"),
    ("pfs",    "CLINICAL_ENDPOINT", "TIME_TO_EVENT", "PFS"),
    ("os",     "CLINICAL_ENDPOINT", "TIME_TO_EVENT", "OS"),
    ("mrd",    "CLINICAL_ENDPOINT", "RESPONSE",      "MRD"),
    ("orr",    "CLINICAL_ENDPOINT", "RESPONSE",      "ORR"),
    ("dor",    "CLINICAL_ENDPOINT", "DURATION",      "DOR"),
    ("ttnt",   "CLINICAL_ENDPOINT", "TIME_TO_EVENT", "TTNT"),
    ("cbr",    "CLINICAL_ENDPOINT", "RESPONSE",      "CBR"),
    ("ttr",    "CLINICAL_ENDPOINT", "TIME_TO_EVENT", "TTR"),

    # ── Study Populations ──────────────────────────────────────────────────
    ("intent-to-treat population",  "STUDY_POPULATION", "ITT",         "ITT Population"),
    ("intention-to-treat",          "STUDY_POPULATION", "ITT",         "ITT Population"),
    ("intent to treat",             "STUDY_POPULATION", "ITT",         "ITT Population"),
    ("modified intent-to-treat",    "STUDY_POPULATION", "mITT",        "mITT Population"),
    ("modified intent to treat",    "STUDY_POPULATION", "mITT",        "mITT Population"),
    ("safety population",           "STUDY_POPULATION", "SAFETY",      "Safety Population"),
    ("safety set",                  "STUDY_POPULATION", "SAFETY",      "Safety Set"),
    ("per-protocol population",     "STUDY_POPULATION", "PER_PROTOCOL","Per Protocol Population"),
    ("per protocol population",     "STUDY_POPULATION", "PER_PROTOCOL","Per Protocol Population"),
    ("full analysis set",           "STUDY_POPULATION", "FAS",         "Full Analysis Set"),
    ("evaluable population",        "STUDY_POPULATION", "EVALUABLE",   "Evaluable Population"),
    ("enrolled population",         "STUDY_POPULATION", "ENROLLED",    "Enrolled Population"),
    # Acronyms
    ("itt",   "STUDY_POPULATION", "ITT",         "ITT Population"),
    ("mitt",  "STUDY_POPULATION", "mITT",        "mITT Population"),
    ("fas",   "STUDY_POPULATION", "FAS",         "Full Analysis Set"),

    # ── Biomarkers / Molecular Targets ────────────────────────────────────
    ("cd38",           "BIOMARKER", "TARGET",     "CD38"),
    ("gprc5d",         "BIOMARKER", "TARGET",     "GPRC5D"),
    ("bcma",           "BIOMARKER", "TARGET",     "BCMA"),
    ("pd-l1",          "BIOMARKER", "CHECKPOINT", "PD-L1"),
    ("pdl1",           "BIOMARKER", "CHECKPOINT", "PD-L1"),
    ("pd-1",           "BIOMARKER", "CHECKPOINT", "PD-1"),
    ("mage-a4",        "BIOMARKER", "TARGET",     "MAGE-A4"),
    ("dll3",           "BIOMARKER", "TARGET",     "DLL3"),
    ("her2",           "BIOMARKER", "TARGET",     "HER2"),
    ("egfr",           "BIOMARKER", "TARGET",     "EGFR"),

    # ── Patient-Reported Outcome Instruments ──────────────────────────────
    ("eortc qlq-c30",      "QUESTIONNAIRE", "PRO", "EORTC QLQ-C30"),
    ("eortc-qlq-c30",      "QUESTIONNAIRE", "PRO", "EORTC QLQ-C30"),
    ("qlq-c30",            "QUESTIONNAIRE", "PRO", "EORTC QLQ-C30"),
    ("eq-5d-5l",           "QUESTIONNAIRE", "PRO", "EQ-5D-5L"),
    ("eq-5d",              "QUESTIONNAIRE", "PRO", "EQ-5D"),
    ("pro-ctcae",          "QUESTIONNAIRE", "PRO", "PRO-CTCAE"),
    ("pgi-s",              "QUESTIONNAIRE", "PRO", "PGI-S"),
    ("mysim-q",            "QUESTIONNAIRE", "PRO", "MySIm-Q"),
    ("promis",             "QUESTIONNAIRE", "PRO", "PROMIS"),
    ("fact-g",             "QUESTIONNAIRE", "PRO", "FACT-G"),
    ("fact-myeloma",       "QUESTIONNAIRE", "PRO", "FACT-Myeloma"),
    ("pgic",               "QUESTIONNAIRE", "PRO", "PGIC"),
    ("myeloma patient outcomes", "QUESTIONNAIRE", "PRO", "MyPOS"),
    ("haem-a-qol",         "QUESTIONNAIRE", "PRO", "Haem-A-QoL"),

    # ── Statistical Methods ────────────────────────────────────────────────
    ("cochran-mantel-haenszel",    "STATISTICAL_METHOD", "TEST",      "Cochran-Mantel-Haenszel"),
    ("cochran mantel haenszel",    "STATISTICAL_METHOD", "TEST",      "Cochran-Mantel-Haenszel"),
    ("kaplan-meier",               "STATISTICAL_METHOD", "SURVIVAL",  "Kaplan-Meier"),
    ("kaplan meier",               "STATISTICAL_METHOD", "SURVIVAL",  "Kaplan-Meier"),
    ("cox proportional hazards",   "STATISTICAL_METHOD", "MODEL",     "Cox PH Model"),
    ("cox regression",             "STATISTICAL_METHOD", "MODEL",     "Cox Regression"),
    ("logistic regression",        "STATISTICAL_METHOD", "MODEL",     "Logistic Regression"),
    ("confidence interval",        "STATISTICAL_METHOD", "ESTIMATE",  "Confidence Interval"),
    ("hazard ratio",               "STATISTICAL_METHOD", "ESTIMATE",  "Hazard Ratio"),
    ("odds ratio",                 "STATISTICAL_METHOD", "ESTIMATE",  "Odds Ratio"),
    ("log-rank test",              "STATISTICAL_METHOD", "TEST",      "Log-rank Test"),
    ("log-rank",                   "STATISTICAL_METHOD", "TEST",      "Log-rank"),
    ("interim analysis",           "STATISTICAL_METHOD", "ANALYSIS",  "Interim Analysis"),
    ("final analysis",             "STATISTICAL_METHOD", "ANALYSIS",  "Final Analysis"),
    ("hierarchical testing",       "STATISTICAL_METHOD", "PROCEDURE", "Hierarchical Testing"),
    ("multiplicity",               "STATISTICAL_METHOD", "PROCEDURE", "Multiplicity Adjustment"),
    ("type i error",               "STATISTICAL_METHOD", "ERROR",     "Type I Error"),
    ("alpha spending",             "STATISTICAL_METHOD", "PROCEDURE", "Alpha Spending"),
    ("o'brien-fleming",            "STATISTICAL_METHOD", "BOUNDARY",  "O'Brien-Fleming"),
    ("obrien-fleming",             "STATISTICAL_METHOD", "BOUNDARY",  "O'Brien-Fleming"),
    ("stratified randomization",   "STATISTICAL_METHOD", "DESIGN",    "Stratified Randomization"),
    ("subgroup analysis",          "STATISTICAL_METHOD", "ANALYSIS",  "Subgroup Analysis"),

    # ── Diseases / Conditions ──────────────────────────────────────────────
    ("relapsed/refractory multiple myeloma", "MEDICAL_CONDITION", "CANCER", "RRMM"),
    ("relapsed and refractory",              "MEDICAL_CONDITION", "STATUS", "Relapsed/Refractory"),
    ("relapsed/refractory",                  "MEDICAL_CONDITION", "STATUS", "Relapsed/Refractory"),
    ("multiple myeloma",                     "MEDICAL_CONDITION", "CANCER", "Multiple Myeloma"),
    ("plasma cell myeloma",                  "MEDICAL_CONDITION", "CANCER", "Multiple Myeloma"),
    ("plasmacytoma",                         "MEDICAL_CONDITION", "CANCER", "Plasmacytoma"),
    ("rrmm",                                 "MEDICAL_CONDITION", "CANCER", "RRMM"),

    # ── Safety Assessments (not clinical endpoints) ────────────────────────
    ("vital signs",              "SAFETY_ASSESSMENT", "VITALS",     "Vital Signs"),
    ("physical examination",     "SAFETY_ASSESSMENT", "EXAM",       "Physical Examination"),
    ("physical examinations",    "SAFETY_ASSESSMENT", "EXAM",       "Physical Examinations"),
    ("neurologic examination",   "SAFETY_ASSESSMENT", "NEURO",      "Neurologic Examination"),
    ("laboratory tests",         "SAFETY_ASSESSMENT", "LAB",        "Laboratory Tests"),
    ("laboratory assessments",   "SAFETY_ASSESSMENT", "LAB",        "Laboratory Assessments"),
    ("ecg",                      "SAFETY_ASSESSMENT", "CARDIAC",    "ECG"),
    ("electrocardiogram",        "SAFETY_ASSESSMENT", "CARDIAC",    "ECG"),
    ("echocardiogram",           "SAFETY_ASSESSMENT", "CARDIAC",    "Echocardiogram"),
    ("bone marrow",              "SAFETY_ASSESSMENT", "PROCEDURE",  "Bone Marrow Assessment"),
    ("performance status",       "SAFETY_ASSESSMENT", "FUNCTIONAL", "Performance Status"),

    # ── Clinical Grading / Severity Scales ────────────────────────────────
    ("nci-ctcae version 5.0",    "CLINICAL_GRADING_SYSTEM", "TOXICITY", "NCI-CTCAE v5.0"),
    ("nci-ctcae v5.0",           "CLINICAL_GRADING_SYSTEM", "TOXICITY", "NCI-CTCAE v5.0"),
    ("nci-ctcae",                "CLINICAL_GRADING_SYSTEM", "TOXICITY", "NCI-CTCAE"),
    ("ctcae",                    "CLINICAL_GRADING_SYSTEM", "TOXICITY", "CTCAE"),
    ("recist",                   "CLINICAL_GRADING_SYSTEM", "RESPONSE", "RECIST"),
    ("recist 1.1",               "CLINICAL_GRADING_SYSTEM", "RESPONSE", "RECIST 1.1"),
    ("imwg criteria",            "CLINICAL_GRADING_SYSTEM", "RESPONSE", "IMWG Criteria"),
    ("ecog performance",         "CLINICAL_GRADING_SYSTEM", "FUNCTIONAL","ECOG PS"),
    ("ecog ps",                  "CLINICAL_GRADING_SYSTEM", "FUNCTIONAL","ECOG PS"),

    # ── Clinical Guidelines / Standards ───────────────────────────────────
    ("astct consensus",          "CLINICAL_GUIDELINE", "GRADING",   "ASTCT Consensus"),
    ("astct",                    "CLINICAL_GUIDELINE", "GRADING",   "ASTCT"),
    ("imwg",                     "CLINICAL_GUIDELINE", "MYELOMA",   "IMWG"),
    ("nccn",                     "CLINICAL_GUIDELINE", "ONCOLOGY",  "NCCN"),
    ("asco",                     "CLINICAL_GUIDELINE", "ONCOLOGY",  "ASCO"),

    # ── Adverse Events ─────────────────────────────────────────────────────
    ("serious adverse event",    "ADVERSE_EVENT", "SAE",  "SAE"),
    ("adverse event",            "ADVERSE_EVENT", "AE",   "Adverse Event"),
    ("treatment-emergent adverse event", "ADVERSE_EVENT", "TEAE", "TEAE"),
    ("dose-limiting toxicity",   "ADVERSE_EVENT", "DLT",  "DLT"),
    ("dose limiting toxicity",   "ADVERSE_EVENT", "DLT",  "DLT"),
    ("infusion-related reaction","ADVERSE_EVENT", "IRR",  "IRR"),
    ("cytokine release syndrome","ADVERSE_EVENT", "CRS",  "CRS"),
    ("sae",  "ADVERSE_EVENT", "SAE",  "SAE"),
    ("teae", "ADVERSE_EVENT", "TEAE", "TEAE"),
    ("dlt",  "ADVERSE_EVENT", "DLT",  "DLT"),
    ("irr",  "ADVERSE_EVENT", "IRR",  "IRR"),
    ("crs",  "ADVERSE_EVENT", "CRS",  "CRS"),

    # ── Regulatory / Administrative ────────────────────────────────────────
    ("fda",           "ORGANIZATION", "REGULATOR", "FDA"),
    ("ema",           "ORGANIZATION", "REGULATOR", "EMA"),
    ("pmda",          "ORGANIZATION", "REGULATOR", "PMDA"),
    ("health canada", "ORGANIZATION", "REGULATOR", "Health Canada"),
    ("irb",           "ORGANIZATION", "ETHICS",    "IRB"),
    ("iec",           "ORGANIZATION", "ETHICS",    "IEC"),
    ("data monitoring committee", "ORGANIZATION", "SAFETY", "DMC"),
    ("data safety monitoring",    "ORGANIZATION", "SAFETY", "DSMB"),
    ("dmc",  "ORGANIZATION", "SAFETY", "DMC"),
    ("dsmb", "ORGANIZATION", "SAFETY", "DSMB"),
    ("dsmc", "ORGANIZATION", "SAFETY", "DSMC"),

    # ── Miscellaneous protocol terms ────────────────────────────────────────
    ("randomized controlled trial", "STUDY_DESIGN", "RCT",        "RCT"),
    ("open-label",                  "STUDY_DESIGN", "BLINDING",   "Open-label"),
    ("double-blind",                "STUDY_DESIGN", "BLINDING",   "Double-blind"),
    ("phase 3",                     "STUDY_DESIGN", "PHASE",      "Phase 3"),
    ("phase 1",                     "STUDY_DESIGN", "PHASE",      "Phase 1"),
    ("phase 2",                     "STUDY_DESIGN", "PHASE",      "Phase 2"),
    ("phase iii",                   "STUDY_DESIGN", "PHASE",      "Phase 3"),
    ("phase i",                     "STUDY_DESIGN", "PHASE",      "Phase 1"),
    ("phase ii",                    "STUDY_DESIGN", "PHASE",      "Phase 2"),
    ("good clinical practice",      "REGULATORY_ID", "GCP",       "GCP"),
    ("gcp",                         "REGULATORY_ID", "GCP",       "GCP"),
    ("ich e6",                      "REGULATORY_ID", "ICH",       "ICH E6"),
    ("ich e9",                      "REGULATORY_ID", "ICH",       "ICH E9"),
    ("21 cfr",                      "REGULATORY_ID", "CFR",       "21 CFR"),
    ("declaration of helsinki",     "REGULATORY_ID", "ETHICS",    "Declaration of Helsinki"),

    # ── PK / PD / Immunogenicity assessments ──────────────────────────────
    ("pharmacokinetics",             "PK_ASSESSMENT", "PK",        "Pharmacokinetics"),
    ("pharmacokinetic",              "PK_ASSESSMENT", "PK",        "Pharmacokinetics"),
    ("pharmacodynamics",             "PD_ASSESSMENT", "PD",        "Pharmacodynamics"),
    ("pharmacodynamic",              "PD_ASSESSMENT", "PD",        "Pharmacodynamics"),
    ("anti-drug antibody",           "IMMUNOGENICITY_ASSESSMENT", "ADA", "Anti-Drug Antibody"),
    ("anti drug antibody",           "IMMUNOGENICITY_ASSESSMENT", "ADA", "Anti-Drug Antibody"),
    ("immunogenicity",               "IMMUNOGENICITY_ASSESSMENT", "IMMUNOGENICITY", "Immunogenicity"),
    ("immunogenicity testing",       "IMMUNOGENICITY_ASSESSMENT", "ADA", "Immunogenicity Testing"),
    ("neutralizing antibody",        "IMMUNOGENICITY_ASSESSMENT", "NAb", "Neutralizing Antibody"),
    # Uppercase acronyms
    ("pk",  "PK_ASSESSMENT", "PK",  "Pharmacokinetics"),
    ("pd",  "PD_ASSESSMENT", "PD",  "Pharmacodynamics"),
    ("ada", "IMMUNOGENICITY_ASSESSMENT", "ADA", "Anti-Drug Antibody"),
    ("nab", "IMMUNOGENICITY_ASSESSMENT", "NAb", "Neutralizing Antibody"),

    # ── Lab Assays / Assessments ───────────────────────────────────────────
    ("immunophenotyping",            "LAB_ASSESSMENT", "FLOW_CYTOMETRY", "Immunophenotyping"),
    ("flow cytometry",               "LAB_ASSESSMENT", "FLOW_CYTOMETRY", "Flow Cytometry"),
    ("mass spectrometry",            "LAB_ASSESSMENT", "MASS_SPEC",      "Mass Spectrometry"),
    ("cytof",                        "LAB_ASSESSMENT", "CYTOF",          "CyTOF"),
    ("elisa",                        "LAB_ASSESSMENT", "ELISA",          "ELISA"),
    ("pcr",                          "LAB_ASSESSMENT", "PCR",            "PCR"),
    ("next-generation sequencing",   "LAB_ASSESSMENT", "NGS",            "NGS"),
    ("ngs",                          "LAB_ASSESSMENT", "NGS",            "NGS"),
    ("serum protein electrophoresis","LAB_ASSESSMENT", "SPEP",           "SPEP"),
    ("spep",                         "LAB_ASSESSMENT", "SPEP",           "SPEP"),
    ("immunofixation",               "LAB_ASSESSMENT", "IFE",            "Immunofixation"),
    ("free light chain",             "LAB_ASSESSMENT", "FLC",            "Free Light Chain"),
    ("creatinine",                   "LAB_ASSESSMENT", "CHEMISTRY",      "Creatinine"),
    ("hemoglobin",                   "LAB_ASSESSMENT", "CBC",            "Hemoglobin"),
    ("complete blood count",         "LAB_ASSESSMENT", "CBC",            "CBC"),
    ("cbc",                          "LAB_ASSESSMENT", "CBC",            "CBC"),
    ("crp",                          "LAB_ASSESSMENT", "INFLAMMATION",   "C-Reactive Protein"),
    ("c-reactive protein",           "LAB_ASSESSMENT", "INFLAMMATION",   "C-Reactive Protein"),
    ("ferritin",                     "LAB_ASSESSMENT", "INFLAMMATION",   "Ferritin"),
    ("frtn",                         "LAB_ASSESSMENT", "INFLAMMATION",   "Ferritin"),
    ("ldh",                          "LAB_ASSESSMENT", "CHEMISTRY",      "LDH"),
    ("beta-2 microglobulin",         "LAB_ASSESSMENT", "CHEMISTRY",      "Beta-2 Microglobulin"),
    ("b2m",                          "LAB_ASSESSMENT", "CHEMISTRY",      "Beta-2 Microglobulin"),
    ("molecular markers",            "BIOMARKER",      "MOLECULAR",      "Molecular Markers"),

    # ── Specimen Types ─────────────────────────────────────────────────────
    ("bone marrow aspirate",   "SPECIMEN_COLLECTION", "BONE_MARROW", "Bone Marrow Aspirate"),
    ("bone marrow biopsy",     "SPECIMEN_COLLECTION", "BONE_MARROW", "Bone Marrow Biopsy"),
    ("bone marrow",            "SPECIMEN_COLLECTION", "BONE_MARROW", "Bone Marrow"),
    ("whole blood",            "SPECIMEN_COLLECTION", "BLOOD",       "Whole Blood"),
    ("pbmc",                   "SPECIMEN_COLLECTION", "BLOOD",       "PBMC"),
    ("serum",                  "SPECIMEN_COLLECTION", "BLOOD",       "Serum"),
    ("plasma",                 "SPECIMEN_COLLECTION", "BLOOD",       "Plasma"),
    ("urine sample",           "SPECIMEN_COLLECTION", "URINE",       "Urine Sample"),
    ("tissue biopsy",          "SPECIMEN_COLLECTION", "TISSUE",      "Tissue Biopsy"),
    ("archival tissue",        "SPECIMEN_COLLECTION", "TISSUE",      "Archival Tissue"),
    ("tumor biopsy",           "SPECIMEN_COLLECTION", "TISSUE",      "Tumor Biopsy"),

    # ── Sampling Timepoints ────────────────────────────────────────────────
    ("end of treatment",       "SAMPLING_TIMEPOINT", "EOT",           "End of Treatment"),
    ("end of study",           "SAMPLING_TIMEPOINT", "EOS",           "End of Study"),
    ("end of follow-up",       "SAMPLING_TIMEPOINT", "EOF",           "End of Follow-up"),
    ("follow-up",              "SAMPLING_TIMEPOINT", "FOLLOWUP",      "Follow-up"),
    ("8 weeks after last dose","SAMPLING_TIMEPOINT", "POST_TREATMENT","8 Weeks After Last Dose"),
    ("30 days after last dose","SAMPLING_TIMEPOINT", "POST_TREATMENT","30 Days After Last Dose"),
    ("predose",                "SAMPLING_TIMEPOINT", "PREDOSE",       "Predose"),
    ("pre-dose",               "SAMPLING_TIMEPOINT", "PREDOSE",       "Predose"),
    ("post-dose",              "SAMPLING_TIMEPOINT", "POSTDOSE",      "Post-dose"),
    ("post dose",              "SAMPLING_TIMEPOINT", "POSTDOSE",      "Post-dose"),
    ("cycle 1 day 1",          "SAMPLING_TIMEPOINT", "CYCLE",         "Cycle 1 Day 1"),
    ("c1d1",                   "SAMPLING_TIMEPOINT", "CYCLE",         "Cycle 1 Day 1"),
    ("eot",                    "SAMPLING_TIMEPOINT", "EOT",           "End of Treatment"),
    ("eos",                    "SAMPLING_TIMEPOINT", "EOS",           "End of Study"),

    # ── Sampling Triggers ──────────────────────────────────────────────────
    ("response-based",                 "SAMPLING_TRIGGER", "RESPONSE",     "Response-based"),
    ("response based",                 "SAMPLING_TRIGGER", "RESPONSE",     "Response-based"),
    ("time of pd",                     "SAMPLING_TRIGGER", "DISEASE_EVENT","Time of PD"),
    ("time of progression",            "SAMPLING_TRIGGER", "DISEASE_EVENT","Time of Disease Progression"),
    ("time of suspected cr or scr",    "SAMPLING_TRIGGER", "RESPONSE",     "Time of Suspected CR or sCR"),
    ("time of suspected scr or cr",    "SAMPLING_TRIGGER", "RESPONSE",     "Time of Suspected CR or sCR"),
    ("time of suspected cr",           "SAMPLING_TRIGGER", "RESPONSE",     "Time of Suspected CR"),
    ("time of confirmed response",     "SAMPLING_TRIGGER", "RESPONSE",     "Time of Confirmed Response"),
    ("at confirmed response",          "SAMPLING_TRIGGER", "RESPONSE",     "At Confirmed Response"),
    ("upon relapse",                   "SAMPLING_TRIGGER", "DISEASE_EVENT","Upon Relapse"),
    ("at time of relapse",             "SAMPLING_TRIGGER", "DISEASE_EVENT","At Time of Relapse"),
    ("at disease progression",         "SAMPLING_TRIGGER", "DISEASE_EVENT","At Disease Progression"),

    # ── Study Visits ───────────────────────────────────────────────────────
    ("post-treatment follow-up visit", "STUDY_VISIT", "FOLLOWUP",   "Post-Treatment Follow-up Visit"),
    ("end of treatment visit",         "STUDY_VISIT", "EOT",        "End of Treatment Visit"),
    ("post-treatment follow-up",       "STUDY_VISIT", "FOLLOWUP",   "Post-Treatment Follow-up"),
    ("eot visit",                      "STUDY_VISIT", "EOT",        "EOT Visit"),
    ("follow-up visit",                "STUDY_VISIT", "FOLLOWUP",   "Follow-up Visit"),
    ("screening visit",                "STUDY_VISIT", "SCREENING",  "Screening Visit"),
    ("baseline visit",                 "STUDY_VISIT", "BASELINE",   "Baseline Visit"),
    ("study visit",                    "STUDY_VISIT", "GENERAL",    "Study Visit"),

    # ── Treatment Phases / Study Periods ──────────────────────────────────
    ("post-treatment follow-up phase", "TREATMENT_PHASE", "PHASE",       "Post-Treatment Follow-up Phase"),
    ("dose escalation phase",          "TREATMENT_PHASE", "ESCALATION",  "Dose Escalation Phase"),
    ("dose expansion phase",           "TREATMENT_PHASE", "EXPANSION",   "Dose Expansion Phase"),
    ("consolidation phase",            "TREATMENT_PHASE", "PHASE",       "Consolidation Phase"),
    ("maintenance phase",              "TREATMENT_PHASE", "PHASE",       "Maintenance Phase"),
    ("induction phase",                "TREATMENT_PHASE", "PHASE",       "Induction Phase"),
    ("observation phase",              "TREATMENT_PHASE", "PHASE",       "Observation Phase"),
    ("treatment period",               "TREATMENT_PHASE", "PERIOD",      "Treatment Period"),
    ("treatment phase",                "TREATMENT_PHASE", "PHASE",       "Treatment Phase"),
    ("run-in period",                  "TREATMENT_PHASE", "PERIOD",      "Run-in Period"),
    ("run-in phase",                   "TREATMENT_PHASE", "PHASE",       "Run-in Phase"),
]

# ── Short acronyms that require the original text to be uppercase ─────────────
# These prevent "pfs" in "helpfulness" → PFS, "or" in "order" → OS, etc.
_UPPERCASE_ONLY: frozenset[str] = frozenset({
    # Clinical endpoints (outcomes / rates)
    "pfs", "pfs2", "os", "mrd", "orr", "dor", "ttnt", "cbr", "ttr",
    # Response criteria acronyms (uppercase; "scr" uses _SPECIFIC_CASE instead)
    "vgpr", "cr", "pr",
    # Study populations
    "itt", "mitt", "fas",
    # Adverse event / safety
    "sae", "teae", "dlt", "irr", "crs",
    # Regulators / ethics
    "fda", "ema", "pmda", "irb", "iec", "dmc", "dsmb", "dsmc",
    # Regulatory standards
    "gcp", "rrmm",
    # PK/PD/immunogenicity acronyms
    "pk", "pd", "ada", "nab",
    # Lab
    "cbc", "crp", "ldh", "b2m", "ngs", "spep", "pcr", "cytof",
    # Sampling
    "eot", "eos", "c1d1",
    # Route-of-administration abbreviations ("IV" excluded — ambiguous with Roman numeral)
    "sc", "po", "im", "subq",
})

# ── Mixed-case acronyms: require an exact specific casing ─────────────────────
# e.g. "sCR" has lowercase 's' + uppercase 'CR' — not matched by _UPPERCASE_ONLY.
_SPECIFIC_CASE: dict[str, str] = {
    "scr": "sCR",
}

# ─────────────────────────────────────────────────────────────────────────────
# Normalization table
#   Maps term_lower → (full_normalized_name, abbreviation)
#   Used to enrich entity output with structured normalized values.
# ─────────────────────────────────────────────────────────────────────────────

_NORMALIZATIONS: dict[str, tuple[str, str]] = {
    # Route-of-administration abbreviations
    "sc":                         ("Subcutaneous",                         "SC"),
    "po":                         ("Oral",                                 "PO"),
    "im":                         ("Intramuscular",                        "IM"),
    "subq":                       ("Subcutaneous",                         "SubQ"),
    "subcutaneous":               ("Subcutaneous",                         "SC"),
    "intravenous":                ("Intravenous",                          "IV"),
    "intramuscular":              ("Intramuscular",                        "IM"),
    # Clinical endpoints
    "pfs":                        ("Progression-Free Survival",            "PFS"),
    "pfs2":                       ("Progression-Free Survival 2",          "PFS2"),
    "os":                         ("Overall Survival",                     "OS"),
    "mrd":                        ("Minimal Residual Disease",             "MRD"),
    "mrd negativity":             ("MRD Negativity",                       "MRD-neg"),
    "mrd negativity status":      ("MRD Negativity Status",                "MRD-neg"),
    "mrd-negative":               ("MRD-Negative",                        "MRD-neg"),
    "mrd negative":               ("MRD-Negative",                        "MRD-neg"),
    "orr":                        ("Overall Response Rate",                "ORR"),
    "vgpr":                       ("Very Good Partial Response",           "VGPR"),
    "cr":                         ("Complete Response",                    "CR"),
    "scr":                        ("Stringent Complete Response",          "sCR"),
    "pr":                         ("Partial Response",                     "PR"),
    "dor":                        ("Duration of Response",                 "DOR"),
    "cbr":                        ("Clinical Benefit Rate",                "CBR"),
    "ttr":                        ("Time to Response",                     "TTR"),
    "ttnt":                       ("Time to Next Treatment",               "TTNT"),
    "minimal residual disease":   ("Minimal Residual Disease",             "MRD"),
    "overall survival":           ("Overall Survival",                     "OS"),
    "progression-free survival":  ("Progression-Free Survival",            "PFS"),
    "progression free survival":  ("Progression-Free Survival",            "PFS"),
    "overall response rate":      ("Overall Response Rate",                "ORR"),
    "duration of response":       ("Duration of Response",                 "DOR"),
    "complete response":          ("Complete Response",                    "CR"),
    "partial response":           ("Partial Response",                     "PR"),
    "very good partial response": ("Very Good Partial Response",           "VGPR"),
    "stringent complete response":("Stringent Complete Response",          "sCR"),
    # Study populations
    "itt":                        ("Intent-to-Treat Population",           "ITT"),
    "intent to treat":            ("Intent-to-Treat Population",           "ITT"),
    "intent-to-treat":            ("Intent-to-Treat Population",           "ITT"),
    "mitt":                       ("Modified Intent-to-Treat Population",  "mITT"),
    "fas":                        ("Full Analysis Set",                    "FAS"),
    "full analysis set":          ("Full Analysis Set",                    "FAS"),
    "safety set":                 ("Safety Set",                           ""),
    "safety population":          ("Safety Population",                    ""),
    # PK/PD
    "pk":                         ("Pharmacokinetics",                     "PK"),
    "pharmacokinetics":           ("Pharmacokinetics",                     "PK"),
    "pd":                         ("Pharmacodynamics",                     "PD"),
    "pharmacodynamics":           ("Pharmacodynamics",                     "PD"),
    # Immunogenicity
    "ada":                        ("Anti-Drug Antibody",                   "ADA"),
    "anti-drug antibody":         ("Anti-Drug Antibody",                   "ADA"),
    "immunogenicity":             ("Immunogenicity",                       ""),
    # Sampling timepoints
    "eot":                        ("End of Treatment",                     "EOT"),
    "eos":                        ("End of Study",                         "EOS"),
    "c1d1":                       ("Cycle 1 Day 1",                        "C1D1"),
    # Statistics
    "kaplan-meier":               ("Kaplan-Meier",                         "KM"),
    "kaplan meier":               ("Kaplan-Meier",                         "KM"),
    "hazard ratio":               ("Hazard Ratio",                         "HR"),
    "confidence interval":        ("Confidence Interval",                  "CI"),
    "cox regression":             ("Cox Regression",                       ""),
    "log-rank":                   ("Log-rank Test",                        ""),
    # Adverse events
    "sae":                        ("Serious Adverse Event",                "SAE"),
    "teae":                       ("Treatment-Emergent Adverse Event",     "TEAE"),
    "dlt":                        ("Dose-Limiting Toxicity",               "DLT"),
    "crs":                        ("Cytokine Release Syndrome",            "CRS"),
    "irr":                        ("Infusion-Related Reaction",            "IRR"),
    # Grading systems
    "nci-ctcae":                  ("NCI Common Terminology Criteria for Adverse Events", "NCI-CTCAE"),
    "ctcae":                      ("Common Terminology Criteria for Adverse Events",     "CTCAE"),
    # Clinical response criteria (acronyms + long forms)
    "cr":                         ("Complete Response",                    "CR"),
    "pr":                         ("Partial Response",                     "PR"),
    "vgpr":                       ("Very Good Partial Response",           "VGPR"),
    "scr":                        ("Stringent Complete Response",          "sCR"),
    "stable disease":             ("Stable Disease",                       "SD"),
    "progressive disease":        ("Progressive Disease",                  "PD"),
    "minor response":             ("Minor Response",                       "MR"),
    "mrd-negative":               ("MRD-Negative",                         "MRD-neg"),
    "mrd negative":               ("MRD-Negative",                         "MRD-neg"),
    # Study visits
    "eot visit":                  ("End of Treatment Visit",               "EOT Visit"),
    "end of treatment visit":     ("End of Treatment Visit",               "EOT Visit"),
    "follow-up visit":            ("Follow-up Visit",                      ""),
    "screening visit":            ("Screening Visit",                      ""),
    "baseline visit":             ("Baseline Visit",                       ""),
    # Treatment phases
    "treatment phase":            ("Treatment Phase",                      ""),
    "treatment period":           ("Treatment Period",                     ""),
    "maintenance phase":          ("Maintenance Phase",                    ""),
    "induction phase":            ("Induction Phase",                      ""),
    "consolidation phase":        ("Consolidation Phase",                  ""),
    "observation phase":          ("Observation Phase",                    ""),
    "dose escalation phase":      ("Dose Escalation Phase",                ""),
    "run-in period":              ("Run-in Period",                        ""),
}

# ── Label family hierarchy ────────────────────────────────────────────────────
# Maps fine-grained labels → top-level family buckets for hierarchical
# filtering, analytics, and UI grouping.
_LABEL_FAMILY: dict[str, str] = {
    # Assessment family
    "PK_ASSESSMENT":             "ASSESSMENT",
    "PD_ASSESSMENT":             "ASSESSMENT",
    "IMMUNOGENICITY_ASSESSMENT": "ASSESSMENT",
    "LAB_ASSESSMENT":            "ASSESSMENT",
    "BIOMARKER":                 "ASSESSMENT",
    # Timepoint family
    "SAMPLING_TIMEPOINT":        "TIMEPOINT",
    "STUDY_VISIT":               "TIMEPOINT",
    "SAMPLING_TRIGGER":          "TIMEPOINT",
    # Specimen family
    "SPECIMEN_COLLECTION":       "SPECIMEN",
    # Response family
    "CLINICAL_RESPONSE":         "RESPONSE",
    # Endpoint family
    "CLINICAL_ENDPOINT":         "ENDPOINT",
    # Intervention family
    "MEDICATION":                "INTERVENTION",
    # Safety family
    "ADVERSE_EVENT":             "SAFETY",
    "SAFETY_ASSESSMENT":         "SAFETY",
    # Condition family
    "MEDICAL_CONDITION":         "CONDITION",
    # Population family
    "STUDY_POPULATION":          "POPULATION",
    # PRO family
    "QUESTIONNAIRE":             "PRO",
    # Statistics family
    "STATISTICAL_METHOD":        "STATISTICS",
    # Design family
    "STUDY_DESIGN":              "DESIGN",
    "TREATMENT_PHASE":           "DESIGN",
    "STUDY_ARM":                 "DESIGN",
    # Regulatory family
    "REGULATORY_ID":             "REGULATORY",
    "ORGANIZATION":              "REGULATORY",
    "CLINICAL_GRADING_SYSTEM":   "REGULATORY",
    "CLINICAL_GUIDELINE":        "REGULATORY",
    "PROTOCOL_ID":               "REGULATORY",
    # Molecular family
    "GENE_PROTEIN":              "MOLECULAR",
}

# ── Pre-sorted term index: longest match first (greedy) ──────────────────────
_TERM_INDEX: list[tuple[str, str, str, str]] = sorted(
    CLINICAL_TERMS, key=lambda t: len(t[0]), reverse=True,
)


def match_entities(text: str) -> list[dict]:
    """
    Whole-word, case-insensitive dictionary scan over *text*.

    Returns entity dicts with the full schema including normalized name and
    abbreviation (when available):
      {text, label, sub_type, canonical, normalized, abbreviation, start, end, score}

    Strategy:
    - Iterate terms longest-first so "Kaplan-Meier" is found before "Kaplan".
    - Enforce whole-word boundaries (no alphanumeric/hyphen adjacent).
    - For short uppercase-only acronyms, require the source text to be uppercase.
    - Skip any position already covered by a longer match (greedy).
    """
    text_lower = text.lower()
    n          = len(text_lower)
    covered    = bytearray(n)   # 1 = position already claimed
    found: list[dict] = []

    for term_lower, label, sub_type, canonical in _TERM_INDEX:
        tlen  = len(term_lower)
        start = 0
        while start <= n - tlen:
            pos = text_lower.find(term_lower, start)
            if pos == -1:
                break
            end = pos + tlen

            # ── Whole-word boundary check ──────────────────────────────────
            pre  = text_lower[pos - 1] if pos > 0  else " "
            post = text_lower[end]     if end < n   else " "
            if pre.isalnum() or pre == "-" or post.isalnum() or post == "-":
                start = pos + 1
                continue

            # ── Case constraint ────────────────────────────────────────────
            if term_lower in _SPECIFIC_CASE:
                if text[pos:end] != _SPECIFIC_CASE[term_lower]:
                    start = pos + 1
                    continue
            elif term_lower in _UPPERCASE_ONLY:
                if text[pos:end] != text[pos:end].upper():
                    start = pos + 1
                    continue

            # ── Skip if span already covered by a longer term ──────────────
            if any(covered[pos:end]):
                start = end
                continue

            # ── Claim this span ────────────────────────────────────────────
            for i in range(pos, end):
                covered[i] = 1

            norm_full, abbrev = _NORMALIZATIONS.get(term_lower, (canonical, ""))

            found.append({
                "text":         text[pos:end],
                "label":        label,
                "sub_type":     sub_type,
                "canonical":    canonical,
                "normalized":   norm_full,
                "abbreviation": abbrev,
                "family":       _LABEL_FAMILY.get(label, "OTHER"),
                "start":        pos,
                "end":          end,
                "score":        1.0,
            })
            start = end

    return found
