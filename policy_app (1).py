import os
import re
import fitz
import nltk
import PyPDF2
import gradio as gr
from dotenv import load_dotenv
from google import genai
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from transformers import pipeline



# NLTK SETUP

def _init_nltk():
    for path, pkg in [
        ("tokenizers/punkt",     "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords",    "stopwords"),
        ("corpora/wordnet",      "wordnet"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

_init_nltk()

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()
_NOISE_WORDS = {"etc", "eg", "i.e", "e.t.c", "viz", "also", "would"}


#  GEMINI 

load_dotenv()
FALLBACK_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY") or FALLBACK_API_KEY

gemini_client   = None
GEMINI_AVAILABLE = False

if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    try:
        gemini_client    = genai.Client(api_key=GEMINI_API_KEY)
        GEMINI_AVAILABLE = True
        print(" Gemini API connected.")
    except Exception as e:
        print(f"  Gemini init failed: {e}. Using local models.")
else:
    print(" No Gemini key. Using local BART + GPT-2.")



# LOCAL MODEL LOADERS

_bart = None

def _get_bart():
    global _bart
    if _bart is None:
        print("⏳ Loading BART (~400 MB, first run only)…")
        _bart = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
        print("✅ BART loaded.")
    return _bart



# PDF EXTRACTION

def extract_text_from_pdf(pdf_file_obj) -> str:
    if pdf_file_obj is None:
        return ""
    file_path = pdf_file_obj if isinstance(pdf_file_obj, str) else pdf_file_obj.name
    try:
        doc  = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        if text.strip():
            return text.strip()
    except Exception:
        pass
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            pages  = [p.extract_text() for p in reader.pages if p.extract_text()]
        return "\n".join(pages).strip()
    except Exception:
        return ""


# NLP PREPROCESSING

def clean_text(text: str) -> str:
    text  = text.lower()
    text  = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = word_tokenize(text)
    words = [
        _lemmatizer.lemmatize(w)
        for w in words
        if w not in _stop_words and w not in _NOISE_WORDS and len(w) > 2
    ]
    return " ".join(words)

def chunk_text(text: str, max_words: int = 600) -> list:
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]



#  SUMMARISATION MODULE


def _clean_for_bart(text: str) -> str:
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [l for l in text.splitlines() if len(l.strip()) > 25 or l.strip() == ""]
    return "\n".join(lines).strip()


def _chunk_sentences(text: str, max_words: int = 450) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current_words, current = [], 0, []
    for sent in sentences:
        wc = len(sent.split())
        if current_words + wc > max_words and current:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(sent)
        current_words += wc
    if current:
        chunks.append(" ".join(current))
    return chunks


def _summarize_gemini(raw_text: str) -> str:
    prompt = (
        "You are a senior government policy analyst producing a comprehensive executive summary "
        "of a policy document for submission to a Cabinet committee.\n\n"

        "REQUIREMENTS:\n"
        "• The summary must be LONG and DETAILED — do not condense or skip any important content.\n"
        "• Begin with a 3–4 sentence INTRODUCTION paragraph that states what the policy is, "
        "who issued it, and what problem it addresses.\n"
        "• Then produce EXACTLY the three numbered sections below.\n"
        "• Every bullet must use this format:\n"
        "    **Bold Label:** Two to three full sentences of explanation drawn directly from the document.\n"
        "• End with a 2–3 sentence CONCLUSION paragraph on expected impact and significance.\n\n"

        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "INTRODUCTION\n"
        "(Write 3–4 sentences here before the numbered sections)\n\n"

        "1. MAIN GOALS OF THE POLICY\n"
        "Write 7–10 bullet points covering ALL major goals in the document.\n"
        "Each bullet: **Goal Title:** 2–3 sentences fully explaining the goal, "
        "including any targets, beneficiaries, or expected outcomes mentioned.\n\n"

        "2. KEY MEASURES & STRATEGIES\n"
        "Write 7–10 bullet points covering ALL significant measures, programmes, "
        "regulations, or strategic approaches described.\n"
        "Each bullet: **Measure Title:** 2–3 sentences fully explaining what the measure "
        "entails, who is responsible, and how it will be implemented.\n\n"

        "3. OVERALL DIRECTION OF THE POLICY\n"
        "Write 5–7 bullet points covering the overarching vision, guiding principles, "
        "long-term ambitions, and values that underpin the policy.\n"
        "Each bullet: **Principle/Direction Title:** 2–3 sentences of explanation.\n\n"

        "CONCLUSION\n"
        "(Write 2–3 sentences summarising the expected impact and significance of the policy)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        "IMPORTANT: Be thorough. If the document is long, cover all sections. "
        "Do not write 'see document for details' — include the details in your summary.\n\n"
        f"POLICY DOCUMENT TEXT:\n{raw_text[:20000]}"
    )
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip()


def _summarize_bart(raw_text: str) -> str:
    bart        = _get_bart()
    light_clean = _clean_for_bart(raw_text)
    chunks      = _chunk_sentences(light_clean, max_words=420)

    raw_summaries = []
    for chunk in chunks[:12]:
        if len(chunk.split()) < 35:
            continue
        try:
            out = bart(
                chunk,
                max_length=220,
                min_length=70,
                do_sample=False,
                truncation=True,
            )
            text = out[0]["summary_text"].strip()
            if text:
                raw_summaries.append(text)
        except Exception:
            continue

    if not raw_summaries:
        return "  Could not extract a summary from this document."

    all_sentences = []
    for block in raw_summaries:
        for sent in re.split(r"(?<=[.!?])\s+", block):
            sent = sent.strip()
            if len(sent) > 20:
                all_sentences.append(sent)

    seen, deduped = set(), []
    for s in all_sentences:
        key = s[:60].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    all_sentences = deduped

    total = len(all_sentences)
    if total == 0:
        return " ".join(raw_summaries)

    cut1 = max(2, round(total * 0.40))
    cut2 = max(cut1 + 2, round(total * 0.75))

    goals_sents     = all_sentences[:cut1]
    measures_sents  = all_sentences[cut1:cut2]
    direction_sents = all_sentences[cut2:]

    def fmt_bullets(sents):
        return "\n".join(f"  • {s}" for s in sents) if sents else "  • (Insufficient content extracted)"

    direction_block = fmt_bullets(direction_sents) if direction_sents else (
        f"  {all_sentences[-1]}" if all_sentences else "  See full document for overall direction."
    )

    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "STRUCTURED POLICY SUMMARY  [BART Local Model]\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "1. MAIN GOALS OF THE POLICY\n"
        f"{fmt_bullets(goals_sents)}\n\n"
        "2. KEY MEASURES & STRATEGIES\n"
        f"{fmt_bullets(measures_sents)}\n\n"
        "3. OVERALL DIRECTION OF THE POLICY\n"
        f"{direction_block}\n\n"
        "─────────────────────────────────────────────────────\n"
        f"ℹ  Generated from {len(chunks)} document sections · "
        f"{total} key points extracted.\n"
        "   Add a Gemini API key for bold-formatted, AI-curated summaries."
    )


def process_file(pdf_file) -> tuple[str, str]:
    if pdf_file is None:
        return "", "  Please upload a PDF file first."

    raw = extract_text_from_pdf(pdf_file)
    if not raw.strip():
        return "", "  No readable text found in this PDF."

    if len(raw.split()) < 30:
        return "", "Text is too short to summarize."

    if GEMINI_AVAILABLE:
        try:
            result = _summarize_gemini(raw)
            return result, " Summary generated via Gemini 2.5 Flash."
        except Exception as e:
            print(f"Gemini summarization failed: {e}. Using BART.")

    result = _summarize_bart(raw)
    return result, " Summary generated via BART (local model)."



# SECTION 7 – SCENARIO DEFINITIONS

SCENARIOS = {
    "Rural Development": {
        "label":       "Rural & Regional Development",
        "description": "Adapt the policy to prioritise rural communities, remote regions, and underserved areas.",
        "lens":        "rural infrastructure gaps, agricultural livelihoods, decentralised service delivery, digital connectivity in remote areas, community participation",
        "audience":    "rural local authorities, village officers, regional ministries, rural community groups",
        "constraints": "must account for low-literacy populations; solutions must be affordable and low-tech where needed; local language accessibility required",
        "tone":        "inclusive, community-centred, practical",
        "icon":        "🌾",
        "color":       "#f59e0b",
    },
    "Post-Disaster Recovery": {
        "label":       "Post-Disaster Recovery & Resilience",
        "description": "Reframe the policy for communities rebuilding after floods, droughts, or other disasters.",
        "lens":        "emergency response, infrastructure rebuilding, psychosocial support, climate-resilient reconstruction, inter-agency coordination",
        "audience":    "Disaster Management Centre, armed forces, INGOs, affected community leaders, public health authorities",
        "constraints": "speed of implementation is critical; equity for most-affected households must be central; avoid creating dependency",
        "tone":        "urgent, compassionate, operationally clear",
        "icon":        "🆘",
        "color":       "#ef4444",
    },
    "Urban & Youth Focus": {
        "label":       "Urban Communities & Youth Empowerment",
        "description": "Adapt the policy to address urban youth, employment, and city-level implementation.",
        "lens":        "youth unemployment, entrepreneurship, digital skills, civic participation, urban inequality, mental health",
        "audience":    "urban local councils, youth ministries, universities, vocational training institutes, private sector",
        "constraints": "must address urban-rural inequality gap; youth data privacy must be protected; gender-inclusive design required",
        "tone":        "dynamic, aspirational, youth-friendly yet formal",
        "icon":        "🏙️",
        "color":       "#3b82f6",
    },
    "Education & Research": {
        "label":       "Education System & Academic Research",
        "description": "Translate the policy into directives for educational institutions and research bodies.",
        "lens":        "curriculum integration, teacher capacity building, research funding, academic partnerships, student outcomes",
        "audience":    "Ministry of Education, university councils, school principals, research institutions, academic staff",
        "constraints": "must align with existing national curriculum framework; equity for under-resourced schools required; evidence-based approach mandatory",
        "tone":        "academic, evidence-based, student-centred",
        "icon":        "🎓",
        "color":       "#8b5cf6",
    },
    "Healthcare & Wellbeing": {
        "label":       "Public Health & Community Wellbeing",
        "description": "Adapt the policy with a focus on health outcomes, healthcare access, and wellbeing.",
        "lens":        "preventive healthcare, equitable access to services, mental health, community health workers, health infrastructure",
        "audience":    "Ministry of Health, hospitals, primary care centres, rural health workers, public health researchers",
        "constraints": "patient data anonymisation mandatory; clinical decisions must involve qualified professionals; equity for low-income communities is non-negotiable",
        "tone":        "compassionate, evidence-based, patient-safety-first",
        "icon":        "🏥",
        "color":       "#10b981",
    },
    "Environment & Sustainability": {
        "label":       "Environmental Protection & Sustainability",
        "description": "Re-orient the policy around climate action, conservation, and sustainable development goals.",
        "lens":        "carbon reduction, biodiversity, climate adaptation, green economy, environmental regulation, SDG alignment",
        "audience":    "Ministry of Environment, conservation authorities, green NGOs, climate researchers, local governments",
        "constraints": "must align with national NDC commitments; polluter-pays principle must apply; indigenous and local community rights must be respected",
        "tone":        "science-driven, forward-looking, regulatory",
        "icon":        "🌿",
        "color":       "#06b6d4",
    },
}

SCENARIO_KEYS = list(SCENARIOS.keys())



#  GENERATION MODULE

def _build_prompt(policy_type: str, scenario_key: str, summary: str, custom_scenario: str = "") -> str:
    if custom_scenario.strip():
        scenario_label  = custom_scenario.strip()
        lens            = f"all aspects specifically relevant to: {scenario_label}"
        audience        = f"key stakeholders and implementing bodies for {scenario_label}"
        constraints     = f"constraints typical to the {scenario_label} context"
        tone            = "formal, authoritative, context-appropriate"
    else:
        s               = SCENARIOS[scenario_key]
        scenario_label  = s["label"]
        lens            = s["lens"]
        audience        = s["audience"]
        constraints     = s["constraints"]
        tone            = s["tone"]

    return f"""You are a senior government policy drafter tasked with adapting an existing policy document \
to a new scenario. You must use ONLY the content from the source policy summary provided below — \
do NOT invent facts, statistics, or measures that are not present in the summary.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE POLICY SUMMARY (your only content source):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{summary[:4000]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ADAPTATION INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DOCUMENT TYPE   : {policy_type}
SCENARIO        : {scenario_label}
ADAPTATION LENS : {lens}
TARGET AUDIENCE : {audience}
KEY CONSTRAINTS : {constraints}
REQUIRED TONE   : {tone}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT STRUCTURE (use exactly these 6 sections):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PREAMBLE
   Explain why this policy, as drawn from the source summary, is specifically relevant \
to the "{scenario_label}" scenario. Reference the policy's core intent.

2. STRATEGIC OBJECTIVES
   List 4–5 objectives that re-focus the source policy's goals through the lens of \
"{scenario_label}". Each objective must be traceable to the source summary.

3. KEY PRIORITY INITIATIVES
   Describe 4–5 concrete programmes or actions derived from the source policy's measures, \
adapted for "{scenario_label}". Name a plausible lead agency or responsible body for each.

4. GOVERNANCE & ACCOUNTABILITY
   Describe oversight arrangements, monitoring mechanisms, and compliance requirements \
appropriate for "{scenario_label}" and consistent with the source policy's direction.

5. IMPLEMENTATION TIMELINE
   Phase 1 (Year 1): Immediate actions
   Phase 2 (Years 2–3): Medium-term programmes
   Phase 3 (Years 4+): Long-term institutionalisation

6. STRATEGIC ALIGNMENT
   Explain how this adapted policy draft connects to the broader goals and vision described \
in the source policy summary.

IMPORTANT RULES:
- Every section must clearly reflect the "{scenario_label}" scenario.
- Derive all content from the SOURCE POLICY SUMMARY above — do not fabricate.
- Use formal, official government policy language throughout.
- Do NOT mention AI, technology, or any domain not present in the source summary \
unless it was already in the summary.
"""


def _generate_gemini(policy_type: str, scenario_key: str, summary: str, custom_scenario: str = "") -> str:
    prompt   = _build_prompt(policy_type, scenario_key, summary, custom_scenario)
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    label = custom_scenario.strip() if custom_scenario.strip() else SCENARIOS[scenario_key]["label"]
    return (
        f"ADAPTED POLICY DRAFT\n"
        f"SCENARIO : {label.upper()}\n"
        f"TYPE     : {policy_type}\n"
        f"{'═' * 55}\n\n"
        f"{response.text.strip()}"
    )


def _extract_summary_parts(summary: str) -> dict:
    goals, measures, direction = [], [], []
    current = None

    for line in summary.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if "main goals" in low or "1." in low[:5]:
            current = "goals"
            continue
        elif "measures" in low or "strategies" in low or "2." in low[:5]:
            current = "measures"
            continue
        elif "overall direction" in low or "3." in low[:5]:
            current = "direction"
            continue

        text = re.sub(r"^[•\-\*\d\.\s]+", "", stripped).strip()
        if len(text) < 10:
            continue
        if current == "goals":
            goals.append(text)
        elif current == "measures":
            measures.append(text)
        elif current == "direction":
            direction.append(text)

    if not goals and not measures:
        all_lines = [
            re.sub(r"^[•\-\*\d\.\s]+", "", l.strip()).strip()
            for l in summary.splitlines()
            if len(l.strip()) > 15
        ]
        mid = max(1, len(all_lines) // 2)
        goals    = all_lines[:mid]
        measures = all_lines[mid:]

    return {
        "goals":      goals,
        "measures":   measures,
        "direction":  direction,
        "all_bullets": goals + measures,
    }


def _structured_template_fallback(
    policy_type: str,
    scenario_key: str,
    summary: str,
    custom_scenario: str = "",
) -> str:
    if custom_scenario.strip():
        label       = custom_scenario.strip()
        audience    = f"all relevant implementing ministries and stakeholders for {label}"
        constraints = f"implementation must be equitable, evidence-based, and context-appropriate for {label}"
        tone_note   = "formal and authoritative"
    else:
        s           = SCENARIOS[scenario_key]
        label       = s["label"]
        audience    = s["audience"]
        constraints = s["constraints"]
        tone_note   = s["tone"]

    parts    = _extract_summary_parts(summary)
    goals    = parts["goals"]
    measures = parts["measures"]
    direction_lines = parts["direction"]

    def pick(lst, idx, fallback=""):
        return lst[idx] if idx < len(lst) else (fallback or (lst[-1] if lst else ""))

    if direction_lines:
        direction_para = " ".join(direction_lines)
    elif goals:
        direction_para = (
            f"The overarching ambition of this policy, as adapted for {label}, "
            f"is to ensure that {goals[0].lower()} This strategic orientation "
            f"shall guide all implementing agencies in their planning, "
            f"resource allocation, and performance reporting obligations."
        )
    else:
        direction_para = (
            f"This policy commits the Government to sustained, coordinated action "
            f"on {label} in a manner consistent with national development priorities "
            f"and international best practice."
        )

    initiative_items = measures if measures else goals
    agencies = [
        "Ministry of Finance and Planning",
        "Ministry of Environment and Natural Resources",
        "Ministry of Local Government and Provincial Councils",
        "Ministry of Education and Higher Education",
        "Ministry of Health and Indigenous Medicine",
        "National Planning Commission",
    ]
    initiative_blocks = ""
    for i, item in enumerate(initiative_items[:5]):
        agency = agencies[i % len(agencies)]
        initiative_blocks += (
            f"\n    {i+1}. {item.rstrip('.')}. "
            f"Lead Agency: {agency}."
        )

    gov_measure  = pick(measures, 0, pick(goals, 0, "effective institutional coordination"))
    gov_measure2 = pick(measures, 1, pick(goals, 1, "transparent resource management"))

    phase1 = pick(goals,    0, "Establish inter-ministerial coordination mechanisms")
    phase2 = pick(measures, 0, "Roll out key programmatic interventions across priority regions")
    phase3 = pick(goals,    -1, "Institutionalise sustainable practices across all implementing bodies")
    if measures:
        phase3 = pick(measures, -1, phase3)

    alignment_text = direction_para if direction_lines else (
        f"This adapted policy directly advances the strategic vision articulated "
        f"in the source document, which emphasises {pick(goals, 0, 'sustainable and equitable development').lower()}. "
        f"By channelling that vision through the specific lens of {label}, "
        f"this draft ensures that national policy commitments translate into "
        f"targeted, context-sensitive action."
    )

    separator = "═" * 60
    draft = f"""ADAPTED POLICY DRAFT
SCENARIO : {label.upper()}
TYPE     : {policy_type}
{separator}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. PREAMBLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Recognising the imperative to adapt existing national policy frameworks to the \
specific circumstances of {label}, the Government hereby sets forth this adapted \
{policy_type}. This document is grounded in the core goals and measures of the \
source policy and reconfigures their strategic orientation to address the distinct \
challenges and opportunities presented by {label}.

The primary purpose of this policy, in its adapted form, is to ensure that \
{pick(goals, 0, 'national priorities are effectively translated into action').lower()}. \
It is further intended to ensure accountability, equitable access, and \
measurable outcomes for all communities and institutions affected by {label}. \
This policy shall be binding upon all relevant ministries, implementing agencies, \
and public bodies identified herein.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. STRATEGIC OBJECTIVES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The following objectives guide all action under this policy, as adapted for {label}:

    i.   To ensure that {pick(goals, 0, 'policy goals are clearly defined and communicated').lower()} \
within the context of {label}.
    ii.  To advance {pick(goals, 1, pick(measures, 0, 'coordinated institutional response')).lower()} \
through targeted programmes and interagency collaboration.
    iii. To establish transparent and accountable governance mechanisms that ensure \
{pick(goals, 2, pick(measures, 1, 'equitable distribution of resources and benefits')).lower()}.
    iv.  To promote evidence-based decision-making and continuous improvement in the \
delivery of outcomes related to {label}.
    v.   To align all implementing activities with the overarching national development \
priorities identified in the source policy.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. KEY PRIORITY INITIATIVES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The following priority initiatives are derived from the key measures identified \
in the source policy and adapted for {label}:{initiative_blocks}

Each initiative shall have a dedicated budget line, a designated lead agency, \
clear key performance indicators, and quarterly reporting obligations to the \
relevant oversight body.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. GOVERNANCE & ACCOUNTABILITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

An inter-ministerial steering committee shall be constituted to provide \
strategic oversight of this policy's implementation in the context of {label}. \
The committee shall convene no less than quarterly and shall include \
representatives from {audience}.

All implementing agencies are required to submit progress reports on a \
six-monthly basis. These reports shall measure performance against the \
objectives set out in Section 2, with particular attention to \
{gov_measure.lower().rstrip('.')} and {gov_measure2.lower().rstrip('.')}.

An independent audit of policy outcomes shall be conducted annually. \
Findings shall be made publicly available and presented to the relevant \
parliamentary committee. Non-compliance with reporting obligations shall \
attract administrative sanctions in accordance with applicable regulations. \
The following constraint shall apply throughout implementation: {constraints}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. IMPLEMENTATION TIMELINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase 1 — Year 1 (Foundations):
    • {phase1.rstrip('.')}.
    • Establish the inter-ministerial steering committee and appoint lead agency focal points.
    • Develop detailed implementation plans with measurable milestones for each initiative.
    • Conduct stakeholder consultations with {audience}.

Phase 2 — Years 2–3 (Implementation):
    • {phase2.rstrip('.')}.
    • Scale priority initiatives to all target regions and implementing bodies.
    • Conduct mid-term review and adjust programmes based on evidence.
    • Strengthen monitoring and evaluation systems and publish first annual audit report.

Phase 3 — Year 4 and Beyond (Consolidation):
    • {phase3.rstrip('.')}.
    • Institutionalise successful models and embed them into regular government operations.
    • Conduct comprehensive policy review and submit recommendations for revision \
or continuation to Cabinet.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. STRATEGIC ALIGNMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{alignment_text}

All implementing agencies are directed to ensure their operational plans are \
consistent with this policy and to report any conflicts or implementation \
barriers to the steering committee without delay.

{separator}
[This draft was generated using structured template synthesis from the uploaded
 policy summary. For AI-generated prose drafts, configure a Gemini API key.]"""

    return draft


def generate_scenario(
    policy_type: str,
    scenario_key: str,
    current_summary: str,
    custom_scenario: str = "",
) -> tuple[str, str, str]:
    """
    Returns: (policy_draft, status_message, scenario_card_html)
    """
    # ── Build the scenario card HTML for the selected scenario ──
    if custom_scenario.strip():
        # Custom scenario: render a simple info card
        label = custom_scenario.strip()
        card_html = f"""
        <div style="
            background: rgba(99,102,241,0.10);
            border: 1px solid rgba(99,102,241,0.4);
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 0 0 2px rgba(99,102,241,0.20);
        ">
            <div style="font-size:1rem; font-weight:700; color:#818cf8; margin-bottom:6px;">
                ✏️&nbsp; Custom Scenario: {label}
            </div>
            <div style="font-size:0.78rem; color:#94A3B8; font-style:italic;">
                This draft has been generated using your custom scenario definition.
                All content is derived exclusively from the uploaded policy document.
            </div>
        </div>"""
    else:
        s = SCENARIOS[scenario_key]
        r, g, b = int(s["color"][1:3], 16), int(s["color"][3:5], 16), int(s["color"][5:7], 16)
        card_html = f"""
        <div style="
            background: rgba({r},{g},{b},0.12);
            border: 1px solid {s['color']};
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 14px;
            box-shadow: 0 0 0 2px {s['color']}40;
        ">
            <div style="font-size:1rem; font-weight:700; color:{s['color']}; margin-bottom:6px;">
                {s['icon']}&nbsp; {s['label']}
            </div>
            <div style="font-size:0.8rem; color:#CBD5E1; margin-bottom:10px; font-style:italic;">
                {s['description']}
            </div>
            <div style="display:grid; gap:6px;">
                <div style="font-size:0.76rem; color:#94A3B8; line-height:1.6;">
                    <span style="
                        display:inline-block;
                        background:rgba({r},{g},{b},0.18);
                        color:{s['color']};
                        font-weight:700;
                        font-size:0.68rem;
                        letter-spacing:0.5px;
                        text-transform:uppercase;
                        padding:2px 8px;
                        border-radius:6px;
                        margin-bottom:3px;
                    ">Audience</span><br>
                    {s['audience']}
                </div>
                <div style="font-size:0.76rem; color:#94A3B8; line-height:1.6; margin-top:6px;">
                    <span style="
                        display:inline-block;
                        background:rgba({r},{g},{b},0.18);
                        color:{s['color']};
                        font-weight:700;
                        font-size:0.68rem;
                        letter-spacing:0.5px;
                        text-transform:uppercase;
                        padding:2px 8px;
                        border-radius:6px;
                        margin-bottom:3px;
                    ">Lens</span><br>
                    {s['lens']}
                </div>
                <div style="font-size:0.76rem; color:#94A3B8; line-height:1.6; margin-top:6px;">
                    <span style="
                        display:inline-block;
                        background:rgba({r},{g},{b},0.18);
                        color:{s['color']};
                        font-weight:700;
                        font-size:0.68rem;
                        letter-spacing:0.5px;
                        text-transform:uppercase;
                        padding:2px 8px;
                        border-radius:6px;
                        margin-bottom:3px;
                    ">Constraints</span><br>
                    {s['constraints']}
                </div>
            </div>
        </div>"""

    # ── Guard: no summary yet ──
    if not current_summary or len(current_summary.strip()) < 20:
        empty_msg = (
            "  No summary found.\n\n"
            "Please upload a PDF and click 'Summarise Policy' in the LEFT panel first.\n"
            "The generated draft will be adapted from your document's summary."
        )
        return empty_msg, "  Please summarise a policy document first.", card_html

    label = custom_scenario.strip() if custom_scenario.strip() else SCENARIOS[scenario_key]["label"]

    if GEMINI_AVAILABLE:
        try:
            result = _generate_gemini(policy_type or "National Strategy", scenario_key, current_summary, custom_scenario)
            return result, f" Draft generated via Gemini 2.5 Flash — Scenario: {label}.", card_html
        except Exception as e:
            print(f"Gemini generation failed: {e}. Using structured template fallback.")

    result = _structured_template_fallback(policy_type or "National Strategy", scenario_key, current_summary, custom_scenario)
    return result, f"Structured draft generated — Scenario: {label}.", card_html



#  GRADIO UI

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 60%, #0F172A 100%) !important;
    min-height: 100vh;
    color: #E2E8F0 !important;
}

textarea, input[type="text"], input[type="password"] {
    background: rgba(15, 23, 42, 0.85) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #CBD5E1 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
textarea:focus, input:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    outline: none !important;
}

label {
    color: #94A3B8 !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

.gr-button, button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    letter-spacing: 0.2px !important;
}
.gr-button.primary, button[variant="primary"] {
    background: linear-gradient(90deg, #3B82F6, #6366F1) !important;
    border: none !important;
    color: #fff !important;
}
.gr-button.primary:hover, button[variant="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}
.gr-button.secondary, button[variant="secondary"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #94A3B8 !important;
}
.gr-button.secondary:hover, button[variant="secondary"]:hover {
    background: rgba(255,255,255,0.13) !important;
    color: #F1F5F9 !important;
}

select, .gr-dropdown select {
    background: rgba(15,23,42,0.85) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #CBD5E1 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
}
.gr-radio label { color: #CBD5E1 !important; font-size: 0.82rem !important; }
.gr-radio input[type="radio"]:checked + label { color: #60A5FA !important; font-weight: 600 !important; }

.panel-wrap {
    background: rgba(30, 41, 59, 0.65) !important;
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 22px 20px !important;
    transition: box-shadow 0.3s ease;
}
.panel-wrap:hover { box-shadow: 0 8px 32px rgba(59, 130, 246, 0.10) !important; }

.gr-accordion {
    background: rgba(15,23,42,0.5) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}

.gr-file {
    background: rgba(15,23,42,0.6) !important;
    border: 1px dashed rgba(99,102,241,0.4) !important;
    border-radius: 10px !important;
    color: #94A3B8 !important;
}
"""

STATUS_ICON = "🟢" if GEMINI_AVAILABLE else "🟡"
STATUS_TEXT = "Gemini 2.5 Flash connected" if GEMINI_AVAILABLE else "Gemini offline — BART (local) active"

HEADER_HTML = f"""
<div style="
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding: 18px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
    border-radius: 14px;
">
    <div>
        <div style="font-size:1.45rem; font-weight:700; color:#F8FAFC; letter-spacing:-0.3px;">
            🏛 Policy Summarizer &amp; Scenario Generator
        </div>
        <div style="font-size:0.78rem; color:#64748B; margin-top:3px;">
            AI-Powered Policy Understanding and Adaptation
        </div>
    </div>
    <div style="display:flex; align-items:center; gap:10px;">
        <div style="
            background: linear-gradient(90deg,#3B82F6,#6366F1);
            color:#fff; font-size:0.68rem; font-weight:700;
            padding:5px 14px; border-radius:20px; letter-spacing:0.6px;
            text-transform:uppercase;"
        
        <div style="
            background:rgba(255,255,255,0.06);
            border:1px solid rgba(255,255,255,0.1);
            color:#94A3B8; font-size:0.72rem;
            padding:5px 12px; border-radius:20px;
        ">{STATUS_ICON} {STATUS_TEXT}</div>
    </div>
</div>
"""

LEFT_HEADER = """
<div style="margin-bottom:14px;">
    <div style="font-size:1.05rem; font-weight:600; color:#F1F5F9; letter-spacing:-0.2px;">
         Policy Summarisation
    </div>
    <div style="font-size:0.76rem; color:#64748B; margin-top:3px;">
        Upload a PDF policy document and generate a structured summary.
    </div>
    <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,0.07),transparent); margin-top:12px;"></div>
</div>
"""

RIGHT_HEADER = """
<div style="margin-bottom:14px;">
    <div style="font-size:1.05rem; font-weight:600; color:#F1F5F9; letter-spacing:-0.2px;">
         Scenario-Based Policy Generation
    </div>
    <div style="font-size:0.76rem; color:#64748B; margin-top:3px;">
        Select a scenario and generate an adapted policy draft. The scenario context card will appear with your results.
    </div>
    <div style="height:1px; background:linear-gradient(90deg,transparent,rgba(255,255,255,0.07),transparent); margin-top:12px;"></div>
</div>
"""

FOOTER_HTML = """
<div style="text-align:center; color:#1E293B; font-size:0.72rem; padding:18px; margin-top:4px;">
    
</div>
"""

# Placeholder HTML shown before any generation
CARD_PLACEHOLDER_HTML = """
<div style="
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    color: #334155;
    font-size: 0.78rem;
    font-style: italic;
    margin-bottom: 14px;
">
    Scenario context card will appear here after generating a draft.
</div>
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(css=CUSTOM_CSS, title="Policy Advisor") as app:

        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):

            # ════════════════════════════════════
            # LEFT PANEL — Summarisation
            # ════════════════════════════════════
            with gr.Column(scale=1, elem_classes=["panel-wrap"]):
                gr.HTML(LEFT_HEADER)

                pdf_input = gr.File(
                    label="Upload Policy Document (PDF)",
                    file_types=[".pdf"],
                    type="filepath",
                )

                with gr.Row():
                    summarise_btn = gr.Button(
                        " Summarise Policy",
                        variant="primary",
                        scale=3,
                    )
                    clear_btn = gr.Button(
                        "✕ Clear",
                        variant="secondary",
                        scale=1,
                    )

                sum_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                )

                summary_output = gr.Textbox(
                    label="Structured Policy Summary",
                    placeholder="Your structured summary will appear here after clicking 'Summarise Policy'…",
                    lines=20,
                    max_lines=30,
                    interactive=True,
                )

            
            # RIGHT PANEL — Generation
            
            with gr.Column(scale=1, elem_classes=["panel-wrap"]):
                gr.HTML(RIGHT_HEADER)

                policy_type_dd = gr.Dropdown(
                    label="Policy Document Type",
                    choices=[
                        "National Strategy",
                        "Ministry Circular",
                        "Government Act",
                        "Gazette Regulation",
                        "Sector Policy Brief",
                        "Cabinet Directive",
                    ],
                    value="National Strategy",
                    allow_custom_value=True,
                )

                scenario_radio = gr.Radio(
                    label="Select Scenario Lens",
                    choices=SCENARIO_KEYS,
                    value=SCENARIO_KEYS[0],
                )

                custom_scenario_input = gr.Textbox(
                    label="✏️ Or define a Custom Scenario (overrides the selection above)",
                    placeholder="e.g.  Climate Adaptation for Coastal Fishing Communities…",
                    lines=2,
                    max_lines=3,
                )

                with gr.Row():
                    generate_btn = gr.Button(
                        " Generate Policy Draft",
                        variant="primary",
                        scale=3,
                    )
                    regenerate_btn = gr.Button(
                        " Regenerate",
                        variant="secondary",
                        scale=1,
                    )

                gen_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                )

                # ── Scenario card appears HERE, above the draft output ──
                scenario_card_display = gr.HTML(CARD_PLACEHOLDER_HTML)

                policy_output = gr.Textbox(
                    label="Generated Policy Draft",
                    placeholder="Your scenario-adapted policy draft will appear here…",
                    lines=20,
                    max_lines=35,
                )

        gr.HTML(FOOTER_HTML)

        
        # EVENT WIRING
        
        summarise_btn.click(
            fn=process_file,
            inputs=[pdf_input],
            outputs=[summary_output, sum_status],
        )

        clear_btn.click(
            fn=lambda: ("", "", "", CARD_PLACEHOLDER_HTML),
            inputs=[],
            outputs=[summary_output, sum_status, policy_output, scenario_card_display],
        )

        generate_btn.click(
            fn=generate_scenario,
            inputs=[policy_type_dd, scenario_radio, summary_output, custom_scenario_input],
            outputs=[policy_output, gen_status, scenario_card_display],
        )

        regenerate_btn.click(
            fn=generate_scenario,
            inputs=[policy_type_dd, scenario_radio, summary_output, custom_scenario_input],
            outputs=[policy_output, gen_status, scenario_card_display],
        )

    return app



#  ENTRY POINT

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  AI Policy Summarizer & Scenario Generator")
    print(f"  Gemini: {'✅ Connected' if GEMINI_AVAILABLE else '⚠️  Offline (local models active)'}")
    print("  URL: http://localhost:7860")
    print("=" * 60 + "\n")
    app = build_app()
    app.launch(server_port=7860, share=False, show_error=True, inbrowser=True)
