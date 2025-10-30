# phase1_api_semantic.py
import os, json, math, re
from typing import List, Dict, Any
import numpy as np
import streamlit as st
import openai
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------
# Load API key
# ---------------------------
load_dotenv()

# Prefer Streamlit Secrets; fall back to environment
API_KEY = (st.secrets.get("OPENAI_API_KEY")
           if hasattr(st, "secrets") else None) or os.getenv("OPENAI_API_KEY")

if not API_KEY:
    st.error("⚠️ OPENAI_API_KEY not found. Add it in Streamlit → Settings → Secrets, or in a local .env.")
    st.stop()

# Make it available to the OpenAI SDK and anything else that reads env
os.environ["OPENAI_API_KEY"] = API_KEY

# Initialize the client WITHOUT passing kwargs (SDK reads from env)
client = OpenAI()

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="SAM 2.4 — Sarcasm (Semantic Context)", page_icon="🗣️", layout="wide")
st.title("🗣️ SAM (Sarcasm Authentication Machine) 2.4 — Sarcasm Detector")

# ---------------------------
# Session state: memory store (vector DB in RAM)
# Each item: {text, label, topic, embedding (np.array)}
# ---------------------------
if "memory" not in st.session_state:
    st.session_state.memory: List[Dict[str, Any]] = []
if "emb_model_name" not in st.session_state:
    st.session_state.emb_model_name = "text-embedding-3-small"

MEM = st.session_state.memory

# ---------------------------
# Models
# ---------------------------
DEFAULT_EMBED_MODEL = "text-embedding-3-small"  # inexpensive, great for retrieval
CHAT_MODEL          = "gpt-4o-mini"             # JSON mode compatible

# ---------------------------
# Embedding helpers
# ---------------------------
def embed(text: str) -> np.ndarray:
    model_name = st.session_state.get("emb_model_name", DEFAULT_EMBED_MODEL)
    resp = client.embeddings.create(model=model_name, input=text)
    return np.array(resp.data[0].embedding, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)

# ---------------------------
# Context retrieval
# ---------------------------
def retrieve_semantic_context(query_text: str, k: int = 8, per_bucket: int = 3):
    """
    Retrieve up to `per_bucket` best matches from facts and sarcasm each (2*per_bucket <= k),
    combining cosine similarity with recency decay. Fill remaining slots with next-best overall.
    Returns (facts_list, sarcasm_list) as bullet strings for the prompt.
    """
    if not MEM:
        return ["- (none)"], ["- (none)"]

    qvec = embed(query_text)

    def recency_decay(idx_from_end: int, base: float = 0.90):
        # 0 = newest
        return base ** idx_from_end

    total = len(MEM)
    scored_facts, scored_sarcasm = [], []
    for idx, item in enumerate(MEM):
        sim = cosine_sim(qvec, item["embedding"])
        age = (total - 1) - idx  # 0 newest
        weight = sim * recency_decay(age)
        if item["label"] == "non-sarcastic":
            scored_facts.append((weight, item))
        else:
            scored_sarcasm.append((weight, item))

    scored_facts.sort(key=lambda x: x[0], reverse=True)
    scored_sarcasm.sort(key=lambda x: x[0], reverse=True)

    top_facts_items = scored_facts[:per_bucket]
    top_sarc_items  = scored_sarcasm[:per_bucket]

    chosen_ids = {id(it) for _, it in (top_facts_items + top_sarc_items)}
    remaining = max(0, k - len(chosen_ids))
    all_scored = sorted(scored_facts + scored_sarcasm, key=lambda x: x[0], reverse=True)
    fill_items = []
    for w, it in all_scored:
        if id(it) not in chosen_ids:
            fill_items.append((w, it))
            chosen_ids.add(id(it))
        if len(fill_items) >= remaining:
            break

    def fmt(items): return [f'- {it["text"]}' for _, it in items]
    top_facts = fmt(top_facts_items) or ["- (none)"]
    top_sarc  = fmt(top_sarc_items) or ["- (none)"]

    if fill_items:
        top_facts += [f'- {it["text"]}' for _, it in fill_items]

    return top_facts, top_sarc

# ---------------------------
# UPDATED SYSTEM PROMPT (rules baked in)
# ---------------------------
SYSTEM = """You are a careful sarcasm analyst for short conversational lines.

Follow this reasoning order:
1) Understand literal meaning.
2) Infer emotional tone (positive/negative/neutral).
3) Use SEMANTIC CONTEXT only to check for contradiction; do not assume sarcasm from prior lines.
4) Derive likely world state from explicit phrases.
5) Compare utterance tone to that world state:
   • Agreement → non-sarcastic
   • Clear contradiction/exaggeration → sarcastic
6) Only label “sarcastic” when there is a clear contradiction, explicit irony markers, or positive gloss over a self-relevant negative event.

Speech-to-text single-utterance rules:
• Humor without contradiction → NON-SARCASTIC. Discourse markers (“speaking of”, “haha”, “lol”), and bare interjections (“sure”, “absolutely”, “right”, “yeah”) do NOT imply sarcasm by themselves.
• Qualifiers ⇒ factual, not ironic. If the negative is introduced by “if only”, “except”, “but”, “though”, “unless”, treat as factual qualification, not sarcasm.
• Self-relevance gate. Positive gloss + negative event is sarcastic only when self-relevant (I/my/me/we/our) or when experiencer is implied by frames like “love when”, “just what I needed”, “now what”.
• Rhetorical thanks. “Thanks/thank you” is sarcastic only if paired with a negative event or contradiction (e.g., “for nothing”, “two days late”).
• Comparative scalar negative. Positive gloss followed by a negative scalar/metric (long delay, high cost, many bugs) signals sarcasm.
• Imperatives. “Please …” or “Do …” are non-sarcastic unless paired with an in-utterance contradiction/negative.
• Hard irony triggers (always sarcasm): “yeah right”, “as if”, “what could go wrong”, “said no one”.
• Default: With no context contradiction and none of the above triggers, return NON-SARCASTIC.

Return ONLY valid JSON with keys:
- label: "sarcastic" | "non-sarcastic"
- confidence: number 0..1 (higher only when a listed condition is satisfied)
- explanation: 1–2 sentences naming the specific contradiction or marker if sarcastic (or the reason for non-sarcastic)
- topic: short string inferred from content
- debug: { "contradiction": true|false, "irony_markers": true|false }
"""

USER_TEMPLATE = """UTTERANCE:
"{utterance}"

SEMANTIC CONTEXT (nearest prior lines):
- Prior facts (literal / non-sarcastic):
{facts_block}
- Prior sarcastic remarks:
{sarcasm_block}
"""

# ---------------------------
# DETERMINISTIC GUARDRAIL (post-processor)
# ---------------------------
# Regex lexicons
POS_GLOSS = [
    r"\blovely\b", r"\bperfect\b", r"\bamazing\b", r"\bwonderful\b",
    r"\bfantastic\b", r"\bawesome\b", r"\bgreat\b", r"\bnice\b", r"\bbrilliant\b", r"\bgenius\b"
]
NEG_EVENTS = [
    r"\brain(ing|s)?\b|\bpour(ing|s)\b|\bstorm(y)?\b|\bwind(y)?\b",
    r"\btraffic\b|\bpile[- ]?up\b|\baccident\b|\bjam\b|\bdelay(ed)?\b|\blate\b",
    r"\bcrash(ed|es|ing)?\b|\bserver down\b|\boutage\b|\bbug(s)?\b|\bfailed?\b",
    r"\bfired\b|\bbroke(n)?\b|\bdied\b|\bflat tire\b|\bsick\b|\bflu\b|\bcold\b",
    r"\bfee\b|\bcharge(d)?\b|\bfine\b|\bovercharge(d)?\b"
]
IRONY_HARD = [
    r"\byeah right\b", r"\bas if\b", r"\bwhat could go wrong\b", r"\bsaid no one\b"
]
HUMOR_MARKERS = [
    r"\bspeaking of\b", r"\bfun fact\b", r"\banyway\b", r"\bha(ha+)?\b", r"\blol\b", r"\blmao\b"
]
QUALIFIERS = [r"\bif only\b", r"\bexcept\b", r"\bbut\b", r"\bthough\b", r"\bunless\b"]
SELF_REF = [r"\bi\b", r"\bme\b", r"\bmy\b", r"\bwe\b", r"\bours?\b"]
DISCOURSE_OK = [
    r"^sure\.?$", r"^absolutely\.?$", r"^right\.?$", r"^yeah\.?$",
    r"^yeah, no(\,? for sure)?\.?$", r"^okay\.?$", r"^ok\.?$"
]
THANKS = [r"^thanks\b", r"^thank you\b"]
SCALAR_NEG = [
    r"\b\d+\s*(days?|hours?|weeks?)\b.*\b(reply|response|wait|delay)\b",
    r"\$\s*\d+", r"\b\d+\s*(crashes|bugs|errors)\b", r"\b(twice|thrice|ten times|dozens)\b"
]
IMPERATIVE_START = [r"^please\b", r"^do\b", r"^go ahead\b", r"^by all means\b"]

def _any(patterns, text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)

def _topic(text: str) -> str:
    if re.search(r"hike|trail|camp|mosquito|outdoors|park", text, re.I): return "outdoors"
    if re.search(r"weather|rain|sunny|wind|breeze|storm|cool breeze", text, re.I): return "weather"
    if re.search(r"traffic|bus|train|jam|accident|commute", text, re.I): return "traffic"
    if re.search(r"server|crash|bug|deploy|build|ticket", text, re.I): return "tech"
    if re.search(r"exam|class|homework|assignment|grade|professor", text, re.I): return "school"
    if re.search(r"boss|meeting|deadline|office|raise|promotion", text, re.I): return "work"
    return "general"

def apply_guardrail_rules(utterance: str, model_json: Dict) -> Dict:
    """
    Deterministic pass to reduce false positives and enforce rules.
    Uses the single utterance primarily; context contradictions are already in 'debug' from the model.
    """
    u = utterance.strip()

    # Discourse-only → non-sarcastic
    if _any(DISCOURSE_OK, u):
        return {
            "label": "non-sarcastic",
            "confidence": max(0.65, float(model_json.get("confidence", 0.0))),
            "explanation": "Bare interjection without contradiction; prosody unavailable in STT.",
            "topic": model_json.get("topic") or _topic(u),
            "debug": {"contradiction": False, "irony_markers": False}
        }

    # Hard irony markers → sarcastic
    if _any(IRONY_HARD, u):
        return {
            "label": "sarcastic",
            "confidence": max(0.92, float(model_json.get("confidence", 0.0))),
            "explanation": "Explicit irony marker in utterance.",
            "topic": model_json.get("topic") or _topic(u),
            "debug": {"contradiction": True, "irony_markers": True}
        }

    pos_gloss     = _any(POS_GLOSS, u)
    mentions_neg  = _any(NEG_EVENTS, u)
    self_rel      = _any(SELF_REF, u)
    qualified_neg = mentions_neg and _any(QUALIFIERS, u)
    humor_only    = _any(HUMOR_MARKERS, u) and not (pos_gloss and mentions_neg and self_rel)

    # Humor without contradiction → non-sarcastic
    if humor_only:
        return {
            "label": "non-sarcastic",
            "confidence": max(0.7, float(model_json.get("confidence", 0.0))),
            "explanation": "Humorous/segue language without clear contradiction.",
            "topic": model_json.get("topic") or _topic(u),
            "debug": {"contradiction": False, "irony_markers": False}
        }

    # Rhetorical thanks
    if _any(THANKS, u):
        if mentions_neg or re.search(r"\bfor nothing\b|\bthat (really )?helped\b", u, re.I) or _any(SCALAR_NEG, u):
            return {
                "label": "sarcastic",
                "confidence": max(0.88, float(model_json.get("confidence", 0.0))),
                "explanation": "Rhetorical thanks paired with negative/contradiction.",
                "topic": model_json.get("topic") or _topic(u),
                "debug": {"contradiction": True, "irony_markers": True}
            }
        else:
            return {
                "label": "non-sarcastic",
                "confidence": max(0.75, float(model_json.get("confidence", 0.0))),
                "explanation": "Genuine thanks; no negative event or contradiction present.",
                "topic": model_json.get("topic") or _topic(u),
                "debug": {"contradiction": False, "irony_markers": False}
            }

    # Comparative scalar negative
    if pos_gloss and _any(SCALAR_NEG, u):
        return {
            "label": "sarcastic",
            "confidence": max(0.88, float(model_json.get("confidence", 0.0))),
            "explanation": "Positive gloss followed by negative scalar metric implies irony.",
            "topic": model_json.get("topic") or _topic(u),
            "debug": {"contradiction": True, "irony_markers": True}
        }

    # Imperatives need contradiction to be sarcastic
    if _any(IMPERATIVE_START, u):
        if mentions_neg:
            return {
                "label": "sarcastic",
                "confidence": max(0.8, float(model_json.get("confidence", 0.0))),
                "explanation": "Imperative paired with negative event suggests ironic intent.",
                "topic": model_json.get("topic") or _topic(u),
                "debug": {"contradiction": True, "irony_markers": True}
            }
        else:
            return {
                "label": "non-sarcastic",
                "confidence": max(0.7, float(model_json.get("confidence", 0.0))),
                "explanation": "Imperative without contradiction or negative event.",
                "topic": model_json.get("topic") or _topic(u),
                "debug": {"contradiction": False, "irony_markers": False}
            }

    # Positive gloss + negative event
    if pos_gloss and mentions_neg:
        if qualified_neg:
            return {
                "label": "non-sarcastic",
                "confidence": max(0.75, float(model_json.get("confidence", 0.0))),
                "explanation": "Qualifier (e.g., 'if only', 'but') makes it a factual contrast, not irony.",
                "topic": model_json.get("topic") or _topic(u),
                "debug": {"contradiction": False, "irony_markers": False}
            }
        if self_rel or re.search(r"\blove when\b|\bjust what i needed\b|\bnow what\b", u, re.I):
            return {
                "label": "sarcastic",
                "confidence": max(0.87, float(model_json.get("confidence", 0.0))),
                "explanation": "Positive gloss over self-relevant negative event or experiencer frame.",
                "topic": model_json.get("topic") or _topic(u),
                "debug": {"contradiction": True, "irony_markers": True}
            }
        # News-like contrast without self-relevance
        return {
            "label": "non-sarcastic",
            "confidence": max(0.7, float(model_json.get("confidence", 0.0))),
            "explanation": "Contrast without self-relevance; treated as non-sarcastic.",
            "topic": model_json.get("topic") or _topic(u),
            "debug": {"contradiction": False, "irony_markers": False}
        }

    # Otherwise: trust the model but keep conservative fallback if it marked sarcastic without justification
    dbg = model_json.get("debug") or {}
    if model_json.get("label") == "sarcastic" and not (dbg.get("contradiction") or dbg.get("irony_markers")):
        return {
            "label": "non-sarcastic",
            "confidence": 0.6,
            "explanation": "Ambiguous tone without contradiction or irony markers; defaulting to non-sarcastic.",
            "topic": model_json.get("topic") or _topic(u),
            "debug": {"contradiction": False, "irony_markers": False}
        }

    # Return model_json as-is (ensure required keys exist)
    return {
        "label": model_json.get("label", "non-sarcastic"),
        "confidence": float(model_json.get("confidence", 0.6)),
        "explanation": model_json.get("explanation", "No explicit irony or contradiction detected."),
        "topic": model_json.get("topic") or _topic(u),
        "debug": model_json.get("debug") or {"contradiction": False, "irony_markers": False}
    }

# ---------------------------
# Sidebar (simplified for deployment)
# ---------------------------
with st.sidebar:
    st.markdown("### Controls")
    auto_add = st.checkbox("Auto-add to memory after classify", value=True)
    if st.button("Clear memory"):
        st.session_state.memory = []
        st.success("Memory cleared.")

    st.caption("Model: gpt-4o-mini · Embeddings: text-embedding-3-small")
    st.caption(f"OpenAI SDK: {openai.__version__}")

    st.divider()
    st.page_link("1_About_SAM.py", label="About SAM 2.4", icon="📘")

# ---------------------------
# Layout
# ---------------------------
left, right = st.columns([1, 1])

# Left: memory preview
with left:
    st.subheader("🧠 Semantic Memory (recent)")
    if not MEM:
        st.info("No memory yet. Classify a sentence to start building context.")
    else:
        # Show last 12 entries
        for i, item in enumerate(MEM[-12:][::-1], 1):
            st.write(f"**{i}.** [{item['label']}] {item['text']}  — _topic: {item['topic']}_")
        st.caption(f"Total stored: {len(MEM)}")

# Right: classifier
with right:
    st.subheader("💬 Classify an Utterance")
    utterance = st.text_input("Enter a sentence:", "")

    if st.button("Classify"):
        try:
            # 1) Retrieve semantic context (balanced + recency)
            facts_list, sarcasm_list = retrieve_semantic_context(utterance, k=8, per_bucket=3)
            facts_block = "\n".join(facts_list) or "- (none)"
            sarcasm_block = "\n".join(sarcasm_list) or "- (none)"

            # Debug: show exactly what was used
            with st.expander("🔎 Debug: retrieved context", expanded=False):
                st.markdown("**Facts used:**\n" + (facts_block if facts_block.strip() else "(none)"))
                st.markdown("**Sarcasm used:**\n" + (sarcasm_block if sarcasm_block.strip() else "(none)"))

            user_msg = USER_TEMPLATE.format(
                utterance=utterance.strip(),
                facts_block=facts_block,
                sarcasm_block=sarcasm_block
            )

            # 2) Classify via Chat Completions JSON mode
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            raw_result = json.loads(content)

            # 2b) Guardrail pass (deterministic rules from our new spec)
            result = apply_guardrail_rules(utterance, raw_result)

            # 3) Show result
            pct = round(float(result["confidence"]) * 100, 1)
            st.json({
                "label": result["label"],
                "confidence_percent": pct,
                "explanation": result["explanation"],
                "topic": result.get("topic", "general")
            })
            if result["label"] == "sarcastic":
                st.error(f"Sarcastic ({pct}%)")
            else:
                st.success(f"Non-sarcastic ({pct}%)")

            # 4) Auto-add to memory (store embedding so future queries find this)
            if auto_add:
                vec = embed(utterance.strip())
                MEM.append({
                    "text": utterance.strip(),
                    "label": result["label"],
                    "topic": (result.get("topic") or "general").strip(),
                    "embedding": vec
                })
                st.toast("Added to semantic memory")

        except Exception as e:
            st.error(f"API error: {e}")


