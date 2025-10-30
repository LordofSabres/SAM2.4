# pages/1_About_SAM.py
import streamlit as st

st.title("📘 About — SAM (Sarcasm Authentication Machine) 2.4")

st.info(
    "SAM 2.4 performs single-utterance sarcasm detection for speech-to-text style inputs. "
    "It blends an LLM JSON classifier with deterministic guardrails that separate humor from irony, "
    "apply qualifier immunity (e.g., 'if only', 'but'), require self-relevance for positive-gloss sarcasm, "
    "and catch hard irony markers."
)

st.subheader("Why sarcasm is tricky")
st.markdown(
    "- Sarcasm often appears as **positive gloss over a negative event** (e.g., “love when the app crashes”).\n"
    "- Context and contrast matter; without context, many lines are simply humor or neutral.\n"
    "- Prosody (tone) is lost in text, so we must rely on **lexical patterns and structure**."
)

st.subheader("What SAM 2.4 does differently")
st.markdown(
    "- **Zero-context safe default**: humor/segues ≠ sarcasm by themselves.\n"
    "- **Qualifier immunity**: “if only / except / but / though / unless” → factual contrast, not irony.\n"
    "- **Self-relevance gate**: positive gloss + negative event is sarcastic only when self-relevant "
    "(I/my/me/we/our) or via experiencer frames (“love when”, “just what I needed”, “now what”).\n"
    "- **Rhetorical thanks & scalar negatives**: catches “Thanks a lot — two days late” and "
    "“Amazing support — 2 days for a reply”.\n"
    "- **Hard markers**: “yeah right”, “as if”, “what could go wrong”, “said no one”."
)

st.subheader("Version history")
st.markdown(
    "- **2.4 (current)** — Working version using OpenAI API as the backend with logic and prompt designed by me.\n"
    "- **2.3** — Another attempt at Version 2.1 (did not work) \n"
    "- **2.2** — The TACO app (initial mobile base where the app listens for the word **TACO** and vibrates. Will be given the logic of SAM later.\n"
    "- **2.1** — Attempted training of phrasing from Hugging Face Sarcastic tweets dataset (did not work)\n"
    "- **2.0** — Made during the hackathon, which utilizes OpenAI and speech to text (shutdown due to lack of credits)\n"
    "- **1.0** — JAMES (Jovial Autism Machine Exploring Sarcasm) was created using a TJBot for my high school capstone."
)

st.subheader("The Future")
st.markdown(
    "- **Speech-to-Text** input for live transcripts.\n"
    "- Add to a mobile app base where it silently vibrates to let the user know if sarcasm was said.\n"
    "- Gather training data and testing data of both phrases, tones, and facial expressions to create a version not requiring OpenAI."
)

st.subheader("Links to past work")
st.markdown(
    "- Devpost from version 2.0: https://devpost.com/software/sam-sarcasm-authentication-model\n"
    "- Github of the SAM (JAMES) 1.0: https://github.com/LordofSabres/J.A.M.E.S-TJBot\n"
    "- Research paper written about the ideal version of SAM: https://tinyurl.com/sarcasmdetectorresearchpaper"
)

st.markdown("---")
st.caption("© 2025 SAM 2.4 — Sarcasm Authentication Machine | Add your contact / GitHub link here.")
