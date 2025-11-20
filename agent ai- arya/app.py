# app.py
import streamlit as st
from datetime import datetime, date
import pandas as pd
import os
import json
from dotenv import load_dotenv
from typing import Optional, List

# --------------------
# Config / env
# --------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # optional; set this in .env to enable Gemini
DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "study_motivation_history.json")
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------
# Optional Gemini (best-effort)
# --------------------
try:
    import google.generativeai as genai  # type: ignore
    GOOGLE_GENAI_AVAILABLE = True
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
        except Exception:
            # configuration failed; mark unavailable
            GOOGLE_GENAI_AVAILABLE = False
except Exception:
    GOOGLE_GENAI_AVAILABLE = False

# --------------------
# Agent class (single-file)
# --------------------
class StudyMotivationAgent:
    """
    Lightweight Study Motivation Agent.
    - local heuristics for analysis & plan
    - optional Gemini enrichment if GEMINI_API_KEY is present
    """
    role = "Study Motivation Agent"
    goal = "Analyze study difficulties, create a motivational plan and daily tasks"

    def __init__(self):
        pass

    def analyze_reason_local(self, subject: str, difficulty: str, reason_text: str, mood: str, time_available_min: int) -> str:
        """Return a short local analysis explaining why the user might struggle."""
        parts: List[str] = []
        diff = difficulty.lower().strip()
        mood_l = mood.lower().strip()
        reason_l = (reason_text or "").strip()

        # difficulty heuristics
        if diff in ("very hard", "hard", "difficult"):
            parts.append("The topic is challenging — it likely needs smaller steps and more practice.")
        elif diff in ("medium", "okay"):
            parts.append("Content is manageable but may need better structure or focused sessions.")
        else:
            parts.append("Material seems easy — the issue may be motivation or routine.")

        # mood heuristics
        if mood_l in ("tired", "stressed", "low motivation", "low"):
            parts.append("Current energy/mood suggests short sessions and self-care first (rest, hydration).")

        # reason-specific hint
        if reason_l:
            # short fallback heuristics
            if "formul" in reason_l or "formula" in reason_l:
                parts.append("Formulas often need active practice and spaced repetition.")
            if "concept" in reason_l or "abstract" in reason_l:
                parts.append("Try visual examples and small analogies to ground abstract concepts.")
            if "practice" in reason_l or "problems" in reason_l:
                parts.append("Practice-based learning will help — start with guided examples.")
        # time heuristics
        if time_available_min < 30:
            parts.append("You have limited time — micro-tasks (10–20 min) are best.")
        elif time_available_min > 180:
            parts.append("You have long time blocks — structure them into deep-work sessions.")

        return " ".join(parts)

    def build_plan_local(self, subject: str, difficulty: str, reason_text: str, mood: str, time_available_min: int) -> dict:
        """
        Return a dict with:
         - motivational_message (short)
         - today_tasks (list)
         - weekly_plan (skeleton list of dict)
        """
        diff = difficulty.lower().strip()
        mood_l = mood.lower().strip()
        subj = subject.strip() or "your topic"

        # motivational message
        msg_parts: List[str] = []
        if diff in ("very hard", "hard", "difficult"):
            msg_parts.append("Break goals into tiny steps — each small win builds momentum.")
        else:
            msg_parts.append("Use focused blocks and active recall to keep progress steady.")

        if mood_l in ("tired", "stressed", "low motivation", "low"):
            msg_parts.append("Start with a 5-minute energizer (stretch/breath) then work in short bursts.")
        else:
            msg_parts.append("Start strong with a short warm-up recap to prime your brain.")

        if time_available_min < 30:
            msg_parts.append("Micro-sessions: strict timers (15–20 min) work best right now.")
        elif time_available_min < 90:
            msg_parts.append("Use 25/5 or 50/10 Pomodoro blocks depending on preference.")
        else:
            msg_parts.append("Combine deep sessions with short breaks and a clear end-of-day review.")

        motivational_message = " ".join(msg_parts)

        # weekly skeleton (generic but descriptive)
        weekly_plan = [
            {"day": "Mon", "focus": "Core concept + 1 guided problem"},
            {"day": "Tue", "focus": "Flashcards & active recall"},
            {"day": "Wed", "focus": "Problem solving + explain out loud"},
            {"day": "Thu", "focus": "Timed practice + error review"},
            {"day": "Fri", "focus": "Mock/test-style practice"},
            {"day": "Sat", "focus": "Revise weak spots"},
            {"day": "Sun", "focus": "Rest + light overview"}
        ]

        # Today tasks based on available time
        t = time_available_min
        today_tasks: List[str] = []
        if t < 30:
            today_tasks = [
                f"Warm-up (5–10 min): quick recap of one key idea from {subj}.",
                "Micro-practice (10–15 min): solve one short problem or make 3 flashcards."
            ]
        elif t < 90:
            today_tasks = [
                f"Warm-up (10 min): recap notes for {subj}.",
                "Focused session (30–40 min): deep work on one sub-topic with practice.",
                "Consolidation (10–15 min): write 5 recall questions and answers."
            ]
        else:
            today_tasks = [
                f"Warm-up & review (15 min): revisit yesterday's weak points in {subj}.",
                "Deep work (2 x 45–50 min): practice problems or concept mapping, with short break.",
                "Reflection (15 min): note mistakes and plan next day's focus."
            ]

        return {
            "motivational_message": motivational_message,
            "today_tasks": today_tasks,
            "weekly_plan": weekly_plan
        }

    def gemini_enrich(self, prompt_extra: str) -> Optional[str]:
        """
        Use Gemini to enrich the plan if available. Returns text or None.
        Wrapped in try/except so failures are silent.
        """
        if not GOOGLE_GENAI_AVAILABLE or not GEMINI_API_KEY:
            return None
        try:
            prompt = (
                "You are an encouraging study coach. Given the user context below, "
                "write a 2-3 sentence motivational paragraph and list 3 practical tasks for today.\n\n"
                + prompt_extra
            )
            resp = genai.generate_text(model="models/text-bison-001", prompt=prompt)
            text = getattr(resp, "text", None) or str(resp)
            return text
        except Exception:
            return None

# --------------------
# Persistence helpers (single-file)
# --------------------
def load_history() -> list:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_entry(entry: dict):
    history = load_history()
    history.append(entry)
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass

def clear_history():
    try:
        with open(DATA_FILE, "w") as f:
            json.dump([], f)
    except Exception:
        pass

# --------------------
# Small quotes utility
# --------------------
QUOTES = [
    "Small consistent steps beat occasional giant leaps.",
    "Focus on progress, not perfection — one topic at a time.",
    "Short, consistent sessions compound into mastery.",
    "Break a task down: tiny wins build momentum.",
    "Energy first: hydrate, breathe, then start.",
    "Active practice > passive reading. Try it for 20 minutes.",
    "A clear plan beats vague intentions. Plan one small victory today."
]

def get_daily_quote() -> str:
    idx = date.today().toordinal() % len(QUOTES)
    return QUOTES[idx]

# --------------------
# UI theming (single-file) - Dark only
# --------------------
def get_theme_css(theme_name: str) -> str:
    # We will only use dark theme (forced)
    return """
    <style>
    .stApp { background: #07080a; color: #e6eef2; }
    .card { background:#0f1720; padding:18px; border-radius:12px; box-shadow: 0 10px 30px rgba(0,0,0,0.6); }
    .header { color:#9be6d9; font-size:30px; font-weight:700; }
    .sub { color:#bfeee0; margin-bottom:8px; }
    .footer-badge { position: fixed; right: 18px; bottom: 14px; background:#0f766e; color:white; padding:8px 12px; border-radius:999px; font-weight:600; z-index:9999; }
    /* make table rows readable in dark mode */
    .stDataFrame table { color: #e6eef2; }
    </style>
    """

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="AI Study Motivation Agent", page_icon="📚", layout="wide")

# Force dark theme (no sidebar)
st.session_state.theme = "Dark"
st.markdown(get_theme_css("Dark"), unsafe_allow_html=True)

# header block
st.markdown(f'<div class="header">📚 AI Study Motivation Agent</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub">{get_daily_quote()}</div>', unsafe_allow_html=True)

# input card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Tell me what's hard right now")
with st.form("study_form", clear_on_submit=False):
    col_left, col_right = st.columns([2,1], gap="large")
    with col_left:
        subject = st.text_input("Subject / Topic", placeholder="e.g., Linear Algebra, Organic Chemistry")
        difficulty = st.selectbox("Difficulty level", ["Very Hard", "Hard", "Medium", "Easy"], index=1)
        reason_text = st.text_area("What specifically is difficult? (short)", placeholder="e.g., too many formulas, lack of practice, concepts feel abstract", height=80)
    with col_right:
        mood = st.selectbox("Current mood/energy", ["Focused", "Tired", "Stressed", "Low Motivation"], index=0)
        time_available = st.slider("Time available today (minutes)", 10, 300, 60)
        save_to_history = st.checkbox("Save this plan to history", value=True)
    submit = st.form_submit_button("Generate Plan")
st.markdown('</div>', unsafe_allow_html=True)

# create agent
agent = StudyMotivationAgent()

# output card
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Your Plan")

if submit:
    # Local analysis & plan
    context_subject = subject.strip() or "your topic"
    analysis = agent.analyze_reason_local(context_subject, difficulty, reason_text, mood, time_available)
    plan = agent.build_plan_local(context_subject, difficulty, reason_text, mood, time_available)

    # Show analysis
    st.markdown("**Why you might be struggling**")
    st.write(analysis)

    # Motivational message
    st.markdown("**Motivational message**")
    st.info(plan["motivational_message"])

    # Today's tasks
    st.markdown("**Today's Tasks**")
    for i, t_task in enumerate(plan["today_tasks"], start=1):
        st.write(f"{i}. {t_task}")

    # Weekly skeleton
    st.markdown("**Weekly Plan (skeleton)**")
    weekly_df = pd.DataFrame(plan["weekly_plan"])
    st.table(weekly_df)

    # Optional Gemini enrichment
    enriched_text = None
    if GOOGLE_GENAI_AVAILABLE and GEMINI_API_KEY:
        try:
            prompt_extra = (
                f"Subject: {context_subject}\nDifficulty: {difficulty}\nReason: {reason_text}\n"
                f"Mood: {mood}\nTime available (min): {time_available}"
            )
            enriched_text = agent.gemini_enrich(prompt_extra)
        except Exception:
            enriched_text = None

    if enriched_text:
        st.markdown("**AI (optional) — enriched suggestion**")
        st.write(enriched_text)

    # Save to history
    if save_to_history:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "subject": context_subject,
            "difficulty": difficulty,
            "reason": reason_text,
            "mood": mood,
            "time_available": time_available,
            "analysis": analysis,
            "today_tasks": plan["today_tasks"]
        }
        save_entry(entry)
        st.success("Saved plan to history.")
else:
    st.info("Enter your topic and press Generate Plan to get a short motivational plan.")

st.markdown('</div>', unsafe_allow_html=True)

# history card (compact)
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Saved Plans (History)")
history = load_history()
if history:
    # show recent items compactly
    df_hist = pd.DataFrame(history)
    df_hist_display = df_hist[["timestamp", "subject", "difficulty", "time_available"]].sort_values("timestamp", ascending=False)
    st.dataframe(df_hist_display, height=220)
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("Export history as CSV"):
            csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="study_motivation_history.csv", mime="text/csv")
    with c2:
        if st.button("Clear history"):
            clear_history()
            st.experimental_rerun()
else:
    st.info("No saved plans yet. Save a plan to see it here.")

st.markdown('</div>', unsafe_allow_html=True)

# floating footer badge (updated name)
st.markdown("<div class='footer-badge'>Created by Arya</div>", unsafe_allow_html=True)

# small footer caption
st.caption("Notes: Gemini is optional (set GEMINI_API_KEY in .env to enable). The app uses local heuristics by default so it works offline.")
