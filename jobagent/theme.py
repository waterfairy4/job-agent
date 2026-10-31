"""The cute look: pink palette from the background picture, translucent rounded chat cards.

APP_THEME=plain in .env switches it off. APP_BACKGROUND points to another picture.
Only CSS on top of Streamlit, so the app works the same with or without it.
"""
import base64
from pathlib import Path

from jobagent import config

ROSE, ROSE_DARK, BLUSH, PLUM = "#d6608a", "#a3415f", "#fbdde7", "#4a2633"

CSS = """
<style>
.stApp {{
  background: {background};
  color: {plum};
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{
  max-width: 880px; margin-left: 3vw; margin-right: auto; padding-top: 2.2rem;
}}
h1, h2, h3 {{ color: {rose_dark} !important; letter-spacing: 0.3px; }}
.stApp, .stApp p, .stApp li, .stApp label {{
  font-family: "Nunito", "Quicksand", "Microsoft YaHei UI", "PingFang SC", "Segoe UI", sans-serif;
}}
[data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] p {{
  color: {rose_dark} !important; font-weight: 600;
  text-shadow: 0 1px 6px rgba(255, 255, 255, 0.9);
}}

/* sidebar: soft frosted pink */
[data-testid="stSidebar"] > div:first-child {{
  background: rgba(255, 238, 244, 0.86);
  backdrop-filter: blur(8px);
  border-right: 1px solid #f3c6d3;
}}

/* chat bubbles: white, translucent, rounded, readable over the picture */
[data-testid="stChatMessage"] {{
  background: rgba(255, 255, 255, 0.80);
  border: 1px solid rgba(255, 255, 255, 0.95);
  border-radius: 22px;
  padding: 0.9rem 1.1rem;
  box-shadow: 0 6px 22px rgba(214, 96, 138, 0.14);
  backdrop-filter: blur(6px);
  margin-bottom: 0.7rem;
}}

/* buttons: pill shaped */
.stButton > button, .stDownloadButton > button {{
  border-radius: 999px;
  border: 1.5px solid #e895ab;
  background: #fff5f8;
  color: {rose_dark};
  font-weight: 600;
  transition: all 0.15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  background: {blush}; border-color: {rose}; color: {rose_dark};
  transform: translateY(-1px); box-shadow: 0 4px 12px rgba(214, 96, 138, 0.25);
}}

/* chat input: a soft rounded bar */
[data-testid="stChatInput"] {{
  border-radius: 999px !important;
  border: 1.5px solid #f0b3c4 !important;
  background: rgba(255, 255, 255, 0.92) !important;
  box-shadow: 0 6px 20px rgba(214, 96, 138, 0.18);
}}
[data-testid="stBottom"] > div {{ background: transparent !important; }}
[data-testid="stBottomBlockContainer"] {{
  background: transparent !important; max-width: 880px; margin-left: 3vw; margin-right: auto;
}}
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] textarea {{ border-radius: 999px !important; }}

/* expanders and alerts inside bubbles */
[data-testid="stExpander"] details {{ border-radius: 14px; border-color: #f3c6d3; background: rgba(255,250,252,0.7); }}
[data-testid="stAlert"] {{ border-radius: 16px; }}
</style>
"""


def _background() -> str:
    path = Path(config.APP_BACKGROUND)
    if path.exists():
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode()
        # a light pink veil on the left keeps text readable; the hearts on the right stay visible
        return (f"linear-gradient(90deg, rgba(252,232,239,0.55) 0%, rgba(252,232,239,0.25) 55%, "
                f"rgba(252,232,239,0) 75%), url(data:{mime};base64,{data}) right center / cover fixed")
    return "linear-gradient(135deg, #fde4ec 0%, #fbd3df 55%, #f8c5d4 100%)"


def apply(st) -> dict:
    """Inject the theme. Returns the texts and avatars the app should use."""
    if config.APP_THEME == "plain":
        return {"title": "🧭 Job Agent", "user": None, "assistant": None, "thinking": "Working...",
                "caption": "Paste a job description or link, find jobs, or ask me to make your CV.",
                "yes": "✅ Approve", "no": "❌ Reject"}
    st.markdown(CSS.format(background=_background(), plum=PLUM, rose=ROSE, rose_dark=ROSE_DARK, blush=BLUSH),
                unsafe_allow_html=True)
    return {"title": "💗 Job Agent", "user": "🌸", "assistant": "🧸", "thinking": "Thinking… 💭",
            "caption": "Your cozy job-hunting buddy ✨ Paste a job or a link, find jobs, or ask me to make your CV.",
            "yes": "💖 Yes, save it", "no": "🙅 Not now"}
