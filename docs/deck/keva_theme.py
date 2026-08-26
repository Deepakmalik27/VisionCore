"""keva_theme.py -- the single palette every deck artefact reads.

Lifted verbatim from the Keva restaurant app so the deck, the flowchart, the
report page and the video overlay all look like the product:
    RESTAURANT-APP/app/globals.css      :root tokens
    RESTAURANT-APP/tailwind.config.ts   font stacks

Do not hand-pick a colour anywhere else in this folder. Add it here.
"""

# ── brand ────────────────────────────────────────────────────────────────
SLATE = "1C1917"        # --keva-slate     primary text / dark ground
FOG = "FAF8F5"          # --keva-fog       page ground
PAPER = "FFFEFB"        # --paper-white    cards
COPPER = "D4622B"       # --keva-brass     THE accent
COPPER_DK = "7C2D12"    # --keva-brass-dark
COPPER_LT = "FCEADB"    # --keva-brass-light
COPPER_TINT = "FDF5EF"  # --keva-brass-tint
SAGE = "8B7E6F"         # --keva-sage      secondary text
OLIVE = "5C6B4F"        # --keva-olive     "good" semantic
ERROR = "DC2626"        # --keva-error
SLATE_LT = "F5F1EB"     # --keva-slate-light
HAIR = "DDD5CB"         # --border, resolved from HSL 30 12% 85%

# ── what the boxes in the annotated video mean (kevacv/helpers.py:57) ────
BOX_STAFF = "EB6834"
BOX_GUEST = "2A78D6"
BOX_FILTERED = "898781"

# ── type ─────────────────────────────────────────────────────────────────
# The app ships Space Grotesk + JetBrains Mono. Both are on Google Fonts, so
# the web deck loads them for real; PowerPoint falls back to what Windows has.
WEB_SANS = "'Space Grotesk',-apple-system,'Segoe UI',sans-serif"
WEB_MONO = "'JetBrains Mono',ui-monospace,Menlo,monospace"
GOOGLE_FONTS = ("https://fonts.googleapis.com/css2?"
                "family=Space+Grotesk:wght@400;500;600;700&"
                "family=JetBrains+Mono:wght@400;500;600&display=swap")

PPT_SANS = "Segoe UI"          # nearest grotesk that ships with Windows
PPT_MONO = "Consolas"

# ── radii (--radius-base / --radius-sharp) ───────────────────────────────
RADIUS = 8
RADIUS_SHARP = 6


def rgb(h):
    """'D4622B' -> (212, 98, 43), for PIL and OpenCV-style callers."""
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def css_dark_tokens():
    """[data-theme='dark'] from the app, for the web deck's dark mode."""
    return {"SLATE": "F5F1EB", "FOG": "1C1917", "PAPER": "22201D",
            "COPPER": "E8844A", "SAGE": "B5ADA1", "HAIR": "35312D",
            "SLATE_LT": "2A2724", "COPPER_TINT": "2A1D17"}
