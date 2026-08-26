"""build_html.py -- the same SLIDES, as a self-contained presentable web page.

Images are embedded as JPEG data URIs (the artifact host blocks external hosts),
so this file is a single portable deck. The 179 MB annotated video is NOT
embedded -- it plays from disk; slide 7 is its cue card.

Design tokens follow the house language already used by the .pptx and the
flowchart: brass on ivory, an optical serif for display, mono for anything
that is a number, a filename or a label.
"""
import base64
import html
import io
import pathlib

from PIL import Image

import keva_theme as T
from build_deck import SLIDES, VIDEO

HERE = pathlib.Path(__file__).parent


def data_uri(name, width=2400, quality=88):
    im = Image.open(HERE / name).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)),
                       Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def video_uri(name="_web.mp4"):
    """The clip, small enough to live inside the page. The 60 MB master stays
    on disk for the projector; this one is for whoever opens the link."""
    return ("data:video/mp4;base64,"
            + base64.b64encode((HERE / name).read_bytes()).decode())


def e(t):
    return html.escape(t).replace("\n", "<br>")


CSS = """
:root{
  --ground:#FAF8F5; --paper:#FFFEFB; --ink:#1C1917; --brass:#D4622B;
  --muted:#8B7E6F; --hair:#DDD5CB; --flag:#7C2D12; --flag-bg:#FCEADB;
  --tint:#FDF5EF; --slatelt:#F5F1EB;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
  --ground:#1C1917; --paper:#22201D; --ink:#F5F1EB; --brass:#E8844A;
  --muted:#B5ADA1; --hair:#35312D; --flag:#E8844A; --flag-bg:#2A1D17;
  --tint:#2A1D17; --slatelt:#2A2724;
  }
}
:root[data-theme="dark"]{
  --ground:#1C1917; --paper:#22201D; --ink:#F5F1EB; --brass:#E8844A;
  --muted:#B5ADA1; --hair:#35312D; --flag:#E8844A; --flag-bg:#2A1D17;
  --tint:#2A1D17; --slatelt:#2A2724;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--ground); color:var(--ink);
  font-family:'Space Grotesk',-apple-system,'Segoe UI',sans-serif;
  -webkit-font-smoothing:antialiased;
  scroll-snap-type:y mandatory; overflow-y:scroll; height:100vh;
}
.slide{
  scroll-snap-align:start; min-height:100vh; display:flex; flex-direction:column;
  justify-content:center; padding:clamp(28px,5vw,80px) clamp(28px,7vw,120px);
  border-bottom:1px solid var(--hair); position:relative; gap:0;
}
/* The two stage panels commit to one look in every theme, so their colours
   are literals on purpose -- a token here flips with the viewer and puts
   dark type on a dark ground. */
.slide.dark{background:#100C09; color:#D8D0C2}
.slide.dark h1, .slide.dark h2{color:#F5F0E5}
.slide.dark .eyebrow, .slide.dark .play, .slide.dark .list .n{color:#C9A470}
.slide.dark .num{color:#7C7467}

.eyebrow{
  font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;
  font-size:.68rem; letter-spacing:.22em; text-transform:uppercase;
  color:var(--brass); margin:0 0 clamp(14px,2vw,26px);
}
h1{
  font-family:'Space Grotesk',-apple-system,'Segoe UI',sans-serif; font-weight:600;
  font-size:clamp(2rem,4.6vw,3.9rem); line-height:1.08; margin:0;
  text-wrap:balance; letter-spacing:-.015em;
}
h2{
  font-family:'Space Grotesk',-apple-system,'Segoe UI',sans-serif; font-weight:600;
  font-size:clamp(1.5rem,3.1vw,2.6rem); line-height:1.14; margin:0;
  text-wrap:balance; letter-spacing:-.01em;
}
p{margin:0; line-height:1.62}
.lede{font-size:clamp(1rem,1.5vw,1.32rem); color:var(--muted);
      max-width:60ch; margin-top:clamp(16px,2vw,26px)}
.body{display:flex; flex-direction:column; gap:.95em; max-width:74ch;
      font-size:clamp(.94rem,1.15vw,1.1rem); color:var(--muted);
      margin-top:clamp(18px,2.4vw,32px)}
.body b{color:var(--ink); font-weight:600}
.rule{height:1px; background:var(--hair); margin:clamp(18px,2.4vw,30px) 0}

.stats{display:flex; flex-wrap:wrap; gap:clamp(24px,4vw,64px);
       margin-top:clamp(20px,3vw,38px)}
.stat .v{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;
         font-size:clamp(1.7rem,3.2vw,2.7rem); color:var(--brass);
         font-variant-numeric:tabular-nums; line-height:1}
.stat .k{font-size:.86rem; color:var(--muted);
         margin-top:.5em}

figure{margin:clamp(16px,2.2vw,28px) 0 0; display:flex; flex-direction:column;
       gap:.9em; min-height:0}
figure img{width:100%; max-height:58vh; object-fit:contain; object-position:left center;
           border-radius:3px}
.slide.full figure img{max-height:70vh; object-position:center}
figcaption{font-size:.86rem; color:var(--muted); max-width:88ch}

.grid{display:grid; gap:clamp(18px,2.2vw,34px); margin-top:clamp(18px,2.4vw,30px)}
.qtable{grid-template-columns:1.1fr 1fr 1.3fr; row-gap:0}
.qtable .h{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace; font-size:.62rem;
           letter-spacing:.18em; text-transform:uppercase; color:var(--brass);
           padding-bottom:.9em; border-bottom:1px solid var(--hair)}
.qtable .c{padding:1.05em 0; border-bottom:1px solid var(--hair);
           font-size:.95rem; color:var(--muted); padding-right:2em}
.qtable .c.q{color:var(--ink); font-size:1.06rem}
.qtable .c.t{font-style:italic}

.cards{grid-template-columns:repeat(3,1fr)}
.card{background:var(--paper); border:1px solid var(--hair); border-radius:4px;
      padding:clamp(18px,1.9vw,28px); display:flex; flex-direction:column; gap:1em}
.card.warnc{background:var(--flag-bg); border-color:var(--flag)}
.card h3{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace; font-size:.64rem;
         letter-spacing:.18em; text-transform:uppercase; margin:0;
         padding-bottom:.85em; border-bottom:1px solid currentColor; color:var(--brass)}
.card.warnc h3{color:var(--flag)}
.card.plain h3{color:var(--ink)}
.card p{font-size:.9rem; color:var(--muted)}

.tiles{grid-template-columns:repeat(4,1fr)}
.tile{background:var(--paper); border:1px solid var(--hair); border-radius:4px;
      padding:clamp(14px,1.5vw,22px)}
.tile .v{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;
         font-size:clamp(1.5rem,2.6vw,2.2rem); font-variant-numeric:tabular-nums}
.tile .k{font-size:.86rem; color:var(--muted); margin-top:.35em}
.tile .t{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace; font-size:.58rem;
         letter-spacing:.16em; text-transform:uppercase; color:var(--brass);
         margin-top:.7em}

.warn{background:var(--flag-bg); border:1px solid var(--flag); border-radius:3px;
      color:var(--flag); font-size:.84rem; font-style:italic;
      padding:.75em 1.1em; margin-top:clamp(14px,1.8vw,22px)}

.list{display:flex; flex-direction:column; gap:.75em;
      margin-top:clamp(14px,1.8vw,22px)}
.list .row{display:flex; gap:1.1em; align-items:baseline}
.list .n{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace; font-size:.8rem;
         color:var(--brass); flex:0 0 2.1em}
.list .row p{font-size:clamp(.9rem,1.1vw,1.02rem)}

.asks{display:flex; flex-direction:column; margin-top:clamp(16px,2.2vw,28px)}
.ask{display:flex; gap:clamp(16px,2vw,30px); padding:clamp(14px,1.7vw,24px) 0;
     border-bottom:1px solid var(--hair)}
.ask .n{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace;
        font-size:clamp(1.2rem,2vw,1.7rem); color:var(--brass); flex:0 0 2ch}
.ask h3{font-family:'Space Grotesk',-apple-system,'Segoe UI',sans-serif; font-weight:600; margin:0 0 .4em;
        font-size:clamp(1.1rem,1.7vw,1.5rem)}
.ask p{font-size:.92rem; color:var(--muted); max-width:80ch}

.clip{width:100%; max-height:56vh; border-radius:6px; display:block;
      background:var(--slatelt); border:1px solid var(--hair);
      margin-top:clamp(14px,1.8vw,22px)}
.play{font-family:'JetBrains Mono',ui-monospace,Menlo,monospace; color:var(--brass);
      font-size:clamp(.95rem,1.5vw,1.3rem); margin-top:clamp(16px,2vw,26px);
      word-break:break-all}
.meta{font-size:.86rem; color:var(--muted); margin-top:.7em}
.slide.dark .rule{background:#3A342C}
.slide.dark .list .row p{color:#D6CDBE}
.slide.dark .lede, .slide.dark .meta{color:#9A9184}

.foot{font-size:.82rem; color:var(--muted);
      margin-top:clamp(14px,1.8vw,24px)}
.num{position:absolute; left:clamp(28px,7vw,120px); bottom:clamp(18px,3vw,42px);
     font-family:'JetBrains Mono',ui-monospace,Menlo,monospace; font-size:.66rem;
     letter-spacing:.16em; color:var(--muted)}
.slide.dark .num{color:#7C7467}

nav{position:fixed; right:22px; top:50%; transform:translateY(-50%);
    display:flex; flex-direction:column; gap:9px; z-index:5}
nav a{width:9px; height:9px; border-radius:50%; border:1px solid var(--hair);
      display:block; transition:background .18s,border-color .18s}
nav a:hover,nav a:focus-visible{background:var(--brass); border-color:var(--brass)}
nav a.on{background:var(--brass); border-color:var(--brass)}
nav a:focus-visible{outline:2px solid var(--brass); outline-offset:3px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;
  scroll-behavior:auto!important}}
@media (max-width:820px){
  .qtable,.cards,.tiles{grid-template-columns:1fr}
  .qtable .h{display:none}
  .qtable .c{border-bottom:none; padding:.3em 0}
  .qtable .c.q{padding-top:1.1em; border-top:1px solid var(--hair)}
  nav{display:none}
}
"""

JS = """
const slides=[...document.querySelectorAll('.slide')];
const dots=[...document.querySelectorAll('nav a')];
let cur=0;
const io=new IntersectionObserver(es=>es.forEach(x=>{
  if(x.isIntersecting){cur=slides.indexOf(x.target);
    dots.forEach((d,j)=>d.classList.toggle('on',j===cur));}
}),{threshold:.55});
slides.forEach(s=>io.observe(s));
const go=i=>slides[Math.max(0,Math.min(slides.length-1,i))]
  .scrollIntoView({behavior:'smooth'});
addEventListener('keydown',ev=>{
  const k=ev.key;
  if(['ArrowRight','ArrowDown','PageDown',' '].includes(k)){ev.preventDefault();go(cur+1);}
  else if(['ArrowLeft','ArrowUp','PageUp'].includes(k)){ev.preventDefault();go(cur-1);}
  else if(k==='Home'){ev.preventDefault();go(0);}
  else if(k==='End'){ev.preventDefault();go(slides.length-1);}
});
"""


def render():
    n = len(SLIDES)
    out = ['<title>Nobody Watches the Cameras</title>',
           '<link rel="preconnect" href="https://fonts.googleapis.com">',
           '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
           '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap">',
           f'<style>{CSS}</style>',
           '<nav aria-label="Slides">' + "".join(
               f'<a href="#s{i+1}" aria-label="Slide {i+1}"></a>'
               for i in range(n)) + '</nav>']

    for i, s in enumerate(SLIDES, 1):
        k = s["kind"]
        dark = False
        cls = "slide" + (" dark" if dark else "") + (" full" if k == "full" else "")
        out.append(f'<section class="{cls}" id="s{i}">')

        if k != "title":
            out.append(f'<p class="eyebrow">{e(s["eyebrow"])}</p>')

        if k == "title":
            out.append(f'<p class="eyebrow">{e(s["eyebrow"])}</p>')
            out.append(f'<h1>{e(s["title"])}</h1>')
            out.append(f'<p class="lede">{e(s["sub"])}</p>')
            out.append('<div class="rule" style="max-width:220px"></div>')

        elif k == "statement":
            out.append(f'<h1>{e(s["title"])}</h1>')
            out.append('<div class="stats">' + "".join(
                f'<div class="stat"><div class="v">{e(v)}</div>'
                f'<div class="k">{e(kk)}</div></div>' for v, kk in s["stats"])
                + '</div>')
            out.append('<div class="rule"></div>')
            out.append('<div class="body">' + "".join(
                f"<p>{e(b)}</p>" for b in s["body"]) + '</div>')

        elif k == "imagelist":
            out.append(f'<h2>{e(s["title"])}</h2>')
            out.append('<figure><img alt="" src="' + data_uri(s["image"]) + '">'
                       '</figure>')
            out.append('<div class="list">' + "".join(
                f'<div class="row"><span class="n">—</span><p>{e(t)}</p></div>'
                for t in s["eye"]) + '</div>')

        elif k in ("image", "full"):
            out.append(f'<h2>{e(s["title"])}</h2>')
            out.append('<figure><img alt="" src="' + data_uri(s["image"]) + '">')
            if s.get("caption"):
                out.append(f'<figcaption>{e(s["caption"])}</figcaption>')
            out.append('</figure>')

        elif k == "table":
            out.append(f'<h2>{e(s["title"])}</h2>')
            cells = "".join(f'<div class="h">{e(h)}</div>' for h in s["head"])
            for row in s["rows"]:
                cells += (f'<div class="c q">{e(row[0])}</div>'
                          f'<div class="c">{e(row[1])}</div>'
                          f'<div class="c t">{e(row[2])}</div>')
            out.append(f'<div class="grid qtable">{cells}</div>')
            out.append(f'<p class="foot">{e(s["foot"])}</p>')

        elif k == "video":
            out.append(f'<h1>{e(s["title"])}</h1>')
            out.append('<video class="clip" controls playsinline preload="metadata" '
                       'src="' + video_uri() + '"></video>')
            out.append(f'<p class="meta">{e(s["meta"])} · '
                       f'full-resolution master: {e(s["file"])}</p>')
            out.append('<div class="rule"></div>')
            out.append('<p class="eyebrow">What to look for</p>')
            out.append('<div class="list">' + "".join(
                f'<div class="row"><span class="n">{j:02d}</span>'
                f'<p>{e(t)}</p></div>' for j, t in enumerate(s["look"], 1))
                + '</div>')

        elif k == "split":
            out.append(f'<h2>{e(s["title"])}</h2>')
            out.append('<div class="grid" style="grid-template-columns:1.15fr 1fr;'
                       'align-items:start">')
            out.append('<figure style="margin-top:0"><img alt="" src="'
                       + data_uri(s["image"]) + '"></figure>')
            out.append('<div><div class="stats" style="margin-top:0;'
                       'flex-direction:column;gap:1.1em">' + "".join(
                           f'<div class="stat"><div class="v">{e(v)}</div>'
                           f'<div class="k">{e(kk)}</div></div>'
                           for v, kk in s["stats"]) + '</div>')
            out.append('<div class="rule"></div><div class="body" '
                       'style="margin-top:0">' + "".join(
                           f"<p>{e(b)}</p>" for b in s["body"])
                       + '</div></div></div>')

        elif k == "report":
            out.append(f'<h2>{e(s["title"])}</h2>')
            out.append(f'<p class="warn">{e(s["warn"])}</p>')
            out.append('<div class="grid tiles">' + "".join(
                f'<div class="tile"><div class="v">{e(v)}</div>'
                f'<div class="k">{e(kk)}</div><div class="t">{e(t)}</div></div>'
                for v, kk, t in s["tiles"]) + '</div>')
            out.append('<p class="eyebrow" style="margin-top:1.6em">'
                       'What needs a human eye</p>')
            out.append('<div class="list">' + "".join(
                f'<div class="row"><span class="n">—</span><p>{e(t)}</p></div>'
                for t in s["eye"]) + '</div>')
            out.append(f'<p class="foot">{e(s["foot"])}</p>')

        elif k == "three":
            out.append(f'<h2>{e(s["title"])}</h2>')
            out.append('<div class="grid cards">')
            for j, (head, items) in enumerate(s["cols"]):
                c = "card warnc" if j == 1 else ("card" if j == 0 else "card plain")
                out.append(f'<div class="{c}"><h3>{e(head)}</h3>'
                           + "".join(f"<p>{e(it)}</p>" for it in items) + '</div>')
            out.append('</div>')
            out.append('<p class="foot">Every figure in the first column is '
                       'measured and reproducible from the repository. '
                       'Nothing on this slide is a projection.</p>')

        elif k == "ask":
            out.append(f'<h1>{e(s["title"])}</h1>')
            out.append('<div class="asks">' + "".join(
                f'<div class="ask"><div class="n">{e(n_)}</div><div>'
                f'<h3>{e(h_)}</h3><p>{e(b_)}</p></div></div>'
                for n_, h_, b_ in s["asks"]) + '</div>')
            out.append(f'<p class="foot" style="font-size:.95rem;color:var(--ink)">'
                       f'{e(s["foot"])}</p>')

        out.append(f'<div class="num">{i:02d} / {n:02d}</div>')
        out.append('</section>')

    out.append(f'<script>{JS}</script>')
    return "\n".join(out)


if __name__ == "__main__":
    doc = render()
    path = HERE / "index.html"
    path.write_text(doc, encoding="utf-8")

    # checks: the page must be self-contained and fit the 16 MB publish cap,
    # and must carry one section per slide -- a dropped section is silent.
    assert doc.count('<section class="slide') == len(SLIDES), "slide count off"
    assert "http://" not in doc, "non-https reference leaked in"
    for host in ("cdn.", "unpkg", "jsdelivr"):
        assert host not in doc, f"external host {host} would be blocked by CSP"
    mb = path.stat().st_size / 1_000_000
    assert mb < 16, f"{mb:.1f} MB exceeds the publish cap"
    assert VIDEO in doc, "the video cue slide lost its file path"
    print(f"wrote index.html · {len(SLIDES)} slides · {mb:.1f} MB")
