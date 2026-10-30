"""Tiny block-based document model rendered to both Markdown and standalone HTML."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Doc:
    title: str
    blocks: list[tuple] = field(default_factory=list)

    def h(self, level: int, text: str) -> None:
        self.blocks.append(("h", level, text))

    def p(self, text: str) -> None:
        self.blocks.append(("p", text))

    def note(self, text: str) -> None:
        self.blocks.append(("note", text))

    def ul(self, items: list[str]) -> None:
        if items:
            self.blocks.append(("ul", items))

    def table(self, header: list[str], rows: list[list], caption: str | None = None) -> None:
        self.blocks.append(("table", header, rows, caption))

    def figure(self, fig: dict) -> None:
        self.blocks.append(("figure", fig))

    # -- renderers ----------------------------------------------------------------
    def markdown(self) -> str:
        out = [f"# {self.title}", ""]
        for b in self.blocks:
            kind = b[0]
            if kind == "h":
                out += [f"{'#' * (b[1] + 1)} {b[2]}", ""]
            elif kind == "p":
                out += [b[1], ""]
            elif kind == "note":
                out += [f"> {b[1]}", ""]
            elif kind == "ul":
                out += [f"- {i}" for i in b[1]] + [""]
            elif kind == "table":
                _, header, rows, caption = b
                if caption:
                    out += [f"*{caption}*", ""]
                out.append("| " + " | ".join(header) + " |")
                out.append("|" + "|".join("---" for _ in header) + "|")
                out += ["| " + " | ".join(_fmt(c).replace("|", "\\|") for c in row) + " |" for row in rows]
                out.append("")
            elif kind == "figure":
                f = b[1]
                out += [f"![Figure {f['number']}]({f['png']})", "", f"*Figure {f['number']}: {f['caption']}* (SVG: `{f['svg']}`)", ""]
        return "\n".join(out)

    def html(self, figure_root: Path) -> str:
        body = []
        for b in self.blocks:
            kind = b[0]
            if kind == "h":
                body.append(f"<h{b[1] + 1}>{html.escape(b[2])}</h{b[1] + 1}>")
            elif kind == "p":
                body.append(f"<p>{html.escape(b[1])}</p>")
            elif kind == "note":
                body.append(f"<p class='note'>{html.escape(b[1])}</p>")
            elif kind == "ul":
                body.append("<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in b[1]) + "</ul>")
            elif kind == "table":
                _, header, rows, caption = b
                cap = f"<caption>{html.escape(caption)}</caption>" if caption else ""
                head = "".join(f"<th>{html.escape(h)}</th>" for h in header)
                trs = "".join("<tr>" + "".join(f"<td>{html.escape(_fmt(c))}</td>" for c in row) + "</tr>" for row in rows)
                body.append(f"<div class='tw'><table>{cap}<thead><tr>{head}</tr></thead><tbody>{trs}</tbody></table></div>")
            elif kind == "figure":
                f = b[1]
                svg_path = figure_root / Path(f["svg"]).name
                svg = svg_path.read_text() if svg_path.exists() else ""
                svg = svg[svg.find("<svg"):] if "<svg" in svg else ""
                body.append(f"<figure>{svg}<figcaption>Figure {f['number']}: {html.escape(f['caption'])}</figcaption></figure>")
        return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(self.title)}</title><style>
:root{{--fg:#14202b;--muted:#5b6b78;--line:#dbe2e8;--accent:#0f766e;--bg:#ffffff}}
body{{font:15px/1.55 -apple-system,Segoe UI,Inter,Helvetica,Arial,sans-serif;color:var(--fg);background:var(--bg);max-width:980px;margin:40px auto;padding:0 20px}}
h1{{font-size:28px;margin-bottom:4px}} h2{{margin-top:36px;border-bottom:2px solid var(--accent);padding-bottom:4px}} h3{{margin-top:22px}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0 16px;min-width:50%}} th,td{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:left;font-variant-numeric:tabular-nums}}
th{{background:#f3f6f8}} caption{{caption-side:top;text-align:left;color:var(--muted);font-style:italic;padding-bottom:4px}}
.tw{{overflow-x:auto}} figure{{margin:16px 0}} figure svg{{max-width:100%;height:auto}} figcaption{{color:var(--muted);font-size:13px;font-style:italic}}
.note{{border-left:3px solid var(--accent);padding:6px 12px;background:#f0f7f6;color:#264}}
</style></head><body><h1>{html.escape(self.title)}</h1>{''.join(body)}</body></html>"""


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v != v:
            return "—"
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        if abs(v) >= 1:
            return f"{v:.4g}" if abs(v) < 100 else f"{v:.2f}"
        return f"{v:.4g}"
    return str(v)
