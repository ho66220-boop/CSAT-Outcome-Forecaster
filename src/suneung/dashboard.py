"""단일 파일 HTML 대시보드 생성.

templates/dashboard.html 의 자리표시자에 데이터(JSON)와 제목을 넣어 서버 없이 열리는 파일 하나를 만듭니다.
font_dir 에 Pretendard OTF 파일이 있으면 쓰인 글자만 잘라 내 파일 안에 넣습니다(fonttools 필요).
없으면 시스템 글꼴(맑은 고딕, Apple SD Gothic Neo)로 표시됩니다.
"""
import base64
import html
import io
import json
from pathlib import Path

FONT_WEIGHTS = [(400, "Regular"), (500, "Medium"), (600, "SemiBold"), (700, "Bold")]


def _font_faces(font_dir, text):
    from fontTools import subset
    from fontTools.ttLib import TTFont

    faces = []
    for w, name in FONT_WEIGHTS:
        f = TTFont(Path(font_dir) / f"Pretendard-{name}.otf")
        opt = subset.Options()
        opt.flavor = "woff2"
        opt.layout_features = ["*"]
        sub = subset.Subsetter(opt)
        sub.populate(text=text)
        sub.subset(f)
        b = io.BytesIO()
        f.flavor = "woff2"
        f.save(b)
        src = base64.b64encode(b.getvalue()).decode()
        faces.append(f'@font-face{{font-family:Pretendard;font-weight:{w};font-display:swap;'
                     f'src:url(data:font/woff2;base64,{src}) format("woff2")}}')
    return "\n".join(faces)


def render(payload, template, out, title, eyebrow="", subtitle="", font_dir=None):
    tpl = Path(template).read_text(encoding="utf-8")
    djs = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    for k, v in (("__TITLE__", title), ("__EYEBROW__", eyebrow), ("__SUBTITLE__", subtitle)):
        tpl = tpl.replace(k, html.escape(v))
    faces = ""
    if font_dir and Path(font_dir).exists():
        text = tpl + djs + "".join(chr(c) for c in range(32, 127)) + "·–×▲▼“”~%±"
        faces = _font_faces(font_dir, text)
    page = tpl.replace("__FONTFACE__", faces).replace("__DATA__", djs)
    Path(out).write_text(page, encoding="utf-8")
    return Path(out)
