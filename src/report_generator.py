from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image as PILImage, ImageDraw
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

from src.deterministic_floorplan import _load_font

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
PAGE_W, PAGE_H = 1240, 1754
M = 72
INK = (31, 51, 44)
SOFT = (82, 101, 92)
TEAL = (47, 111, 98)
TEAL_DARK = (33, 83, 72)
SAGE = (221, 233, 225)
PAPER = (250, 251, 248)
LINE = (203, 215, 207)
CORAL = (193, 84, 60)
AMBER = (217, 142, 59)
WHITE = (255, 255, 255)


def _font(size: int):
    return _load_font(size)


def _wrap(draw, text: str, font, max_width: int) -> list[str]:
    text = str(text or "")
    lines = []
    for raw in text.split("\n"):
        if not raw:
            lines.append("")
            continue
        cur = ""
        for ch in raw:
            cand = cur + ch
            if draw.textlength(cand, font=font) <= max_width or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
    return lines or [""]


def _text(draw, xy, text, size=26, fill=INK, max_width=None, line_gap=8, anchor=None):
    font = _font(size)
    x, y = xy
    if max_width is None:
        draw.text((x, y), str(text), font=font, fill=fill, anchor=anchor)
        return y + int(size * 1.35)
    for line in _wrap(draw, text, font, max_width):
        draw.text((x, y), line, font=font, fill=fill)
        y += size + line_gap
    return y


def _card(draw, box, fill=WHITE, outline=LINE, radius=18, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _section_title(draw, y, num: str, title: str):
    _text(draw, (M, y), f"{num}. {title}", 34, TEAL_DARK)
    draw.line((M, y + 52, PAGE_W - M, y + 52), fill=TEAL, width=3)
    return y + 78


def _footer(draw, page_no: int, total: int = 4):
    draw.rectangle((0, PAGE_H - 62, PAGE_W, PAGE_H), fill=TEAL_DARK)
    _text(draw, (M, PAGE_H - 49), "안넘어집", 22, WHITE)
    _text(draw, (PAGE_W - M - 65, PAGE_H - 49), f"{page_no} / {total}", 19, WHITE)


def _load_static(rel_path: str | None):
    if not rel_path:
        return None
    p = STATIC_DIR / rel_path
    try:
        im = PILImage.open(p).convert("RGB")
        return im
    except Exception:
        return None


def _fit_image(canvas_img, source, box):
    if source is None:
        return
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    src = source.copy()
    src.thumbnail((bw, bh))
    x = int(x0 + (bw - src.width) / 2)
    y = int(y0 + (bh - src.height) / 2)
    canvas_img.paste(src, (x, y))


def _positive_factors(context: dict, limit=5):
    factors = [f for f in (context.get("main_factors") or []) if isinstance(f, dict)]
    positives = [f for f in factors if float(f.get("shap_value", 0)) > 0]
    positives.sort(key=lambda x: float(x.get("shap_value", 0)), reverse=True)
    return positives[:limit]


def _page1(context: dict) -> PILImage.Image:
    img = PILImage.new("RGB", (PAGE_W, PAGE_H), PAPER)
    d = ImageDraw.Draw(img)
    _text(d, (M, 70), "안넘어집", 26, TEAL_DARK)
    _text(d, (M, 135), "안넘어집 맞춤 공간 안전 리포트", 48, TEAL_DARK)
    _text(d, (M, 205), "낙상 위험 예측과 실제 공간 개선 제안", 25, SOFT)

    generated = context.get("generated_at") or datetime.now()
    room = context.get("room_label", "-")
    _card(d, (M, 270, PAGE_W-M, 385), fill=(245,248,244))
    cols = [(M+28, "생성일", generated.strftime("%Y.%m.%d")), (430, "분석 공간", room), (750, "분석 방식", "설문 + SHAP + 공간배치 + RAG/AI")]
    for x, a, b in cols:
        _text(d, (x, 292), a, 18, TEAL_DARK)
        _text(d, (x, 329), b, 20, INK, max_width=360)

    _card(d, (M, 430, PAGE_W-M, 735), fill=WHITE)
    _text(d, (M+34, 458), "낙상 예측 결과", 28, TEAL_DARK)
    score = float(context.get("risk_score") or 0)
    level = str(context.get("risk_level") or "모델 판정 결과 없음")
    threshold = float(context.get("threshold_percent") or 0)
    _text(d, (M+34, 515), f"{score:.1f}%", 68, CORAL if score >= threshold else TEAL)
    _text(d, (M+270, 540), level, 24, INK, max_width=780)
    bar_x0, bar_y, bar_x1 = M+34, 625, PAGE_W-M-34
    d.rounded_rectangle((bar_x0,bar_y,bar_x1,bar_y+28), 14, fill=(230,235,231))
    frac = max(0, min(1, score/100.0))
    d.rounded_rectangle((bar_x0,bar_y,bar_x0+(bar_x1-bar_x0)*frac,bar_y+28), 14, fill=CORAL if score >= threshold else TEAL)
    if threshold > 0:
        tx = bar_x0 + (bar_x1-bar_x0)*min(1,threshold/100.0)
        d.line((tx, bar_y-8, tx, bar_y+40), fill=AMBER, width=4)
        _text(d, (tx+8, bar_y+42), f"모델 기준 {threshold:.1f}%", 16, SOFT)

    factors = _positive_factors(context, 3)
    sections = context.get("hazard_sections") or []
    confirmed_count = len(sections)
    _card(d, (M, 780, 395, 1048), fill=WHITE)
    _text(d,(M+24,808),"주요 개인 위험요인",22,TEAL_DARK)
    if factors:
        yy=855
        for i,f in enumerate(factors,1):
            _text(d,(M+26,yy),f"{i}. {f.get('label','')}",19,INK,max_width=270)
            yy += 52
    else:
        _text(d,(M+26,855),"양의 SHAP 위험기여 요인이 없습니다.",18,SOFT,max_width=270)

    _card(d, (420, 780, 790, 1048), fill=WHITE)
    _text(d,(445,808),"분석한 공간",22,TEAL_DARK)
    _text(d,(445,870),room,38,TEAL)
    _text(d,(445,930),"사용자가 배치한 가구와\n확인된 공간 위험을 분석했습니다.",18,SOFT,max_width=310)

    _card(d, (815, 780, PAGE_W-M, 1048), fill=WHITE)
    _text(d,(840,808),"개선 우선순위",22,TEAL_DARK)
    _text(d,(840,870),f"{confirmed_count}개",38,CORAL)
    _text(d,(840,930),"현재 분석에서 도출된\n위험요인 단위 개선 항목",18,SOFT,max_width=270)

    _card(d,(M,1095,PAGE_W-M,1325),fill=(246,249,246))
    _text(d,(M+30,1125),"핵심 개선 방향",22,TEAL_DARK)
    ai_comment = str(context.get("ai_comment") or "").strip()
    if ai_comment:
        _text(d,(M+30,1170),ai_comment,18,INK,max_width=PAGE_W-2*M-60)
    else:
        _text(d,(M+30,1170),"분석 결과와 선택한 개선안을 다음 페이지에서 확인할 수 있습니다.",18,INK,max_width=PAGE_W-2*M-60)
    _text(d,(M+30,1270),"※ 이 문서는 현재 서비스에서 실제 계산·확인된 결과만 사용합니다.",16,SOFT,max_width=PAGE_W-2*M-60)
    _footer(d,1)
    return img


def _page2(context: dict) -> PILImage.Image:
    img=PILImage.new("RGB",(PAGE_W,PAGE_H),PAPER); d=ImageDraw.Draw(img)
    y=_section_title(d,70,"1","개인 위험요인 분석")
    factors=_positive_factors(context,5)
    _card(d,(M,y,PAGE_W-M,y+410),fill=WHITE)
    _text(d,(M+28,y+25),"SHAP 양의 기여도 상위 요인",22,TEAL_DARK)
    yy=y+80
    maxv=max([float(f.get('shap_value',0)) for f in factors] or [1])
    for i,f in enumerate(factors,1):
        label=str(f.get('label',''))
        val=float(f.get('shap_value',0))
        _text(d,(M+30,yy),f"{i}. {label}",18,INK,max_width=330)
        bx=M+410; bw=560
        d.rounded_rectangle((bx,yy+4,bx+bw,yy+25),10,fill=(236,239,236))
        d.rounded_rectangle((bx,yy+4,bx+bw*(val/maxv),yy+25),10,fill=CORAL)
        _text(d,(bx+bw+20,yy-2),f"{val:+.3f}",17,SOFT)
        yy+=62
    if not factors:
        _text(d,(M+30,yy),"현재 결과에서 양의 SHAP 위험기여 요인이 없습니다.",19,SOFT)

    y2=y+455
    y2=_section_title(d,y2,"2","분석 공간의 확인된 위험 포인트")
    sections=context.get("hazard_sections") or []
    left=M; right=PAGE_W-M
    box_h=145
    for idx,sec in enumerate(sections[:4]):
        top=y2+idx*(box_h+14)
        _card(d,(left,top,right,top+box_h),fill=WHITE)
        _text(d,(left+24,top+20),f"{sec.get('priority','-')}순위",20,CORAL)
        _text(d,(left+130,top+18),sec.get('floorplan_problem',''),21,INK,max_width=780)
        factor=sec.get('shap_factor','')
        if factor:
            _text(d,(left+130,top+62),f"연결된 개인 위험요인: {factor}",16,SOFT,max_width=760)
        source="AI 독자 제안" if str(sec.get('recommendation_source','')).startswith('AI_ONLY') else "RAG 근거 기반"
        _text(d,(left+130,top+99),f"해결안 출처: {source}",16,TEAL)
    if not sections:
        _card(d,(left,y2,right,y2+170),fill=WHITE)
        _text(d,(left+24,y2+40),"현재 분석에서 확인된 공간 개선 우선순위가 없습니다.",20,SOFT)

    note=str(context.get("room_note") or "").strip()
    if note:
        top=min(PAGE_H-260, y2+min(len(sections),4)*(box_h+14)+18)
        _card(d,(M,top,PAGE_W-M,top+135),fill=(246,249,246))
        _text(d,(M+24,top+18),"사용자가 입력한 공간 메모",19,TEAL_DARK)
        _text(d,(M+24,top+58),note,17,INK,max_width=PAGE_W-2*M-48)
    _footer(d,2)
    return img


def _page3(context: dict) -> PILImage.Image:
    img=PILImage.new("RGB",(PAGE_W,PAGE_H),PAPER); d=ImageDraw.Draw(img)
    y=_section_title(d,70,"3","우선순위 개선 가이드 (검토된 대안 전체)")
    _text(d,(M,y-8),"사용자의 개인 위험요인과 공간의 확인된 위험을 연결해 도출된 해결책입니다. 검토된 대안을 모두 표시하고, 선택하신 대안은 강조 표시했습니다.",15,SOFT,max_width=PAGE_W-2*M)
    sections=context.get("hazard_sections") or []
    yy=y+34
    ALT_ROW_H=56
    for sec in sections:
        alts=sec.get("alternatives") or []
        shown_alts=alts[:3]
        n=max(1,len(shown_alts))
        box_h=78+n*ALT_ROW_H+14
        if yy+box_h > PAGE_H-100:
            remaining=len(sections)-sections.index(sec)
            _text(d,(M,yy+10),f"※ 지면 제약으로 이후 {remaining}개 위험요인은 생략했습니다. 전체 내용은 서비스 화면에서 확인해 주세요.",15,SOFT,max_width=PAGE_W-2*M)
            break
        _card(d,(M,yy,PAGE_W-M,yy+box_h),fill=WHITE)
        d.ellipse((M+22,yy+20,M+66,yy+64),fill=TEAL_DARK)
        _text(d,(M+44,yy+31),str(sec.get('priority','-')),19,WHITE,anchor='mm')
        source="AI 독자 제안 · RAG 미매핑" if str(sec.get('recommendation_source','')).startswith('AI_ONLY') else "RAG 근거 기반"
        _text(d,(M+84,yy+16),sec.get('floorplan_problem',''),19,INK,max_width=760)
        _text(d,(PAGE_W-M-24,yy+20),source,14,TEAL,anchor='ra')
        factor=sec.get('shap_factor','')
        if factor:
            _text(d,(M+84,yy+50),f"개인 위험요인: {factor}",13,SOFT,max_width=880)

        ay=yy+82
        for i,alt in enumerate(shown_alts):
            selected=bool(alt.get('selected'))
            if selected:
                d.rounded_rectangle((M+18,ay-6,PAGE_W-M-18,ay+ALT_ROW_H-16),10,fill=SAGE)
            prefix="[선택함] " if selected else f"{i+1}. "
            _text(d,(M+36,ay),f"{prefix}{alt.get('text','')}",16,TEAL_DARK if selected else INK,max_width=980)
            meta=[]
            if alt.get('estimated_minutes') is not None: meta.append(f"약 {alt['estimated_minutes']}분")
            if alt.get('difficulty_label'): meta.append(f"난이도 {alt['difficulty_label']}")
            meta.append("도움 필요" if alt.get('requires_helper') else "혼자 가능")
            if alt.get('price_note'): meta.append(str(alt['price_note']))
            if alt.get('height_note'): meta.append(f"※ {alt['height_note']}")
            _text(d,(M+36,ay+24)," · ".join(meta),13,SOFT,max_width=980)
            ay+=ALT_ROW_H
        if len(alts)>3:
            _text(d,(M+36,ay-6),f"그 외 대안 {len(alts)-3}개는 서비스 화면에서 확인할 수 있습니다.",12,SOFT)
        yy+=box_h+16

    _text(d,(M,PAGE_H-90),"※ 본 리포트는 서비스의 예측·SHAP·공간 입력·RAG/AI 추천 결과를 정리한 참고용 문서입니다.",15,SOFT,max_width=PAGE_W-2*M)
    _footer(d,3)
    return img


def _page4(context: dict) -> PILImage.Image:
    img=PILImage.new("RGB",(PAGE_W,PAGE_H),PAPER); d=ImageDraw.Draw(img)
    y=_section_title(d,70,"4","개선 위치와 실사형 참고 이미지")
    _text(d,(M,y-8),"사용자가 만든 실제 평면도를 기준으로 개선 번호를 표시하고, 같은 번호를 실사형 참고 이미지와 대응시킵니다.",15,SOFT,max_width=PAGE_W-2*M)

    after2d=_load_static(context.get("after_2d_path"))
    box_top=y+38; box_bottom=box_top+430
    _card(d,(M,box_top,PAGE_W-M,box_bottom),fill=WHITE)
    _text(d,(M+18,box_top+14),"현재 배치 기준 개선 위치 평면도",18,TEAL_DARK)
    _fit_image(img,after2d,(M+20,box_top+52,PAGE_W-M-20,box_bottom-16))

    py=box_bottom+28
    before_photo=_load_static(context.get("photoreal_before_image_path"))
    after_photo=_load_static(context.get("photoreal_after_image_path") or context.get("photoreal_image_path"))
    if before_photo and after_photo:
        _text(d,(M,py),"현재 배치 도면 · AI 개선 후 참고 이미지",21,TEAL_DARK)
        _text(d,(M,py+34),"왼쪽은 실제 입력 좌표를 코드로 렌더링한 현재 배치 도면이고, 오른쪽은 선택한 개선안을 반영해 AI가 생성한 개선 후 참고 이미지입니다.",13,SOFT,max_width=PAGE_W-2*M)
        pbox_top=py+66; pbox_bottom=min(PAGE_H-300,pbox_top+330)
        gap=16; mid=(M + PAGE_W-M)//2
        _card(d,(M,pbox_top,mid-gap//2,pbox_bottom),fill=WHITE)
        _card(d,(mid+gap//2,pbox_top,PAGE_W-M,pbox_bottom),fill=WHITE)
        _text(d,(M+16,pbox_top+12),"BEFORE · 현재 배치 도면",15,CORAL)
        _text(d,(mid+gap//2+16,pbox_top+12),"AFTER · AI 개선 적용",15,TEAL_DARK)
        _fit_image(img,before_photo,(M+14,pbox_top+44,mid-gap//2-14,pbox_bottom-14))
        _fit_image(img,after_photo,(mid+gap//2+14,pbox_top+44,PAGE_W-M-14,pbox_bottom-14))
        py=pbox_bottom+24
    elif after_photo:
        _text(d,(M,py),"AI 생성 개선 후 참고 이미지",21,TEAL_DARK)
        pbox_top=py+44; pbox_bottom=min(PAGE_H-300,pbox_top+330)
        _card(d,(M,pbox_top,PAGE_W-M,pbox_bottom),fill=WHITE)
        _fit_image(img,after_photo,(M+18,pbox_top+18,PAGE_W-M-18,pbox_bottom-18))
        py=pbox_bottom+24

    supplementary=context.get("supplementary_space_advice") or {}
    if supplementary.get("requests") and py < PAGE_H-320:
        _text(d,(M,py),"사용자 추가 요청사항",20,TEAL_DARK)
        sy=py+38
        for item in supplementary.get("requests",[])[:3]:
            _text(d,(M+18,sy),f"• {item.get('text','')}",14,INK,max_width=PAGE_W-2*M-36)
            sy+=42
        advice=str(supplementary.get("advice") or "").strip()
        if advice:
            _card(d,(M,sy,PAGE_W-M,min(PAGE_H-120,sy+130)),fill=WHITE)
            _text(d,(M+18,sy+18),advice,13,INK,max_width=PAGE_W-2*M-36)

    _text(d,(M,PAGE_H-90),"※ BEFORE 도면이 좌표 기준의 기준 자료이며, AFTER AI 이미지는 개선 방향 이해를 돕는 보조 시각화입니다.",14,SOFT,max_width=PAGE_W-2*M)
    _footer(d,4)
    return img


def build_report_pdf(context: dict[str, Any]) -> bytes:
    pages=[_page1(context),_page2(context),_page3(context),_page4(context)]
    out=io.BytesIO(); c=canvas.Canvas(out,pagesize=A4)
    pw,ph=A4
    for page in pages:
        bio=io.BytesIO(); page.save(bio,format='PNG'); bio.seek(0)
        c.drawImage(ImageReader(bio),0,0,width=pw,height=ph,preserveAspectRatio=False,mask='auto')
        c.showPage()
    c.save(); return out.getvalue()
