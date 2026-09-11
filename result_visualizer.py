from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import pandas as pd


JsonLike = str | Path | Mapping[str, Any]
TableLike = str | Path | pd.DataFrame | Sequence[Mapping[str, Any]]

HIDDEN_SHAP_FEATURES = {
    "RES_MAR_미혼",
    "RES_MAR_별거",
    "RES_MAR_사별",
    "RES_MAR_유배우",
    "RES_MAR_이혼",
    "Q1_기타",
    "Q1_단독주택",
    "Q1_아파트",
    "Q1_연립다세대",
}

FEATURE_LABEL_MAP = {
    # 신체기능
    "근력상태_의자나 침대에 앉았다가 일어나기 5회 반복": "하지 근력",
    "동작수행 어려움_운동장 한 바퀴(400m)정도 뛰기": "400m 뛰기 어려움",
    "동작수행 어려움_운동장 한 바퀴(400m)정도 걷기": "400m 걷기 어려움",
    "동작수행 어려움_쉬지않고 10계단 오르기": "계단 오르기 어려움",
    "동작수행 어려움_몸 구부리거나 쭈그려 앉거나 무릎 꿇기": "쪼그려 앉기 어려움",
    "동작수행 어려움_머리보다 높은 곳에 있는 것 손 뻗쳐 닿기": "높은 곳 손 뻗기 어려움",
    "동작수행 어려움_쌀 1말(8kg) 정도 물건 들어 올리거나 옮기기": "무거운 물건 들기 어려움",

    # 건강 및 기능
    "Nutritional": "영양 상태",
    "ADL": "일상생활 수행능력",
    "IADL": "도구적 일상생활 수행능력",
    "평소 운동 여부": "운동 여부",
    "GDS": "우울 점수",
    "인지기능_총점": "인지기능",

    # 환경 및 만족도
    "노인이 생활하는데 편리함 정도": "생활 편의성",
    "현재 살고있는 주택에 대한 만족도": "주거 만족도",
    "지역사회 환경 만족도_지역사회 환경 전반": "지역사회 만족도",
    "만족도_경제상태": "경제상태 만족도",
    "만족도_건강상태": "건강상태 만족도",

    # 인구학적 특성
    "노인 조사 대상자 만연령": "나이",
    "노인 조사 대상자 성별": "성별",
    "노인 조사 대상자 교육연수": "교육연수",
    "노인가구형태": "가구 형태",

    # 생활습관
    "현재 경제활동 여부": "경제활동 여부",
    "평소의 건강상태": "주관적 건강상태",
    "지난 1년 간 음주 빈도": "음주 빈도",

    # 감각기능
    "일상생활의 불편함_시력": "시력 불편",
    "일상생활의 불편함_청력": "청력 불편",

    # 약물
    "현재 3개월 이상 복용하고 있는 의사처방약(종류)": "복용 약물 수",

    # 만성질환
    "의사진단 만성질환 총 수": "만성질환 수",
    "의사진단 만성질환 유무_뇌졸중(중풍, 뇌경색)": "뇌졸중",
    "의사진단 만성질환 유무_협심증, 심근경색증": "협심증·심근경색",
    "의사진단 만성질환 유무_골관절염(퇴행성관절염), 류머티즘 관절염": "관절염",
    "의사진단 만성질환 유무_골다공증": "골다공증",
    "의사진단 만성질환 유무_요통, 좌골신경통": "요통·좌골신경통",
    "의사진단 만성질환 유무_골절, 탈골 및 사고 후유증": "골절 후유증",
    "의사진단 만성질환 유무_만성기관지염, 폐기종": "만성기관지염",
    "의사진단 만성질환 유무_파킨슨병": "파킨슨병",
    "의사진단 만성질환 유무_불면증": "불면증",
    "의사진단 만성질환 유무_요실금": "요실금",
    "의사진단 만성질환 유무_빈혈": "빈혈",

    # 치료
    "치료 여부_우울증": "우울증 치료",
    "치료 여부_불면증": "불면증 치료",
    "치료 여부_전립선 비대증": "전립선비대증 치료",

    # 배우자 상태
    "RES_MAR_미혼": "미혼",
    "RES_MAR_별거": "별거",
    "RES_MAR_사별": "사별",
    "RES_MAR_유배우": "배우자 있음",
    "RES_MAR_이혼": "이혼",

    # 주거형태
    "Q1_기타": "기타 주택",
    "Q1_단독주택": "단독주택",
    "Q1_아파트": "아파트",
    "Q1_연립다세대": "연립·다세대",
}

def _pretty_feature_name(name: str) -> str:
    for original, pretty in FEATURE_LABEL_MAP.items():
        if original in name:
            name = name.replace(original, pretty)
    return name

def _set_korean_font() -> None:
    """설치된 한글 폰트가 있으면 자동으로 적용합니다."""
    candidates = (
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "NanumGothic",
        "Malgun Gothic",
        "AppleGothic",
        "Arial Unicode MS",
    )
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return

    # Linux 환경에서는 Noto CJK가 .ttc(폰트 컬렉션)로만 설치되어 있고
    # matplotlib의 기본 폰트 캐시에 한글 패밀리명으로 잡히지 않는 경우가 많습니다.
    # 이 경우 파일 경로를 직접 찾아 폰트 매니저에 등록합니다.
    font_file_candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansMonoCJK-Regular.ttc",
    )
    for path in font_file_candidates:
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                family_name = font_manager.FontProperties(fname=path).get_name()
                plt.rcParams["font.family"] = family_name
                plt.rcParams["axes.unicode_minus"] = False
                return
            except Exception:
                continue

    plt.rcParams["axes.unicode_minus"] = False


def _load_json(value: JsonLike) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_table(value: TableLike) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Path)):
        return pd.DataFrame(list(value))

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    if path.suffix.lower() == ".json":
        data = _load_json(path)
        if isinstance(data.get("ranked_actions"), list):
            return pd.DataFrame(data["ranked_actions"])
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame([data])

    return pd.read_csv(path, encoding="utf-8-sig")


def _shorten(text: object, limit: int) -> str:
    value = str(text)
    if len(value) <= limit:
        return value
    return value[: max(1, limit - 1)] + "…"


def _wrap(text: object, width: int) -> str:
    return textwrap.fill(str(text), width=width, break_long_words=False)


def _ensure_output(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def plot_local_shap(
    shap_result: JsonLike,
    output_path: str | Path,
    *,
    top_n: int = 15,
    positive_only: bool = False,
    label_max_length: int = 40,
    dpi: int = 160,
) -> Path:
    """
    01_shap_result.json의 Local SHAP main effect를 시각화합니다.

    positive_only=False:
        절댓값 기준 상위 N개를 표시하며, 음수/양수 방향을 모두 보여줍니다.
    positive_only=True:
        낙상위험을 높이는 양의 SHAP만 표시합니다.
    """
    if top_n < 1:
        raise ValueError("top_n은 1 이상이어야 합니다.")

    data = _load_json(shap_result)
    rows = data.get("main_effects")
    if not isinstance(rows, list):
        raise ValueError("main_effects 배열이 없습니다.")

    parsed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        
        feature = str(row.get("feature", ""))
        
         # 배우자 상태 변수는 그래프에서 제외
        if feature in HIDDEN_SHAP_FEATURES:
            continue
        
        value = float(row.get("shap_value", 0.0))
        if positive_only and value <= 0:
            continue
        parsed.append(
            {
                "feature": feature,
                "feature_value": row.get("feature_value"),
                "shap_value": value,
            }
        )

    if not parsed:
        raise ValueError("표시할 SHAP 값이 없습니다.")

    parsed.sort(
        key=lambda item: item["shap_value"],
        reverse=True,
    )

    selected = parsed[:top_n]

    labels = [
        _shorten(
            _pretty_feature_name(item["feature"]),
            label_max_length,
        )
        for item in selected
]
    values = [item["shap_value"] for item in selected]

    _set_korean_font()
    output = _ensure_output(output_path)

    fig_height = max(5.5, 0.52 * len(selected) + 2.4)
    fig = plt.figure(figsize=(8, fig_height))
    ax = fig.add_subplot(111)

    positions = list(range(len(selected)))
    colors = [
        "#E76F51" if value > 0 else "#4F7CAC"
        for value in values
    ]

    bars = ax.barh(
        positions,
        values,
        color=colors,
        height=0.55,
    )
    
    ax.invert_yaxis()
    ax.set_yticks(positions)
    ax.set_yticklabels(
        labels,
        fontsize=13,
        fontweight="bold",
    )
    
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("Local SHAP 값")
    ax.set_ylabel("")
    ax.set_title("")

    subtitle: list[str] = []
    if data.get("risk_label") is not None:
        subtitle.append(f"판정: {data['risk_label']}")
    if isinstance(data.get("calibrated_probability"), (int, float)):
        subtitle.append(f"보정확률: {data['calibrated_probability']:.2%}")
    if isinstance(data.get("selected_threshold"), (int, float)):
        subtitle.append(f"기준값: {data['selected_threshold']:.2%}")
    if subtitle:
        ax.text(
            0.0,
            1.015,
            " | ".join(subtitle),
            transform=ax.transAxes,
            va="bottom",
            fontsize=10,
        )

    max_abs = max(abs(value) for value in values) or 1.0
    offset = max_abs * 0.02
    for bar, value in zip(bars, values):
        x = value + offset if value >= 0 else value - offset
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.4f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(-max_abs * 1.28, max_abs * 1.28)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.tight_layout()

    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)
    return output


def plot_shap_interactions(
    shap_result: JsonLike,
    output_path: str | Path,
    *,
    top_n: int = 15,
    positive_only: bool = False,
    label_max_length: int = 72,
    dpi: int = 160,
) -> Path:
    """
    01_shap_result.json의 Local SHAP interaction을 시각화합니다.

    주의:
        이 그래프는 상위 interaction 자체를 보여줍니다.
        실제 공간 위험 우선순위에 반영되는 것은 승인된 interaction 매핑뿐입니다.
    """
    if top_n < 1:
        raise ValueError("top_n은 1 이상이어야 합니다.")

    data = _load_json(shap_result)
    rows = data.get("interactions")
    if not isinstance(rows, list):
        raise ValueError("interactions 배열이 없습니다.")

    parsed: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue

        value = float(row.get("interaction_value", 0.0))
        if positive_only and value <= 0:
            continue

        raw_feature_1 = str(row.get("feature_1", ""))
        raw_feature_2 = str(row.get("feature_2", ""))

        # 둘 중 하나라도 배우자 상태 변수면 그래프에서 제외
        if (
            raw_feature_1 in HIDDEN_SHAP_FEATURES
            or raw_feature_2 in HIDDEN_SHAP_FEATURES
        ):
            continue

        feature_1 = _pretty_feature_name(raw_feature_1)
        feature_2 = _pretty_feature_name(raw_feature_2)

        pair = f"{feature_1} × {feature_2}"

        parsed.append(
            {
                "pair": pair,
                "interaction_value": value,
            }
        )

    if not parsed:
        raise ValueError("표시할 SHAP interaction 값이 없습니다.")

    parsed.sort(
        key=lambda item: item["interaction_value"],
        reverse=True,
    )

    selected = parsed[:top_n]

    labels = [_shorten(item["pair"], label_max_length) for item in selected]
    values = [item["interaction_value"] for item in selected]

    _set_korean_font()
    output = _ensure_output(output_path)

    fig_height = max(6.0, 0.62 * len(selected) + 2.6)
    fig = plt.figure(figsize=(8, fig_height))
    ax = fig.add_subplot(111)

    positions = list(range(len(selected)))
    colors = [
        "#E76F51" if value > 0 else "#4F7CAC"
        for value in values
    ]

    bars = ax.barh(
        positions,
        values,
        color=colors,
        height=0.55,
    )
    
    ax.invert_yaxis()
    ax.set_yticks(positions)
    ax.set_yticklabels(
        labels,
        fontsize=13,
        fontweight="bold",
    )
    ax.axvline(0.0, linewidth=1)
    ax.set_xlabel("SHAP interaction 값")
    ax.set_ylabel("")
    ax.set_title("")

    # status = data.get("interaction_priority_status")
    # if status:
    #     ax.text(
    #         0.0,
    #         1.015,
    #         f"우선순위 적용 상태: {status}",
    #         transform=ax.transAxes,
    #         va="bottom",
    #         fontsize=10,
    #     )

    max_abs = max(abs(value) for value in values) or 1.0
    offset = max_abs * 0.02
    for bar, value in zip(bars, values):
        x = value + offset if value >= 0 else value - offset
        ax.text(
            x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.4f}",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(-max_abs * 1.28, max_abs * 1.28)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.tight_layout()

    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    fig.savefig(
        output,
        dpi=dpi,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)
    return output


def _extract_mapping_rows(final_plan: JsonLike | TableLike) -> list[dict[str, Any]]:
    """
    05_final_ranked_action_plan.json 또는 05_selected_solutions.csv를
    공통 매핑 행 구조로 변환합니다.
    """
    if isinstance(final_plan, Mapping):
        data = dict(final_plan)
        actions = data.get("ranked_actions", [])
    elif isinstance(final_plan, (str, Path)) and Path(final_plan).suffix.lower() == ".json":
        data = _load_json(final_plan)
        actions = data.get("ranked_actions", [])
    else:
        table = _load_table(final_plan)
        actions = table.to_dict(orient="records")

    rows: list[dict[str, Any]] = []

    for action in actions:
        if not isinstance(action, Mapping):
            continue

        rank = action.get(
            "final_priority_rank",
            action.get("recommendation_rank", action.get("preliminary_priority_rank", "")),
        )
        hazard = action.get("hazard_code", "UNKNOWN_HAZARD")
        room = action.get("room_type", "")
        object_id = action.get("object_id", "")
        score = action.get(
            "final_priority_score",
            action.get("preliminary_priority_score", action.get("priority_score")),
        )

        source_labels: list[str] = []

        for item in action.get("matched_main_factors", []) or []:
            if isinstance(item, Mapping):
                feature = item.get("feature")
                factor_code = item.get("factor_code")
                contribution = item.get("contribution")
                label = feature or factor_code
                if label:
                    if isinstance(contribution, (int, float)):
                        source_labels.append(f"{label}\nSHAP={contribution:.4f}")
                    else:
                        source_labels.append(str(label))

        for item in action.get("matched_interactions", []) or []:
            if isinstance(item, Mapping):
                feature_1 = item.get("feature_1", "")
                feature_2 = item.get("feature_2", "")
                contribution = item.get("contribution")
                label = f"{feature_1} × {feature_2}".strip(" ×")
                if label:
                    if isinstance(contribution, (int, float)):
                        source_labels.append(f"{label}\nInteraction={contribution:.4f}")
                    else:
                        source_labels.append(label)

        if not source_labels:
            csv_features = action.get("matched_features")
            need_codes = action.get("need_codes", action.get("matched_need_codes"))
            if isinstance(csv_features, str) and csv_features.strip():
                source_labels = [part.strip() for part in csv_features.split("|") if part.strip()]
            elif isinstance(need_codes, list):
                source_labels = [str(item) for item in need_codes]
            elif isinstance(need_codes, str) and need_codes.strip():
                source_labels = [part.strip() for part in need_codes.split("|") if part.strip()]
            else:
                source_labels = ["개인 위험기여"]

        evidence = action.get("selected_evidence", {})
        if not isinstance(evidence, Mapping):
            evidence = {}

        recommendation = (
            evidence.get("recommendation")
            or action.get("recommendation")
            or "선택된 솔루션 없음"
        )
        evidence_id = evidence.get("evidence_id") or action.get("evidence_id") or ""
        expected_effect = evidence.get("expected_effect") or action.get("expected_effect") or ""
        tier = evidence.get("tier") or action.get("tier") or ""

        rows.append(
            {
                "rank": rank,
                "source": "\n+\n".join(source_labels),
                "hazard": str(hazard),
                "room": str(room),
                "object_id": str(object_id),
                "score": score,
                "recommendation": str(recommendation),
                "evidence_id": str(evidence_id),
                "expected_effect": str(expected_effect),
                "tier": str(tier),
            }
        )

    rows.sort(key=lambda row: int(row["rank"]) if str(row["rank"]).isdigit() else 9999)
    return rows


def plot_solution_mapping(
    final_plan: JsonLike | TableLike,
    output_path: str | Path,
    *,
    top_n: int = 7,
    source_wrap: int = 28,
    recommendation_wrap: int = 42,
    dpi: int = 220,
) -> Path:
    """
    개인 위험요인/interaction → 실제 공간위험 → RAG 문헌 솔루션
    매핑을 3단 흐름도로 저장합니다.

    입력:
        - 05_final_ranked_action_plan.json
        - 또는 05_selected_solutions.csv
        - 또는 동일 구조의 dict/DataFrame
    """
    if top_n < 1:
        raise ValueError("top_n은 1 이상이어야 합니다.")

    rows = _extract_mapping_rows(final_plan)[:top_n]
    if not rows:
        raise ValueError("ranked_actions 또는 solution 매핑 행이 없습니다.")

    _set_korean_font()
    output = _ensure_output(output_path)

    row_height = 1.75
    figure_height = max(6.0, row_height * len(rows) + 2.3)
    fig = plt.figure(figsize=(18, figure_height))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(rows) + 1.2)
    ax.axis("off")

    ax.set_title(
        "개인 위험기여 → 실제 공간위험 → 근거 기반 솔루션 매핑",
        fontsize=16,
        pad=20,
    )

    x_positions = (0.03, 0.365, 0.66)
    widths = (0.27, 0.22, 0.31)

    ax.text(x_positions[0] + widths[0] / 2, len(rows) + 0.75, "개인 SHAP / Interaction", ha="center", weight="bold")
    ax.text(x_positions[1] + widths[1] / 2, len(rows) + 0.75, "공간위험·우선순위", ha="center", weight="bold")
    ax.text(x_positions[2] + widths[2] / 2, len(rows) + 0.75, "RAG 문헌 솔루션", ha="center", weight="bold")

    for index, row in enumerate(rows):
        y = len(rows) - index - 0.05
        box_height = 0.98

        source_text = _wrap(row["source"], source_wrap)

        hazard_lines = [f"{row['rank']}순위 · {row['hazard']}"]
        location = " / ".join(part for part in [row["room"], row["object_id"]] if part)
        if location:
            hazard_lines.append(location)
        if isinstance(row["score"], (int, float)):
            hazard_lines.append(f"우선순위 점수: {row['score']:.4f}")
        hazard_text = "\n".join(hazard_lines)

        solution_lines = [_wrap(row["recommendation"], recommendation_wrap)]
        metadata = " · ".join(
            part for part in [
                f"Evidence: {row['evidence_id']}" if row["evidence_id"] else "",
                f"Tier: {row['tier']}" if row["tier"] else "",
            ]
            if part
        )
        if metadata:
            solution_lines.append(metadata)
        if row["expected_effect"]:
            solution_lines.append("기대효과: " + _wrap(row["expected_effect"], recommendation_wrap))
        solution_text = "\n".join(solution_lines)

        box_texts = (source_text, hazard_text, solution_text)

        for x, width, text in zip(x_positions, widths, box_texts):
            patch = FancyBboxPatch(
                (x, y - box_height / 2),
                width,
                box_height,
                boxstyle="round,pad=0.012",
                linewidth=1.2,
                fill=False,
            )
            ax.add_patch(patch)
            ax.text(
                x + width / 2,
                y,
                text,
                ha="center",
                va="center",
                fontsize=9,
            )

        ax.annotate(
            "",
            xy=(x_positions[1] - 0.01, y),
            xytext=(x_positions[0] + widths[0] + 0.01, y),
            arrowprops={"arrowstyle": "->", "linewidth": 1.2},
        )
        ax.annotate(
            "",
            xy=(x_positions[2] - 0.01, y),
            xytext=(x_positions[1] + widths[1] + 0.01, y),
            arrowprops={"arrowstyle": "->", "linewidth": 1.2},
        )

    ax.text(
        0.03,
        0.18,
        "※ 위험 우선순위는 SHAP·승인된 interaction·공간 탐지 결과로 계산되며, "
        "RAG 유사도나 문헌 점수는 위험 순위를 변경하지 않습니다.",
        fontsize=9,
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def visualize_all_pipeline_results(
    *,
    shap_result: JsonLike,
    final_plan: JsonLike | TableLike,
    output_dir: str | Path,
    top_n_shap: int = 15,
    top_n_interactions: int = 15,
    top_n_solutions: int = 7,
    formats: Sequence[str] = ("png",),
) -> dict[str, list[str]]:
    """
    SHAP, SHAP interaction, solution mapping 시각화를 한 번에 생성합니다.

    Returns
    -------
    dict
        생성된 파일 경로 목록.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    allowed = {"png", "svg", "pdf"}
    requested = [fmt.lower().lstrip(".") for fmt in formats]
    invalid = [fmt for fmt in requested if fmt not in allowed]
    if invalid:
        raise ValueError(f"지원하지 않는 형식입니다: {invalid}. 지원 형식: {sorted(allowed)}")

    created: dict[str, list[str]] = {
        "local_shap": [],
        "shap_interactions": [],
        "solution_mapping": [],
    }

    for fmt in requested:
        local_path = out / f"viz_01_local_shap.{fmt}"
        interaction_path = out / f"viz_02_shap_interactions.{fmt}"
        mapping_path = out / f"viz_03_solution_mapping.{fmt}"

        plot_local_shap(
            shap_result,
            local_path,
            top_n=top_n_shap,
        )
        plot_shap_interactions(
            shap_result,
            interaction_path,
            top_n=top_n_interactions,
        )
        plot_solution_mapping(
            final_plan,
            mapping_path,
            top_n=top_n_solutions,
        )

        created["local_shap"].append(str(local_path))
        created["shap_interactions"].append(str(interaction_path))
        created["solution_mapping"].append(str(mapping_path))

    manifest = out / "visualization_manifest.json"
    manifest.write_text(
        json.dumps(created, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    created["manifest"] = [str(manifest)]
    return created


if __name__ == "__main__":
    # 현재 작업 폴더에 파이프라인 산출물이 있을 때의 단독 실행 예시
    result = visualize_all_pipeline_results(
        shap_result="01_shap_result.json",
        final_plan="05_final_ranked_action_plan.json",
        output_dir="visualizations",
        formats=("png", "svg"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
