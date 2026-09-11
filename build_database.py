import json
import os
import re
import time
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# 1. 기본 경로 및 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# 논문 PDF를 넣는 폴더
PAPERS_DIR = BASE_DIR / "papers"

# 결과물을 저장하는 폴더
DATABASE_DIR = BASE_DIR / "data"

# 최종 데이터베이스
OUTPUT_CSV = DATABASE_DIR / "fall_prevention_database.csv"

# 이미 처리한 PDF 목록
PROCESSED_LOG = DATABASE_DIR / "processed_files.txt"

# 오류가 발생한 PDF 목록
ERROR_LOG = DATABASE_DIR / "error_log.txt"


# 사용할 OpenAI 모델
# 필요하면 .env에서 OPENAI_MODEL 값을 따로 지정할 수 있음
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-mini")

# 논문 한 편에서 OpenAI로 보낼 최대 문자 수
MAX_TEXT_LENGTH = 80_000

# API 오류 발생 시 재시도 횟수
MAX_RETRIES = 3


# =========================================================
# 2. OpenAI API 설정
# =========================================================

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "\nOPENAI_API_KEY가 설정되지 않았습니다.\n\n"
        "프로젝트 폴더의 .env 파일에 다음과 같이 입력하세요.\n\n"
        "OPENAI_API_KEY=본인의_API_KEY\n"
    )

client = OpenAI(api_key=api_key)


# 폴더가 없으면 자동 생성
PAPERS_DIR.mkdir(exist_ok=True)
DATABASE_DIR.mkdir(exist_ok=True)


# =========================================================
# 3. 위험요인 목록
# =========================================================
#
# 반드시 네 머신러닝에서 사용한 최종 변수명과 동일하게 맞추는 것이 좋음.
# 나중에 변수명이 바뀌면 이 부분만 수정하면 됨.
# =========================================================

RISK_FACTORS = [
    "근력상태_의자나 침대에 앉았다가 일어나기 5회 반복",
    "동작수행 어려움_운동장 한 바퀴(400m)정도 걷기",
    "동작수행 어려움_쉬지않고 10계단 오르기",
    "동작수행 어려움_몸 구부리거나 쭈그려 앉거나 무릎 꿇기",
    "동작수행 어려움_머리보다 높은 곳에 있는 것 손 뻗쳐 닿기",
    "동작수행 어려움_쌀 1말(8kg) 정도 물건 들어 올리거나 옮기기",
    "Nutritional",
    "ADL",
    "IADL",
    "시력",
    "청력",
    "우울",
    "인지기능",
    "수면",
    "규칙적 운동",
    "만성질환",
    "복용약물",
    "독거 또는 사회적 고립",
    "음주",
    "BMI 또는 비만",
    "연령",
    "낙상 경험",
]


# =========================================================
# 4. 최종 CSV 열
# =========================================================
#
# 기존 테스트 CSV의 열:
# id, risk_factor, space, design_strategy,
# expected_effect, evidence_type
#
# 새로 추가한 열:
# target_object, source, doi
# =========================================================

CSV_COLUMNS = [
    "id",
    "risk_factor",
    "space",
    "design_strategy",
    "expected_effect",
    "evidence_type",
    "target_object",
    "source",
    "doi",
]

# GPT가 반환해야 하는 열
# id는 파이썬이 자동 생성하므로 제외
GPT_COLUMNS = [
    "risk_factor",
    "space",
    "design_strategy",
    "expected_effect",
    "evidence_type",
    "target_object",
    "source",
    "doi",
]


# =========================================================
# 5. Structured Outputs용 JSON Schema
# =========================================================

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "risk_factor": {
                        "type": "string",
                        "description": (
                            "제공된 위험요인 목록 중 하나와 정확히 같은 값"
                        ),
                    },
                    "space": {
                        "type": "string",
                        "description": (
                            "개선 전략을 적용할 공간. "
                            "예: 침실, 거실, 욕실, 주방, 복도, 계단, 현관"
                        ),
                    },
                    "design_strategy": {
                        "type": "string",
                        "description": (
                            "논문에서 근거를 확인할 수 있는 구체적인 "
                            "공간 또는 환경 개선 전략"
                        ),
                    },
                    "expected_effect": {
                        "type": "string",
                        "description": (
                            "해당 공간 개선으로 기대되는 결과 또는 "
                            "논문에서 관찰된 효과"
                        ),
                    },
                    "evidence_type": {
                        "type": "string",
                        "description": (
                            "연구 유형. 예: 무작위 대조시험, 코호트 연구, "
                            "단면연구, 체계적 문헌고찰, 메타분석, 가이드라인"
                        ),
                    },
                    "target_object": {
                        "type": "string",
                        "description": (
                            "개선 대상 물체나 요소. "
                            "예: 침대, 의자, 손잡이, 조명, 바닥, 문턱"
                        ),
                    },
                    "source": {
                        "type": "string",
                        "description": "논문 또는 문서의 제목",
                    },
                    "doi": {
                        "type": "string",
                        "description": (
                            "DOI. 본문에서 확인되지 않으면 빈 문자열"
                        ),
                    },
                },
                "required": GPT_COLUMNS,
                "additionalProperties": False,
            },
        }
    },
    "required": ["rows"],
    "additionalProperties": False,
}


# =========================================================
# 6. PDF에서 텍스트 추출
# =========================================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    PDF의 모든 페이지에서 텍스트를 추출한다.

    일반 논문 PDF나 PowerPoint를 PDF로 변환한 초록 파일은
    대부분 텍스트 추출이 가능하다.

    스캔 이미지로만 구성된 PDF는 추출되지 않을 수 있다.
    """

    page_texts = []

    with fitz.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            page_text = page.get_text("text")

            if page_text and page_text.strip():
                page_texts.append(
                    f"\n--- PAGE {page_number} ---\n"
                    f"{page_text.strip()}"
                )

    full_text = "\n".join(page_texts).strip()

    # 반복되는 공백 정리
    full_text = re.sub(r"[ \t]+", " ", full_text)

    # 지나치게 많은 줄바꿈 정리
    full_text = re.sub(r"\n{4,}", "\n\n\n", full_text)

    return full_text


# =========================================================
# 7. 너무 긴 논문 텍스트 줄이기
# =========================================================

def shorten_paper_text(text: str) -> str:
    """
    논문이 지나치게 긴 경우 앞부분과 뒷부분을 사용한다.

    앞부분:
    제목, 초록, 서론, 방법 등이 있을 가능성이 높음

    뒷부분:
    결과, 논의, 결론 등이 있을 가능성이 높음
    """

    if len(text) <= MAX_TEXT_LENGTH:
        return text

    front_length = int(MAX_TEXT_LENGTH * 0.6)
    back_length = MAX_TEXT_LENGTH - front_length

    return (
        text[:front_length]
        + "\n\n"
        + "[논문 중간 부분은 길이 제한으로 생략되었습니다.]"
        + "\n\n"
        + text[-back_length:]
    )


# =========================================================
# 8. GPT에게 전달할 프롬프트 생성
# =========================================================

def make_prompt(
    paper_text: str,
    source_filename: str,
) -> str:
    """
    논문에서 공간디자인 전략을 추출하기 위한 프롬프트를 만든다.
    """

    risk_factor_text = "\n".join(
        f"{index}. {factor}"
        for index, factor in enumerate(RISK_FACTORS, start=1)
    )

    return f"""
당신은 노인 낙상 예방, 주거환경 개선 및 공간디자인 문헌을
데이터베이스 형식으로 정리하는 연구 보조자입니다.

아래 논문 또는 논문 초록에서 개인의 건강·기능적 위험요인과
연결할 수 있는 공간 및 환경 개선 전략을 추출하십시오.

분석 중인 파일명:
{source_filename}


[사용 가능한 위험요인]

{risk_factor_text}


[가장 중요한 규칙]

1. risk_factor에는 반드시 위 위험요인 중 하나를 그대로 작성하십시오.

2. 논문에서 공간, 환경, 주거, 가구, 설비, 조명, 바닥,
   동선 또는 보조시설과 관련된 개선 전략만 추출하십시오.

3. 운동 프로그램, 약물 치료, 영양식 제공, 상담처럼
   공간 또는 환경과 무관한 개입은 제외하십시오.

4. 논문에서 직접적으로 확인되지 않은 수치, 결과, 연구유형,
   논문 제목 또는 DOI를 만들지 마십시오.

5. 같은 전략이 여러 위험요인에 적용될 수 있다면
   위험요인별로 행을 각각 만들 수 있습니다.

6. 하나의 논문에서 여러 공간개선 전략이 확인된다면
   전략별로 행을 나누십시오.

7. 단순하고 추상적인 표현을 피하고 실행 가능한 수준으로 작성하십시오.

나쁜 예:
- 환경을 개선한다.
- 조명을 밝게 한다.
- 안전한 가구를 사용한다.

좋은 예:
- 침대에서 화장실까지 이어지는 이동 경로에 야간 센서등을 설치한다.
- 침대 높이를 사용자의 무릎 높이에 맞춰 앉고 일어서기 쉽게 조절한다.
- 욕실 변기 옆에 앉고 일어설 때 잡을 수 있는 안전손잡이를 설치한다.

8. 공간 또는 환경을 개선했더니 우울, 수면, 인지기능,
   신체활동 등의 상태가 개선된 연구도 포함할 수 있습니다.
   다만 예상 효과에 낙상 감소라고 임의로 덧붙이지 마십시오.

9. 논문에서 근거를 찾지 못했다면 rows를 빈 배열로 반환하십시오.

10. DOI가 확인되지 않으면 doi는 빈 문자열로 작성하십시오.

11. 연구유형이 확인되지 않으면 evidence_type은 빈 문자열로 작성하십시오.

12. 논문 제목이 확인되지 않으면 source에는 파일명을 작성하십시오.


[각 열의 작성 방법]

risk_factor:
위 위험요인 목록 중 하나와 정확히 동일한 값

space:
개선 전략이 적용되는 공간
예: 침실, 거실, 욕실, 주방, 복도, 계단, 현관,
주거공간 전반, 병실, 요양시설

design_strategy:
구체적인 공간 또는 환경 개선 방법

expected_effect:
논문에서 보고하거나 제안한 효과
예: 기립 안정성 향상, 야간 시인성 향상,
보행 부담 감소, 우울 증상 완화

evidence_type:
논문에서 확인되는 연구유형
예: 무작위 대조시험, 비무작위 중재연구, 코호트 연구,
단면연구, 체계적 문헌고찰, 메타분석, 가이드라인,
공공기관 권고

target_object:
개선 대상이 되는 가구 또는 환경 요소
예: 침대, 의자, 변기, 욕조, 손잡이, 조명,
바닥, 수납장, 문턱, 계단

특정 물체를 정하기 어렵다면 빈 문자열

source:
논문 또는 문서의 제목

doi:
논문에서 확인되는 DOI
확인되지 않으면 빈 문자열


[분석할 논문 또는 초록]

{paper_text}
""".strip()


# =========================================================
# 9. OpenAI로 논문 분석
# =========================================================

def analyze_paper_with_openai(
    paper_text: str,
    source_filename: str,
) -> list[dict]:
    """
    논문 텍스트를 OpenAI에 보내 구조화된 결과를 받는다.
    """

    prompt = make_prompt(
        paper_text=paper_text,
        source_filename=source_filename,
    )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 연구논문에서 근거 기반 공간디자인 "
                            "전략을 구조화하여 추출하는 연구 보조자입니다."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "fall_prevention_database",
                        "strict": True,
                        "schema": OUTPUT_SCHEMA,
                    }
                },
            )

            if not response.output_text:
                raise ValueError(
                    "OpenAI 응답이 비어 있습니다."
                )

            result = json.loads(response.output_text)

            rows = result.get("rows", [])

            if not isinstance(rows, list):
                raise ValueError(
                    "OpenAI 응답의 rows가 리스트가 아닙니다."
                )

            return rows

        except Exception as error:
            last_error = error

            print(
                f"  OpenAI 호출 오류 "
                f"({attempt}/{MAX_RETRIES}): {error}"
            )

            if attempt < MAX_RETRIES:
                wait_seconds = attempt * 3

                print(
                    f"  {wait_seconds}초 후 다시 시도합니다."
                )

                time.sleep(wait_seconds)

    raise RuntimeError(
        f"OpenAI 분석에 최종 실패했습니다: {last_error}"
    )


# =========================================================
# 10. GPT 결과 정리
# =========================================================

def normalize_row(
    row: dict,
    source_filename: str,
) -> dict:
    """
    GPT가 반환한 한 행을 CSV 형식에 맞게 정리한다.
    """

    normalized = {}

    for column in GPT_COLUMNS:
        value = row.get(column, "")

        if value is None:
            value = ""

        if isinstance(value, (list, dict)):
            value = json.dumps(
                value,
                ensure_ascii=False,
            )

        normalized[column] = str(value).strip()

    # source가 비어 있으면 파일명 사용
    if not normalized["source"]:
        normalized["source"] = source_filename

    # DOI URL 앞부분 제거
    doi = normalized["doi"]

    doi = re.sub(
        r"^https?://(?:dx\.)?doi\.org/",
        "",
        doi,
        flags=re.IGNORECASE,
    )

    normalized["doi"] = doi.strip()

    return normalized


# =========================================================
# 11. 기존 데이터베이스 불러오기
# =========================================================

def load_existing_database() -> pd.DataFrame:
    """
    기존 CSV가 있으면 불러오고,
    없으면 빈 데이터프레임을 만든다.
    """

    if not OUTPUT_CSV.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)

    try:
        dataframe = pd.read_csv(
            OUTPUT_CSV,
            encoding="utf-8-sig",
            dtype=str,
        )

    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=CSV_COLUMNS)

    # 기존 CSV에 새 열이 없는 경우 자동으로 추가
    for column in CSV_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = ""

    dataframe = dataframe[CSV_COLUMNS].fillna("")

    return dataframe


# =========================================================
# 12. 처리 완료 파일 목록
# =========================================================

def load_processed_files() -> set[str]:
    """
    이미 분석한 PDF 파일명을 불러온다.
    """

    if not PROCESSED_LOG.exists():
        return set()

    lines = PROCESSED_LOG.read_text(
        encoding="utf-8"
    ).splitlines()

    return {
        line.strip()
        for line in lines
        if line.strip()
    }


def mark_as_processed(filename: str) -> None:
    """
    처리 완료한 PDF 파일명을 기록한다.
    """

    with PROCESSED_LOG.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(filename + "\n")


# =========================================================
# 13. 오류 기록
# =========================================================

def write_error_log(
    filename: str,
    error: Exception,
) -> None:
    """
    오류가 발생한 파일과 오류 내용을 기록한다.
    """

    with ERROR_LOG.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            f"{filename}\t"
            f"{type(error).__name__}\t"
            f"{error}\n"
        )


# =========================================================
# 14. 데이터베이스 저장
# =========================================================

def save_database(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    중복을 제거하고 ID를 다시 부여한 뒤 CSV로 저장한다.
    """

    if dataframe.empty:
        dataframe = pd.DataFrame(
            columns=CSV_COLUMNS
        )

    else:
        # 누락된 열이 있으면 추가
        for column in CSV_COLUMNS:
            if column not in dataframe.columns:
                dataframe[column] = ""

        dataframe = dataframe.fillna("")

        # 동일한 전략이 중복 저장되는 것을 방지
        dataframe = dataframe.drop_duplicates(
            subset=[
                "risk_factor",
                "space",
                "design_strategy",
                "source",
                "doi",
            ],
            keep="first",
        )

        dataframe = dataframe.reset_index(drop=True)

        # ID를 1부터 다시 부여
        dataframe["id"] = range(
            1,
            len(dataframe) + 1,
        )

        dataframe = dataframe[CSV_COLUMNS]

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    return dataframe


# =========================================================
# 15. 전체 PDF 분석 및 DB 구축
# =========================================================

def build_database() -> None:
    """
    papers 폴더의 PDF를 하나씩 분석하고
    결과를 CSV에 누적 저장한다.
    """

    # 하위 폴더까지 포함하여 PDF 검색
    pdf_files = sorted(
        PAPERS_DIR.rglob("*.pdf")
    )

    if not pdf_files:
        print()
        print("PDF 파일이 없습니다.")
        print(f"다음 폴더에 PDF를 넣으세요:")
        print(PAPERS_DIR)
        return

    processed_files = load_processed_files()
    database = load_existing_database()

    # 같은 이름의 파일 충돌을 막기 위해
    # papers 폴더 기준 상대경로로 기록
    target_files = []

    for pdf_path in pdf_files:
        relative_name = str(
            pdf_path.relative_to(PAPERS_DIR)
        )

        if relative_name not in processed_files:
            target_files.append(pdf_path)

    print()
    print("=" * 65)
    print("낙상 예방 공간디자인 데이터베이스 구축")
    print("=" * 65)
    print(f"사용 모델: {MODEL_NAME}")
    print(f"전체 PDF: {len(pdf_files)}개")
    print(f"이미 처리한 PDF: {len(processed_files)}개")
    print(f"이번에 처리할 PDF: {len(target_files)}개")
    print(f"현재 DB 행 수: {len(database)}개")
    print(f"결과 파일: {OUTPUT_CSV}")
    print("=" * 65)

    if not target_files:
        print()
        print("새롭게 처리할 PDF가 없습니다.")
        return

    for index, pdf_path in enumerate(
        target_files,
        start=1,
    ):
        relative_name = str(
            pdf_path.relative_to(PAPERS_DIR)
        )

        print()
        print(
            f"[{index}/{len(target_files)}] "
            f"{relative_name}"
        )

        try:
            # 1. PDF 텍스트 추출
            paper_text = extract_text_from_pdf(
                pdf_path
            )

            if len(paper_text.strip()) < 100:
                raise ValueError(
                    "PDF에서 충분한 텍스트를 추출하지 "
                    "못했습니다. 스캔 이미지 PDF일 수 있습니다."
                )

            print(
                f"  텍스트 추출 완료: "
                f"{len(paper_text):,}자"
            )

            # 2. 너무 긴 논문 축약
            paper_text = shorten_paper_text(
                paper_text
            )

            print(
                f"  OpenAI 전달 텍스트: "
                f"{len(paper_text):,}자"
            )

            # 3. OpenAI 분석
            rows = analyze_paper_with_openai(
                paper_text=paper_text,
                source_filename=pdf_path.stem,
            )

            # 4. 결과 정리
            normalized_rows = []

            for row in rows:
                if not isinstance(row, dict):
                    continue

                normalized = normalize_row(
                    row=row,
                    source_filename=pdf_path.stem,
                )

                # 핵심값이 비어 있는 행 제외
                if not normalized["risk_factor"]:
                    continue

                if not normalized["design_strategy"]:
                    continue

                normalized_rows.append(normalized)

            # 5. 데이터베이스에 추가
            if normalized_rows:
                new_dataframe = pd.DataFrame(
                    normalized_rows
                )

                # id는 저장할 때 자동 생성
                new_dataframe["id"] = ""

                new_dataframe = new_dataframe[
                    CSV_COLUMNS
                ]

                database = pd.concat(
                    [
                        database,
                        new_dataframe,
                    ],
                    ignore_index=True,
                )

                database = save_database(
                    database
                )

                print(
                    f"  추출된 전략: "
                    f"{len(normalized_rows)}행"
                )

            else:
                print(
                    "  관련된 공간·환경 개선 전략을 "
                    "찾지 못했습니다."
                )

            # 관련 근거가 없더라도 분석은 완료했으므로 기록
            mark_as_processed(relative_name)
            processed_files.add(relative_name)

            print(
                f"  현재 데이터베이스: "
                f"{len(database)}행"
            )

        except Exception as error:
            print(f"  처리 실패: {error}")

            write_error_log(
                filename=relative_name,
                error=error,
            )

    # 마지막으로 한 번 더 저장
    database = save_database(database)

    print()
    print("=" * 65)
    print("데이터베이스 구축 완료")
    print("=" * 65)
    print(f"최종 데이터 수: {len(database)}행")
    print(f"CSV 위치: {OUTPUT_CSV}")

    if ERROR_LOG.exists():
        print(f"오류 기록: {ERROR_LOG}")

    print("=" * 65)


# =========================================================
# 16. 실행
# =========================================================

if __name__ == "__main__":
    build_database()