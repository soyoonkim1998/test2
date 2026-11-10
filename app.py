import base64
import datetime
import hashlib
import io
import os
import re
import threading
import warnings

import streamlit as st
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)
from PIL import Image, ImageOps, UnidentifiedImageError

load_dotenv()

ALLOWED_TYPES = ["jpg", "jpeg", "png", "webp"]
DEFAULT_MODEL = "gpt-4o-mini"
MAX_FILE_BYTES = 10 * 1024 * 1024
# 용량은 작아도 해상도가 큰 이미지(예: 12000x12000 PNG)는 펼칠 때 수백 MB를 쓰므로 화소 수로도 막는다.
MAX_PIXELS = 50_000_000
MAX_IMAGE_SIDE = 1024
# 공개 배포 시 API 비용 보호용 기본값. secrets/.env 로 바꿀 수 있다.
DEFAULT_MAX_REQUESTS_PER_SESSION = 10
DEFAULT_MAX_DAILY_REQUESTS = 200
REQUEST_TIMEOUT = 60
NO_FACE = "NO_FACE"

SYSTEM_PROMPT = f"""당신은 수십 년 경력의 친근한 한국 관상가입니다.
사용자가 보낸 얼굴 사진을 보고 재미와 엔터테인먼트 목적의 관상 풀이를 해 주세요.

[출력 형식] 반드시 아래 Markdown 형식과 순서를 지키고, 한국어로 작성하세요.

## 부위별 관상 · 部位
- **이마**: (2~3문장)
- **눈**: (2~3문장)
- **코**: (2~3문장)
- **입**: (2~3문장)
- **귀**: (2~3문장, 사진에서 보이지 않으면 보이지 않는다고 짧게 언급)
- **얼굴형**: (2~3문장)

## 종합 운세 · 運勢
- **재물운**: (1~2문장)
- **애정운**: (1~2문장)
- **직업운**: (1~2문장)
- **건강운**: (1~2문장)

## 한 줄 요약 · 總評
(한 문장)

[지켜야 할 원칙]
- 긍정적이고 유쾌한 톤을 유지하고, 외모를 비하하거나 단점을 부각하지 마세요.
- 인종, 성별, 나이, 국적, 종교 등 민감한 속성을 추측하거나 단정하지 마세요.
- 사진 속 인물이 누구인지 추정하거나 실명을 언급하지 마세요.
- 사진에 사람의 얼굴이 없거나, 얼굴을 알아보기 어려울 정도로 불분명하면 풀이하지 말고 정확히 {NO_FACE} 한 단어만 출력하세요.
"""

USER_PROMPT = "이 사람의 관상을 봐주세요."

REFUSAL_MARKERS = (
    "i'm sorry",
    "i am sorry",
    "i can't",
    "i cannot",
    "can't help",
    "도와드릴 수 없",
    "분석할 수 없",
    "제공할 수 없",
)


class FaceReadingError(Exception):
    """사용자에게 그대로 보여줄 수 있는 오류 메시지를 담는다."""


def get_config(key: str, default: str | None = None) -> str | None:
    """st.secrets(Streamlit Cloud)를 먼저 보고, 없으면 환경변수(.env)를 사용한다."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        # secrets.toml이 없으면 st.secrets 접근 시 예외가 발생한다.
        pass
    return os.getenv(key, default)


def get_int_config(key: str, default: int) -> int:
    try:
        return int(get_config(key, default))
    except (TypeError, ValueError):
        return default


def validate_image(data: bytes) -> str | None:
    """이미지를 펼치지 않고 헤더만 읽어 검사한다. 문제가 있으면 사용자용 오류 메시지를 반환한다."""
    if len(data) > MAX_FILE_BYTES:
        return "파일이 너무 큽니다. 10MB 이하의 사진을 올려주세요."
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                width, height = image.size
                if width * height > MAX_PIXELS:
                    return (
                        f"사진 해상도가 너무 큽니다({width}×{height}). "
                        "5천만 화소 이하의 사진을 올려주세요."
                    )
                image.verify()
    except Image.DecompressionBombError:
        return "사진 해상도가 너무 큽니다. 5천만 화소 이하의 사진을 올려주세요."
    except Exception:
        return "올바른 이미지 파일이 아닙니다. JPG, PNG, WEBP 사진을 올려주세요."
    return None


def prepare_image(data: bytes) -> str:
    """이미지 바이트를 리사이즈된 JPEG base64 data URL로 변환한다. 디스크에는 쓰지 않는다."""
    error = validate_image(data)
    if error:
        raise FaceReadingError(error)
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise FaceReadingError("올바른 이미지 파일이 아닙니다. JPG, PNG, WEBP 사진을 올려주세요.") from exc

    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@st.cache_resource
def _daily_counter() -> dict:
    """모든 사용자가 공유하는 하루 요청 수. 앱 프로세스가 재시작되면 초기화된다."""
    return {"date": None, "count": 0, "lock": threading.Lock()}


def reserve_request() -> str | None:
    """요청 한도를 확인하고 1회를 차감한다. 한도를 넘으면 사용자용 메시지를 반환한다."""
    per_session = get_int_config("MAX_REQUESTS_PER_SESSION", DEFAULT_MAX_REQUESTS_PER_SESSION)
    daily = get_int_config("MAX_DAILY_REQUESTS", DEFAULT_MAX_DAILY_REQUESTS)

    if st.session_state.get("request_count", 0) >= per_session:
        return f"한 번 방문에 최대 {per_session}번까지 관상을 볼 수 있습니다."

    counter = _daily_counter()
    today = datetime.date.today().isoformat()
    with counter["lock"]:
        if counter["date"] != today:
            counter["date"], counter["count"] = today, 0
        if counter["count"] >= daily:
            return "오늘 관상소의 운영 한도에 도달했습니다. 내일 다시 찾아주세요."
        counter["count"] += 1

    st.session_state["request_count"] = st.session_state.get("request_count", 0) + 1
    return None


def to_plain_text(markdown: str) -> str:
    """.txt 저장용으로 Markdown 기호(##, **, -)를 걷어낸다."""
    text = re.sub(r"^#{1,6}\s*", "", markdown, flags=re.MULTILINE)
    text = text.replace("**", "")
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.MULTILINE)
    return text.strip() + "\n"


def is_refusal(text: str) -> bool:
    lowered = text.strip().lower()
    return len(lowered) < 200 and any(marker in lowered for marker in REFUSAL_MARKERS)


def analyze_face(data_url: str, api_key: str, model: str) -> str:
    """OpenAI 이미지 분석 모델에 관상 풀이를 요청하고 Markdown 결과를 반환한다."""
    client = OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
    except AuthenticationError as exc:
        raise FaceReadingError("API 키가 올바르지 않습니다. OPENAI_API_KEY 설정을 확인해주세요.") from exc
    except RateLimitError as exc:
        raise FaceReadingError(
            "요청이 너무 많거나 사용량 한도를 초과했습니다. 잠시 후 다시 시도하거나 OpenAI 사용량 한도를 확인해주세요."
        ) from exc
    except (APITimeoutError, APIConnectionError) as exc:
        # APITimeoutError는 APIConnectionError의 하위 클래스라 함께 처리한다.
        raise FaceReadingError("OpenAI 서버에 연결하지 못했습니다. 네트워크 상태를 확인하고 다시 시도해주세요.") from exc
    except NotFoundError as exc:
        raise FaceReadingError(
            f"모델을 찾을 수 없습니다. OPENAI_MODEL 설정({model})이 이미지 입력을 지원하는 모델 이름인지 확인해주세요."
        ) from exc
    except BadRequestError as exc:
        raise FaceReadingError(
            f"요청이 거부되었습니다. 이미지 형식이나 모델 설정(OPENAI_MODEL={model})을 확인해주세요."
        ) from exc
    except Exception as exc:
        raise FaceReadingError("관상 분석 중 알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해주세요.") from exc

    message = response.choices[0].message
    if getattr(message, "refusal", None):
        return ""
    return (message.content or "").strip()


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@600;700;900&family=Gowun+Batang:wght@400;700&display=swap');

:root { --ink: #2A211B; --seal: #A3262A; --paper: #F4EDE0; --paper-light: #FBF7EE; --line: #B9A07E; }

h1, h2, h3, .hero-title, .seal { font-family: 'Noto Serif KR', serif !important; }

/* 헤더: 붉은 낙관(도장) + 먹색 제목 */
.hero { text-align: center; padding: 0.5rem 0 0.75rem; }
.hero-row { display: flex; align-items: center; justify-content: center; gap: 0.9rem; }
.seal {
    writing-mode: vertical-rl; background: var(--seal); color: var(--paper) !important;
    font-weight: 900; font-size: 1.15rem !important; letter-spacing: 0.15em;
    padding: 0.4rem 0.3rem; border-radius: 3px; transform: rotate(-4deg);
    box-shadow: inset 0 0 0 2px var(--paper), 0 0 0 2px var(--seal); margin: 0 !important;
}
.hero-title {
    font-size: clamp(2rem, 8vw, 2.7rem) !important; font-weight: 900;
    color: var(--ink); margin: 0 !important; line-height: 1.2; letter-spacing: 0.04em;
}
.hero-sub { color: #6B5A4A; margin: 0.6rem 0 0.9rem !important; font-family: 'Gowun Batang', serif; }
.steps {
    display: flex; flex-wrap: wrap; justify-content: center; gap: 0.25rem 1rem;
    border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); padding: 0.45rem 0;
}
.step { font-size: 0.9rem; color: var(--ink); white-space: nowrap; font-family: 'Gowun Batang', serif; }
.step b { color: var(--seal); font-family: 'Noto Serif KR', serif; margin-right: 0.25rem; }
@media (max-width: 480px) {
    .steps { gap: 0.25rem 0.6rem; }
    .step { font-size: 0.82rem; }
}

/* 결과 카드: st.container(key="result_card")가 붙이는 .st-key-result_card 클래스로 지정.
   바깥 실선 + 안쪽 실선으로 전통 서책 테두리 느낌을 낸다. */
.st-key-result_card {
    background: var(--paper-light);
    border: 1px solid var(--line) !important;
    box-shadow: inset 0 0 0 4px var(--paper-light), inset 0 0 0 5px rgba(185, 160, 126, 0.6);
    padding: 1rem 1.4rem !important;
}
.st-key-result_card h2 {
    font-size: 1.35rem !important; color: var(--ink) !important;
    border-left: 4px solid var(--seal); padding: 0.1rem 0 0.1rem 0.6rem !important; margin-top: 0.6rem;
}
.st-key-result_card li, .st-key-result_card p {
    font-family: 'Gowun Batang', serif; line-height: 1.85 !important;
}
.st-key-result_card strong { color: var(--seal); }

.footer { text-align: center; color: #8A7A69; font-size: 0.8rem; line-height: 1.6; }
</style>
"""


def render_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <div class="hero-row">
            <span class="seal">觀相</span>
            <span class="hero-title">AI 관상소</span>
          </div>
          <div class="hero-sub">얼굴에 새겨진 복(福)과 운(運)을 풀어드립니다</div>
          <div class="steps">
            <span class="step"><b>一</b>사진 올리기</span>
            <span class="step"><b>二</b>관상 보기</span>
            <span class="step"><b>三</b>풀이 확인</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model: str) -> None:
    with st.sidebar:
        st.header("관상소 안내")
        st.write(
            "얼굴 사진을 올리면 AI가 이마·눈·코·입·귀·얼굴형을 살펴보고 "
            "재물운·애정운·직업운·건강운을 풀이해 드립니다."
        )
        st.caption(f"사용 모델: `{model}`")
        st.divider()
        st.subheader("주의사항")
        st.markdown(
            "- 결과는 **재미로만** 봐주세요. 과학적 근거가 없습니다.\n"
            "- 이 앱은 사진을 **저장하지 않습니다.** 다만 풀이를 위해 사진이 **OpenAI로 전송**됩니다.\n"
            "- 다른 사람의 사진은 동의를 받고 사용해주세요."
        )


def render_footer() -> None:
    st.divider()
    st.markdown(
        '<p class="footer">⚠️ 본 결과는 재미를 위한 것이며 과학적 근거가 없습니다.<br>'
        "이 앱은 사진을 저장하지 않으며, 풀이를 위해 OpenAI로 전송됩니다.</p>",
        unsafe_allow_html=True,
    )


def reset() -> None:
    """결과를 지우고, 위젯 key를 바꿔 업로더/카메라 입력도 비운다."""
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)
    st.session_state.pop("image_id", None)
    st.session_state["input_version"] = st.session_state.get("input_version", 0) + 1


def get_input_image():
    """업로드 탭과 카메라 탭 중 입력된 이미지를 반환한다. 둘 다 있으면 카메라 사진을 우선한다."""
    version = st.session_state.get("input_version", 0)
    upload_tab, camera_tab = st.tabs(["📁 사진 업로드", "📷 카메라 촬영"])
    with upload_tab:
        uploaded = st.file_uploader(
            "얼굴 사진을 올려주세요 (JPG·PNG·WEBP, 10MB 이하)",
            type=ALLOWED_TYPES,
            key=f"uploader_{version}",
        )
    with camera_tab:
        # 탭은 숨겨져 있어도 렌더링되므로, 켜기 전에는 카메라 권한을 요청하지 않도록 토글로 감싼다.
        captured = None
        if st.toggle("카메라 켜기", key=f"camera_on_{version}"):
            captured = st.camera_input("정면을 바라보고 촬영해주세요", key=f"camera_{version}")
    st.caption("💡 촬영 팁: 정면을 바라보고, 밝은 곳에서, 얼굴 전체가 보이게 찍어주세요.")
    return captured or uploaded


def show_result(result: str) -> None:
    if result == NO_FACE or NO_FACE in result:
        st.warning("사진에서 얼굴을 찾지 못했습니다. 얼굴이 잘 보이는 정면 사진을 올려주세요.")
    elif not result or is_refusal(result):
        st.warning("AI가 이 사진의 관상 풀이를 거절했습니다. 얼굴이 잘 보이는 다른 사진으로 다시 시도해주세요.")
    else:
        with st.container(border=True, key="result_card"):
            st.markdown(result)
        st.download_button(
            "📥 결과 저장 (.txt)",
            data=to_plain_text(result),
            file_name="관상_결과.txt",
            mime="text/plain",
            width="stretch",
        )
        with st.expander("📋 결과 텍스트 복사하기"):
            st.code(result, language=None, wrap_lines=True)
    st.button("🔄 다시 하기", on_click=reset, width="stretch")


def main() -> None:
    st.set_page_config(page_title="AI 관상소", page_icon="📜", layout="centered")
    render_header()

    api_key = get_config("OPENAI_API_KEY")
    if not api_key:
        st.error(
            "OpenAI API 키가 설정되지 않았습니다.\n\n"
            "- 로컬: 프로젝트 폴더의 `.env` 파일에 `OPENAI_API_KEY=sk-...` 를 추가하세요.\n"
            "- Streamlit Cloud: App settings → Secrets 에 `OPENAI_API_KEY = \"sk-...\"` 를 추가하세요."
        )
        render_footer()
        st.stop()
    model = get_config("OPENAI_MODEL") or DEFAULT_MODEL
    render_sidebar(model)

    image_file = get_input_image()
    image_bytes = image_file.getvalue() if image_file is not None else None

    # 새 이미지가 들어오면 이전 결과를 지운다.
    image_id = hashlib.sha256(image_bytes).hexdigest() if image_bytes else None
    if st.session_state.get("image_id") != image_id:
        st.session_state["image_id"] = image_id
        st.session_state.pop("result", None)
        st.session_state.pop("error", None)

    if image_bytes is not None:
        error = validate_image(image_bytes)
        if error:
            st.error(error)
            image_bytes = None
        else:
            # 데스크톱에서 가운데 약 400px, 모바일에서는 열이 쌓이며 전체 폭이 된다.
            _, center, _ = st.columns([1, 3, 1])
            with center:
                st.image(image_bytes, caption="분석할 사진", width="stretch")

    per_session = get_int_config("MAX_REQUESTS_PER_SESSION", DEFAULT_MAX_REQUESTS_PER_SESSION)
    remaining = max(per_session - st.session_state.get("request_count", 0), 0)

    if st.button(
        "관상 보기",
        type="primary",
        width="stretch",
        disabled=image_bytes is None or remaining == 0,
    ):
        st.session_state.pop("error", None)
        limit_message = reserve_request()
        if limit_message:
            st.session_state["error"] = ("warning", limit_message)
        else:
            try:
                with st.spinner("관상가가 얼굴을 살피는 중..."):
                    data_url = prepare_image(image_bytes)
                    st.session_state["result"] = analyze_face(data_url, api_key, model)
            except FaceReadingError as exc:
                st.session_state["error"] = ("error", str(exc))
        # 버튼은 클릭 전 횟수로 이미 그려졌으므로, 다시 그려서 남은 횟수/비활성 상태를 맞춘다.
        st.rerun()

    if "error" in st.session_state:
        level, message = st.session_state["error"]
        (st.warning if level == "warning" else st.error)(message)

    if remaining == 0:
        st.caption(f"이번 방문에서 볼 수 있는 관상 {per_session}회를 모두 사용했습니다.")
    else:
        st.caption(f"남은 횟수: {remaining}/{per_session}회")

    if "result" in st.session_state:
        show_result(st.session_state["result"])

    render_footer()


if __name__ == "__main__":
    main()
