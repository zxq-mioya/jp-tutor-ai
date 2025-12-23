import json
import streamlit as st
from typing import List, Literal
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# -----------------------------
# 1) 结构化输出（纠错结果的固定格式）
# -----------------------------
ErrorType = Literal[
    "grammar", "vocabulary", "politeness", "particle", "tense",
    "kanji_kana", "naturalness", "other"
]

class CorrectionItem(BaseModel):
    original: str = Field(..., description="用户原句中的问题片段")
    corrected: str = Field(..., description="对应的修正片段")
    error_type: ErrorType
    reason_zh: str = Field(..., description="中文简短理由（1-2句）")
    reason_ja: str = Field(..., description="日文简短理由（1-2句）")
    tip: str = Field(..., description="可复用的小技巧（短）")

class TutorTurn(BaseModel):
    reply_ja: str = Field(..., description="AI对话回复（日语）")
    corrected_sentence_ja: str = Field(..., description="用户整句修正（日语）")
    more_natural_ja: str = Field(..., description="更自然表达（日语）")
    corrections: List[CorrectionItem] = Field(default_factory=list)
    mini_lesson_ja: str = Field(..., description="只讲一个点，2-4句日语")
    next_question_ja: str = Field(..., description="引导下一句（日语）")
    fluency_score: int = Field(..., ge=0, le=100, description="自然度评分0-100")

# -----------------------------
# 2) Streamlit 页面
# -----------------------------
st.set_page_config(page_title="日本語 会話×添削AI", layout="wide")
st.title("日本語 会話×添削AI")

with st.sidebar:
    st.subheader("设置")
    level = st.selectbox("目标水平", ["N5", "N4", "N3", "N2", "N1"], index=2)
    tone = st.selectbox("语气", ["カジュアル（友達）", "丁寧（店員・面接）"], index=1)
    strictness = st.slider("纠错严格度", 1, 5, 3)
    topic = st.text_input("对话主题", value="日常生活（学校・買い物・趣味）")
    explain_lang = st.selectbox("解释语言", ["中文+日文", "只用日文"], index=0)
    model = st.text_input("模型名", value="gpt-4o-mini")

if "history" not in st.session_state:
    st.session_state.history = []  # [{"role":"user"/"assistant","content":"..."}]
if "mistakes" not in st.session_state:
    st.session_state.mistakes = []  # [{"type":..., "original":..., "corrected":...}]

import os
import streamlit as st

##
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
##

client = OpenAI()

SYSTEM_PROMPT = f"""
你是一个“日语会话老师+逐句纠错器”。
目标：让学习者通过对话练习，并对每一句进行纠错、自然改写、简短讲解，再用一个问题引导继续对话。

对话设定：
- 学习者目标水平：{level}
- 对话语气：{tone}
- 主题：{topic}
- 纠错严格度：{strictness}/5

输出必须严格符合 TutorTurn 结构：
- reply_ja：日语继续对话（2-5句），符合水平
- corrected_sentence_ja：把用户整句修正（尽量保持原意）
- more_natural_ja：更自然的另一种表达（不改变核心意思）
- corrections：列出具体问题点（如果几乎没错，也给1-2个更自然的改进）
- mini_lesson_ja：只讲一个点，2-4句日语，易懂
- next_question_ja：一个问题推动下一轮
- fluency_score：0-100

如果用户输入不是日语：
- reply_ja：提醒尽量用日语，并给一句可直接套用的日语模板
- corrected_sentence_ja / more_natural_ja：把用户想表达的内容改成简短日语
"""

def render_turn(result: TutorTurn):
    st.divider()
    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("AIの返事（会話）")
        st.write(result.reply_ja)

        st.subheader("次の質問")
        st.write(result.next_question_ja)

    with right:
        st.subheader("添削")
        st.write("修正（整句）")
        st.code(result.corrected_sentence_ja, language="text")
        st.write("より自然")
        st.code(result.more_natural_ja, language="text")
        st.write(f"スコア: {result.fluency_score}/100")

        st.subheader("ポイント")
        if result.corrections:
            for i, c in enumerate(result.corrections, 1):
                st.write(f"{i}. [{c.error_type}] {c.original} → {c.corrected}")
                if explain_lang == "中文+日文":
                    st.write(f"  理由(中): {c.reason_zh}")
                    st.write(f"  理由(日): {c.reason_ja}")
                else:
                    st.write(f"  理由: {c.reason_ja}")
                st.write(f"  tip: {c.tip}")
        else:
            st.write("大きな間違いはありません。")

        st.subheader("ミニレッスン")
        st.write(result.mini_lesson_ja)

def call_model(user_text: str) -> TutorTurn:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += st.session_state.history[-10:]
    messages.append({"role": "user", "content": user_text})

    # 优先：结构化解析
    try:
        resp = client.responses.parse(
            model=model,
            input=messages,
            text_format=TutorTurn,
        )
        return resp.output_parsed
    except Exception:
        # 兼容：让模型只输出 JSON，再手动解析
        fallback_prompt = (
            SYSTEM_PROMPT
            + "\n\n重要：只输出一个 JSON 对象，不要输出任何额外文字。"
            + "\n必须满足 TutorTurn 的全部字段。"
        )
        messages2 = [{"role": "system", "content": fallback_prompt}]
        messages2 += st.session_state.history[-10:]
        messages2.append({"role": "user", "content": user_text})

        resp2 = client.responses.create(
            model=model,
            input=messages2,
        )
        text = resp2.output_text.strip()

        # 处理模型偶尔加```json```的情况
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            text = text.replace("json", "", 1).strip()

        data = json.loads(text)
        return TutorTurn.model_validate(data)

st.write("输入一句日语开始练习：")
user_text = st.text_input("あなたの文", value="", placeholder="例：今日は学校でプレゼンがありました。")

col_a, col_b = st.columns([1, 1])
with col_a:
    send = st.button("送信")
with col_b:
    clear = st.button("清空对话")

if clear:
    st.session_state.history = []
    st.session_state.mistakes = []
    st.rerun()

if send and user_text.strip():
    try:
        result = call_model(user_text)

        # 写入历史（用于对话连贯）
        st.session_state.history.append({"role": "user", "content": user_text})
        st.session_state.history.append({"role": "assistant", "content": result.reply_ja})

        # 保存错题
        for c in result.corrections:
            st.session_state.mistakes.append(
                {"type": c.error_type, "original": c.original, "corrected": c.corrected}
            )

        render_turn(result)

    except ValidationError as ve:
        st.error("模型返回的格式不符合预期（结构化解析失败）。")
        st.write(ve)
    except Exception as e:
        st.error("运行失败：请检查 API Key / 网络 / 模型名。")
        st.exception(e)

with st.expander("错题本（最近30条）", expanded=False):
    if st.session_state.mistakes:
        for i, m in enumerate(st.session_state.mistakes[-30:], 1):
            st.write(f"{i}. [{m['type']}] {m['original']} → {m['corrected']}")
    else:
        st.write("暂无。")
