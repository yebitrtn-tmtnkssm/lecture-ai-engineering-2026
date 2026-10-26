"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def ask(question: str, temperature: float | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    # response = client.models.generate_content(model=MODEL, contents=question)
    if temperature is None:
        response = client.models.generate_content(model=MODEL, contents=question)
    else:
        response = client.models.generate_content(
            model=MODEL,
            contents=question,
            config={"temperature": temperature},
        )

    # TODO(第1回 演習): response.usage_metadata から
    #   入力トークン数と出力トークン数を表示してみよう（Copilotに頼んでよい）
    usage = response.usage_metadata
    if usage is None:
        print("[トークン数] 使用量情報を取得できませんでした")
    else:
        prompt_tokens = usage.prompt_token_count
        response_tokens = usage.candidates_token_count
        print(f"[入力トークン数] {prompt_tokens if prompt_tokens is not None else '不明'}")
        print(f"[出力トークン数] {response_tokens if response_tokens is not None else '不明'}")

    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--temperature",
        type=float,
        help="生成時のtemperature（省略時はモデル/APIのデフォルト）",
    )
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    temperature_label = (
        str(args.temperature)
        if args.temperature is not None
        else "省略（モデル/APIのデフォルト）"
    )
    print(f"[temperature] {temperature_label}")
    print(f"[質問] {args.question}\n")
    try:
        print(ask(args.question, temperature=args.temperature))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
