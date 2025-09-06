import os
import json
import logging
import urllib.request
import urllib.error
import time
from decimal import Decimal
from typing import List, Dict, Any, Optional

import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ===== Settings =====
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
LINE_ACCESS_TOKEN = os.environ["LINE_ACCESS_TOKEN"]
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "5"))
DDB_TABLE_NAME = os.environ.get("DDB_TABLE_NAME", "chatbot-history")

# ===== Clients =====
bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(DDB_TABLE_NAME)

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
        ev = (body.get("events") or [None])[0] or {}

        reply_token = ev.get("replyToken")
        user_text = ((ev.get("message") or {}).get("text") or "").strip()

        # 会話のキー：基本はユーザー単位。グループ/ルームならそのIDを使うとスレッド共有になる
        source = ev.get("source") or {}
        user_id = source.get("userId")
        group_id = source.get("groupId")
        room_id  = source.get("roomId")
        conv_pk  = user_id or group_id or room_id or reply_token  # フォールバック

        if not reply_token or not user_text:
            return _ok({"message": "no text"})

        # これまでの全履歴をDynamoDBから読み出す（ページネーションで最後まで）
        history_turns = _load_all_history(conv_pk)

        # Anthropic messages に変換して、今回のユーザー発話を末尾に追加
        messages = _to_anthropic_messages(history_turns)
        messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})

        # Bedrock呼び出し（Anthropicはmax_tokens必須のため最小限を指定）
        ai_text = _ask_bedrock(messages)

        # 返信
        _reply_line(reply_token, ai_text)

        # 履歴保存（先にuser、続けてassistant）
        _save_message(conv_pk, role="user", text=user_text)
        _save_message(conv_pk, role="assistant", text=ai_text)

        return _ok({"message": "ok"})
    except Exception as e:
        logger.exception("Unhandled")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }

# ===== Bedrock =====
def _ask_bedrock(messages: List[Dict[str, Any]]) -> str:
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,  # Anthropic仕様上必須（数値は任意に調整可）
        "messages": messages,
    }
    resp = bedrock.invoke_model(
        modelId=MODEL_ID,
        accept="application/json",
        contentType="application/json",
        body=json.dumps(payload),
    )
    data = json.loads(resp["body"].read())
    content = data.get("content") or []
    if content and isinstance(content[0], dict):
        return content[0].get("text") or "（応答なし）"
    return "（応答なし）"

# ===== LINE Reply =====
def _reply_line(reply_token: str, text: str):
    url = "https://api.line.me/v2/bot/message/reply"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}",
    }
    payload = {
        "replyToken": reply_token,
        "messages": [{"type": "text", "text": text}]
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        _ = resp.read()

# ===== DynamoDB Helpers =====
def _now_ms() -> int:
    return int(time.time() * 1000)

def _save_message(pk: str, role: str, text: str):
    item = {
        "pk": pk,                           # 会話キー（ユーザー/グループ/ルーム）
        "sk": Decimal(str(_now_ms())),      # ソートキー（ミリ秒）
        "role": role,                       # "user" or "assistant"
        "text": text,
    }
    table.put_item(Item=item)

def _load_all_history(pk: str) -> List[Dict[str, Any]]:
    """
    指定pkの全件を古い→新しい順で返す。
    DynamoDBのQueryは1MBページングされるので、LastEvaluatedKeyで走り切る。
    """
    items: List[Dict[str, Any]] = []
    last_evaluated_key: Optional[Dict[str, Any]] = None

    while True:
        kwargs = {
            "KeyConditionExpression": Key("pk").eq(pk),
            "ScanIndexForward": True,   # 古い→新しい
        }
        if last_evaluated_key:
            kwargs["ExclusiveStartKey"] = last_evaluated_key

        resp = table.query(**kwargs)
        batch = resp.get("Items") or []
        items.extend(batch)

        last_evaluated_key = resp.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break

    # 返却フォーマットは [{"role":"user/assistant","text":"..."}...]
    return [{"role": i.get("role", "user"), "text": i.get("text", "")} for i in items]

def _to_anthropic_messages(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    for turn in history:
        role = turn.get("role") or "user"
        text = turn.get("text") or ""
        if not text:
            continue
        if role not in ("user", "assistant"):
            role = "user"
        messages.append({"role": role, "content": [{"type": "text", "text": text}]})
    return messages

# ===== HTTP Helper =====
def _ok(payload: dict):
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }
