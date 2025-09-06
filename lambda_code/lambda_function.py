# lambda_function.py
import os
import json
import logging
import urllib.request
import urllib.error
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ===== Settings =====
BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "us-east-1")
MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
LINE_ACCESS_TOKEN = os.environ["LINE_ACCESS_TOKEN"]
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "5"))
LINE_MAX = 5000

# ===== Clients =====
bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
        ev = (body.get("events") or [None])[0] or {}
        reply_token = ev.get("replyToken")
        user_text = ((ev.get("message") or {}).get("text") or "").strip()
        if not reply_token or not user_text:
            return _ok({"message": "no text"})

        # Claude 3.5 メッセージ形式で1回だけ呼び出し
        ai_text = _ask_bedrock(user_text)
        _reply_line(reply_token, ai_text[:LINE_MAX])
        return _ok({"message": "ok"})
    except Exception as e:
        logger.exception("Unhandled")
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }

def _ask_bedrock(user_text: str) -> str:
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ],
    }
    resp = bedrock.invoke_model(
        modelId=MODEL_ID,
        accept="application/json",
        contentType="application/json",
        body=json.dumps(payload),
    )
    data = json.loads(resp["body"].read())
    content = data.get("content") or []
    return (content[0].get("text") if content and isinstance(content[0], dict) else "（応答なし）")

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

def _ok(payload: dict):
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload),
    }
