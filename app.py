# app.py
# Flask server，接收 Twilio WhatsApp webhook

import logging
import traceback
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

# 設置日誌，方便調試
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 延遲載入 chatbot，確保 Flask 能優雅地處理導入錯誤
chatbot_available = False
try:
    from chatbot import handle_message
    chatbot_available = True
    logger.info("✓ Chatbot module loaded successfully")
except Exception as e:
    logger.error(f"✗ Failed to load chatbot module: {e}")
    traceback.print_exc()

app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_reply():
    """Twilio 每次收到訊息都會 POST 到這裡"""
    try:
        user_id      = request.form.get("From", "unknown")
        user_message = request.form.get("Body", "").strip()

        logger.info(f"[MESSAGE] {user_id}: {user_message}")

        if not chatbot_available:
            reply = "Bot initialization failed. Please try again later."
        elif not user_message:
            reply = "Sorry, I didn't catch that. What are you looking for?"
        else:
            try:
                reply = handle_message(user_id, user_message)
            except Exception as e:
                logger.error(f"[CHATBOT_ERROR] {user_id}: {e}")
                traceback.print_exc()
                reply = "Sorry, I encountered an error. Please try again."

        logger.info(f"[REPLY] {reply}")

        # 用 Twilio 的格式回覆
        resp = MessagingResponse()
        resp.message(reply)
        return str(resp), 200

    except Exception as e:
        logger.error(f"[WEBHOOK_ERROR] {e}")
        traceback.print_exc()
        # 確保 Twilio 收到有效的 TwiML 回覆
        resp = MessagingResponse()
        resp.message("Sorry, an unexpected error occurred.")
        return str(resp), 200

@app.route("/", methods=["GET"])
def health_check():
    """確認 server 正在運行"""
    status = "✅ Chatbot server is running!"
    if not chatbot_available:
        status = "⚠️ Server running but chatbot not loaded"
    return status, 200

@app.errorhandler(404)
def not_found(e):
    return "Not found", 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"[500 ERROR] {e}")
    return "Internal server error", 500

if __name__ == "__main__":
    print("🚀 啟動 Flask server，端口 5000...")
    if not chatbot_available:
        print("⚠️  WARNING: Chatbot not loaded. Check the console for errors.")
    app.run(debug=False, port=5000, host="0.0.0.0")
