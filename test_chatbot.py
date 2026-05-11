#!/usr/bin/env python
# test_chatbot.py
# 本地測試 chatbot，無需 Twilio

from chatbot import handle_message

def run_conversation(user_id, messages, label=""):
    print(f"\n{'=' * 60}")
    print(f"🤖 {label}")
    print("=" * 60)
    for i, msg in enumerate(messages, 1):
        print(f"\n[Turn {i}]")
        print(f"👤 User: {msg}")
        try:
            response = handle_message(user_id, msg)
            print(f"🤖 Bot: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    return True

def test_conversation():
    """模擬完整對話流程"""
    ok = True

    # ── Test 1: 基本流程 ──────────────────────────────────────
    ok &= run_conversation("test_user_001", [
        "I'm looking for cheap Italian food in Fishtown",
        "with outdoor seating",
        "yes that sounds great",
    ], "Test 1: Basic flow")

    # ── Test 2: noise_level keyword matching ──────────────────
    ok &= run_conversation("test_user_002", [
        "I want a quiet restaurant in Center City",
        "quiet",           # should fill noise_level via keyword matching
        "no preference",   # skip next slot
        "yes",
    ], "Test 2: noise_level keyword matching")

    # ── Test 3: noise_level skip ──────────────────────────────
    ok &= run_conversation("test_user_003", [
        "Looking for sushi in Old City",
        "skip",            # skip noise_level when asked
        "yes",
    ], "Test 3: noise_level skip")

    print("\n" + "=" * 60)
    if ok:
        print("✅ All tests completed successfully!")
    else:
        print("❌ Some tests failed.")
    print("=" * 60)
    return ok

if __name__ == "__main__":
    success = test_conversation()
    exit(0 if success else 1)
