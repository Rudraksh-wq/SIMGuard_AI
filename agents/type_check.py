import laya, time
CHECKPOINT = "english"   # <-- the one you picked in Step 2

QUESTIONS = {"is_fraud": {"type": "noul",
    "instructions": "Does this SIM replacement request look like a SIM-swap fraud attempt?"}}

fraud_text = ("SIM replacement request. IMEI changed: yes. Request hour: 03:12 (night). "
              "Location gap: 1200 km (very large). Failed authentications: 4 (high). "
              "Support calls in 24h: 2 (elevated). Days since last swap: 5 (very recent). "
              "OTP burst count: 6 (high).")

normal_text = ("SIM replacement request. IMEI changed: no. Request hour: 14:05 (day). "
               "Location gap: 3 km (small). Failed authentications: 0 (none). "
               "Support calls in 24h: 0 (none). Days since last swap: 400 (long ago). "
               "OTP burst count: 1 (normal).")

print("Loading Laya... (first time can take a few minutes)")
agent = laya.load(CHECKPOINT)

print("\n=== WAY 1: plain string ===")
try:
    print(agent.predict(fraud_text, QUESTIONS))
except Exception as e:
    print("WAY 1 FAILED:", e)

print("\n=== WAY 2: {'body': text} ===")
try:
    print(agent.predict({"body": fraud_text}, QUESTIONS))
except Exception as e:
    print("WAY 2 FAILED:", e)

print("\n=== NORMAL event (plain string) ===")
try:
    print(agent.predict(normal_text, QUESTIONS))
except Exception as e:
    print("FAILED:", e)
