1. Looks like fraud
SIM was swapped 6 minutes after an OTP request. New device, IMSI and ICCID both changed, and the SIM type changed to eSIM. Location similarity is 0.03. The SIM was activated 1 day ago, with 4 SIM changes in the last 30 days. The previous holder only had the number for 12 days. The account is 2 months old. IP address changed.
Expected: all 11 values are read, the case is flagged, Laya's gate says block (p = 1.0), and the score is high.

2. Looks normal
Customer replaced a damaged SIM in a store about three days after the OTP. Same device, IMSI unchanged, ICCID changed because of the new card, same SIM type, same IP. Location similarity 95%. SIM activated 400 days ago, no other SIM changes. The previous holder kept the number for 5 years and the account is 8 years old.
Expected: all 11 values are read, the case is not flagged, and Laya's gate says allow (p = 0.000). The Isolation Forest anomaly shows 1.00 here because very long account and tenure values are unusual in the training data. LightGBM and Laya both rate it legitimate, so the case is not flagged.

3. Incomplete, to test the missing-values panel
SIM was changed 2 hours after the OTP. Device changed, IMSI unchanged. Location similarity 0.5. Two SIM changes in the last 30 days. The account is 3 years old.
Expected: "I understood 6 of the 11 values. 5 are still missing", and it asks for these five:
- SIM type changed
- ICCID changed
- how long ago the SIM was activated
- the previous holder's tenure
- whether the IP changed