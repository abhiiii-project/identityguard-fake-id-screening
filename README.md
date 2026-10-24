# IdentityGuard – AI-Based Fake Identity & Document Screening

Screens an applicant's declared data plus an uploaded ID image and returns an
**explainable** risk score (0-100) and a decision: `APPROVE`, `MANUAL_REVIEW` or `REJECT`.

## What it checks

| Layer | Checks |
|---|---|
| Structural validation | Aadhaar Verhoeff checksum, PAN format/holder-type/surname letter, passport MRZ (ICAO 9303 TD3) check digits, DOB / issue / expiry sanity |
| Synthetic-identity signals | placeholder / gibberish names, disposable e-mail domains, random-looking e-mails, invalid or patterned phone numbers, PIN format |
| Image forensics | edit-software metadata, edge-normalised error-level analysis, noise inconsistency, copy-move detection verified by duplicated sensor noise, blur / resolution |
| Learned model | gradient-boosted tamper classifier over the forensic feature vector |
| OCR cross-check | Tesseract reads name / DOB / Aadhaar / PAN from the image and compares with what was typed (OCR-tolerant) |
| History | salted-hash store detects the same Aadhaar/PAN under a different identity, shared phone/e-mail, re-used document images, submission velocity |

Scores combine findings with a noisy-OR, so independent signals reinforce each
other. Any single *critical* finding (e.g. failed Aadhaar checksum) forces `REJECT`.

## Run

```bash
pip install -r requirements.txt        # also needs the tesseract binary for OCR
python train_model.py 80               # optional: (re)train the demo ML model
python make_samples.py                 # writes demo cards to ./samples
python app.py                          # http://localhost:5000
python tests/test_screening.py         # 16 tests
```

API:

```bash
curl -F name="Priya Sharma" -F dob=14/03/1994 -F aadhaar=358813998697 -F pan=ABCPS1234D \
     -F phone=9822012345 -F email=priya@example.com -F document=@samples/genuine_card.jpg \
     http://localhost:5000/api/screen
```

## Important limitations (read before production use)

* The ML model and the pixel-forensics thresholds were calibrated on **synthetic demo cards**. The
  near-perfect score on the synthetic test set does **not** predict real-world accuracy. Retrain
  `train_model.py` on your own labelled genuine/fraudulent submissions and re-tune thresholds on real scans.
* Checksums prove a number is *well-formed*, not that it belongs to the applicant. Real verification needs
  the issuer's official service (e.g. DigiLocker / UIDAI / NSDL APIs) under the applicable law.
* AI-generated or fully re-rendered documents can pass pixel forensics. Treat this as a triage layer
  that routes risky cases to a human, not as sole decision-maker.
* Face detection is informational only (OpenCV 5 removed the Haar API); add a face-match / liveness
  model if you need selfie-to-ID verification.
* Set `IDSCREEN_SALT` to a secret value; store only hashes, apply retention limits, and follow your
  data-protection obligations (DPDP Act in India) for any real applicant data.
