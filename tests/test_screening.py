import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from screening import ScreeningPipeline, forensics, synthetic as syn, validators as v
from screening.store import HistoryStore

rng = random.Random(42)


def codes(findings):
    return {f.code for f in findings}


class TestValidators(unittest.TestCase):
    def test_aadhaar_valid_roundtrip(self):
        for _ in range(50):
            self.assertEqual(v.validate_aadhaar(syn.valid_aadhaar(rng)), [])

    def test_aadhaar_bad_checksum(self):
        a = syn.valid_aadhaar(rng)
        bad = a[:-1] + str((int(a[-1]) + 1) % 10)
        self.assertIn("AADHAAR_CHECKSUM", codes(v.validate_aadhaar(bad)))

    def test_aadhaar_format_and_pattern(self):
        self.assertIn("AADHAAR_FORMAT", codes(v.validate_aadhaar("1234")))
        self.assertIn("AADHAAR_PATTERN", codes(v.validate_aadhaar("222222222222")))

    def test_pan(self):
        self.assertEqual(v.validate_pan("ABCPS1234D", "Sharma"), [])
        self.assertIn("PAN_FORMAT", codes(v.validate_pan("12345ABCDE")))
        self.assertIn("PAN_HOLDER_TYPE", codes(v.validate_pan("ABCZS1234D")))
        self.assertIn("PAN_SURNAME_MISMATCH", codes(v.validate_pan("ABCPK1234D", "Sharma")))

    def test_mrz_icao_specimen(self):
        # ICAO 9303 specimen (Anna Maria Eriksson)
        l1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
        l2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
        self.assertEqual(v.validate_mrz_td3(l1, l2), [])
        tampered = l2[:13] + "7408132" + l2[20:]
        self.assertIn("MRZ_CHECKSUM", codes(v.validate_mrz_td3(l1, tampered)))

    def test_dates(self):
        self.assertIn("DOB_FUTURE", codes(v.validate_dates("01/01/2999")))
        self.assertIn("UNDERAGE", codes(v.validate_dates("01/01/2020")))
        self.assertIn("EXPIRY_BEFORE_ISSUE", codes(v.validate_dates("01/01/1990", "01/01/2020", "01/01/2019")))


class TestForensics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.card = syn.make_card("Priya Sharma", "14/03/1994", syn.valid_aadhaar(rng), "ABCPS1234D", seed=5)
        cls.genuine = syn.to_jpeg(cls.card, 85)

    def test_genuine_has_no_high_findings(self):
        _, f, _ = forensics.analyse_image(self.genuine)
        self.assertFalse([x for x in f if x.severity in ("high", "critical")], [x.code for x in f])

    def test_editing_software_detected(self):
        data = syn.to_jpeg(self.card, 85, software="Adobe Photoshop 25.0")
        _, f, _ = forensics.analyse_image(data)
        self.assertIn("EDITING_SOFTWARE", codes(f))

    def test_clone_detected(self):
        bad = syn.to_jpeg(syn.tamper_clone(forensics.load_image(self.genuine)), 85)
        _, f, _ = forensics.analyse_image(bad)
        self.assertIn("COPY_MOVE", codes(f))

    def test_dhash_stable_under_recompression(self):
        a = forensics.dhash(forensics.load_image(self.genuine))
        b = forensics.dhash(forensics.load_image(syn.to_jpeg(self.card, 60)))
        self.assertLessEqual(forensics.hamming(a, b), 10)


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipe = ScreeningPipeline(store=HistoryStore(os.path.join(self.tmp, "t.db")))
        self.aad = syn.valid_aadhaar(rng)
        self.pan = "ABCPS1234D"
        self.good = {"name": "Priya Sharma", "dob": "14/03/1994", "aadhaar": self.aad, "pan": self.pan,
                     "phone": "9822012345", "email": "priya.sharma@example.com", "pincode": "431605"}

    def test_clean_application_approved(self):
        card = syn.to_jpeg(syn.make_card("Priya Sharma", "14/03/1994", self.aad, self.pan, seed=9), 85)
        r = self.pipe.screen(self.good, card)
        self.assertEqual(r.decision, "APPROVE", [f.code for f in r.findings])

    def test_fake_identity_rejected(self):
        fake = dict(self.good, name="test user 99", dob="01/01/2015", aadhaar="123456789012",
                    email="x8k2q9z7w1@mailinator.com", phone="9999999999")
        r = self.pipe.screen(fake)
        self.assertEqual(r.decision, "REJECT")
        self.assertGreater(r.risk_score, 80)

    def test_tampered_doc_scores_higher_and_dob_mismatch_found(self):
        card = syn.make_card("Priya Sharma", "14/03/1994", self.aad, self.pan, seed=9)
        genuine = syn.to_jpeg(card, 85)
        tampered = syn.to_jpeg(syn.tamper_text(forensics.load_image(syn.to_jpeg(card, 70)), "01/01/1980"), 85)
        g = self.pipe.screen(self.good, genuine, record=False)
        t = self.pipe.screen(self.good, tampered, record=False)
        self.assertGreater(t.risk_score, g.risk_score)
        self.assertNotEqual(t.decision, "APPROVE")

    def test_identity_reuse_detected(self):
        self.pipe.screen(self.good)
        thief = dict(self.good, name="Rahul Verma", dob="02/02/1990", phone="9822099999", email="rahul@example.com")
        r = self.pipe.screen(thief)
        self.assertIn("AADHAAR_REUSED", codes(r.findings))
        self.assertEqual(r.decision, "REJECT")

    def test_same_person_resubmitting_is_not_flagged(self):
        self.pipe.screen(self.good)
        r = self.pipe.screen(self.good)
        self.assertNotIn("AADHAAR_REUSED", codes(r.findings))

    def test_garbage_upload(self):
        r = self.pipe.screen(self.good, b"not an image", record=False)
        self.assertIn("IMAGE_UNREADABLE", codes(r.findings))


if __name__ == "__main__":
    unittest.main(verbosity=2)
