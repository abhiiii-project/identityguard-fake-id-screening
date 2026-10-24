"""Write demo cards to ./samples so you can try the web UI.

    python make_samples.py
Prints the matching form values for each card.
"""
import os
import random

from screening import forensics, synthetic as syn

rng = random.Random(2026)
os.makedirs("samples", exist_ok=True)
name, dob = "Priya Sharma", "14/03/1994"
aad, pan = syn.valid_aadhaar(rng), "ABCPS1234D"
card = syn.make_card(name, dob, aad, pan, seed=11)

open("samples/genuine_card.jpg", "wb").write(syn.to_jpeg(card, 85))
open("samples/tampered_dob.jpg", "wb").write(
    syn.to_jpeg(syn.tamper_text(forensics.load_image(syn.to_jpeg(card, 70)), "01/01/1980"), 85))
open("samples/cloned_region.jpg", "wb").write(
    syn.to_jpeg(syn.tamper_clone(forensics.load_image(syn.to_jpeg(card, 85))), 85))
open("samples/edited_in_photoshop.jpg", "wb").write(syn.to_jpeg(card, 85, software="Adobe Photoshop 25.0"))

print(f"Form values for the cards:\n  name={name}\n  dob={dob}\n  aadhaar={aad}\n  pan={pan}")
print("Files written to ./samples")
