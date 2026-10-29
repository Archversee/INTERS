"""Write the demo manuals (fictional organisations) as PDF and plain text, with no PDF library.

    python samples/manuals/make_manuals.py

Edit the texts below and run it again to remake the files.
"""
import textwrap
from pathlib import Path

OUT = Path(__file__).parent
OUT.mkdir(parents=True, exist_ok=True)

LAND_PDF = [  # pages of (style, text): "t" title, "h" heading, "p" paragraph, "s" small print
    [("t", "NORTHFIELD DRIVING CENTRE"),
     ("s", "Fictional organisation - sample document for demonstration only."),
     ("h", "Parallel Parking - Test Standards, Revision 3"),
     ("h", "1. Purpose"),
     ("p", "This document sets the standards examiners apply to the parallel parking manoeuvre on the "
           "Northfield test circuit. Candidates should practise the manoeuvre until it is smooth and controlled."),
     ("h", "2. Approach"),
     ("p", "Drive towards the parking bay at a steady speed. Approach the parked car at no more than 20 km/h."),
     ("p", "Switch on the left indicator before you stop alongside the parked car."),
     ("p", "Stop with your car level with the parked car, about half a metre to one metre away from it."),
     ("h", "3. Reversing into the bay"),
     ("p", "Select reverse gear and check all round, looking over both shoulders before you move."),
     ("p", "Reverse at walking pace. Your speed must not exceed 4 km/h while reversing."),
     ("p", "Do not reverse more than 10 m from where you stopped alongside the parked car.")],
    [("h", "4. Steering"),
     ("p", "Steer fully to the left when the rear of your car is level with the rear of the parked car. "
           "Straighten the wheels at about 45 degrees, then steer fully to the right to bring the front in."),
     ("p", "The car must never be more than 55 degrees to the kerb at any point while reversing."),
     ("h", "5. Finishing position"),
     ("p", "The car must finish parallel to the kerb, within 3 degrees."),
     ("p", "Apply the handbrake before you release the foot brake."),
     ("p", "Cancel the indicator once the car is parked."),
     ("p", "A good finishing position is normally about 30 cm from the kerb."),
     ("h", "6. Assessment"),
     ("p", "The examiner may stop the test at any time if the manoeuvre becomes unsafe.")],
]

AIR_PDF = [
    [("t", "RIVERSIDE AERO CLUB"),
     ("s", "Fictional organisation - sample document for demonstration only."),
     ("h", "Cessna 172 Circuit Procedures, Issue 2"),
     ("h", "1. Before take-off"),
     ("p", "Complete the before take-off checks at the holding point."),
     ("p", "The strobe and the landing light must be ON before entering the runway."),
     ("p", "Rotate at about 55 KIAS and climb at 70 to 80 KIAS."),
     ("h", "2. Climb and crosswind"),
     ("p", "Do not turn crosswind below 500 ft AGL."),
     ("h", "3. Circuit height"),
     ("p", "Circuits are flown at 1,000 ft AGL. Never exceed 1,100 ft AGL on the downwind leg.")],
    [("h", "4. Flaps"),
     ("p", "Do not select flaps beyond 10 degrees above 85 KIAS."),
     ("p", "The maximum speed with 10 degrees of flap is 110 KIAS."),
     ("h", "5. Airspeed limits"),
     ("p", "Never exceed 163 KIAS."),
     ("h", "6. After landing"),
     ("p", "Retract the flaps once clear of the runway."),
     ("p", "Set the parking brake before shutdown."),
     ("p", "Taxi slowly and keep a good lookout for other aircraft.")],
]

LAND_TXT = """NORTHFIELD DRIVING CENTRE
Fictional organisation - sample document for demonstration only.

Examiner briefing notes - parallel parking

General
Be calm and give clear instructions. Allow the candidate time to observe before each move.

Speed on the circuit
The maximum speed anywhere on the test circuit is 30 km/h.

Moving off
The candidate must release the handbrake before moving off.

Observation
Watch whether the candidate checks the mirrors and blind spots, and note any hesitation.
"""

AIR_TXT = """RIVERSIDE AERO CLUB
Fictional organisation - sample document for demonstration only.

Instructor notes - circuit training

Approach
On final approach, the rate of descent must not exceed 800 fpm.

General advice
Encourage students to look outside and to call each turn. Keep the circuit tidy and predictable,
and debrief every circuit while it is fresh.
"""

STYLE = {"t": ("F2", 18, 26), "h": ("F2", 13, 22), "p": ("F1", 11, 15), "s": ("F1", 9, 14)}


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, pages) -> None:
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", None,
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"]
    kids = []
    for page in pages:
        y, ops = 790, []
        for style, text in page:
            font, size, lead = STYLE[style]
            if style == "h":
                y -= 8
            for line in textwrap.wrap(text, int(475 / (size * 0.5))):
                ops.append(f"BT /{font} {size} Tf 60 {y} Td ({_esc(line)}) Tj ET")
                y -= lead
            y -= 4
        stream = "\n".join(ops).encode("latin-1")
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        content_id = len(objs)
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                    b"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents %d 0 R >>" % content_id)
        kids.append(len(objs))
    objs[1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (" ".join(f"{k} 0 R" for k in kids).encode(), len(kids))
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    path.write_bytes(bytes(out))


write_pdf(OUT / "land_northfield_parking_test_standards.pdf", LAND_PDF)
write_pdf(OUT / "air_riverside_circuit_procedures.pdf", AIR_PDF)
(OUT / "land_northfield_examiner_notes.txt").write_text(LAND_TXT, encoding="utf-8")
(OUT / "air_riverside_instructor_notes.txt").write_text(AIR_TXT, encoding="utf-8")
print("written:", sorted(p.name for p in OUT.glob("*.*") if p.suffix in (".pdf", ".txt")))
