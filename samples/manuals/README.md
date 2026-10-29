# Demo manuals for "Read a manual with AI"

Four short official documents from **fictional** organisations, written to show the AI reading a manual on the
*Official documents* page. They have no limits list, so the rules have to be found by the AI. Each mixes hard
rules with advice the AI should leave out, and a few sentences that tempt a wrong reading, so the checks have
something to catch.

| File | Procedure | What is in it |
|---|---|---|
| `land_northfield_parking_test_standards.pdf` | Parallel parking | 2 pages: speed limits, a distance limit, an angle limit, required actions, and advice ("about 45 degrees", "normally about 30 cm") |
| `land_northfield_examiner_notes.txt` | Parallel parking | A circuit speed limit and one required action among general advice |
| `air_riverside_circuit_procedures.pdf` | Cessna 172 circuit | 2 pages: circuit height, flap and airspeed limits (with "1,100 ft" and "10 degrees" of flap), required lights, flaps up, parking brake, and advice ("about 55 KIAS") |
| `air_riverside_instructor_notes.txt` | Cessna 172 circuit | A descent-rate limit on final among general advice |

## How to demo

1. Turn the AI writer on in the sidebar and pick the matching procedure.
2. *Load sample data* on Home first, so the review can show how many recorded runs would break each proposed rule.
3. *Official documents* → **Read a manual with AI** → upload a file → **Find the rules with AI** (about 10–15 s for a
   PDF with the network Ollama, `qwen2.5:7b-instruct`).
4. Walk through the table: each rule quotes the manual; the "Recorded … breaking it" column checks it against the
   data; open **Set aside by the checks** to show what was refused and why.
5. **Add the manual with N rules**: the guidelines are re-checked and the review pops up.

## What to expect (qwen2.5:7b-instruct, 29 Oct 2026)

Runs vary a little; this is one real run.

**Aviation PDF**: all nine kept rules correct.

- Height AGL max 1,100 ft on downwind; airspeed max 163 kt; airspeed max 110 kt with flaps 10°; flaps no further
  than 10° above 85 kt. That last one is broken by **1 of 5** sample flights, the deliberate overspeed flight.
- Required: strobe on, landing light on, flaps up, parking brake set.
- "Circuits are flown at 1,000 ft AGL" proposed as a minimum is broken by 5 of 5 flights, so it is **unticked**
  automatically.

**Parking PDF**: 8 of 10 right.

- Speed max 20 km/h on the approach, max 4 km/h reversing (broken by the fast sample drive), distance reversed max
  10 m, and the required indicator, handbrake and cancel-signal actions.
- "55 degrees to the kerb" put on steering angle instead of angle to the kerb is broken by 5 of 5 drives, so it is
  **unticked**.
- "Straighten the wheels at about 45 degrees" is **set aside** as a guide.

**Text files**: circuit speed max 30 km/h and "release the handbrake" (parking); descent rate on final
(aviation). The descent rate comes out as vertical speed max 800 fpm instead of min −800. Nothing breaks it, so
only the reviewer can catch it.

## What the checks do

| Check | Catches |
|---|---|
| The quote must be in the manual word for word | invented rules |
| The value must be written in the quote (a flap position by its name, "1,100" with its comma) | converted or guessed numbers |
| Wording sets the direction ("no more than" = max, "never below" = min; a negative value flips it) | a maximum turned into a minimum |
| "about", "normally", "should" … in the quote | advice taken as a hard rule |
| An acronym in the quote must match the signal ("ft AGL" is height AGL) | wrong signal; repaired when the quote's unit points to one signal |
| Phases must be named in the quote; a phase written as a condition moves to the phases | rules narrowed or misplaced |
| Recorded runs that would break it ("5 of 5" is unticked) | a wrong signal or direction the text checks cannot see |

To change the texts, edit `make_manuals.py` and run `python samples/manuals/make_manuals.py`.
