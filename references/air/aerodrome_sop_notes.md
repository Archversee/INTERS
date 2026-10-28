---
id: aerodrome_sop
title: Aerodrome circuit SOP (sample)
kind: sop
revision: "2026-1"
limits:
  - id: circuit_height_max
    signal: agl_ft
    max: 1100
    phases: [crosswind, downwind]
    section: "Circuit height"
    text: "Circuit height 1000 ft AGL; do not exceed 1100 ft AGL"
  - id: crosswind_min_height
    signal: agl_ft
    min: 500
    phases: [crosswind]
    at: trigger
    section: "Crosswind turn"
    text: "Do not turn crosswind below 500 ft AGL"
requirements:
  - id: strobe_on_runway
    key: "toggle:light_strobe:1"
    section: "Runway entry"
    text: "Strobe on before entering the runway"
  - id: landing_light_takeoff
    key: "toggle:light_landing:1"
    section: "Runway entry"
    text: "Landing light on for take-off"
  - id: parking_brake_shutdown
    key: "toggle:parking_brake:1"
    section: "After landing"
    text: "Parking brake set before shutdown"
---
# Aerodrome circuit SOP (sample)

> SAMPLE. A made-up aerodrome SOP for development. Replace with the real aerodrome or
> school SOP. Lines starting with ">" are not indexed.

## Runway entry
Before entering the runway: strobe on, landing light on, transponder ALT. Line up on the
centreline and check the windsock.

## Crosswind turn
Climb straight ahead on the runway heading. Do not turn crosswind below 500 ft AGL.

## Circuit height
Circuits are flown left-hand at 1000 ft AGL. Do not exceed 1100 ft AGL in the circuit.

## Downwind
Fly the downwind leg parallel to the runway. Complete the before-landing checks abeam the
threshold.

## After landing
Vacate the runway at the first suitable exit, flaps up, landing light and strobe off.
Taxi to parking and set the parking brake before shutdown.
