---
id: driving_test_sop
title: Driving test — parallel parking (sample)
kind: sop
revision: "2026-1"
limits:
  - id: reverse_speed_max
    signal: speed_kmh
    max: 5
    when: "reverse == 1"
    phases: [reverse_in]
    section: "Reversing speed"
    text: "Reverse at walking pace, no faster than 5 km/h"
  - id: circuit_speed_max
    signal: speed_kmh
    max: 25
    phases: [approach]
    section: "Test circuit speed"
    text: "Do not exceed 25 km/h on the test circuit"
  - id: kerb_angle_max
    signal: kerb_angle_deg
    max: 50
    phases: [reverse_in]
    section: "Angle to the kerb"
    text: "Do not swing the car more than 50° to the kerb"
  - id: reverse_distance_max
    signal: back_m
    max: 11
    phases: [reverse_in]
    section: "Space behind"
    text: "Do not reverse more than 11 m from where you stopped (the car behind)"
  - id: final_parallel
    signal: kerb_angle_deg
    min: -5
    max: 5
    phases: [secure]
    section: "Final position"
    text: "Finish parallel to the kerb, within 5°"
requirements:
  - id: signal_left
    key: "toggle:indicator_left:1"
    section: "Signalling"
    text: "Signal left before stopping to park"
  - id: handbrake_on
    key: "toggle:hand_brake:1"
    section: "Securing the car"
    text: "Apply the handbrake once parked"
  - id: signal_off
    key: "toggle:indicator_left:0"
    section: "Securing the car"
    text: "Cancel the signal once parked"
---
# Driving test — parallel parking (sample)

> SAMPLE. A made-up test-circuit SOP for development. Replace with the driving centre's
> own test rules. Lines starting with ">" are not indexed.

## Signalling
Signal left in good time before slowing down to stop alongside the parked car, and keep
the signal on until the car is parked.

## Reversing speed
Reverse at walking pace, no faster than 5 km/h, so the car can be stopped at once if a
pedestrian or vehicle appears.

## Test circuit speed
Do not exceed 25 km/h anywhere on the test circuit.

## Angle to the kerb
When reversing in, do not let the car swing more than 50° to the kerb: the front swings out
into the road and the rear wheel may hit the kerb.

## Space behind
The space behind the stop point is limited by the car parked behind. Do not reverse more than
11 m from where you stopped alongside the parked car.

## Final position
Finish parallel to the kerb, within 5°. A car left at an angle fails the manoeuvre.

## Securing the car
Once the car is parallel to the kerb and stopped, apply the handbrake before releasing the
foot brake. Cancel the signal.
