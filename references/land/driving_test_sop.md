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
requirements:
  - id: signal_left
    key: "toggle:indicator_left:1"
    section: "Signalling"
    text: "Signal left before stopping to park"
  - id: handbrake_on
    key: "toggle:hand_brake:1"
    section: "Securing the car"
    text: "Apply the handbrake once parked"
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

## Securing the car
Once the car is parallel to the kerb and stopped, apply the handbrake before releasing the
foot brake. Cancel the signal.
