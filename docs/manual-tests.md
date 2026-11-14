# Milestone 4 manual acceptance checklist

Record tester, date, submitted commit, machine, and artifact versions here.
Replace Not run with actual observations and Pass/Fail. Blocked items remain unmet.

| ID | Scenario and steps | Expected result | Actual result | Status |
| --- | --- | --- | --- | --- |
| M01 | Start with documented Docker command; connect a local client; submit a valid PNG. | Backend returns score and availability fields. | No execution evidence yet. | Not run |
| M02 | Send malformed bytes, then send a valid image on the same connection. | Clear input error followed by a valid response. | No execution evidence yet. | Not run |
| M03 | Connect a fresh stream; send its first valid frame. | Learned/fused score is unavailable with a reason; rule score remains separately available when calculated. | No execution evidence yet. | Not run |
| M04 | Send consecutive frames from a representative clip. | Tracking state and responses continue; no unexpected crash. | No execution evidence yet. | Not run |
| M05 | Run local camera demonstration with matching artifacts; inspect objects, IDs and status text; quit with q. | Display updates; exit works. | No execution evidence yet. | Not run |
| M06 | Save an event through the UI; restart services; reload the event. | Stored information survives restart and appears in the UI. | UI/persistence unfinished in reviewed checkout. | Blocked |
| M07 | Clone on another member's computer; copy .env.example; run docker compose up --build. | Entire application starts from documented prerequisites. | Compose/configuration unfinished in reviewed checkout. | Blocked |
