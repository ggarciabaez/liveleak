# Manual test checklist (Milestone 4)

Start the backend (`docker compose up --build`) and the desktop app (`python -m app.app`) first.
Fill in **Result** with Pass or Fail, plus date and tester.

| # | Scenario                        | Steps                                                            | Expected result                                                                       | Result |
|---|---------------------------------|------------------------------------------------------------------|---------------------------------------------------------------------------------------|--------|
| 1 | Start a session on a video file | Source = video path; press Start Session                         | Status turns LIVE MONITORING; frames show boxes, IDs and PPE tags; risk gauge updates |        |
| 2 | Person without PPE              | Play footage with a person lacking hardhat or vest               | Person tag turns red; PPE Compliance rule pill lights; DetNG score rises              |        |
| 3 | Empty scene                     | Play footage with no objects                                     | Status shows NO_DETECTIONS; risk shows `---`, not 0; app keeps running                |        |
| 4 | Invalid input                   | Enter a bad video path; press Start Session                      | ERROR badge "Failed to open video source"; app stays usable                           |        |
| 5 | Backend down                    | Stop the backend; press Start Session                            | ERROR badge "Connection failed"; no crash                                             |        |
| 6 | Save and reload                 | Run a session, stop it, open Retroactive tab, select the session | Session listed; timeline, histogram and incident log match; CSV export has the rows   |        |
| 7 | Restart persistence             | Close and reopen the app                                         | Previous sessions still load from `risk.db`                                           |        |
