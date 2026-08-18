# Architecture Decision — Batch vs Real-Time

## Current Architecture: BATCH

```
Video chunks (3GB each) → Kaggle notebook → Process offline → REPORT.md + Excel
```

- Processing happens AFTER the footage is recorded
- One notebook run = one chunk = ~1 hour of footage
- Results available next day (or whenever someone runs it)
- No alerts, no live feedback

---

## What Would Change for Real-Time

| Aspect | Batch (current) | Real-time |
|--------|----------------|-----------|
| Latency | Next-day | < 60 seconds |
| Infrastructure | Kaggle (free GPU) | Dedicated server with GPU 24/7 |
| Cost | ~$0 (free tier) | $200-500/month (cloud GPU) |
| Complexity | One script | Streaming pipeline + state management + alert system |
| Reliability | Manual trigger | Must never crash, auto-restart, monitoring |
| Alert example | "Last night desk was empty 3x" | "Desk has been empty for 10 minutes RIGHT NOW" |

---

## Decision: Stay Batch, Design for Real-Time Later

**Reasons:**
1. No business requirement for real-time alerts yet
2. Batch lets us iterate faster (change code → re-run → compare)
3. Accuracy must be solved first — a real-time system that counts wrong is worse than no system
4. The triage planner (Ph5a) already processes segments, which is the same primitive a streaming system would use

**What we do now to not block real-time later:**
- Keep processing functions stateless (they already are)
- Triage planner works on segments, not whole files
- The kevacv package is importable — a streaming wrapper just calls the same functions
- Profile/config is data, not notebook cells

**When to revisit:**
- When someone says "I need to know RIGHT NOW if the desk is empty"
- When there are multiple cameras that need coordinated processing
- After accuracy targets are met in batch mode

---

## The Honest One-Liner

Real-time is a deployment problem, not an accuracy problem. Solve accuracy first in batch, then deploy the same logic as a stream. The code doesn't need to change; the infrastructure does.
