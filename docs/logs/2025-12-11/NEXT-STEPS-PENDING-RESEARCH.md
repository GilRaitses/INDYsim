# Next Steps - Pending Research Clarification

## Current Status

The hazard model is validated and documented. Pre-simulation validation is complete.

| Completed | Result |
|-----------|--------|
| Matched validation (all events) | Rate ratio = 0.97 PASS |
| Time-rescaling test | p < 0.001 (mild fail) |
| Event filtering | 319 / 1,407 = 23% are "true turns" |
| Calibration documentation | Rate normalization explained |

## Pending Decision Points

The following require strategic input before proceeding:

### 1. Event Definition (BLOCKING)

**Decision needed**: Use all 1,407 events or refit on 319 filtered events?

| Option | Pros | Cons |
|--------|------|------|
| All events | Validated, no refitting | 77% are micro-movements |
| Filtered only | Biologically meaningful | Requires refitting, fewer events |
| Post-hoc subsample | No refitting | Arbitrary, may miss temporal structure |

**Impact**: Affects all downstream simulation and publication framing.

### 2. Trajectory Simulation Scope (BLOCKING)

**Decision needed**: What level of trajectory simulation is required?

| Level | Components | Effort |
|-------|------------|--------|
| Minimal | Event times only (current) | Done |
| Basic | + Turn angles from empirical distribution | 1 day |
| Full | + Run kinematics, speed, edge avoidance | 3-5 days |
| Complete | + Head sweeps, reversals, pauses | 1-2 weeks |

**Impact**: Determines scope of deliverables and paper framing.

### 3. Scientific Direction (IMPORTANT)

**Decision needed**: What is the primary contribution?

| Direction | Key Figure | Effort | Impact |
|-----------|------------|--------|--------|
| Methods | Kernel shape + validation metrics | Low | Medium |
| Biology | Timescale interpretation + circuit comparison | Medium | High |
| Platform | Full trajectory simulation + demos | High | High |

### 4. Time-Rescaling Response (LOW PRIORITY)

**Decision needed**: Accept mild Poisson violation or add refractory?

- Current deviation: 13% (mean 0.87 vs expected 1.0)
- Adding refractory: +1-2 parameters, likely marginal improvement
- Recommendation: Accept and report transparently

### 5. Condition Generalization (FUTURE)

**Decision needed**: Attempt generalization or stay condition-specific?

- Current: 0→250 PWM, 10s ON / 20s OFF
- Available: 50→250 PWM, temperature conditions
- Recommendation: Defer to future work, note as limitation

---

## Proposed Execution Order

Pending research clarification, the likely sequence is:

```
1. RESEARCH RESPONSE (external)
   └── Clarify event definition strategy
   └── Clarify trajectory scope
   └── Clarify scientific direction

2. EVENT DECISION (1 hour)
   └── If refit: Rerun GLM on 319 filtered events
   └── If keep: Document limitation

3. TRAJECTORY SIMULATION (1-3 days)
   └── Extract turn angle distribution from data
   └── Implement basic RUN/TURN state machine
   └── Validate trajectory statistics

4. PUBLICATION PREP (1-2 days)
   └── Generate key figures
   └── Draft methods section
   └── Prepare supplementary materials

5. CONDITION GENERALIZATION (optional, 2-3 days)
   └── Test intensity scaling hypothesis
   └── Compare across conditions
```

---

## Files for Research Agent

The research prompt is at:
```
docs/logs/2025-12-11/RESEARCH-PROMPT-NEXT-STEPS-STRATEGY.md
```

Supporting context:
```
docs/MODEL_SUMMARY.md                    # Full model specification
data/validation/matched_validation.json  # Validation metrics
data/validation/time_rescaling.json      # Time-rescaling results
```

---

## Questions Awaiting Answer

1. Which event definition strategy is standard?
2. What is the minimal viable trajectory model?
3. Which scientific direction has highest impact-to-effort?
4. Is 13% time-rescaling deviation acceptable?
5. Should we attempt condition generalization?

