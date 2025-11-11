# Daily Agent Handoff Protocol

## Daily Structure Setup

### Every New Day:

**1. Get System Date**
```bash
powershell -Command "Get-Date -Format 'yyyy-MM-dd'"
```

**2. Create Script Folder**
```bash
cd D:\mechanosensation\scripts
New-Item -ItemType Directory -Name "YYYY-MM-DD"
```

**3. Create Agent Handoffs Subfolder**
```bash
cd scripts/YYYY-MM-DD
New-Item -ItemType Directory -Name "agent-handoffs"
```

**4. Copy This Protocol**
```bash
cp ../previous-date/agent-handoffs/DAILY_PROTOCOL.md ./agent-handoffs/
```

**5. Create Daily Log**
```bash
# Create: docs/logs/YYYY-MM-DD.md
```

---

## File Naming Convention

**All handoff files:**
```
<author>_<recipient>_<timestamp>_<subject-4-5-words>.md

Where:
- author: Agent codename (lowercase-hyphenated)
- recipient: Target agent codename
- timestamp: YYYYMMDD-HHMMSS from system (NO hallucination!)
- subject: 4-5 descriptive words (lowercase-hyphenated)
```

**Examples:**
```
larry_osito-tender_20251020-091500_fix-rendering-performance-issue.md
conejo-code_larry_20251020-143000_test-results-all-passing.md
```

---

## Required Handoff Structure

Every handoff must include:

```markdown
# Handoff: <Subject>

**From:** <author>  
**To:** <recipient>  
**Date:** YYYY-MM-DD HH:MM:SS  
**Priority:** [High/Medium/Low]  
**Status:** [Pending/In Progress/Complete/Blocked]

## Context
[What led to this handoff]

## Task/Question/Report
[The actual content]

## Deliverables (if task)
[What needs to be produced]

## Questions (if inquiry)
[Specific questions needing answers]

## Results (if completion)
[What was accomplished]

## Next Steps
[What happens after this]

---

**<Author Name>** <emoji>  
**Date:** YYYY-MM-DD HH:MM:SS
```

---

## Daily Log Requirements

**File:** `docs/logs/YYYY-MM-DD.md`

**Must include:**
- Objective for the day
- Active agents and their tasks
- Completed work from previous day
- Current blockers
- Milestone status
- Next steps

**Template:**
```markdown
# Daily Log - Month DD, YYYY

## Objective
[What we're trying to accomplish today]

## Carryover from [Previous Date]
[Summary of where we left off]

## Active Agents
| Agent | Task | Status | Handoff |
|-------|------|--------|---------|
| ...   | ...  | ...    | ...     |

## Progress Today
[What got done]

## Blockers
[What's preventing progress]

## Milestone Status
M1: [status]
M2: [status]
...

## Next Steps
[What's next]

---

**Status:** [summary]  
**Next Session:** [what to tackle]
```

---

## Agent Roster (Current)

**Senior:**
- 🎖️ **larry** - Architecture, coordination, review

**Specialists:**
- 🐻 **osito-tender** - MATLAB class structure
- 🐱 **gatito-cheer** - Figma UI/UX design
- 🐰 **conejo-code** - Python/MCP backend
- 🐦 **pajaro-bright** - Electron frontend
- 🦋 **mari-test** - Testing & QA

---

## Status Tracking

**File:** `agent-handoffs/STATUS.md` (updated daily)

Track all active handoffs:
```markdown
| From | To | Subject | Status | File |
|------|----|---------
|--------|------|
| ...  | ...| ...     | ...    | ...  |
```

---

## Cross-Day References

**Referencing previous days:**
```markdown
See: scripts/2025-10-16/agent-handoffs/larry_osito-tender_...
```

**Continuing work:**
```markdown
Continued from: 2025-10-16
Previous handoff: larry_osito-tender_20251016-161604_...
```

---

## Development Workflow: Self-Contained Scripts Folder Structure

### Concept
All daily work including scripts, analysis, outputs, and reports must be developed within the daily scripts folder (`scripts/YYYY-MM-DD/`) as a self-contained unit. This structure enables easy archiving and reuse as templates.

### Development Structure

**During Development:**
```
scripts/YYYY-MM-DD/
├── agent-handoffs/
│   └── [handoff documents]
├── report/
│   ├── report_YYYY-MM-DD.qmd
│   ├── report_YYYY-MM-DD.pdf
│   ├── figures/ (if any)
│   └── data/ (if any)
├── [analysis scripts]
│   ├── script1.py
│   ├── script2.py
│   └── ...
├── [output folders]
│   ├── output1/
│   ├── output2/
│   └── ...
└── [other work subdirectories]
```

### Key Principles

1. **Self-Containment:** All scripts, outputs, and report materials must be within `scripts/YYYY-MM-DD/`
2. **Report Location:** Reports go in `scripts/YYYY-MM-DD/report/` subdirectory
3. **Organization:** Keep work organized in logical subdirectories within the daily folder
4. **Relative Paths:** Scripts should use relative paths referencing other files within the daily folder
5. **Archive Ready:** Structure should be ready for copying to `docs/reports/YYYY-MM-DD/` as complete archive

### Report Development

**Location:** `scripts/YYYY-MM-DD/report/`

**Requirements:**
- QMD file with proper YAML header
- Style rules from `docs/style_rules.yaml` strictly followed
- Cinnamoroll color palette (reference: `scripts/cinnamoroll_palette.py`)
- Avenir Ultralight font configuration
- Self-contained: all figures and data references work within the folder structure

**Report Structure:**
- Title page with date and project name
- Executive summary
- Analysis sections (incorporating findings from other tasks)
- Conclusions and next steps
- All supporting materials (figures, data) in report subdirectory

### Archive Process

**When Work is Complete:**
1. Copy entire `scripts/YYYY-MM-DD/` folder to `docs/reports/YYYY-MM-DD/`
2. Include `docs/style_rules.yaml` in archive for reference
3. Archive becomes reusable template with:
   - All scripts needed to regenerate analysis
   - All output folders with results
   - Complete report with figures and data
   - Style rules for consistency

**Archive Structure:**
```
docs/reports/YYYY-MM-DD/
├── report/
│   ├── report_YYYY-MM-DD.qmd
│   ├── report_YYYY-MM-DD.pdf
│   ├── figures/
│   └── data/
├── [analysis scripts]
├── [output folders]
├── agent-handoffs/
└── style_rules.yaml (copied for reference)
```

### Benefits

1. **Reproducibility:** Complete archive contains everything needed to regenerate results
2. **Template Reuse:** Previous day's structure can be copied as starting point
3. **Organization:** Clear separation between development (scripts/) and documentation (docs/)
4. **Self-Containment:** No broken references when archive is moved or shared
5. **Traceability:** All work for a day stays together with handoffs and reports

### Example Workflow

1. **Morning:** Create `scripts/2025-11-11/` folder structure
2. **Development:** All scripts, analysis, outputs created within `scripts/2025-11-11/`
3. **Report:** Report developed in `scripts/2025-11-11/report/` referencing work in same folder
4. **End of Day:** Copy complete `scripts/2025-11-11/` to `docs/reports/2025-11-11/` with style_rules.yaml
5. **Archive:** Complete self-contained report archive ready for reuse

---

## Rules

1. ✅ Always get fresh system timestamp
2. ✅ Follow exact naming convention
3. ✅ Update STATUS.md
4. ✅ Create daily log
5. ✅ Reference related work
6. ✅ Clear deliverables
7. ✅ No hallucinated timestamps!
8. ✅ Keep all daily work self-contained in `scripts/YYYY-MM-DD/`
9. ✅ Develop reports in `scripts/YYYY-MM-DD/report/` subdirectory
10. ✅ Structure for eventual archive to `docs/reports/YYYY-MM-DD/`

---

**Last Updated:** 2025-11-11  
**Purpose:** Maintain clean, traceable agent communication across days and self-contained development workflow
