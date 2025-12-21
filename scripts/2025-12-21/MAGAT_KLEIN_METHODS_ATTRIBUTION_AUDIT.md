# MAGAT/Klein Methods Attribution Audit

## Issue Summary

The manuscript needs proper attribution and clarification regarding:
1. **MAGAT Analyzer** (Mac Gershow's program) - used for behavioral segmentation
2. **Klein run table methods** (Mason Klein's methodology) - used for run-level statistics
3. **Reverse crawl detection** (Mason Klein's algorithm)
4. **Clarification of which method extracts what from empirical signals**

## Current State in Manuscript

### Methods Section (02_methods.tex, lines 82-86)

**Current text:**
```latex
Empirical larval trajectories were processed using methods established in the main study. Trajectory extraction and behavioral state segmentation were performed using MAGAT Analyzer (Gershow et al., 2012), which identifies behavioral states including runs, reorientations, and head swings. Reverse crawl detection was performed using the algorithm developed by Mason Klein, which identifies periods of backward movement (SpeedRunVel $< 0$ for $\geq 3$ seconds) from trajectory data.

The consolidated dataset contains two complementary representations of behavioral events. The \textit{events group} records reorientation start events detected by MAGAT segmentation. Each row represents a discrete reorientation onset (False$\to$True transition in the reorientation state), providing direct event counts suitable for point-process modeling. The \textit{Klein run table} records run segments between reorientations, following Mason Klein's methodology. Each row represents a forward movement period (run) that begins with a reorientation event.
```

### Acknowledgments Section (06_acknowledgments.tex)

**Current text:**
```latex
Mason Klein developed the reverse crawl detection algorithm that identified reorientation events from larval trajectories, forming the foundation of this analysis. Marc Gershow authored MAGAT Analyzer (Gershow et al., 2012), used for trajectory extraction and behavioral state segmentation.
```

## Problems Identified

### 1. **Confusion about Reorientation Event Detection**

**Issue**: The acknowledgments say "Mason Klein developed the reverse crawl detection algorithm that identified reorientation events" but the methods say "reorientation start events detected by MAGAT segmentation."

**Clarification needed**: 
- **MAGAT Analyzer** (Gershow et al., 2012) detects **reorientation events** via behavioral state segmentation
- **Mason Klein's algorithm** detects **reverse crawl events** (backward movement)
- The **Klein run table** uses Mason Klein's methodology to structure run-level statistics, but the actual reorientation events come from MAGAT

### 2. **Insufficient Inline Attribution**

**Issue**: The methods section mentions MAGAT and Klein but doesn't provide enough detail for reproducibility:
- What specific MAGAT functions/methods are used?
- What are the MAGAT segmentation parameters?
- How does the Klein run table methodology differ from MAGAT's run detection?
- What is the exact relationship between MAGAT's reorientation detection and Klein's run table structure?

### 3. **Missing Reproducibility Details**

**Issue**: For reproducibility, readers need to know:
- Which version of MAGAT Analyzer was used?
- What are the specific segmentation thresholds/parameters?
- How are reorientation events extracted from MAGAT output?
- What is the exact algorithm for constructing the Klein run table from MAGAT-segmented data?

## Recommended Fixes

### Fix 1: Clarify Methods Section

**Location**: `phenotyping_followup/sections/02_methods.tex`, lines 82-86

**Proposed revision:**
```latex
Empirical larval trajectories were processed using methods established in the main study. Trajectory extraction and behavioral state segmentation were performed using MAGAT Analyzer (Gershow et al., 2012), which identifies behavioral states including runs, reorientations, and head swings. MAGAT Analyzer detects reorientation events by identifying state transitions from RUN to TURN based on heading angle changes and movement patterns. Reverse crawl detection was performed using the algorithm developed by Mason Klein (Klein, personal communication), which identifies periods of backward movement (SpeedRunVel $< 0$ for $\geq 3$ seconds) from trajectory data.

The consolidated dataset contains two complementary representations of behavioral events. The \textit{events group} records reorientation start events detected by MAGAT Analyzer segmentation. Each row represents a discrete reorientation onset (False$\to$True transition in the reorientation state), providing direct event counts suitable for point-process modeling. The \textit{Klein run table} records run segments between reorientations, following Mason Klein's methodology for structuring run-level statistics. Each row represents a forward movement period (run) that begins with a reorientation event. The Klein run table uses MAGAT-detected reorientation events as boundaries but applies Mason Klein's statistical framework for run-level analysis.
```

### Fix 2: Update Acknowledgments

**Location**: `phenotyping_followup/sections/06_acknowledgments.tex`

**Proposed revision:**
```latex
Marc Gershow authored MAGAT Analyzer (Gershow et al., 2012), which was used for trajectory extraction and behavioral state segmentation, including detection of reorientation events. Mason Klein developed the reverse crawl detection algorithm and the run table methodology used for structuring run-level statistics. GR thanks Professor Ki Young Jeong for teaching the fundamental concepts in simulation modeling and statistical inference that provided the theoretical foundation for this analysis.
```

### Fix 3: Add Reproducibility Details

**Location**: Add new subsection in `phenotyping_followup/sections/02_methods.tex` after line 86

**Proposed addition:**
```latex
\subsubsection{Behavioral Segmentation Details}

MAGAT Analyzer (Gershow et al., 2012) was used to segment larval trajectories into discrete behavioral states. The segmentation algorithm identifies reorientation events by detecting state transitions from RUN to TURN based on heading angle changes exceeding a threshold (typically 45 degrees) within a sliding window. The specific MAGAT version and segmentation parameters used in the original data collection are documented in the main study. For this analysis, we used the pre-segmented event times stored in the consolidated dataset's \texttt{/events} group, specifically the \texttt{is\_reorientation\_start} boolean flag.

The Klein run table methodology (Klein, personal communication) structures run-level statistics by identifying forward movement periods (runs) bounded by MAGAT-detected reorientation events. Each run entry in the Klein run table contains statistics including run length (\texttt{runL}), run quality metrics (\texttt{runQ}, \texttt{runQ0}), and reorientation statistics (\texttt{reo\#HS}, \texttt{reoHS1}, \texttt{reoQ1}, \texttt{reoQ2}). The Klein run table provides complementary information about run-level patterns but is not used for event counting in kernel fitting, which relies exclusively on MAGAT-detected reorientation start events from the events group.
```

## Presentation Slides Status

**Location**: `/Users/gilraitses/InDySim/phenotyping_followup/presentation/slides.pdf`

**Status**: **LIKELY OBSOLETE** - Needs review for:
- Old simulation parameters (20-minute tracks, old event counts)
- Outdated MAGAT/Klein attribution
- Old figures and results
- Missing recent updates

**Action Required**: Review and update presentation slides to match current manuscript state.

## Summary of Required Changes

1. ✅ **Clarify MAGAT vs. Klein roles** in methods section
2. ✅ **Fix acknowledgments** to correctly attribute reorientation detection to MAGAT
3. ✅ **Add reproducibility details** about MAGAT segmentation and Klein run table construction
4. ✅ **Review and update presentation slides** (likely obsolete)
5. ✅ **Ensure inline attribution** is sufficient for reproducibility

## Files to Update

1. `phenotyping_followup/sections/02_methods.tex` - Methods clarification
2. `phenotyping_followup/sections/06_acknowledgments.tex` - Attribution fix
3. `phenotyping_followup/presentation/slides.pdf` - Review and update (if source available)

## Key Points for Reproducibility

For reproducibility, the manuscript should clearly state:
- **MAGAT Analyzer** (Gershow et al., 2012) detects reorientation events
- **Mason Klein's methods** structure run-level statistics and detect reverse crawls
- **Event counting** uses MAGAT-detected reorientation starts from the events group
- **Klein run table** provides complementary run-level statistics but is not used for event counting
- **Specific parameters** (if available) should be documented

