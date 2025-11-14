# What to Send to Sandbox Films

## Recommended Package

### 1. **Video File**
- Original unedited source file: `IMG_3788.mov`
- Delivered via WeTransfer, Dropbox, or similar service

### 2. **Client-Facing Analysis Report** ✅
**File:** `video_quality_analysis_report_client.qmd` (rendered to PDF)

**What it includes:**
- Executive summary with key findings
- Technical specifications
- Visual quality metrics (with figures)
- Content significance analysis
- Narrative significance (CRITICAL score)
- Usability analysis
- **Valuation analysis showing \$2,226 - \$2,713 range**
- Professional conclusion

**What it omits:**
- ❌ Internal negotiation recommendations ("start with conservative as floor")
- ❌ Detailed methodology appendix (too technical)
- ❌ "For Licensing Negotiations" section (internal strategy)
- ❌ "For Production Use" recommendations (they know how to use footage)

### 3. **Licensing Agreement** ✅
**File:** `licensing_agreement.qmd` (rendered to PDF)

**Keep as-is** - This is the standard agreement with:
- License fee: \$2,500.00 (within your range)
- All rights and terms clearly defined
- Technical specifications appendix
- Valuation breakdown appendix

## What NOT to Send

### ❌ Full Technical Report (`video_quality_analysis_report.qmd`)
- Contains internal negotiation strategy
- Has "Recommendations for Licensing Negotiations" section
- Includes detailed methodology that might give them negotiation leverage
- Too technical/internal-facing

### ❌ Python Scripts or Analysis Code
- Not relevant for them
- Internal tools only

## Tone Adjustments Made

### ✅ Professional & Factual
- Focuses on value proposition
- Shows objective analysis
- Presents valuation range transparently
- Emphasizes narrative significance

### ✅ What's Removed
- "Start with conservative estimate as floor" → Too tactical
- "Factor into negotiations" → Implies you're negotiating against them
- Internal production recommendations → They're the experts
- Detailed score computation → Too technical, might be used to negotiate down

## Key Numbers to Highlight

| Item | Value | Why It Matters |
|------|-------|----------------|
| Usable Seconds | 15.83 (100%) | All footage is usable |
| Narrative Score | 100/100 (CRITICAL) | This is THE key moment |
| Multiple-Use Potential | Yes | Can be used throughout doc |
| Value Range | \$2,226 - \$2,713 | Transparent, fair range |
| License Fee | \$2,500.00 | Within range, fair value |

## Suggested Email/Message Template

```
Subject: Flaco Footage - IMG_3788.mov

Hi [Name],

Attached is the original unedited source file (IMG_3788.mov) along with a technical analysis report and licensing agreement.

The footage documents Flaco's first landing on Fifth Avenue on the night he was freed from the Central Park Zoo - a critical moment in his story. The analysis shows:

- 15.83 seconds of usable footage (100% usable)
- Full HD resolution suitable for broadcast
- Critical narrative significance (100/100 score)
- Multiple-use potential throughout the documentary

Based on the technical analysis and narrative value, the licensing fee is \$2,500.00 for worldwide, perpetual use in your documentary production.

Please review the attached documents and let me know if you have any questions.

Best,
[Your Name]
Fifth Avenue Studios
```

## Files to Render and Send

1. `quarto render video_quality_analysis_report_client.qmd --to pdf`
2. `quarto render licensing_agreement.qmd --to pdf`
3. Send both PDFs + video file

## Summary

**Send:**
- ✅ Video file (IMG_3788.mov)
- ✅ Client-facing analysis report (shows value, omits strategy)
- ✅ Licensing agreement (with \$2,500 fee)

**Don't Send:**
- ❌ Full technical report with negotiation recommendations
- ❌ Analysis scripts or code
- ❌ Internal notes or strategy documents

The client-facing version keeps it professional, factual, and focused on the value of the footage while showing how you arrived at the \$2,500 fee.

