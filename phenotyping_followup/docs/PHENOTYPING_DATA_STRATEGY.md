# Phenotyping Data Strategy

## Current Situation

- **79 complete tracks** (20-minute duration)
- **~622 incomplete tracks** (10-19 minutes, excluded)
- **4 factorial conditions**

## Key Questions

### 1. Should we include incomplete tracks?

**Answer: YES, but with conditions**

**Criteria for inclusion:**
- ✅ **Duration ≥ 10 minutes** (likely sufficient for kernel fitting)
- ✅ **≥ 50 events** (minimum for reliable fitting)
- ✅ **Proper censoring handling** (account for duration)

**Benefits:**
- 9× increase in sample size (79 → ~700 tracks)
- Better phenotype discovery
- More robust clustering

**Challenges:**
- Need to handle censoring properly
- May need duration-normalized features
- More complex analysis

### 2. Should we generate more complete tracks?

**Answer: YES, definitely**

**Recommendation:**
- **Generate 200-300 tracks per condition** (800-1200 total)
- **Current:** 75 per condition (300 total)
- **Benefit:** 3-4× increase in sample size

**Why:**
- Better phenotype discovery
- More robust clustering
- Better statistical power
- Clean, complete data (no censoring issues)

### 3. What's the point of phenotyping?

**Answer: Discover unknown variability**

**But:**
- Need **reliable data** to discover it
- Need **sufficient sample size** for robust discovery
- Need **proper statistical methods** to validate findings

**Incomplete tracks can help IF:**
- They're long enough for reliable fitting
- They show different phenotypes
- Censoring is properly handled

---

## Recommended Strategy

### Phase 1: Generate More Complete Tracks (IMMEDIATE)

**Action:**
```bash
python3 generate_extended_phenotyping_tracks.py \
    --n-tracks-per-condition 250 \
    --output-dir data/simulated_phenotyping_extended
```

**Result:**
- 1000 complete tracks (250 per condition)
- 3× increase in sample size
- Better phenotype discovery

**Time:** ~20-30 minutes

### Phase 2: Assess Incomplete Tracks (NEXT)

**Action:**
```bash
python3 assess_incomplete_tracks.py \
    --complete-tracks-dir data/simulated_phenotyping \
    --output-dir results/incomplete_tracks_analysis
```

**Questions to answer:**
1. What's minimum reliable duration? (likely 10 min)
2. Are incomplete tracks systematically different?
3. Is censoring informative?

**Decision:**
- If incomplete tracks add unique phenotypes → Include them
- If incomplete tracks are just shorter versions → May not add value

### Phase 3: Combined Analysis (IF INCOMPLETE ADD VALUE)

**Action:**
- Include incomplete tracks (≥ 10 min) with proper handling
- Use duration as covariate
- Weight by track length
- Duration-normalized features

---

## Implementation Plan

### Immediate (This Week)

1. **Generate extended complete tracks:**
   - 250 tracks per condition
   - 1000 total tracks
   - Run phenotyping analysis
   - Compare to 300-track results

2. **Assess incomplete tracks:**
   - Test minimum duration requirements
   - Compare complete vs incomplete
   - Decide on inclusion

### Short-term (Next Week)

3. **Update analysis pipeline:**
   - Add duration filtering
   - Handle censoring
   - Duration-normalized features

4. **Combined analysis:**
   - Complete + incomplete (if warranted)
   - Proper statistical handling
   - Validate findings

---

## Expected Outcomes

### With More Complete Tracks (1000 tracks)

- **Better phenotype discovery:** More robust clustering
- **Better statistical power:** More reliable differences
- **Better coverage:** More complete parameter space
- **Cleaner analysis:** No censoring complications

### With Incomplete Tracks (if included)

- **Even larger sample:** ~1700 tracks total
- **More phenotypes:** May discover early-termination types
- **Better coverage:** Different behavioral trajectories

---

## Recommendations Summary

1. **✅ Generate more complete tracks** (250 per condition)
2. **✅ Assess incomplete tracks** (test if they add value)
3. **✅ Include incomplete tracks IF** they show unique phenotypes
4. **✅ Use proper censoring handling** if including incomplete

**Priority:** Generate more complete tracks first (immediate benefit, no complications).

