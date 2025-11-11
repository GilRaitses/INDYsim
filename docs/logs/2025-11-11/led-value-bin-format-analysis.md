# LED Value Bin File Format Analysis

**Date:** 2025-11-11  
**Agent:** osito-tender  
**Task:** Task 0.1 - Investigate Track Extraction Software LED Value Format

## Overview

This document describes the LED value bin file format and the method for aligning LED values with track timecode. LED values are written to binary files during track extraction and must be properly aligned with maggot track timecode to enable accurate stimulus-response analysis.

## LED Value Storage Architecture

### Data Flow

1. **LabVIEW Acquisition:** LED values are acquired and combined into an MMF (Memory-Mapped File) in LabVIEW
2. **Track Extraction Software:** The Track Extraction Software (`D:\magniphyq\codebase\Track-Extraction-Software`) writes LED values to binary files
3. **Metadata Storage:** Per-frame metadata including LED frame numbers and buffer numbers are stored in `.mdat` files
4. **Alignment:** MATLAB code aligns LED values with track timecode using buffer number mapping

### Key Components

- **LED Value Bin Files:** Binary files containing LED intensity values (bytes or bits)
- **Metadata Files (.mdat):** Tab-separated files containing per-frame metadata including `ledFrame`, `bufnum`, `ledNBytesOut`
- **Track Timecode:** Elapsed time index (ETI) associated with each track frame

## Metadata File Structure

The `.mdat` file contains per-frame metadata with the following key fields:

| Field | Description |
|-------|-------------|
| `FrameNumber` | Frame number in the track (0-indexed) |
| `bufnum` | Buffer number from camera/tracking system |
| `bufnum_time` | Timestamp for buffer number |
| `ledFrame` | LED frame number (starts at 0, can be negative before LED starts) |
| `ledFrame_time` | Timestamp for LED frame |
| `ledNBytesOut` | Cumulative total bytes written to LED bin file (or `ledNBitsOut` for bits) |
| `ledTime` | LED timestamp |
| `ledFP1`, `ledFP2` | LED frame positions |

### Example Metadata Entry

```
FrameNumber: 25
bufnum: 36463
ledFrame: 0
ledNBytesOut: 2
```

This indicates that at track frame 25, the LED frame counter was 0, and 2 bytes had been written to the LED bin file.

## LED Value Bin File Format

### File Structure

LED values are stored as raw binary data:
- **Byte-based format:** Values stored as `uint8` (unsigned 8-bit integers)
- **Bit-based format:** Values stored as individual bits (`ubit1`)

The file contains sequential LED intensity values with no header or frame markers. The position in the file corresponds to LED frame number multiplied by bytes/bits per frame.

### Reading LED Values

**MATLAB Implementation:**
```matlab
% Byte-based reading
fid = fopen(bitfilename, 'rb');
bits = fread(fid, inf, 'uint8=>double');
fclose(fid);

% Bit-based reading
fid = fopen(bitfilename, 'rb');
bits = fread(fid, inf, 'ubit1=>double');
fclose(fid);
```

## Alignment Method

### Core Concept

LED values must be aligned with track timecode because:
1. LED values are written at a different rate than track frames
2. LED acquisition may start before or after track recording begins
3. Buffer numbers (`bufnum`) provide the mapping between LED frames and track frames

### Alignment Algorithm

The MATLAB implementation (`createLedTableFromBitFile.m`) uses the following method:

#### Step 1: Find First LED Frame

Find the track frame where `ledFrame == 0` (first LED frame):

```matlab
firstFrame = find(ds.ledFrame == 0, 1, 'first');
```

If `ledFrame == 0` is not found, use polynomial fitting to estimate:
```matlab
x = ds.bufnum;
y = ds.ledFrame;
p = polyfit(y(isfinite(y)), x(isfinite(y)), 1);
firstFrame = find(x == round(p(2)));
```

#### Step 2: Calculate Bytes/Bits Per Frame

Calculate the number of bytes (or bits) per track frame:

```matlab
% For byte-based format
ind = find(isfinite(ds.ledNBytesOut) & isfinite(ds.bufnum), 1, 'last');
bytesperframe = round((ds.ledNBytesOut(ind)) / (ds.bufnum(ind) - ds.bufnum(firstFrame)));

% For bit-based format
ind = find(isfinite(ds.ledNBitsOut) & isfinite(ds.bufnum), 1, 'last');
bitsperframe = round((ds.ledNBitsOut(ind)) / (ds.bufnum(ind) - ds.bufnum(firstFrame)));
```

**Key Formula:**
```
bytesperframe = total_bytes_written / (last_bufnum - first_bufnum)
```

This calculates how many bytes in the bin file correspond to each track frame.

#### Step 3: Map Buffer Numbers to Byte Positions

Create a mapping from buffer numbers to byte positions in the bin file:

```matlab
bufaxis = (ds.bufnum - ds.bufnum(firstFrame)) * bytesperframe;
```

**Explanation:**
- `bufnum - bufnum(firstFrame)`: Number of buffers elapsed since first LED frame
- Multiply by `bytesperframe`: Convert to byte position in bin file
- `bufaxis`: Array of byte positions corresponding to each track frame

#### Step 4: Interpolate Time to Byte Positions

Interpolate elapsed time (ETI) to byte positions:

```matlab
valid = isfinite(bufaxis) & isfinite(expt.elapsedTime);
bufaxis = bufaxis(valid);
et = expt.elapsedTime(valid);

% Truncate to valid byte range
bitaxis = min(bufaxis):(min(length(bits)-1, max(bufaxis)));
timaxis = interp1(bufaxis, et, bitaxis);
```

**Result:**
- `timaxis`: Elapsed time for each byte position in the bin file
- `bitaxis`: Byte positions (0-indexed)

#### Step 5: Read and Align LED Values

Read LED values from bin file and align to time axis:

```matlab
allbits = zeros(size(timaxis));
inds = (bitaxis >= 0);
allbits(inds) = bits(1:nnz(inds));
```

### Complete MATLAB Code Pattern

```matlab
% Find first frame
firstFrame = find(ds.ledFrame == 0, 1, 'first');

% Calculate bytes per frame
ind = find(isfinite(ds.ledNBytesOut) & isfinite(ds.bufnum), 1, 'last');
bytesperframe = round((ds.ledNBytesOut(ind)) / (ds.bufnum(ind) - ds.bufnum(firstFrame)));

% Map buffer numbers to byte positions
bufaxis = (ds.bufnum - ds.bufnum(firstFrame)) * bytesperframe;
valid = isfinite(bufaxis) & isfinite(expt.elapsedTime);
bufaxis = bufaxis(valid);
et = expt.elapsedTime(valid);

% Read bin file
fid = fopen(bitfilename, 'rb');
bits = fread(fid, inf, 'uint8=>double');
fclose(fid);

% Interpolate time to byte positions
bitaxis = min(bufaxis):(min(length(bits)-1, max(bufaxis)));
timaxis = interp1(bufaxis, et, bitaxis);

% Align LED values
allbits = zeros(size(timaxis));
inds = (bitaxis >= 0);
allbits(inds) = bits(1:nnz(inds));
```

## Key Relationships

### ledFrame and bufnum

- `ledFrame`: LED frame counter (starts at 0 when LED acquisition begins)
- `bufnum`: Camera buffer number (increments with each track frame)
- Relationship: `ledFrame` may be negative before LED starts, then increments independently

### Buffer Axis to Bit Axis Mapping

- **Buffer axis (`bufaxis`):** Byte positions in bin file calculated from buffer numbers
- **Bit axis (`bitaxis`):** Actual byte positions in bin file (0-indexed)
- **Time axis (`timaxis`):** Elapsed time interpolated to byte positions

### Bytes Per Frame Calculation

The `bytesperframe` value represents:
- How many bytes in the bin file correspond to each track frame
- Calculated from total bytes written divided by buffer range
- Used to map buffer numbers to byte positions

## Current Python Implementation Issue

The current Python implementation (`engineer_dataset_from_h5.py`) reads LED values directly from H5 files:

```python
data['led1Val'] = gq_item['yData'][:]
```

**Problem:** This assumes LED values are already aligned with track timecode, which may not be correct. The H5 file may contain LED values that need temporal alignment using the `ledFrame`/`bufnum` mapping method.

## Required Metadata Fields

To implement proper alignment in Python, the following metadata fields must be available:

1. **From H5 metadata or .mdat file:**
   - `ledFrame`: LED frame numbers for each track frame
   - `bufnum`: Buffer numbers for each track frame
   - `ledNBytesOut` or `ledNBitsOut`: Cumulative bytes/bits written

2. **From track data:**
   - `elapsedTime` or `eti`: Elapsed time index for each track frame

3. **From bin file:**
   - LED intensity values (raw bytes or bits)

## Implementation Notes

### Edge Cases

1. **LED starts before tracking:** `ledFrame` may be negative for early track frames
2. **Missing frames:** Some track frames may not have corresponding LED values
3. **File truncation:** Bin file may be shorter than expected (handled by `min(length(bits)-1, max(bufaxis))`)

### Interpolation Method

The MATLAB code uses linear interpolation (`interp1` with default 'linear' method) to map elapsed time to byte positions. This assumes a constant frame rate.

### Average in Preceding Bin

The final alignment uses `GlobalLookupTable.averageInPrecedingBin` to average LED values within time bins, which handles the case where multiple LED values map to the same track time point.

## References

- **MATLAB Code:**
  - `D:\magniphyq\codebase\Matlab-Track-Analysis-SkanataLab\@GlobalLookupTable\createLedTableFromBitFile.m`
  - `D:\magniphyq\codebase\Matlab-Track-Analysis-SkanataLab\@GlobalLookupTable\GlobalLookupTable.m`

- **Track Extraction Software:**
  - `D:\magniphyq\codebase\Track-Extraction-Software\source code\StackLoader.cpp`
  - `D:\magniphyq\codebase\Track-Extraction-Software\Image-Stack-Compressor\StackReader.cpp`
  - `D:\magniphyq\codebase\Track-Extraction-Software\Image-Stack-Compressor\ExtraDataWriter.cpp`

- **Example Metadata:**
  - `D:\magniphyq\codebase\Matlab-Track-Analysis-SkanataLab\user specific\Yiming\20250115_1115_T_Bl_Sq_A27h@Chr(3).mdat`

## Next Steps

1. **Task 0.2:** Implement this alignment method in Python
2. **Verify:** Compare Python-aligned LED values with MATLAB reference
3. **Validate:** Check that stimulus-response dynamics are biologically plausible after alignment

---

**Status:** Complete  
**Handoff:** Ready for conejo-code to implement Python alignment method (Task 0.2)

