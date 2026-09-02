# Project briefing: Pneumoperitoneum CT classification (VISTA3D encoder transfer learning)

## Goal
Binary classification (pneumoperitoneum positive/negative) from CT volumes (.nii.gz),
using VISTA3D's pretrained image encoder as a feature extractor (transfer learning),
per mentor's suggestion, rather than training a 3D CNN from scratch.

## Dataset (on this PC)
- Path: `C:\Users\User1\pneumo_dataset\merged\Pneumo_Positive\` and `Pneumo_Negative\`
- 363 .nii.gz volumes total: 145 positive, 218 negative (~40:60 imbalance)
- Ground-truth report also exists (xlsx, "Analysis_Result_clean_gas_result_with_free_air_location")
  with columns: 匿名病歷號 (patient ID), Pneumoperitoneum_Label (Yes/No),
  Keep_In_Dataset (True/False), Exclude_Reason, 電腦斷層掃描日期 (scan date),
  Confidence, Review_Flag. Filtering Keep_In_Dataset==True gives exactly the 363.
  NOTE: this report file needs to be re-obtained/re-uploaded on this PC if not present.

## Key prior findings (already established, don't need to redo the investigation)
1. **9 patients have 2 scans each** (multi-scan). 3 of those have DIFFERENT labels
   between their two scans (e.g. negative then positive weeks later -- plausible
   clinical progression). Filename pattern is `{patientID}-{scanIndex}.nii.gz`
   (e.g. `20251028002-1.nii.gz`). scan_idx (1,2) corresponds to chronological
   order (sorted by report's scan date per patient) -- verified with zero
   unmatched cases and zero label mismatches once matched correctly.
   IMPORTANT: multi-scan patients must stay entirely within ONE split
   (train/val/test) to avoid leakage.
2. **No corrupt files, no NaN/Inf/constant volumes** in the 363 -- clean dataset.
3. **Bimodal FOV split**: NOT "full-body vs abdomen" as originally assumed --
   more precisely two acquisition protocols. Extent_z (mm) = shape_z * spacing_z:
   - ~105 volumes: abdomen-only, extent_z 150-370mm (few slices, 30-74 count)
   - ~258 volumes: extended-FOV (chest+abdomen+pelvis), extent_z 385-685mm
     (34-512 slices)
   - Real gap in the histogram sits at 328-417mm (only 12 volumes fall there) --
     use ~375mm as the threshold, not an arbitrary guess.
   - This split is genuine (correlates with slice COUNT, r=0.77, not slice
     THICKNESS, r=0.10 -- both groups use ~5mm spacing) so it's not a resampling
     artifact.
   - This matters because naively resizing both types to the same voxel grid
     makes the model risk learning "scan protocol" as a shortcut instead of the
     pathology. Plan was to crop extended-FOV volumes down to the abdominal
     region using VISTA3D segmentation output as an ROI localizer BEFORE
     feeding into the classifier -- this still applies for the encoder approach.
4. **HU intensity**: min consistently ~-1000 to -1024 (correct raw HU, not
   pre-windowed), max up to 3071 (contrast/metal, not a bug).
5. **Orientation**: 5 different codes present (LPS, LIP, LAS, LPI, PIR) --
   must canonicalize (e.g. to RAS) during preprocessing.
6. **On-disk dtype**: mixed int16 and float64 -- standardize on load.
7. **7 volumes have unusually thick slices** (10.4-15.2mm vs dataset mode of
   5mm): 6 of 7 are positive-labeled. Report confirms all 7 are Confidence:High,
   Review_Flag:No, so labels aren't in question, but worth tracking these as
   potential outliers during training/eval.

## VISTA3D bundle
- Was tried via MONAI Label server (`monailabel start_server --app apps\monaibundle
  --studies studies --conf models vista3d`) but hit repeated version-mismatch bugs
  in monailabel 0.8.5's BundleInferTask (`'BundleInferTask' object has no attribute
  'description'`, then `'...' has no attribute '_config'`). Was mid-way through
  trying monailabel==0.8.1 as a fix when the approach pivoted.
- DECISION: skip MONAI Label server entirely. We have direct access to
  `model.pt` + `configs/inference.json` + `configs/metadata.json` (standard
  MONAI bundle layout: .cache, configs, docs, eval, models, scripts folders).
  Need to download/set up this bundle fresh on this PC.
- VISTA3D architecture (relevant for encoder extraction): built from an
  **image encoder** (processes the CT volume into features -- this is what we
  want), a **prompt encoder** (point/class-ID prompts, for interactive
  segmentation -- NOT needed for classification), and a **decoder** (fuses
  image features + prompt -- NOT needed for classification).
- Real label ID -> organ name mapping (from metadata.json), for defining the
  abdominal ROI (used to crop extended-FOV volumes):
  ```
  1 liver, 2 kidney, 3 spleen, 4 pancreas, 5 right kidney, 7 inferior vena cava,
  8 right adrenal gland, 9 left adrenal gland, 10 gallbladder, 11 esophagus
  (lower portion near diaphragm -- relevant for subphrenic free air),
  12 stomach, 13 duodenum, 14 left kidney, 15 bladder,
  17 portal vein and splenic vein, 18 rectum, 19 small bowel, 62 colon
  ```
  (full 132-class list also available in metadata.json if needed later)
- Was about to inspect `model.pt`'s state_dict structure (top-level submodule
  prefixes) to identify exactly which weight keys belong to the image encoder,
  so it can be cleanly extracted with its pretrained weights and a new
  classification head attached. This inspection has NOT been done yet.

## Hardware constraint (NEW on this PC)
- RTX 3060, 12GB VRAM (previous plan assumed a 4090 24GB -- crop/patch sizes,
  batch size, and precision (fp16/bf16, gradient checkpointing) need to be
  planned around this smaller budget, especially once fine-tuning the encoder
  (not just linear probing).

## Method decision (from mentor + discussion)
- Use VISTA3D's image encoder (pretrained) + new classification head (global
  pool -> small MLP -> 1 logit) instead of training a 3D CNN from scratch.
- Strategy: **linear probe first** (freeze encoder entirely, train only the
  head) to get a fast, honest signal on whether the pretrained features
  separate positive/negative at all. Only proceed to fine-tuning (gradually
  unfreezing, low LR) if the linear probe result is promising. Rationale:
  fine-tuning the full encoder on ~290 training volumes risks destroying
  pretrained features via overfitting; linear probe is cheap and diagnostic.
- Track AUC/sensitivity/specificity, not just accuracy, given class imbalance.
- Compare against a from-scratch 3D CNN baseline eventually, for a real
  ablation number.

## Ordered plan from here (fresh start on this PC)
1. Set up venv, install torch/monai/nibabel/pandas/tqdm etc.
2. Download/set up the VISTA3D bundle (model.pt + configs) on this PC.
3. Re-run the full dataset audit (geometry, spacing, orientation, HU, file
   size, extent_z, bimodal FOV split, data quality flags) against the new
   dataset path -- reuse the same methodology as before (see "Key prior
   findings" above for what to expect / validate against).
4. Re-obtain and cross-check against the ground-truth report if available on
   this PC (same crosscheck logic: match filename scan_idx to report rows
   sorted by date per patient, NOT a naive patient_id-only merge -- that
   caused duplicate-row bugs before).
5. Inspect model.pt's state_dict structure to identify the image encoder's
   weight keys.
6. Extract the image encoder as a standalone module, build the classification
   head, set up linear-probe training (frozen encoder).
7. ROI cropping for extended-FOV volumes still needed before classification
   (via VISTA3D segmentation of abdominal organs, OR reuse of the encoder
   features directly -- decide once encoder structure is known) -- avoids the
   scan-protocol-shortcut risk.
8. Patient-level leak-free train/val/test split (respecting the multi-scan
   patient grouping).
9. Train linear probe, evaluate (AUC/sens/spec), decide on fine-tuning next.
