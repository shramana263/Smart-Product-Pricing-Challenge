# 🏗️ Try4 Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  train.csv   │  │  test.csv    │  │ image_links  │          │
│  │  (~390k)     │  │  (~26k)      │  │              │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 1: FEATURE EXTRACTION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────────────────────────────────────────────┐          │
│  │   1A. DeBERTa-v3-large (Text Embeddings)          │          │
│  │   • Input: catalog_content                        │          │
│  │   • Output: 1024-dim embeddings                   │          │
│  │   • Fine-tuned with regression head               │          │
│  │   • 5-fold CV ensemble                            │          │
│  └───────────────────────────────────────────────────┘          │
│                            │                                     │
│  ┌───────────────────────────────────────────────────┐          │
│  │   1B. CLIP ViT-Large (Image Embeddings)           │          │
│  │   • Input: image_link URLs                        │          │
│  │   • Output: 768-dim embeddings                    │          │
│  │   • Pre-trained vision encoder                    │          │
│  │   • Handles missing images (zero vector)          │          │
│  └───────────────────────────────────────────────────┘          │
│                            │                                     │
│  ┌───────────────────────────────────────────────────┐          │
│  │   1C. Tabular Features (Engineered)               │          │
│  │   • Text stats: length, density, etc.             │          │
│  │   • Brand & category: target encoded              │          │
│  │   • Quantity & units: extracted & normalized      │          │
│  │   • Price features: log, sqrt, binned             │          │
│  │   • Output: ~40 handcrafted features              │          │
│  └───────────────────────────────────────────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         STAGE 2: OUTLIER DETECTION (Priority++)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ Isolation Forest │  │   IQR Method     │                    │
│  │  (5% outliers)   │  │  (1.5×IQR)       │                    │
│  └──────────────────┘  └──────────────────┘                    │
│           │                      │                              │
│           └──────────┬───────────┘                              │
│                      │                                          │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │    Z-Score       │  │     DBSCAN       │                    │
│  │   (|z| > 3)      │  │  (Density-based) │                    │
│  └──────────────────┘  └──────────────────┘                    │
│           │                      │                              │
│           └──────────┬───────────┘                              │
│                      ▼                                          │
│            ┌──────────────────┐                                 │
│            │ Ensemble Voting  │                                 │
│            │ (2+ methods      │                                 │
│            │  agree = outlier)│                                 │
│            └──────────────────┘                                 │
│                      │                                          │
│                      ▼                                          │
│            ┌──────────────────┐                                 │
│            │ Outlier Flags &  │                                 │
│            │ Confidence Scores│                                 │
│            └──────────────────┘                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         STAGE 3: OUTLIER TREATMENT (Priority++)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Strategy 1: Winsorization     → Cap at 1%/99% percentiles      │
│  Strategy 2: Robust Scaling    → Use quantile ranges (5-95%)    │
│  Strategy 3: Log Transform     → Compress extreme values        │
│  Strategy 4: Separate Modeling → Train dedicated outlier models │
│  Strategy 5: Confidence Weight → Weight by detection confidence │
│                                                                  │
│                      ▼                                          │
│            ┌──────────────────┐                                 │
│            │ Treated Features │                                 │
│            │  + Metadata      │                                 │
│            └──────────────────┘                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│         STAGE 4: FEATURE FUSION & TRAINING                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐                   │
│  │ DeBERTa   │  │   CLIP    │  │  Tabular  │                   │
│  │  (1024)   │  │   (768)   │  │   (~40)   │                   │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                   │
│        │              │              │                          │
│        └──────────────┴──────────────┘                          │
│                       │                                         │
│                       ▼                                         │
│            ┌────────────────────┐                               │
│            │  Concatenate       │                               │
│            │  (~1850 features)  │                               │
│            └────────────────────┘                               │
│                       │                                         │
│                       ▼                                         │
│            ┌────────────────────┐                               │
│            │   LightGBM         │                               │
│            │   • 5-Fold CV      │                               │
│            │   • Outlier weights│                               │
│            │   • Early stopping │                               │
│            └────────────────────┘                               │
│                       │                                         │
│                       ▼                                         │
│            ┌────────────────────┐                               │
│            │  Trained Models    │                               │
│            │  (5 folds)         │                               │
│            └────────────────────┘                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 5: PREDICTION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────┐                 │
│  │  Ensemble Prediction (Average 5 folds)     │                 │
│  └────────────────────────────────────────────┘                 │
│                       │                                         │
│        ┌──────────────┴──────────────┐                          │
│        │                             │                          │
│        ▼                             ▼                          │
│  ┌──────────┐                  ┌──────────┐                     │
│  │   OOF    │                  │   Test   │                     │
│  │Predictions│                 │Predictions│                    │
│  └──────────┘                  └──────────┘                     │
│        │                             │                          │
│        ▼                             ▼                          │
│  ┌──────────┐                  ┌──────────┐                     │
│  │ SMAPE:   │                  │Submission│                     │
│  │ <35%     │                  │  File    │                     │
│  └──────────┘                  └──────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
catalog_content (text)  ──┐
                          │
                          ├──→ [DeBERTa] ──→ 1024-dim ──┐
                          │                              │
image_link (URLs)  ───────┼──→ [CLIP] ─────→ 768-dim ───┤
                          │                              │
                          └──→ [Engineer] ──→ ~40 feat ──┤
                                                         │
                                                         ▼
                                              ┌──────────────────┐
                                              │ Feature Matrix   │
                                              │  (~1850 cols)    │
                                              └────────┬─────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │ Outlier Detection│
                                              │  (4 methods)     │
                                              └────────┬─────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │ Outlier Treatment│
                                              │  (5 strategies)  │
                                              └────────┬─────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │   LightGBM       │
                                              │  (5-Fold CV)     │
                                              └────────┬─────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │   Predictions    │
                                              │   (SMAPE <35%)   │
                                              └──────────────────┘
```

---

## Component Interactions

```
┌──────────────────────────────────────────────────────────┐
│                    MAIN PIPELINE                          │
│                  (main_pipeline.py)                       │
└───────────────────┬──────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│Feature Extraction│ Preprocessing│  │   Modeling    │
└─────────────┘ └─────────────┘ └─────────────┘
        │           │           │
        │           │           │
        ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ DeBERTa     │ │ Outlier     │ │ LightGBM    │
│ CLIP        │ │ Detection   │ │ Training    │
│ Tabular     │ │ Treatment   │ │             │
└─────────────┘ └─────────────┘ └─────────────┘
        │           │           │
        └───────────┴───────────┘
                    │
                    ▼
            ┌──────────────┐
            │   Outputs    │
            │  (Embeddings,│
            │   Models,    │
            │  Predictions)│
            └──────────────┘
```

---

## Feature Dimension Breakdown

```
┌──────────────────────────────────────────────────────┐
│            FINAL FEATURE VECTOR                       │
│              (~1850 dimensions)                       │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌────────────────────────────────────────────┐     │
│  │  DeBERTa Embeddings (1024 dims)            │     │
│  │  55.7% of total features                   │     │
│  │  - Semantic text understanding             │     │
│  │  - Contextual word representations         │     │
│  │  - Fine-tuned on price prediction          │     │
│  └────────────────────────────────────────────┘     │
│                                                       │
│  ┌────────────────────────────────────────────┐     │
│  │  CLIP Embeddings (768 dims)                │     │
│  │  41.8% of total features                   │     │
│  │  - Visual product features                 │     │
│  │  - Pre-trained vision encoder              │     │
│  │  - Zero vectors for missing images         │     │
│  └────────────────────────────────────────────┘     │
│                                                       │
│  ┌────────────────────────────────────────────┐     │
│  │  Tabular Features (~40 dims)               │     │
│  │  2.2% of total features                    │     │
│  │  - Text statistics (8)                     │     │
│  │  - Brand/category encoding (4)             │     │
│  │  - Quantity features (5)                   │     │
│  │  - Price features (5)                      │     │
│  │  - Image features (3)                      │     │
│  │  - Target encodings (15)                   │     │
│  └────────────────────────────────────────────┘     │
│                                                       │
│  ┌────────────────────────────────────────────┐     │
│  │  Treatment Metadata (~18 dims)             │     │
│  │  0.3% of total features                    │     │
│  │  - Outlier flags (4)                       │     │
│  │  - Confidence scores (2)                   │     │
│  │  - Segment labels (2)                      │     │
│  │  - Winsorized/log prices (4)               │     │
│  │  - Quality scores (6)                      │     │
│  └────────────────────────────────────────────┘     │
│                                                       │
└──────────────────────────────────────────────────────┘
```

---

## Outlier Handling Pipeline

```
┌──────────────────────────────────────────┐
│        DETECTION PHASE                    │
├──────────────────────────────────────────┤
│                                           │
│  Sample 1  ──┐                            │
│  Sample 2  ──┼──→ [Isolation Forest] ──→ Flag A
│  Sample 3  ──┼──→ [IQR Method]       ──→ Flag B
│  ...       ──┼──→ [Z-Score]          ──→ Flag C
│  Sample N  ──┘   [DBSCAN]            ──→ Flag D
│                                           │
│              Ensemble: A+B+C+D ≥ 2?      │
│                     │                     │
│                     ▼                     │
│              Outlier Count & Score        │
│                                           │
└──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│        TREATMENT PHASE                    │
├──────────────────────────────────────────┤
│                                           │
│  Normal Samples (85%)                     │
│  ├─→ Robust scaling                       │
│  ├─→ Standard training                    │
│  └─→ High confidence weights              │
│                                           │
│  Outlier Samples (15%)                    │
│  ├─→ Winsorization                        │
│  ├─→ Log transform                        │
│  ├─→ Separate model (optional)            │
│  └─→ Lower confidence weights             │
│                                           │
│              ▼                            │
│     Treated Feature Matrix                │
│                                           │
└──────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────┐
│        PREDICTION PHASE                   │
├──────────────────────────────────────────┤
│                                           │
│  Prediction = α·P_normal + β·P_outlier   │
│                                           │
│  where α, β based on confidence scores    │
│                                           │
└──────────────────────────────────────────┘
```

---

## Performance Improvement Strategy

```
Baseline (try3):           42.0% SMAPE
                              │
                              ▼
        ┌─────────────────────────────────┐
        │  Upgrade DeBERTa (+5-7%)        │  ──→  35-37% SMAPE
        └─────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────┐
        │  Add CLIP Images (+3-5%)        │  ──→  30-34% SMAPE
        └─────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────┐
        │  Outlier Handling (+5-10%)      │  ──→  20-29% SMAPE
        └─────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────┐
        │  Engineered Features (+2-3%)    │  ──→  17-27% SMAPE
        └─────────────────────────────────┘
                              │
                              ▼
Target (try4):          <35% SMAPE  ✅
                (Conservative estimate)

Optimistic target:      <30% SMAPE  🎯
                (With perfect tuning)
```

---

## Execution Timeline

```
Stage 1A: DeBERTa      [████████████████████] 60-90 min  (GPU)
Stage 1B: CLIP         [████████████████    ] 45-60 min  (GPU)
Stage 1C: Tabular      [████                ] 5-10 min   (CPU)
Stage 2:  Detection    [████                ] 10-15 min  (CPU)
Stage 3:  Treatment    [██                  ] 2-5 min    (CPU)
Stage 4:  Training     [████████████████    ] 40-60 min  (GPU)
                       ─────────────────────────────────
Total:                 ~2.5-4 hours (GPU)
                       ~20-30 hours (CPU)
```

---

This architecture provides:
- ✅ Robustness through multi-strategy outlier handling
- ✅ Rich feature space through multi-modal fusion
- ✅ Flexibility through modular design
- ✅ Scalability through efficient implementation
- ✅ Interpretability through feature analysis
