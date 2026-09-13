# Dysarthria Speech Screening

A machine learning tool that screens speech recordings for acoustic markers
associated with dysarthria, using classical acoustic/prosodic features
(MFCCs, jitter, shimmer, harmonics-to-noise ratio, formants) and a Support
Vector Machine classifier.

## Important
This is a **screening aid**, not a diagnostic tool. It is intended to flag
speech patterns worth follow-up with a speech-language pathologist — it does
not replace clinical evaluation.

## Data
- [TORGO Database](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) — dysarthric/typical speech from speakers with cerebral palsy and ALS, plus matched controls. Academic use only.
- [UASpeech](https://www.kaggle.com/datasets/aryashah2k/noise-reduced-uaspeech-dysarthria-dataset) (noise-reduced) — dysarthric/typical speech corpus. CC BY-NC 4.0.

## Method
1. Acoustic features extracted per utterance: 13 MFCCs (+ delta, delta-delta),
   jitter, shimmer, harmonics-to-noise ratio, formants F1/F2, zero-crossing
   rate, RMS energy (85 features total).
2. Classifier: SVM (RBF kernel), trained on combined TORGO + UASpeech data.
3. Evaluated with leave-one-speaker-out cross-validation to test genuine
   speaker-independent generalization (not just in-corpus accuracy).

## Results
| Setup | Accuracy |
|---|---|
| In-corpus (TORGO only, random split) | 94% |
| Speaker-independent (TORGO only, LOGO-CV) | 57–60% |
| Speaker-independent (TORGO + UASpeech, LOGO-CV) | 75% |

Combining corpora for greater speaker diversity substantially improved
genuine generalization to unseen speakers, confirming that speaker-pool
size — not feature or model choice — was the primary bottleneck.

## Limitations
- Both source datasets are restricted to academic/non-commercial use.
- Speaker pool, while improved, is still modest (38 unique speakers).
- Screening only — not validated for diagnostic use.

## Future Work
- Multimodal (audio + video) screening using orofacial motor cues.
- Integration into a broader speech-therapy adherence application with
  exercise-level correctness scoring.

## Reproducing
Training was conducted on Kaggle due to dataset size (9.5GB + 11.5GB).
See the linked Kaggle notebook for the full feature-extraction and
training pipeline. This repo contains only the exported model artifacts
and inference app.
