# ASC AI v1.2 training and frozen-test benchmark

## Training

v1.2 uses the same recipe as v1 and v1.1: ImageNet-pretrained ResNet18 with separate clear/cloudy and rain/no-rain heads; random resized crops, horizontal flips, and color jitter for training; class-weighted cross-entropy; AdamW (learning rate 0.0003, weight decay 0.0001); cosine schedule; batch size 16; seed 42; 30 epochs. The selected checkpoint is epoch 21, chosen by validation sky accuracy + rain F1.

The labeled dataset had 3,545 frames: 3,017 train and 528 validation frames. Validation reuses the 13 v1 dates and adds September 11 and 17, 2026, as whole-date holds. Best validation sky accuracy was 94.51% and rain F1 was 0.9167.

The new patch contained 921 fully labeled frames. Four frames with notes (bird, overexposed, or underexposed) were removed from the labeling pool; their half-hour archive originals are preserved. Of the 917 accepted frames, 774 entered training, 96 entered validation, and 47 were assigned to the added test date.

## Test-set expansion and policy

The official benchmark has 225 frames: the original 178-frame manifest plus all 47 unflagged September 5 frames. September 5 labels are 19 clear and 28 cloudy, including 18 rain-positive frames. The 47 were held out as one complete date to avoid splitting same-date conditions between train/validation and test. September 5 was selected from the new patch because it was an unflagged complete date; this was a deliberate date selection, not a random sample. The original 178-frame slice is reported separately because the added date is easier for all three models and lifts the combined score.

## Same expanded Test: 225 frames

| Model | Sky accuracy | Rain accuracy | Rain F1 | GO/NO-GO accuracy | False OK | False NO-GO |
|---|---:|---:|---:|---:|---:|---:|
| v1 | 94.22% | 93.78% | 0.9186 | 92.89% | 11/225 | 5/225 |
| v1.1 | 91.56% | 92.44% | 0.8994 | 90.22% | 16/225 | 6/225 |
| v1.2 | 95.11% | 95.11% | 0.9364 | 92.89% | 12/225 | 4/225 |

GO means predicted clear with no rain. False OK is an actual cloudy or rainy frame predicted GO. On all 225 frames, v1.2 ties v1 at 92.89% gate accuracy (209/225); v1.2 has one more false OK (12 vs 11) and one fewer false NO-GO (4 vs 5). v1.2 improves sky accuracy from 94.22% to 95.11%, rain accuracy from 93.78% to 95.11%, and rain F1 from 0.9186 to 0.9364 compared with v1.

## Original and added test portions

| Model | Slice | n | Sky acc | Rain acc | Rain F1 | Gate acc | False OK |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 | Original frozen Test | 178 | 93.82% | 92.70% | 0.9051 | 92.13% | 10/178 |
| v1 | Added September 5 | 47 | 95.74% | 97.87% | 0.9714 | 95.74% | 1/47 |
| v1.1 | Original frozen Test | 178 | 89.89% | 91.01% | 0.8806 | 88.20% | 15/178 |
| v1.1 | Added September 5 | 47 | 97.87% | 97.87% | 0.9714 | 97.87% | 1/47 |
| v1.2 | Original frozen Test | 178 | 94.38% | 94.94% | 0.9353 | 91.57% | 11/178 |
| v1.2 | Added September 5 | 47 | 97.87% | 95.74% | 0.9412 | 97.87% | 1/47 |

On the original 178 frames alone, v1.2 gate accuracy is 91.57% (163/178), compared with v1 at 92.13% (164/178). The added date scores 46/47 gate-correct for v1.2. The expansion is useful as a separate recent-date check, but the higher/easier September 5 slice means the 225-frame aggregate should not replace the original 178-frame comparison when judging progress over time.

## Reproducibility and artifacts

The full checkpoint, data split, label snapshot, command log, predictions, and manifests are saved locally under `ASC Label Tool/model_backups/v1.2/`. Model weights and camera images are not included in Git. The repository contains the compact manifests, source, per-frame benchmark predictions, metrics, and this report. The test manifest hash and checkpoint hashes are recorded in the JSON results.
