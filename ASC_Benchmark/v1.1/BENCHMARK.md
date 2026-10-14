# ASC AI v1.1 training and frozen-test benchmark

## How v1 was trained

The saved v1 script and checkpoint show a ResNet18 initialized from ImageNet weights, with separate clear/cloudy and rain/no-rain heads. The model used random resized crops, horizontal flips, and color jitter for training; validation used resize to 256 and center crop to 224. Sky and rain each had class-weighted cross-entropy loss. The optimizer was AdamW (learning rate 0.0003, weight decay 0.0001) with cosine annealing. The script defaults to batch size 16 and seed 42, and the saved history shows 30 epochs. The best checkpoint was selected by validation sky accuracy plus rain F1; v1 selected epoch 19. Its 2,135 labeled frames were split by date into 1,712 training and 423 validation frames. The separate official Test folder had 178 frames and was excluded from training and checkpoint selection.

## v1.1 run

v1.1 used the same architecture, transforms, optimizer, batch size, learning rate, seed, 30-epoch schedule, and checkpoint score. To ensure the 540 newly labeled frames contribute to training, it reused v1's validation **dates** rather than recomputing the latest quarter of dates. This put 2,243 frames in training and 432 in validation, including 531 new frames in training and 9 new frames in validation. The best checkpoint was epoch 14. The full data split, label snapshot, and command log are saved beside this report.

## Official frozen Test: 178 identical frames

- **Sky accuracy:** v1 93.82% (167/178); v1.1 89.89% (160/178), down 3.93 percentage points.
- **Rain accuracy:** v1 92.70% (165/178); v1.1 91.01% (162/178), down 1.69 points.
- **Rain F1:** v1 0.9051; v1.1 0.8806.
- **GO/NO-GO accuracy:** v1 92.13% (164/178); v1.1 88.20% (157/178), down 3.93 points. GO means predicted clear with no rain.
- **False OK:** v1 10/178; v1.1 15/178. These are actual NO-GO frames predicted OK.
- **False NO-GO:** v1 4/178; v1.1 6/178.

On the GO/NO-GO decision, v1.1 corrected 5 frames v1 missed but became wrong on 12 frames v1 got right. Its validation sky accuracy was 93.29% and rain F1 was 0.8776, similar to v1's best validation scores, but that did not carry over to this test set. This run does **not** support replacing v1 with v1.1.

The evaluator reproduced v1's previously saved official benchmark metrics exactly before comparing v1.1. The Test filenames match the frozen manifest, and no Test filename overlaps the v1.1 training or validation list. Both checkpoints were evaluated with the same resize, center crop, normalization, labels, and prediction rules. `benchmark/test_benchmark_results.json` and the two per-frame prediction CSVs contain the full results. The test set is small, and this comparison is one trained run; the result is descriptive rather than proof of a general performance difference.
