# Test-only follow-up findings

## 1. Metric provenance

The current result is the reproducible, test-only result:

- Data YAML: `dataset/yolo/carla.yaml`
- Test split: `dataset/yolo/test.txt`, 994 sequence-disjoint frames
- Pretrained checkpoint: `yolo11n.pt`
- Fine-tuned checkpoint: `runs/detect/outputs/training/guardian_yolo11n-3/weights/best.pt`
- Ultralytics call: `model.val(data=dataset/yolo/carla.yaml, split="test")`

The active checkpoint was produced by the `guardian_yolo11n-3` run, not the
older `guardian_yolo11n` run. Its exact saved arguments are copied to
`guardian_yolo11n-3_args.yaml`. That run used the same YAML, `split: val` during
training, 50 epochs, `imgsz: 1280`, and batch size 2.

The older pair (mAP50 91.96%, mAP50-95 61.95%) has no surviving artifact that
binds it to a YAML, split, or checkpoint. It therefore cannot be reproduced or
compared fairly. It is marked **SUPERSEDED / untrusted** in `provenance.json`;
the current test-only values are 85.30% and 58.32%.

## 2. Paired distance comparison

Only the 946 ground-truth pedestrians independently matched by both models are
included. Errors are predicted distance minus ground-truth distance.

| View | Model | MAE (m) | AbsRel | Bias (m) | Median error (m) |
|---|---|---:|---:|---:|---:|
| Overall | Pretrained | 2.36 | 13.15% | -0.21 | -0.81 |
| Overall | Fine-tuned | 2.44 | 13.38% | -1.33 | -1.56 |
| 0-10 m | Pretrained | 1.25 | 17.41% | +0.64 | -0.32 |
| 0-10 m | Fine-tuned | 1.25 | 16.91% | +0.12 | -0.72 |
| 10-20 m | Pretrained | 2.00 | 13.81% | +0.46 | -0.78 |
| 10-20 m | Fine-tuned | 2.11 | 14.58% | -0.31 | -1.31 |
| 20-30 m | Pretrained | 2.53 | 10.33% | -0.52 | -1.65 |
| 20-30 m | Fine-tuned | 2.53 | 10.39% | -1.76 | -2.06 |
| 30+ m | Pretrained | 4.03 | 10.86% | -1.88 | -2.74 |
| 30+ m | Fine-tuned | 4.29 | 11.41% | -4.11 | -3.92 |

Fine-tuned bias becomes strongly negative for small boxes: -4.38 m for
0-30 px, -1.83 m for 30-60 px, and approximately -0.39 m for boxes at least
120 px high. By truncation, its bias is -1.46 m for untruncated objects,
+0.50 m for truncation in (0, 0.25], and +0.97 m for (0.25, 0.5].

## 3. Geometric height assumption

Training labels contain 7,479 pedestrians. The labelled 3D pedestrian height
has mean 1.775 m and median 1.86 m. The median fitted geometric height
`box_height * distance / fy` is 1.876 m. The median visible-silhouette ratio
(`observed box height / projected labelled height`) is 1.010.

On fine-tuned test matches, fixed 1.7 m gives MAE 3.40 m, AbsRel 15.18% and
bias -0.69 m. A training-only learned height of 1.876 m improves MAE to
2.49 m and AbsRel to 11.86%, but over-corrects the signed bias to +1.65 m.
The training-only affine correction has near-zero bias (-0.02 m) but worse MAE
3.57 m and AbsRel 19.66%. The height assumption is therefore biased, especially
for small/truncated boxes; the learned constant is the better simple correction
in this test.

## 4. Critical denominators and confidence intervals

| Model | View | Recall numerator / denominator | Recall | False objects / predicted critical | False object rate | False frames / frames | False frame rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Pretrained | All | 88 / 160 | 55.0% | 35 / 123 | 28.5% | 10 / 994 | 1.01% |
| Fine-tuned | All | 96 / 160 | 60.0% | 42 / 138 | 30.4% | 21 / 994 | 2.11% |
| Pretrained | 2 m corridor | 23 / 54 | 42.6% | 18 / 41 | 43.9% | 9 / 994 | 0.91% |
| Fine-tuned | 2 m corridor | 28 / 54 | 51.9% | 22 / 50 | 44.0% | 14 / 994 | 1.41% |

The sequence-bootstrap 95% CIs are in `critical_denominators.json`. They are
wide for corridor recall because only 54 critical pedestrians are in that
subset.

## 5. 0-10 m recall diagnosis

The fine-tuned model missed 58 ground-truth pedestrians in the 0-10 m bin.
Among those misses: 35 extend past an image border, 42 overlap another
pedestrian at IoU > 0.1, and occlusion counts are 16 unoccluded, 21 level-1,
and 21 level-2. The raw truncation distribution is retained in
`zero_to_ten_diagnosis.json`.

Clipping ground-truth boxes to the 1242x375 image before matching produced
1,528 matched and 264 unmatched ground-truth objects over the complete
fine-tuned test set. The 30 worst 0-10 m misses are in `worst_misses/`; red
boxes are ground truth and green boxes are predictions.

## 6. Corrected distance-bin protocol

Matched pairs are assigned to exactly one bin using ground-truth distance:

| Ground-truth bin | Matched | Unmatched GT |
|---|---:|---:|
| 0-10 m | 276 | 58 |
| 10-20 m | 347 | 33 |
| 20-30 m | 429 | 31 |
| 30+ m | 464 | 154 |

Unmatched predictions are reported separately by their predicted geometric
distance: 13 in 0-10 m, 5 in 10-20 m, 12 in 20-30 m, and 23 in 30+ m.

## Bug fixes

See `BUG_FIXES.txt`. No checkpoint was retrained, replaced, or modified.
