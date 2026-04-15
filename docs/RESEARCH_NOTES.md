# Research Notes

## Project: Skeleton-Guided ASL Recognition

**Author:** Paran Kafle  
**Institution:** University of Technology Sydney  
**Level:** Master's Thesis  
**Date:** 2025–2026

---

## Research Motivation

American Sign Language (ASL) is the primary language for approximately 500,000 deaf and hard-of-hearing individuals in the United States. Automated ASL recognition reduces communication barriers but remains challenging due to:

1. **Intra-class variability** — Same sign performed differently across individuals.
2. **Inter-class similarity** — Some signs are visually very similar (e.g., M/N, R/U).
3. **Lighting and background** — Real-world image noise degrades model accuracy.
4. **Limited datasets** — Large-scale annotated ASL datasets are scarce.

---

## Key Design Decisions

### Why Skip MediaPipe?

MediaPipe has known compatibility issues with certain Python/TensorFlow versions and platforms. This project instead uses:

- **OpenCV** for image preprocessing and skeleton visualization.
- **Custom keypoint extraction** via a lightweight CNN-based approach.
- **Synthetic skeleton graphs** derived from image features for GCN input.

This makes the pipeline more self-contained and reproducible without external binary dependencies.

### Why Sign Language MNIST?

- Publicly available on Kaggle, no license barriers.
- Standardized 28×28 grayscale images — easy to benchmark.
- 24 classes (A–Z excluding J and Z, which require motion).
- Widely used in research, enabling direct comparison with published results.

### Why Graph Neural Networks?

Hand keypoints form a natural graph (anatomical connections between finger joints). GCNs can exploit this relational structure, which flat CNN feature maps cannot capture directly. This is the **novel contribution** of this thesis.

---

## Model Architecture Rationale

### Custom CNN (Baseline)
- Simple 3-conv-layer network establishes lower-bound performance.
- Trained from scratch to understand dataset difficulty without pretraining.

### MobileNetV2
- Designed for mobile/web deployment with depthwise separable convolutions.
- Pretrained on ImageNet; fine-tuned on upscaled (96×96) ASL images.
- Best latency/accuracy trade-off for web deployment.

### EfficientNetB0
- Compound scaling balances depth, width, resolution uniformly.
- Expected highest raw accuracy among individual models.

### Skeleton GCN
- 21-node graph following standard hand topology (wrist + 4 joints per finger).
- Edge adjacency matrix encodes anatomical finger connections.
- Uses graph convolution layers (Kipf & Welling, 2017).
- **Expected advantage:** Rotation and scale invariance if skeleton is normalized.

### Attention CNN
- Squeeze-and-Excitation blocks add channel attention (Hu et al., 2018).
- Spatial attention refines focus to hand region.
- Ablation study isolates the contribution of each attention type.

### Ensemble
- Weighted average of CNN + transfer model + GCN predictions.
- Weights tuned on validation set.
- Expected to achieve best overall accuracy and robustness.

---

## Evaluation Strategy

| Method | Purpose |
|--------|---------|
| 80/10/10 stratified split | Preserve class balance in all splits |
| Early stopping (patience=10) | Prevent overfitting, select best epoch |
| Bootstrap CI (n=1000) | Quantify uncertainty in accuracy estimates |
| McNemar's test | Statistical significance between model pairs |
| Confusion matrix | Identify confused sign pairs |
| Per-class F1 | Find classes where model underperforms |
| Robustness tests | Assess real-world degradation |

---

## Expected Results (Literature Baselines)

| Model | Expected Accuracy (Sign MNIST) |
|-------|-------------------------------|
| Simple CNN | 92–95% |
| MobileNetV2 (fine-tuned) | 96–98% |
| EfficientNetB0 | 97–99% |
| GCN on skeleton | 88–94% |
| Ensemble | 97–99% |

*Note: Actual results will vary. GCN accuracy depends on keypoint extraction quality.*

---

## Known Challenges & Mitigations

| Challenge | Mitigation |
|-----------|-----------|
| J and Z require motion | Excluded from dataset (static-only task) |
| Similar signs (M/N, R/U) | Per-class analysis + attention mechanisms |
| Small dataset for GCN | Data augmentation + dropout |
| Web model size | Model quantization + TF.js conversion |
| Reproducibility | Fixed random seeds (42 throughout) |

---

## Future Work

1. Extend to dynamic signs (J, Z) using LSTM/Transformer on video sequences.
2. Incorporate 3D depth information from RGBD cameras.
3. Real-time continuous sentence recognition (not just isolated signs).
4. Cross-lingual extension to other sign languages (BSL, Auslan).
5. Federated learning for privacy-preserving model improvement.

---

## References

1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.
2. Howard, A. et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.
3. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
4. Hu, J. et al. (2018). Squeeze-and-Excitation Networks. *CVPR*.
5. Jochen, T. (2018). Sign Language MNIST. *Kaggle Dataset*.
