
**SkinDisNet: A lightweight multi-scale dual-attention network for skin disease classification**

Official implementation of the paper **“SkinDisNet: A lightweight multi-scale dual-attention network for skin disease classification.”**

SkinDisNet is a lightweight deep learning model for dermatological image classification. The architecture combines a MobileNetV3-Small backbone with three task-specific modules: a Multi-Scale Context Block (MSCB), a Local Refinement Block (LRB), and an Adaptive Dual Attention Module (DAM). The model is designed to improve classification performance while maintaining low computational cost for deployment in resource-constrained environments.
## Highlights

- Lightweight architecture with **1.04M parameters**
- Multi-scale feature fusion through **MSCB**
- Local texture refinement through **LRB**
- Parallel channel-spatial attention through **DAM**
- Post-hoc calibration using **temperature scaling**
- Explainability analysis using **Grad-CAM**
- Suitable for edge and web-based deployment

## Overview

Skin diseases often exhibit high inter-class similarity and intra-class variation in color, texture, and lesion structure. SkinDisNet addresses these challenges through:

- **Backbone feature extraction** using MobileNetV3-Small
- **Multi-scale contextual fusion** to aggregate shallow and deep features
- **Local refinement** to preserve fine-grained texture cues
- **Adaptive dual attention** to enhance lesion-relevant channel and spatial responses

## Architecture

SkinDisNet consists of the following components:

1. **MobileNetV3-Small backbone**
2. **MSCB** – Multi-Scale Context Block
3. **LRB** – Local Refinement Block
4. **DAM** – Adaptive Dual Attention Module
5. **Classification head**




