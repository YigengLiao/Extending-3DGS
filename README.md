## Extending 3DGS: Color Prediction via MLP, Transmittance-Driven Visibility, and Tile-Based Training Rasterization

> A research-only code release, derived from [**3D Gaussian Splatting (3DGS)**](https://github.com/graphdeco-inria/gaussian-splatting).

* **Color via MLP**: replaces the spherical-harmonics color head with a compact MLP that directly predicts RGB from hash-encoded positional features, an encoded unit view-direction vector, and a learnable per-Gaussian material tensor of coefficients.
* **Transmittance-driven visibility**: substitutes α-sorting with a learned, segment-wise transmittance product along the clipped camera ray inside a unit cube.
* **Tile-based training rasterization**: enables **batched multi-image-tile** rendering (akin to NeRF random-ray training) by stacking tiles along a virtual batch axis for parallelism.
> **MLP & HashGridEncoder** are implemented via the Python interface of [**tiny-cuda-nn**](https://github.com/nvlabs/tiny-cuda-nn#pytorch-extension). (If you encounter installation issues with tiny-cuda-nn, you may refer to [this issue](https://github.com/NVlabs/tiny-cuda-nn/issues/195#issuecomment-1316275803) for solutions.)

> **Status**: early sample / WIP. Generalization is limited. Gaussian add/clone, pruning/merging, and LR-decay are not yet included.
---

## Contact

Please contact me if you have any questions at: [ygliao@tju.edu.cn](mailto:ygliao@tju.edu.cn).
