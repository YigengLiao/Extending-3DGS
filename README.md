## Extending 3DGS: Color Prediction via MLP, Transmittance-Driven Visibility, and Tile-Based Training Rasterization 
 
> A research-only code release, derived from [**3D Gaussian Splatting (3DGS)**](https://github.com/graphdeco-inria/gaussian-splatting). 
 
* Provides an implementation that replaces the **color head** and the **transmittance/visibility pipeline** (as an alternative to α-sorting), and extends the **rasterizer** to support **multi-image-tile parallel rendering** for training (analogous to NeRF’s random-ray training). 
* **MLP & HashGridEncoder** are implemented via the Python interface of [**tiny-cuda-nn**](https://github.com/nvlabs/tiny-cuda-nn#pytorch-extension). (If you encounter installation issues with tiny-cuda-nn, you may refer to [this issue](https://github.com/NVlabs/tiny-cuda-nn/issues/195#issuecomment-1316275803) for solutions.) 
 
> **Status**: early sample / WIP. Generalization is limited (see Performance). Gaussian add/clone, pruning/merging, and LR-decay are not yet included. 
---

## Contents 
 
* [Color Estimation](#color-estimation) 
* [Transmittance Computation](#transmittance-computation) 
* [Multi-Image-Tile Rendering](#multi-image-tile-rendering) 

---
## Color Estimation 
 
Color is predicted with a compact MLP driven by directional and positional features plus a per-Gaussian material code. 
 
Let $v\in\mathbb{R}^3$ be the (normalized) view direction. Let $\Phi_{\mathrm{SH}}(v)$ be SH features up to degree $D$ and $\psi(x)$ be the hash-encoded position features. With per-Gaussian material vector $m$, 
 
$$
\begin{aligned}
f_{\mathrm{in}}(x,v) &= [\, \Phi_{\mathrm{SH}}(v)\ \Vert\ \psi(x)\ \Vert\ m \,],\\
\tilde{c} &= \mathrm{MLP}\big(f_{\mathrm{in}}\big),\qquad
c = \sigma(\tilde{c})\in(0,1)^3.
\end{aligned}
$$

---
## Transmittance Computation 
 
α-sorting is replaced by estimating a **segment transmittance product** along the ray segment clipped to the unit box $\mathcal{B}=[0,1]^3$. 
 
#### 1) Clip a ray segment to the unit cube 
 
Given start point $\mathbf{p}\in\mathbb{R}^3$ and direction $\mathbf{v}\in\mathbb{R}^3$, consider $\mathbf{r}(t)=\mathbf{p}+t\mathbf{v}$ for $t\in[0,1]$.  
For each axis $j\in\{x,y,z\}$ and planes $x_j\in\{0,1\}$, 
 
$$
t^{(0)}_j=\frac{0-p_j}{v_j},\quad
t^{(1)}_j=\frac{1-p_j}{v_j},\quad
t^{\min}_j=\min\{\,t^{(0)}_j,\,t^{(1)}_j\,\},\quad
t^{\max}_j=\max\{\,t^{(0)}_j,\,t^{(1)}_j\,\}.
$$
 
Aggregate entrance/exit: 
 
$$
t_{\mathrm{enter}}=\max_j\, t^{\min}_j,\qquad
t_{\mathrm{exit}}=\min_j\, t^{\max}_j.
$$
 
Clamp to $[0,1]$: 
 
$$
t_0=\mathrm{clip}(t_{\mathrm{enter}},0,1),\quad
t_1=\mathrm{clip}(t_{\mathrm{exit}},0,1),\quad
\lambda=\max\{\,t_1-t_0,\,0\,\}.
$$
 
If $\lambda>0$, the **clipped segment** is 
 
$$
\mathbf{s}=\mathbf{p}+t_0\mathbf{v},\qquad
\mathbf{w}=\lambda\,\mathbf{v},\qquad
L=\| \mathbf{w} \|_2.
$$
 
#### 2) Midpoint sampling and transmittance product 
 
Choose a step parameter $\sigma>0$.  
Number of subsegments: 
 
$$
K=\left\lceil \frac{L}{\sigma}\right\rceil,\quad
\Delta s=\frac{L}{K},\quad
s_{\mathrm{factor}}=\frac{\Delta s}{\sigma}.
$$
 
Midpoint samples: 
 
$$
\tilde{t}_k=\frac{k+\tfrac12}{K},\quad
\mathbf{x}_k=\mathbf{s}+\tilde{t}_k\,\mathbf{w},\quad k=0,\ldots,K-1.
$$
 
Per-point features $\mathbf{z}_k=[\,\phi(\mathbf{x}_k),\,\mathbf{u}\,]$ (hash feature + extra attr).  
Scalar network output $a_k=\mathrm{net}(\mathbf{z}_k)$.  
Per-step factor: 
 
$$
\rho_k=\big(\sigma_{\mathrm{sig}}(a_k)\big)^{s_{\mathrm{factor}}},\quad
\sigma_{\mathrm{sig}}(a)=\frac{1}{1+e^{-a}}.
$$
 
**Segment transmittance:** 
 
$$
T=\prod_{k=0}^{K-1}\rho_k.
$$
 
#### 3) Weighting and color accumulation  
Let per-Gaussian opacity (after our transmittance) be $\alpha_i = \mathrm{opacity}_i \cdot T_i$.  
For a pixel, accumulate 
 
$$
\mathbf{C} = \sum_i \alpha_i\,\mathbf{c}_i,\qquad
S=\sum_i \alpha_i.
$$
 
Final pixel: 
 
$$
\mathbf{I}=
\begin{cases}
\mathbf{C}/S,& S>1\\
\mathbf{C}+(1-S)\,\mathbf{b},& S\le 1
\end{cases}
$$
 
where $\mathbf{b}$ is the background color. 

---
## Multi-Image-Tile Rendering 
 
The renderer processes $\{\mathcal{B}_t\}_{t=1}^K$ square tiles, each of side length $S$, **in a single batched pass** by stacking tiles along a virtual batch axis. 
 
#### 1) Setup 
 
Partition the full image plane into tiles $\{\mathcal{B}_t\}_{t=1}^K$, where each tile $t$ is an axis-aligned square *(any integer-rounded padding introduced during preprocessing is marked and excluded from the loss during training)*: 
 
$$
\mathcal{B}_t=\{\,\mathbf{p}\in\mathbb{R}^2 \mid \mathbf{o}_t \le \mathbf{p} < \mathbf{o}_t+(S,S)\,\}.
$$
 
with elementwise inequalities and an integer tile origin $\mathbf{o}_t\in\mathbb{Z}^2$.  
Assume each Gaussian $i$ has a **screen-space center** $\mathbf{x}_i$ and a **support radius** $r_i>0$ (e.g., derived upstream from its 2D footprint). 
 
#### 2) Tile-relative shifting 
 
For tile $t$, define tile-relative centers 
 
$$
\mathbf{x}_i^{(t)}=\mathbf{x}_i-\mathbf{o}_t .
$$
 
A Gaussian $i$ is a **candidate for tile $t$** if its support overlaps the tile: 
 
$$
\mathcal{P}_t=\{\, i \mid \mathrm{dist}(\mathbf{x}_i,\mathcal{B}_t)\le r_i \,\}.
$$
 
#### 3) Batched launch and memory layout 
 
A 3D CUDA grid covers per-tile pixels on $(x,y)$ and indexes tiles along the **$z$-axis**. The output tensor is 
 
$$
\mathbf{Y}\in\mathbb{R}^{K\times C\times S\times S},
$$
 
stored contiguously by tile; the base write offset for tile $t$ is $t\cdot(C S S)$. 

---
## Contact 
 
Please contact me if you have any questions at: [ygliao@tju.edu.cn](mailto:ygliao@tju.edu.cn).
