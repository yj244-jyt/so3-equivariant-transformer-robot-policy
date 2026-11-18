# Third-Party Notices

This repository is an experiment portfolio and reproducibility record built on
the following research projects.

## EquiBot

- Project: **EquiBot: SIM(3)-Equivariant Diffusion Policy for Generalizable and
  Data Efficient Learning**
- Repository: <https://github.com/yjy0625/equibot>
- License: MIT
- Copyright: 2024 Stanford Interactive Perception and Robot Learning Lab

The files under `assets/equibot/` are figures assembled from locally generated
rollouts or locally computed experiment summaries using the EquiBot simulator.
The three `e0_*_original_s0_ckpt01999_ep00.gif` files are direct format
conversions of frozen local evaluation videos. No EquiBot source code is
redistributed here.

## RiEMann

- Project: **RiEMann: Near Real-Time SE(3)-Equivariant Robot Manipulation
  without Point Cloud Segmentation**
- Repository: <https://github.com/HeegerGao/RiEMann>
- License: MIT
- Copyright: 2023 PKU-EPIC

`assets/riemann/official_web_teaser.gif` is copied from the official RiEMann
repository under its MIT License. The SAPIEN GIFs under
`assets/riemann/sapien/` are format conversions of locally recorded expert
demonstration videos from the reconstructed environment. The low-quality
review GIFs and audit charts are derived from those same local recordings.
The proxy results are not the paper's private Mug-on-Rack benchmark.

The GIFs under `assets/rlbench/` are format conversions of locally recorded
RLBench `PickUpCup` rollout videos from the paired v5 evaluation. They are
included as reproducibility media; no RLBench source code is redistributed.

## ET-SEED

- Project: **ET-SEED: Efficient Trajectory-Level SE(3) Equivariant Diffusion
  Policy**
- Project page: <https://et-seed.github.io/>
- Paper: <https://proceedings.iclr.cc/paper_files/paper/2025/file/969a11f489c59c50c2cfe8c407c68a9b-Paper-Conference.pdf>
- Code: <https://github.com/yuechen0614/ET-SEED>

`assets/isaac_lab/et_seed_paper_simulation_environment.png` is a downloaded
copy of the teaser image published on the ET-SEED project page and is included
only as a cited environment reference. Copyright and attribution remain with
the ET-SEED authors.

## Scope Exclusion

No implementation source, checkpoint, training data, or architecture asset was
copied from the local `e2tsp-escnn` worktree. Architectural descriptions in the
README are limited to facts already recorded in the frozen experiment reports.
