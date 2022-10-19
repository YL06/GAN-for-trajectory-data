# Generative Adversarial Networks for Trajectory Data

Evaluation implementation of privacy and utility of Two-Stage GAN (TSG) Trajectory Generation.

### Introduction

This repository contains the code for my Bachelor's thesis ***Generative Adversarial Networks for Trajectory Data.***
For the sake to run the project this contains some files **not** edited/written by me. Those files will be listed in a later section.
The original implementation can be found at ([Link](https://github.com/XingruiWang/Two-Stage-Gan-in-trajectory-generation)).



### Train

#### Dataset

Trajectory data in Porto, [available on Kaggle](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i)

#### Prepare the data

1. Transform trajectory data into grids format
	- cvs file to npy: run `/prepare_data/to_npy.py`
	- trajectory to grid form: run `First_stage_gan/to_grid.py`
2. Prepare the corresponding map images: 

   - go to `pre_process/map_generation/`

   - run `screen_shot.py`
   
   - run `cut.py`
   
   - run `merge.py`
   
#### First stage GAN

1. go to `First_stage_gan/`.

2. run: 

```sh
python WGANGP.py \
--dataroot ./grid32/ \
--labelroot ./traj_all.txt \
--outf ./output \
--batchSize 64 \
--n_critic 1 \
--netG ./output_IN/netG_epoch_320.pth \
--netD ./output_IN/netD_epoch_320.pth \
--cuda \
--start_iter 320 \
--niter 350
```

#### Second stage GAN

1. preprocess data:
	- run `TSG/save_points.py`
	- run `TSG/prep_data_2.py`
	- run `prepare_data/p_tonyp.py`
2. get map dict: run `prepare_data/connect.py`
3. run `Second_stage_gan/train.py`

#### Generate trajectory data


1. Coarse result generated from First stage GAN

```sh
cd First_stage_gan
python generate_fake_data.py --large_num 200 --model_path ./output_IN/netG_epoch_260.pth --output_path ../output_generated_coarse
```
2. Preprocess generated data
```sh
cd Connect
python preprocess_grid.py
```

3. Final result `TSG/pred.py`

   **Configurations**
   
   - `step_1_output` path to the result of first stage GAN
   - `map_dir` path to the map data
   - `checkpoint` model result of second stage GAN
   
### Privacy Evaluation

1. go to `mia/`
2. run `python attack_WGANgp.py --cuda`
   
### Utility Evaluation

#### Trajectory Length Preservation

run `Utility/traj_length_pres.py`

#### Similarity Measures

1. Hausdorff Distance: run `Utility/path_similarity.py`

2. Dynamic Time Warping: run `Utility/DTW.py`

#### plot Trajectory line

run `Utility/plot_line.py` 


### Unedited original files

List of unedited files:

In folder `First_stage_gan/`
- `generate_fake_data.py`
- `to_grid.py`
- `top_50.py`
- `vis_trj.py`
- `WGANGP.py`
- `traj_all.txt`

In folder `pre_process/`
- `map_generation/merge.py`
- `map_generation/screen_shot.py`
- `process_trajectory_data/generate_traj_plot.py`
- `process_trajectory_data/np_json.py`
- `process_trajectory_data/process_traj.py`
- `process_trajectory_data/transform.py`
- `process_trajectory_data/template/template.html`

In folder `Second_stage_gan/`
- `utils.py`
- `vis.py`
- `run.sh`
- `roadnetwork_best.pth`

In folder `TSG/`
- `connect.py`
- `get_length.py`
- `utils.py`
- `vis.py`
- `Untitled.ipynb`



### Edited files

List of edited files:

- `pre_process/map_generation/cut.py`
- `pre_process/map_generation/map.html`
- `Second_stage_gan/model.py`
- `Second_stage_gan/dataset.py`
- `Second_stage_gan/train.py`
- `TSG/model.py`
- `TSG/pred.py`


