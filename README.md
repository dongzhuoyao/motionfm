# Motion Flow Matching for Human Motion Synthesis and Editing

**Arxiv**


[![Website](doc/badges/badge-website.svg)](https://taohu.me/mfm)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/abs/2312.08895)
[![GitHub](https://img.shields.io/github/stars/dongzhuoyao/motionfm?style=social)](https://github.com/dongzhuoyao/motionfm)
[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)


# Citation
```
@inproceedings{motionfm,
  title = {Motion Flow Matching for Human Motion Synthesis and Editing},
  author = {Hu, Vincent Tao and Yin, Wenzhe and Ma, Pingchuan and Chen, Yunlu and Fernando, Basura and Asano, Yuki M. and Gavves, Efstratios and Mettes, Pascal and Ommer, Björn and Snoek, Cees G.M.},
  year = {2024},
 booktitle = {Arxiv},
}
```

# Training


##  KIT Dataset

```python
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 train_acc.py name=kit_trans_enc_512_4gpu dataset=kit training.eval_during_training=0 model.cond_mask_prob=0.1 guidance_param=2.5  training.overwrite=1 training.log_interval=1000 training.num_steps=300000 num_workers=12 input_text=./assets/example_text_prompts.txt   is_debug=0
```


##  Humanml Dataset


```python
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc-per-node=4  train.py name=humanml_trans_enc_512_4gpu_600k dataset=humanml training.eval_during_training=0 model.cond_mask_prob=0.1 guidance_param=2.5 training.overwrite=1 training.log_interval=1000 batch_size=128 training.num_steps=600000 num_workers=8  input_text=./assets/example_text_prompts.txt    is_debug=0
```





# Editing


```python
python edit.py dataset=humanml --model_path=./pretrained/humanml_trans_enc_512/model000200000.pt --edit_mode in_between
```


# Generation 

```python
python generate.py   dataset=kit model_path=./pretrained/kit_trans_enc_512/model000400000.pt input_text=./assets/example_text_prompts.txt 
python generate.py dataset=humanml model_path=./pretrained/humanml_trans_enc_512/model000475000.pt input_text=./assets/example_text_prompts.txt 
```


## Evaluation 




```python 
python eval_humanml.py dataset=kit dynamic=flow model_path=./outputs/kit_trans_enc_512_4gpu/07-09-2023/17-49-00/model000200000.pt guidance_param=2.5 eval_mode=mm_short ode_kwargs.step_size=0.02 is_debug=0
```

```python 
python eval_humanml.py dataset=kit dynamic=diffusion model_path=./pretrained/kit_trans_enc_512/model000400000.pt eval_mode=mm_short  guidance_param=2.5  diffusion_steps_sample=500 use_ddim=1 is_debug=0
```



```python 
python eval_humanml.py dataset=humanml dynamic=flow model_path=./outputs/humanml_trans_enc_512_3gpu_600k/08-09-2023/17-39-14/model000300000.pt guidance_param=2.5 eval_mode=wo_mm diffusion_steps_sample=-1 is_debug=0
```



## Visualize

```python 
python -m visualize.render_mesh --input_path demo_data/humanml_trans_enc_512/samples_humanml_trans_enc_512_000475000_seed10_example_text_prompts/sample00_rep00.mp4
```



## Environment Preparation

```
conda create -n motionfm  python=3.10
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pytorch-lightning torchdiffeq  h5py  diffusers accelerate loguru blobfile ml_collections ipdb
pip install hydra-core  einops scikit-learn --upgrade
conda install -c conda-forge ffmpeg
pip install numpy==1.23.0
pip install clearml wandb  sentencepiece transformers
pip install spacy  clip smplx chumpy
python -m spacy download en_core_web_sm
pip install matplotlib==3.2.0 #necessary for plot_3d_motion, https://github.com/GuyTevet/motion-diffusion-model/issues/41
pip install git+https://github.com/openai/CLIP.git
pip install gdown #downloading dataset needs it.
install ffmpeg
pip install wandb==0.14.2
```



## Dataset Preparation

KIT-ML dataset 

```bash
download from gdrive
unzip KIT-ML-20230906T121325Z-001.zip
 rar x new_joint_vecs.rar new_joint_vecs
 rar x new_joints.rar new_joints
 rar x texts.rar texts
```

Prepare the following data following [MDM](https://github.com/GuyTevet/motion-diffusion-model)
```bash
HumanML3D.tar.gz 
./pretrained 
./body_models 
./glove 
./kit 
./t2m 
```




## Acknowledgments

This code is standing on the shoulders of giants. We want to thank the following contributors
that our code is based on:

[guided-diffusion](https://github.com/openai/guided-diffusion), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [text-to-motion](https://github.com/EricGuo5513/text-to-motion), [actor](https://github.com/Mathux/ACTOR), [joints2smpl](https://github.com/wangsen1312/joints2smpl), [MoDi](https://github.com/sigal-raab/MoDi),[MDM](https://github.com/GuyTevet/motion-diffusion-model).

## License
This code is distributed under an [MIT LICENSE](LICENSE).

Note that our code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.
