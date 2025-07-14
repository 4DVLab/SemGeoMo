# Semgeomo
This is the official implement for the paper: Semgeomo: Dynamic contextual human motion generation with semantic and geometric guidance.

<br>
<p align="center">
<h1 align="center"><strong>Semgeomo: Dynamic contextual human motion generation with semantic and geometric guidance
</strong></h1>
<p align="center">
  <a href="https://4dvlab.github.io/project_page/semgeomo/"><b>📖 Project Page</b></a> |
  <a href="https://arxiv.org/pdf/2503.01291"><b>📄 Arxiv Paper </b></a> |
</p>


## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+
- PyTorch 1.8+

### Environment Setup 🛠️

1. Clone the repository:
```bash
git clone https://github.com/your-repo/SemGeoMo.git
cd SemGeoMo
```

2. Create and activate conda environment:
```bash
conda create -n semgeomo python=3.8
conda activate semgeomo
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
Download **[Long_CLIP](https://github.com/beichenzbc/Long-CLIP.git)** and **[Pytorch3d](https://github.com/facebookresearch/pytorch3d.git)** package.


## Data Preparation 

* Preprocessed Data (.pkl Files)

We provide preprocessed .pkl files for the FullBodyManipulation dataset. Each .pkl file contains a dictionary with the following structure:

```python
motion_dict[name] = {
    "motion": T x 263,           # Motion features
    "length": len(motion),        # Number of frames
    "text": str,                  # Annotated text description
    "fine_text": str,             # Fine-grained texts from LLM
    "joint": T x 22 x 3,         # Joint positions
    "seq_name": str,              # Sequence name
    "id": int,                    # Sequence ID
    "obj_name": str,              # Object name
    "pc": T x 1024 x 3,          # Point cloud
    "root_trans": T x 3,         # Root translation
    "dis": T x 1024 x 2,         # Distance features
    "pc_bps": T x 1024 x 3,      # BPS features
    "obj_rot_mat": T x 3 x 3,    # Object rotation matrix
    "obj_trans": T x 3,          # Object translation
    "obj_scale": T                # Object scale
}
```

**Note**: If this is your first time training Stage 1, the `manip/data/hand_contact_data.py` will automatically process the original data into .pkl files. Preprocessed .pkl files are already available in the `/data_pkl` folder.

* Download relative package and models from the **[Google Drive.](https://drive.google.com/drive/folders/1iKANCKEV_FdAwv_0KNJ0JIDEAOIext30?usp=sharing)**


* Data Structure

The project expects the following directory structure:
```
SemGeoMo/
├── data_pkl/          # Preprocessed .pkl files
  ├── omomo_fps15/
├── pretrain/          # Pretrained models
├── bps/
├── glove/
├── body_models/
  ├──smpl
├──smpl_all_models/
  ├──smplx
├── Long_CLIP/
├── pytorch3d/
├── exp/               # Training outputs
├── semgeomo/  
  ├── t2m/       
```

## Training

### Stage 1: Geometric guidance Training

Train the first stage using hand-object contact data:

```bash
# Option 1: Use the provided script
bash scripts/train_stage1_omomo.sh

# Option 2: Run directly
python trainer_contact.py \
  --window=100 \
  --batch_size=64 \
  --project="./exp" \
  --exp_name="omomo-stage1" \
  --dataset_name="omomo" \
  --text=True
```

### Stage 2: Full-Body Motion Training

Train the second stage using pretrained MDM models:

```bash
# Option 1: Use the provided script
bash scripts/train_stage2_omomo.sh

# Option 2: Run directly
python trainer_fullbody.py \
  --window=100 \
  --batch_size=16 \
  --project="./exp" \
  --exp_name="omomo-stage2" \
  --dataset_name="omomo" \
  --save_dir="path/to/save" \
  --pretrained_path="path/to/pretrain/models"
```

**Note**: Ensure you have the correct path for `--pretrained_path`, which should point to the pretrained MDM models stored in the `/pretrain` folder.

## Testing

### Stage 1


```bash
# Option 1: Use the provided script
bash scripts/test_stage1.sh

# Option 2: Run directly
python sample_stage1.py \
  --window=100 \
  --batch_size=64 \
  --project="./exp" \
  --exp_name="omomo-stage1" \
  --add_hand_processing \
  --test_sample_res \
  --dataset_name="omomo" \
  --checkpoint="path/to/stage1/checkpoint"\
  --for_quant_eval \
  --text=True \
  --joint_together=True
```

### Full Pipeline Testing

```bash
# Option 1: Use the provided script
bash scripts/test_pipeline.sh

# Option 2: Run directly
python sample_pipeline.py \
  --window=100 \
  --batch_size=1 \
  --project="./exp" \
  --exp_name="omomo-test" \
  --dataset_name="omomo" \
  --run_whole_pipeline \
  --test_sample_res \
  --checkpoint="path/to/stage1/checkpoint" \
  --model_path="path/to/model" \
  --text=True \
  --for_quant_eval \
  --use_posterior
```

**Note**: 
- Fill in the appropriate paths for `--checkpoint` and `--model_path` when running the test pipeline
- The test pipeline evaluates 100 randomly selected samples
- For evaluation on the entire test set, run `bash scripts/test_all.sh`


## Citation 🖊️

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{cong2025semgeomo,
  title={Semgeomo: Dynamic contextual human motion generation with semantic and geometric guidance},
  author={Cong, Peishan and Wang, Ziyi and Ma, Yuexin and Yue, Xiangyu},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={17561--17570},
  year={2025}
}
```

## License 🎫

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the authors of [MDM](https://github.com/GuyTevet/motion-diffusion-model), [OMOMO](https://lijiaman.github.io/projects/omomo/), [InterControl](https://github.com/zhenzhiwang/intercontrol) and other related works for their contributions to the field.
