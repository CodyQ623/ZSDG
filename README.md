# Zero-shot Defect Generation for Industrial Anomaly Detection
**Official implementation of ZSDG: A zero-shot defect generation framework using a Latent Diffusion Model to create realistic industrial anomalies.**


## Dataset
1. [RealIAD](https://huggingface.co/datasets/Real-IAD/Real-IAD/tree/main/realiad_256) (To pair with the corresponding MVTEC dataset, we only need the following 3 categories of RealIAD dataset: **toothbrush, vcpill, and zipper**)
2. [MVTEC AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
3. [PCB Dataset](https://pan.baidu.com/s/1PN70ACOY_gYBvmHakcoImg?pwd=1x4e)
4. [Steel Dataset](https://pan.baidu.com/s/1_BORNJrO4msD0OPEcVSc-Q?pwd=9m10#list/path=%2FCPANet) (We only use the **FSSD-12** dataset from this link)
5. [Texture-AD Dataset](https://huggingface.co/datasets/texture-ad/Texture-AD-Benchmark)
**P.S.**
**1. The following tutorial is based on the RealIAD dataset, but can be adapted to other datasets with minor modifications.**
**2. All the following bash commands should be run in the root folder**

## Env Install
```bash
conda env create -f environment.yaml
conda activate zsdg
```

## Data Preparation

1. **Download Dataset**: Create a new folder in the root directory named data, and 2 subfolders: mvtec and realiad. 
   - Download RealIAD dataset and unzip it under `./data/realiad`. Copy `reorganize.py` under `./prepare/realiad` to `./data/realiad` and run the following command to reorganize the dataset:
      ```bash
      cd data/realiad/
      python reorganize.py
      ```

      The generated data will be in `./data/realiad/realiad_reorg`. Now you can delete the original RealIAD dataset folders.
      The structure of `realiad_reorg` should look like this:
      ```
      ./realiad_reorg
      ├── result.xlsx
      ├── toothbrush
      │   ├── ground_truth
      │   ├── test
      │   └── train
      ├── vcpill
      ...
      ```
   
   - Download MVTEC AD dataset and unzip it under `./data/mvtec` and **rename pill to vcpill**. Copy `combine_mvtec.py` under `./prepare/realiad` to `./data/mvtec` and run the following command:
      ```bash
      cd data/mvtec/
      python combine_mvtec.py
      ```

      The generated data will be in `./data/mvtec/mvtec_reorg`. 
      The structure of `mvtec_reorg` should look like this:
      ```
      ./mvtec_reorg
      ├── toothbrush
      │   ├── combined
      │   └── source
      ├── vcpill
      ...
      ```
2. **Generate Foreground Masks**: Run the following command to generate foreground masks for the RealIAD dataset:
   ```bash
   cd prepare/realiad/
   python gen_fg.py   
   ```
   The generated data will be in `./data/realiad/realiad_fg`.

3. **Generate Cut-and-Paste Defects**: Run the following command to generate cut-and-paste synthetic defects for cold start training:
   ```bash
   cd prepare/realiad/
   python gen_fake_img.py
   ```
   The generated data will be in `./data/realiad/realiad_fake`, the structure of which should look like this:
   ```
   ./realiad_fake
   ├── toothbrush
   │   ├── mask
   │   ├── source
   │   └── target
   ├── vcpill
   ...
   ```

4. **Generate 4D Semantic Maps**: Run the following command to generate 4D semantic maps for the RealIAD dataset:
   ```bash
   cd prepare/realiad/
   python gen_real_semap.py
   ```
   This will create a folder `semap` and `prompt.json` under `./data/realiad/realiad_fake/$ITEMS`. The structure of `./data/realiad/realiad_fake` should now look like this:
   ```
   ./realiad_fake
   ├── toothbrush
   │   ├── prompt.json
   │   ├── mask
   │   ├── source
   │   ├── target
   │   └── semap
   ├── vcpill
   ...
   ```

5. 
   **P.S.**
   - **Make sure that you have changed the paths in all the scripts above, you can search for `TODO` in each script to locate the path configurations!**
   - You can modify the attribute vector in `./prepare/realiad/reorganize.py`, Line 10, by changing the corresponding values in the `attribute_vector` list. 
   - You can use `./prepare/realiad/vis_semap.py` to visualize the generated semantic maps by running:
      ```bash
      cd prepare/realiad/
      python vis_semap.py /path/to/semap
      ``` 



## Training

1. **Download SD Model**: Download [v1-5-pruned.ckpt](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt) and put it in `./models/`.

2. **Convert SD Weights**:
   ```bash
   python tool_add_control.py ./models/v1-5-pruned.ckpt ./models/control_sd15_ini.ckpt
   ```

3. **Start Cold Start Training**:
   ```bash
   python train_cs_realiad.py --include toothbrush
   ```
   There are other parameters you can adjust, see details in the script.
   The saved checkpoints will be in `./checkpoints_realiad/toothbrush_cs/`.


## Mask Generation Network

1. **Generate 6D Semantic Maps**: Run the following command to generate 6D semantic maps for mask generation:
   ```bash
   cd prepare/realiad/
   python gen_6d_semap.py
   ```
   The generated data will be in `./data/realiad/realiad_6dsemap`.

2. **Train Mask Generation Network**: Train Pix2Pix network for mask generation:
   ```bash
   cd pix2pix/
   python pix2pix.py --data_root /path/to/realiad_6dsemap
   ```
   There are other parameters you can adjust, see details in the script.
   The saved checkpoints will be in `./pix2pix/saved_models`.

3. **Test Mask Generation Network**: Test Pix2Pix network to generate masks:
   ```bash
   cd pix2pix/
   python test_realiad.py
   ```
   The generated masks will be in `./pix2pix/output_realiad`

## Discriminator Training
1. **Train Discriminator**: Train the discriminator network for anomaly detection:
   ```bash
   python train_discriminator.py --item toothbrush
   ```
   The saved checkpoints will be in `./GAN/checkpoints_discriminator/realiad/toothbrush`.

2. **Test Discriminator**: Test the discriminator network to evaluate its performance:
   ```bash
   python test_discriminator.py --checkpoint /path/to/discriminator/checkpoint --image /path/to/test/image
   ```
## Testing

1.  **Generate Test Data**: Run the following command to generate test data for the MVTEC dataset:
      ```bash
      python test_realiad.py --item toothbrush
      ```
      There are other parameters you can adjust, see details in the script.
      The generated synthetic defects will be in `./output/realiad/toothbrush`
