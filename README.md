# BrainLMM
Code of article "BrainLMM: A Label-Free Framework for Mapping Multi-Semantic Representation in the Human Visual Cortex"

## Install Dependencies

Ensure you have the necessary packages by referring to the requirements file. Set up a virtual environment and install the required packages:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install `torch` and `torchvision` from [PyTorch](https://pytorch.org/) tailored to your system configuration.

## Data acquisition

To proceed with the analysis, you'll need specific datasets. For a streamlined setup, consider downloading the required datasets directly from their respective sources, such as the [natural scene datasets](https://naturalscenesdataset.org/) and the NOD

## Model Training

This part is located within BrainLMM/model-training directory

### Configure Paths

Examine the sample `config.cfg` for guidance on setting up paths. Update it to include the local paths for your NSD data, NOD data, coco, etc.

The directory structure of result should be organized as follows where `ROOT` shall be your working directory:

```
ROOT/result/
├── output/          # For output files like encoding models and bootstrap results.
├── figures/         # For diagrams and images.
└── features/        # For files associated with model features.
```

### One-command run

`project_commands.sh` is designed to run all models for all subjects by default. 

To execute the script:

```bash
sh project_commands.sh
```

### Run single models

#### On NSD

To run a specific model on the NSD, utilize the command:

```bash
python run_model/run_model.py --subj $subj --roi SELECTIVE_ROI --model clip_vit
python run_model/run_model.py --subj $subj --roi SELECTIVE_ROI --model clip_visual_resnet
python run_model/run_alex.py --subj $subj
python run_model/run_resnet.py --subj $subj
```

#### On NOD

For the NOD dataset, run the model using:

```bash
python run_model/RN50_NOD.py --subj $subj
python run_model/ViT_NOD.py --subj $subj
```

### Bootstrap

To process bootstrap result, you can use:

``` bash
python analyze/process_bootstrap_results.py --subj $subj --model clip_vit --roi SELECTIVE_ROI
```

### Visualization

For visualizing results, execute the following scripts located in the `/plot` directory:

- **Noise Ceiling**: run `plot_noise_ceiling.py`
- **Model Performance**:
  - `plot_box_graph.py` generates a box plot illustrating the performance of all models on a specified subject.
  - `plot_multiple_models.py` displays the mean RSQ (R-squared) and its standard deviation for all models across various data sessions on a specific subject.
  - `plot_single_model.py` displays the mean RSQ and its standard deviation for a single model across different data sessions on a specific subject.
  - `plot_single_model_Pearson.py` assesses model performance using Pearson correlation coefficients.
- **Flatmap Visualization**: Utilize the script `plot_flatmap.sh`.


## Dissect

### Config Settings

You need to replace the dataset path, diffusion model path, BLIP path and tokensizer path with your local path before running the dissection model. Also, you can choose your running device, cpu or cuda


### Run for NSD

You can choose your target model and subject by running `describe_neurons.py` for NSD dataset
the resutls are shown in  `BrainLMM/run/results/NSD`

### Run for NOD

You can choose your target model and subject by running `describe_neurons.py` for NOD dataset
the resutls are shown in  `BrainLMM/run/results/NOD`
