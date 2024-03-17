
<div align="center">
<img src="https://dl3.pushbulletusercontent.com/qzGTBQatkBVyqaRaJt1rgaLuBPXLmS08/ncalogo-removebg-preview.png" alt="logo" width="50%"></img>
</div>

# Master's Thesis: AI Generating Algorithms with Self-Organizing Neural Cellular Automata
<a target="_blank" href="https://colab.research.google.com/drive/1HYKttER_0I6HD1y1oDdg_MLG0D21vMxB?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a> <a target="_blank" href="https://www.python.org/dev/peps/pep-0008/">
  <img src="https://img.shields.io/badge/code%20style-PEP%208-blueviolet.svg" alt="python style"/>
</a> <a target="_blank" href="https://nbviewer.org/urls/nca-exp10.s3.amazonaws.com/NonUniform_NCA_v1_1_colab_7Dec.ipynb">
  <img src="https://user-images.githubusercontent.com/2791223/29387450-e5654c72-8294-11e7-95e4-090419520edb.png" alt="python style"/>
</a> 




## The Project
<a target="_blank" href="https://www.hiof.no/english/">
  <img src="https://dl3.pushbulletusercontent.com/IeVNnAuYX4XQODiVTL9Ln3VWATf5azCc/hio.png" alt="HiOF logo" width="200px"/>
</a><br/>
This repo is code supplement for MSc. thesis at <a href="https://www.hiof.no/english/">HiØ</a> under supervision of prof <a href="https://www.nichele.eu/">Stefano Nichele</a> for the June, 2024.

This project implements the Neural CA Frameowork where self-replication with mutation is the only way for cells to live longer. We study and Analyse the growht in Phenotypic Diversity (PD) and Genetic Diversity (GD) over longer time steps. Specifically, we intend to study, formalise and propose three tools to analyse GD and four tools for PD. Specifically, GD level tools are (1) Random Weight Selection Plot (RWSP), (2) Clustering Neural Weights Approach (CNWA) and (3) Genotypic Hash Coloring (GHC). PD level tools include (1) Cellular Type Frequency Plot (CTFP), (2) Global Entropy Plot (GEP), (3) Gross Cell Variance Plot (GCVP), and (4) Cell Local Organisation Global Variance (CLOGV).

## Quick Demo NCA Video (Redirects YouTube)
[![NCA Video](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/f4dfb09b-014e-46a1-9490-b7b9969825f7)](https://youtu.be/tNrphmZuk0Y)

## CTFP
![CTFP](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/d001e49a-1d8e-4b5f-9c98-bb085da85a95)

## GEP+GCVP+CLOGV - Phenotypic Diversity
![GEP+GCVP+CLOGV](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/8cbd6761-de48-4d7e-bfd4-4e8dfa11ed34)

Research Questions:
1. What are the key species that emerge, survive, reproduce and becomes extinct (dead)
in the evolved NCA model?
2. How does the genotypic diversity evolve over successive generations of the NCA
model?
3. What is the impact of the self-replication and self-maintenance processes, characteristic
of autopoiesis, on the genetic makeup of the evolving NCAs?
4. How do phenotypic variations manifest within the NCA model, and what are the
underlying factors contributing to this diversity?
5. To what extent do the cellular activities within the NCA exhibit sensitivity to initial
conditions (for example number of time steps) and how does this influence the genetic
and phenotypic outcomes?

Objectives:
1. Analyze the species composition within the evolved NCA model using Cellular Type
Frequency Plot and identify dominant and rare species.
2. Implement Genotypic Hash Coloring and Random Weight Selection Plot to quantify
and visualize changes in genotypic diversity across multiple generations of NCAs.
3. Apply Clustering Neural Weights Approach to understand how self-replication and
autopoiesis contribute to the organization and maintenance of genetic traits within
the NCAs.
4. Utilize Phenotypic Diversity tools, including Global Entropy Plot and Gross Cell
Variance Plot, to assess and quantify the diversity in emergent phenotypes over the
course of NCA evolution.
5. Investigate the impact of dynamic landscapes on the evolution of smart and interesting
phenomena, exploring adaptability and evolvability within the NCA model.
6. Evaluate the effectiveness of the proposed tools in providing fine-grained and coarse-
grained analysis of Non-Uniform Self-Replicating NCAs.
7. Assess the potential applications of the research findings in advancing the understanding
of life, biology, and complex systems, as well as their implications for artificial general
intelligence.

Please refer to report document for further conceptual and implementation details.

### Contents
* [Open in Colab](#open-in-colab)
* [Using SLURM for Larger Experiments](#using-slurm-for-larger-experiments)
* [Configuration on Simula eX3](#configuration-on-simula-ex3)
* [SageMaker instance](#sagemaker-instance)
* [Cite this](#cite-this)
* [Sample 900 steps](#sample-900-steps)
* [Working pipeline](#working-pipeline)

## Open in Colab
<a target="_blank" href="https://colab.research.google.com/drive/1HYKttER_0I6HD1y1oDdg_MLG0D21vMxB?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Use the colab button above to directly start reproducing the code. You can also import [1_colab.ipynb](https://github.com/s4nyam/MasterThesis/blob/main/1_colab.ipynb) into your notebook / ipython environment. The advantage of Google Colab is to debug and quickly test the actual framework. We set the following parameters in first cell of the notebook and rest of the cooking material evolves the NCA and generate corresponding results.

```python
import torch
precision = 1
torch.set_printoptions(precision=precision)
WIDTH, HEIGHT = 30,30
grid_size = (WIDTH, HEIGHT)
print("Width and Height used are {} and {}".format(WIDTH, HEIGHT))
INIT_PROBABILITY = 0.02
min_pixels = max(0, int(WIDTH * HEIGHT * INIT_PROBABILITY))
NUM_LAYERS = 2  # One hidden and one alpha
ALPHA = 0.6  # To make other cells active
INHERTIANCE_PROBABILITY = 0.2  # probability that neighboring cells will inherit by perturbation.
parameter_perturbation_probability = 0.2
print("Numbers of layers used are {}".format(NUM_LAYERS))
print("1 for alpha layer and rest {} for hidden".format(NUM_LAYERS-1))
NUM_STEPS = 90
num_steps = NUM_STEPS
print("Numbers of Time Steps are {}".format(NUM_STEPS))
DEVICE = torch.device("cuda"  if torch.cuda.is_available() else  "cpu")
print(f"Using device: {DEVICE}")
activation = 'sigmoid'  # ['relu','sigmoid','tanh','leakyrelu']
frequency_dicts = []
FPS = 2  # Speed of display for animation of NCA and plots
marker_size = 2  # for plots
everystep_weights = [] # Stores weigths of the NNs from every time step.
```
With this configuration it simulates the proposed NCA framework. And then results a simulation like:
![sim](https://github.com/s4nyam/MasterThesis/assets/13884479/6854c5f4-dc25-404d-874a-28e97f363df0)


Beyond that it produces more results that are related to Phenotypic Diversity (PD) and Genotypic Diversity. PD tools are shown below:
<div align="center">
<img src="https://dl3.pushbulletusercontent.com/7bI309gDF8A9kVN8bv73gQKMwkRo22io/pdtools.gif" width=100% alt="logo"></img>
</div>

And GD tools for example 
* Genotypic Hash Coloring
<div align="center">
<img src="https://dl3.pushbulletusercontent.com/GRf9FgtTCiyHPbIvemcVvS9qgDNpAItq/download.gif" width=50% alt="logo"></img>
</div>

* Speciation Plot using Clustering Neural Weights Approach:
<div align="center">
<img src="https://dl3.pushbulletusercontent.com/0HSjDgttZncmwsen0wyc186r7h3fYBUs/download.png" width=50% alt="logo"></img>
</div>

* Random Weight Selection Plot 
<div align="center">
<img src="https://dl3.pushbulletusercontent.com/VIXuT827El7mWMtapTpfZh5Nahqaw5LN/rwsp.gif" width=50% alt="logo"></img>
</div>

## Using SLURM for Larger Experiments
We use Simula eX3 cluster in support of Research Council Norway. We use a100q, dgx2q and hgx2q machines to run our larger experiments. The batch script looks like:
```bash
#!/bin/bash
#SBATCH -p a100q ## hgx2q, a100q,  dgx2q (old)
#SBATCH --output=run.out    ## Output file
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -t 13-00:00 # time (D-HH:MM)
#SBATCH --mail-user=XXXXX
## module load cuda11.0/toolkit/11.0.3 
## module load cudnn8.0-cuda11.0
#module load cuda11.3/toolkit/11.3.0 
#module load cudnn8.0-cuda11.3
module use  /cm/shared/ex3-modules/0.4.1/modulefiles
##module load slurm/20.02.7
##module load python-3.9.15
# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate new_env
srun python -u run.py 
```
However the run.py file contains similar code that we have seen in Google Colab, but only produces large archives for evolved weights and the grid, which can further be analysed using the SageMaker notebook. For instance, the ```run.py``` file has following ingredient:
```python
precision = 1
WIDTH, HEIGHT = 100,100
grid_size = (WIDTH, HEIGHT)
INIT_PROBABILITY = 0.0030
min_pixels = max(0, int(WIDTH * HEIGHT * INIT_PROBABILITY))
NUM_LAYERS = 2 # One hidden and one alpha
ALPHA = 0.6 # To make other cells active
INHERTIANCE_PROBABILITY  = 0.2 # probability that neighboring cells will inherit by perturbation.
parameter_perturbation_probability = 0.2
NUM_STEPS = 1000, 2000, 3000, ..., 10000
num_steps = NUM_STEPS
activation = 'sigmoid' # ['relu','sigmoid','tanh','leakyrelu']
FPS = 1 # Speed of display for animation of NCA and plots
marker_size = 2 # for plots
frequency_dicts = []
everystep_weights = [] 
ca_grids_for_later_analysis_layer0 = []
ca_grids_for_later_analysis_layer1 = []
...
# Evolve NCA weights and grid
...
with open('frequency_dicts.pkl', 'wb') as f1, open('everystep_weights.pkl', 'wb') as f2, open('ca_grids_layer0.pkl', 'wb') as f3, open('ca_grids_layer1.pkl', 'wb') as f4:
        pickle.dump(frequency_dicts, f1)
        pickle.dump(everystep_weights, f2)
        pickle.dump(ca_grids_for_later_analysis_layer0, f3)
        pickle.dump(ca_grids_for_later_analysis_layer1, f4)
```
We release all required code in the ```main runs``` folder. The requirements to start with the SLURM code is as follows. 
+ Create environment and install using ```requirements.txt```
```
conda create --name myenv 
conda activate myenv 
conda install --file requirements.txt
```
+ ```requirements.txt```
```
# NumPy for numerical operations 
numpy==1.21.0 
# Matplotlib for plotting 
matplotlib==3.4.3 
# PyTorch for machine learning 
torch==1.10.0 
# Pandas for data manipulation and analysis 
pandas==1.3.3
```
## Configuration on Simula eX3
Following configuration was used with multiple nodes of each platform to fasten the multi-fold experiments. 
| Platform  | Model   | GPU            | Processor                                      | 
|-----------|---------|----------------|------------------------------------------------|
| x64_64/A100 | hgx2q | g002 | DualProcessor AMD EPYC Milan 7763 64-core w/ 8 qty Nvidia Volta A100/80GB |
| x64_64/V100 | dgx2q | g001 | DualProcessor Intel Xeon Scalable Platinum 8176 w/ 16 qty Nvidia Volta V100 |

## SageMaker instance
We use AWS SageMaker for the easiness to process experimental results. Specifically, for the exports that we receive from SLURM experiments, which are as pickle files containing evolved weights and stored grids. SageMaker notebook instance used with almost 500 GB of storage and following system configurations as per the size of experiment exports.
| Instance Type    | vCPU | Memory  | Price per Hour | Experiment size |
|------------------|------|---------|-----------------|-------------|
| ml.r5.large      | 2    | 16 GiB  | $0.151         |<500 MB|
| ml.r5.xlarge     | 4    | 32 GiB  | $0.302         |<1 GB|
| ml.r5.2xlarge    | 8    | 64 GiB  | $0.605         |<2 GB|
| ml.r5.4xlarge    | 16   | 128 GiB | $1.21          |<10 GB|
| ml.r5.8xlarge    | 32   | 256 GiB | $2.419         |<15 GB|
| ml.r5.12xlarge   | 48   | 384 GiB | $3.629         |<20 GB|
| ml.r5.16xlarge   | 64   | 512 GiB | $4.838         |<30 GB|
| ml.r5.24xlarge   | 96   | 768 GiB | $7.258         |<100 GB|

This classification of Experiment Size is done to avoid runtime collapse because of the memory outage during the processing of the results for the experiments. We could have produced results on runtime while experimenting using SLURM experiments, however to modularise and balance the workload, we preferred to save the weights for one time, and then process results later if required. This also brought flexibility to improve existing code bases for processing results and add more tools later for our project. We release ```2_sagemaker.ipynb``` to process the experiments. This notebook takes public URLs of the experimentation and process results. We also release public URLs of the experiments that can be processed using this notebook.

| Experiment | Download Link                                      |
|------------|-----------------------------------------------------|
| exp1s      | [Download exp1s.tar](https://nca-exp10.s3.amazonaws.com/exp1s.tar)  |
| exp2s      | [Download exp2s.tar](https://nca-exp10.s3.amazonaws.com/exp2s.tar)  |
| exp3s      | [Download exp3s.tar](https://nca-exp10.s3.amazonaws.com/exp3s.tar)  |
| exp4s      | [Download exp4s.tar](https://nca-exp10.s3.amazonaws.com/exp4s.tar)  |
| exp5s      | [Download exp5s.tar](https://nca-exp10.s3.amazonaws.com/exp5s.tar)  |
| exp6s      | [Download exp6s.tar](https://nca-exp10.s3.amazonaws.com/exp6s.tar)  |

Further we release results for all these raw experiments here at this [Drive Link](https://bit.ly/smallruns)

## Cite this
To cite this repository:

```
@software{aiganca2023github,
  author = {Sanyam Jain and Stefano Nichele},
  title = {{AIGA}: Self-Replicating NCA with Coarse Grained Analysis},
  url = {http://github.com/s4nyam/MasterThesis},
  version = {1.0.1},
  year = {2023},
}
```


## Sample 900 steps

https://github.com/s4nyam/MasterThesis/assets/13884479/57d5e84b-9381-445e-b9d3-342f065d89b7


## Working pipeline

![IMG_2405](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/00ff2275-04b9-486c-a97a-686e249748d8)

