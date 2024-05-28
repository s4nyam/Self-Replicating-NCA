
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

This project implements the Neural CA Frameowork where self-replication with mutation is the only way for cells to live longer. We study and Analyse the growht in Phenotypic Diversity (PD) and Genetic Diversity (GD) over longer time steps. Specifically, we intend to study, formalise and propose three tools to analyse GD and four tools for PD. Specifically, GD level tools are (1) Random Weight Selection Plot (RWSP), (2) Genotypic Hash Coloring (GHC) and (3) Combined count plot of unique colors. PD level tools include (1) Cellular Type Frequency Plot (CTFP), (2) Global Entropy Plot (GEP), (3) Gross Cell Variance Plot (GCVP), and (4) Cell Local Organisation Global Variance (CLOGV).

## Quick Demo Configuration:
```python
WIDTH, HEIGHT = 300,300
INIT_PROBABILITY = 0.1
NUM_LAYERS = 2
ALPHA = 0.5 
INHERTIANCE_PROBABILITY  = 0.2
parameter_perturbation_probability = 0.2
NUM_STEPS = 300
activation sigmoid on board
FPS = 10
budget = 4  # Cells die after 4 generations
```

Visit youtube channel for Demos - https://www.youtube.com/@growingnca/playlists
![Screenshot 2024-05-28 at 12 18 03 PM](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/cb0567c2-c8ce-4c83-833d-afb074b5adfd)

## Quick Demo NCA Video (Redirects YouTube)
https://www.youtube.com/watch?v=kD6vfKnRA-I&list=PL_IJ0j36aBCbmN7_OvWWdjNVw-QhXFfZo&index=1&pp=gAQBiAQB8AUB

## CTFP
![Screenshot 2024-05-28 at 12 19 20 PM](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/1fe35895-83c4-42f7-941b-59731bd7a49d)

## GEP+GCVP+CLOGV - Phenotypic Diversity
![Screenshot 2024-05-28 at 12 19 32 PM](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/8498b34a-746f-4126-941d-3660e2f6a62b)


## Quick Demo RWSP Video (Redirects YouTube)
https://www.youtube.com/watch?v=2fDUCPR4kk8&list=PL_IJ0j36aBCbmN7_OvWWdjNVw-QhXFfZo&index=2&pp=gAQBiAQB8AUB

## Quick Demo GHC Video (Redirects YouTube)
https://www.youtube.com/watch?v=KbvLTyEpMYM&list=PL_IJ0j36aBCbmN7_OvWWdjNVw-QhXFfZo&index=3&pp=gAQBiAQB8AUB

## RWSP and GHC Count Plot
![Screenshot 2024-05-28 at 12 20 09 PM](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/12ced9af-88dd-4be2-b16f-30138bdb71b0)

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
* [Experiments](#experiments)
* [Cite this](#cite-this)

## Open in Colab
<a target="_blank" href="https://colab.research.google.com/drive/1HYKttER_0I6HD1y1oDdg_MLG0D21vMxB?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Use the colab button above to directly start reproducing the code. You can download the colab and import into your notebook / ipython environment. The advantage of Google Colab is to debug and quickly test the actual framework. We set the following parameters in first cell of the notebook and rest of the cooking material evolves the NCA and generate corresponding results.

### Experiment 1
```python
precision = 1
torch.set_printoptions(precision=precision)
WIDTH, HEIGHT = 30,30
grid_size = (WIDTH, HEIGHT)
print("Width and Height used are {} and {}".format(WIDTH, HEIGHT))
INIT_PROBABILITY = 0.1
min_pixels = max(0, int(WIDTH * HEIGHT * INIT_PROBABILITY))
NUM_LAYERS = 2 # rest hidden and one alpha
ALPHA = 0.5 # To make other cells active (we dont go with other values below 0.6 to avoid dead cells and premature livelihood)
INHERTIANCE_PROBABILITY  = 0.2 # probability that neighboring cells will inherit by perturbation.
parameter_perturbation_probability = 0.2
print("Numbers of layers used are {}".format(NUM_LAYERS))
print("1 for alpha layer and rest {} for hidden".format(NUM_LAYERS-1))
NUM_STEPS = 90
num_steps = NUM_STEPS
at_which_step_random_death = 9999999999 # Set this to infinity or high value if you never want to enter catastrophic deletion (random death happens at this generation)
probability_death = 0.004 # 40 pixels die every generation
print("Numbers of Time Steps are {}".format(NUM_STEPS))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
activation = 'sigmoid' # ['sigmoid','tanh','noact']
frequency_dicts = []
FPS = 10 # Speed of display for animation of NCA and plots
marker_size = 1 # for plots
everystep_weights = [] # Stores weigths of the NNs from every time step.
enable_annotations_on_nca = True
budget_per_cell = 3
fixed_value = 0
budget_counter_grid = np.zeros((WIDTH, HEIGHT)) + fixed_value
```
With this configuration it simulates the proposed NCA framework. And then results a simulation like:
![sim](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/9217f724-5785-4ede-b026-13f63ac9301b)


Beyond that it produces more results that are related to Phenotypic Diversity (PD) and Genotypic Diversity. PD tools are shown below:

* CTFP Plot - Frequency Count Plot

![download](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/f494caca-4dc0-4efa-8cbc-3a2cc86ffdbf)

* GEP+GCVP+CLOGV (Phenotypic Diversity and Randomness)

![download](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/398631f9-b709-4ed3-b56a-0c6b41db30e5)

* GD Unique Color Count Plots (GHC and RWSP)

![download-1](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/c399e50e-9e1c-492a-8fb3-96f176447423)

* RWSP Animation

![download-ezgif com-video-to-gif-converter](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/8b4eda32-c45f-42fd-8ca1-208e4929418e)

* GHC Animation

![download-ezgif com-video-to-gif-converter](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/6b1b91f0-23a3-41e4-bf31-b65ee8ba576c)


### Experiment 2
```python
precision = 1
torch.set_printoptions(precision=precision)
WIDTH, HEIGHT = 30,30
grid_size = (WIDTH, HEIGHT)
print("Width and Height used are {} and {}".format(WIDTH, HEIGHT))
INIT_PROBABILITY = 0.09
min_pixels = max(0, int(WIDTH * HEIGHT * INIT_PROBABILITY))
NUM_LAYERS = 2 # rest hidden and one alpha
ALPHA = 0.5 # To make other cells active (we dont go with other values below 0.6 to avoid dead cells and premature livelihood)
INHERTIANCE_PROBABILITY  = 0.1 # probability that neighboring cells will inherit by perturbation.
parameter_perturbation_probability = 0.02
print("Numbers of layers used are {}".format(NUM_LAYERS))
print("1 for alpha layer and rest {} for hidden".format(NUM_LAYERS-1))
NUM_STEPS = 90
num_steps = NUM_STEPS
at_which_step_random_death = 9999999999 # Set this to infinity or high value if you never want to enter catastrophic deletion (random death happens at this generation)
probability_death = 0.004 # 40 pixels die every generation
print("Numbers of Time Steps are {}".format(NUM_STEPS))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
activation = 'sigmoid' # ['sigmoid','tanh','noact']
frequency_dicts = []
FPS = 10 # Speed of display for animation of NCA and plots
marker_size = 1 # for plots
everystep_weights = [] # Stores weigths of the NNs from every time step.
enable_annotations_on_nca = True
budget_per_cell = 999999999
fixed_value = 0
budget_counter_grid = np.zeros((WIDTH, HEIGHT)) + fixed_value
```

With this configuration it simulates the proposed NCA framework. And then results a simulation like:

![download-ezgif com-video-to-gif-converter](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/6b092716-56d6-41e3-bf07-53d63d84b378)


Beyond that it produces more results that are related to Phenotypic Diversity (PD) and Genotypic Diversity. PD tools are shown below:

* CTFP Plot - Frequency Count Plot

![download](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/326a4bf1-21f2-4f13-b7e8-15985399fe91)

* GEP+GCVP+CLOGV (Phenotypic Diversity and Randomness)

![download](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/9b804ac1-8ffd-4b18-bdd2-bcc90095abce)


* GD Unique Color Count Plots (GHC and RWSP)

![download-2](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/a5e3f88c-20f2-4090-947f-3181c9a9d24d)

* RWSP Animation
* 
![download-ezgif com-video-to-gif-converter](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/0dda3c5a-91fa-4ca9-a3ff-39e3a6467946)

* GHC Animation

![download-ezgif com-video-to-gif-converter](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/fd6e125e-401a-47fc-820f-4efd8276d9f2)



## Using SLURM for Larger Experiments
We use Simula eX3 cluster in support of Research Council Norway. We use a100q, dgx2q and hgx2q machines to run our larger experiments. The batch script looks like:
```bash
#!/bin/bash
#SBATCH -p a100q ## hgx2q, a100q,  dgx2q (old), genomaxq, xeonmazq
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
| Platform      | Model   | GPU  | Processor                                         | 
|---------------|---------|------|---------------------------------------------------|
| x64_64/A100   | hgx2q   | g002 | DualProcessor AMD EPYC Milan 7763 64-core w/ 8 qty Nvidia Volta A100/80GB |
| x64_64/V100   | dgx2q   | g001 | DualProcessor Intel Xeon Scalable Platinum 8176 w/ 16 qty Nvidia Volta V100 |
| x86_64/cpu | xeonmaxq     | n022 | DualProcessor Intel Xeon Max Q with mix configuration | 
| x86_64/cpu | genoaxq     | n021 | DualProcessor (Epyc or Xeon) with mix configuration | 


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

This classification of Experiment Size is done to avoid runtime collapse because of the memory outage during the processing of the results for the experiments. We could have produced results on runtime while experimenting using SLURM experiments, however to modularise and balance the workload, we preferred to save the weights for one time, and then process results later if required. This also brought flexibility to improve existing code bases for processing results and add more tools later for our project. We release ```2_sagemaker.ipynb``` to process the experiments. This notebook takes public URLs of the experimentation and process results. We release these large experiments as public urls below. These publics links are also self-contained in order to check results of those expierments. You can download sagemaker notebook from this link - https://drive.google.com/file/d/1lvhCRLtoIFvBiwAEwROmBaDzjSYLHUdP/view?usp=sharing




## Experiments

| Experiment | Download Link                                      |
|------------|-----------------------------------------------------|
| 100by100, 0.1init, 2layers, 0.5alpha, 0.2ip, 0.2ppp, 500generations, sigmoid, b4      | [Download exp1.tar](https://drive.google.com/file/d/15KuHi7IDjZpbj01AERP1c1OaGpxXfmVH/view?usp=sharing)  |
| 100by100, 0.1init, 2layers, 0.5alpha, 0.2ip, 0.2ppp, 1000generations, sigmoid, b4      | [Download exp2.tar](https://drive.google.com/file/d/1XzvSMeqY86RdhHC0L8AFLkmeDNme4n3u/view?usp=sharing)  |
| 200by200, 0.1init, 2layers, 0.5alpha, 0.2ip, 0.2ppp, 100generations, sigmoid, b4      | [Download exp3.tar](https://drive.google.com/file/d/19BbSy7lBj4c6fg9FZcgnFpKdS66lwgnJ/view?usp=sharing)  |
| 300by300, 0.1init, 2layers, 0.5alpha, 0.2ip, 0.2ppp, 100generations, sigmoid, b4      | [Download exp4.tar](https://drive.google.com/file/d/154mO_AcRLj9dNwg86gmpJQjx6o25Ekrr/view?usp=sharing)  |

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


## Working pipeline Demo

![IMG_2405](https://github.com/s4nyam/Self-Replicating-NCA/assets/13884479/00ff2275-04b9-486c-a97a-686e249748d8)

