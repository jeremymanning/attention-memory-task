# Overview

This repository contains code and data pertaining to [Category-based and
location-based volitional covert attention affect memory at different
timescales](https://psyarxiv.com/2ps6e) by Kirsten Ziman, Madeline R. Lee,
Alejandro R. Martinez, Ethan D. Adner, and Jeremy R. Manning.

The repository is organized as follows:
```
root
├── code: analysis code used in the paper
├── data (created when the notebooks are run; data must be downloaded via the notebooks): all data analyzed in the paper
│   ├── sustained: data from the sustained attention condition (30 participants)
│   └── variable: data from the variable attention condition (23 participants)
├── experiment: code for setting up and running the experiment
│   ├── stimuli.zip: contains face, place, and composite images used in the experiment, along with attention cue graphics
│   ├── stimulus_generation_code: code for generating composite images
│   ├── sustained_attention_experiment: code and materials related to the sustained attention condition
│   └── variable_attention_experiment: code and materials related to the variable attention condition
└── paper: all files needed to generate a PDF of the paper and supplement
    └── figs: PDF copies of all figures
```

Our project uses [davos](https://github.com/ContextLab/davos) to improve shareability and compatability across systems.

# Analysis setup instructions

Note: we have tested these instructions on MacOS and Ubuntu (Linux) systems.  We *think* they are likely to work on Windows systems too, but we haven't explicitly verified Windows compatability.

We recommend running all of the analyses in a fresh Python 3.10 conda environment.  To set up your environment:
  1. Install [Anaconda](https://www.anaconda.com/)
  2. Clone this repository by running the following in a terminal: `git clone https://github.com/ContextLab/attention-memory-task` and change the working directory to the newly cloned repository (e.g., `cd attention-memory-task`)
  3. Create a new (empty) virtual environment by running the following (in the terminal): `conda create --name attention-memory python=3.10` (follow the prompts)
  4. Navigate (in terminal) to the activate the virtual environment (`conda activate attention-memory`)
  5. Install support for jupyter notebooks (`conda install -c anaconda ipykernel jupyter`) and then add the new kernel to your notebooks (`python -m ipykernel install --user --name=attention-memory`).  Follow any prompts that come up (accepting the default options should work).
  6. Navigate to the `code` directory (`cd code`) in terminal
  7. Start a notebook server (`jupyter notebook`) and click on the notebook you want to run in the browser window that comes up.  The `analysis.ipynb` notebook is a good place to start.  
  8. Before running any of the notebooks, always make sure the notebook kernel is set to `attention-memory` (indicated in the top right).  If not, in the `Kernel` menu at the top of the notebook, select "Change kernel" and then "attention-memory".  Selecting "Restart & Run All" from the "Kernel" menu will automatically run all cells.
  9. To stop the notebook server, send the "kill" command in terminal (e.g., `ctrl` + `c` on a Mac or Linux system).
  10. To "exit" the virtual environment, type `conda deactivate`.

Notes:
- After setting up your environment for the first time, you can skip steps 1, 2, 3, and 5 when you wish to re-enter the analysis environment in the future.
- To run any notebook:
  - Select the desired notebook from the Jupyter "Home Page" menu to open it in a new browser tab
  - Verify that the notebook is using the `attention-memory` kernel, using the above instructions to adjust the kernel if needed.
  - Select "Kernel" $\rightarrow$ "Restart & Run All" to execute all of the code in the notebook.
- The eyetracking dataset is too large to store directly on GitHub.  The relevant datasets will be downloaded automatically when you run the notebooks.

To remove the `attention-memory` environment from your system, run `conda
remove --name attention-memory --all` in the terminal and follow the prompts.
(If you remove the `attention-memory` environment, you will need to repeat the
initial setup steps if you want to re-run any of the code in the repository.)

Each notebook contains embedded documentation that describes what the various
code blocks do. Any figures you generate will end up in `paper/figs/source`.
Statistical results are printed directly inside each notebook when you run it.

# Experiment setup instructions

Our experiment is implemented using [Psychopy](http://psychopy.org/). There are
two versions of the experiment code, corresponding to the two conditions in our
paper: a block design version (the sustained attention condition), and a
trial-wise cuing version (the variable attention condition). Both begin with an
initial practice task to orient the participant to the task instructions,
followed by a series of trial blocks, each comprising a presentation and memory
phase.  Task details may be found in the main text of our paper.

# Directory Organization

The `sustained_attention_experiment` and `variable_attention_experiment`
directories contain the code for running the experiments. The
`stimulus_generation` directory contains the code to process the single images
and to create the composite image stimuli that appear in the experiments, and
the paradigm figure directory simply contains an image of the schematic
displayed above.

# Running the task

To run a behavioral participant, first unzip `stim.zip` so you can access all
of the stimuli for the experiment. Then, simply open attention_memory.py in
Psychopy (in either the `sustained_attention_experiment` or
`variable_attention_experiment` directory, as desired), hit the green "run"
button in the code viewer, and enter the subejct ID and the experiment should
start! Make sure to set `practice = True` if you would like the participant to
complete the practice trials before beginning (we strongly recommend this).

Note that the subject name or number can be arbitrarily chosen by the
experimenter (any number or string), so be careful not to enter a subject
number that exists in the data you've already collected.

The run number indicates which run to display next. If the subject is just
beginning the experiment, the run number should be 0. A run number of 0 will
initiate the experiment on the first run, and loop over presentation and memory
runs until the total number of desired runs (outlined in the parameters in
attention_memory.py) has been presented, pausing after each memory block for
eye tracker recalibration. If the experiment is interrupted, you can pick up
where you left off by setting the run number to a different value as needed.
("Run" is what we refer to as "task block" in our paper.)

Our participants sat 60 cm from the screen (the distance at which the Eye Tribe
eye tracker we used has 0.1 degrees visual angle root mean squared error). Our
Psychopy monitor center was set to the original default settings (including the
assumed 57 cm viewing distance). Thus, to replicate the experiment exactly, you
may either: (1) use our exact code and display (27 inch iMac display at a
resolution of 2048 x 1152 pixels), and place participants 60cm from the screen,
using the default monitor settings, or, (2) change the stimulus sizes,
specified in degrees visual angle, directly to match those in the paper (see
the `attention_memory.py` files in `sustained_attention_experiment` and
`variable_attention_experiment`).

# Acknowledgements

Special thanks to Megan deBettencourt for providing the image processing script
available in this repository (stimulus_generation_code/process_images.m) and
for recommending the stimulus sets!

# References:
Face stimuli: J. Xiao, J. Hays, K. Ehinger, A. Oliva, and A. Torralba.
SUN Database: Large-scale Scene Recognition from Abbey to Zoo.
IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

Place stimuli: P. Jonathon Phillips, Harry Wechsler, Jeffrey Huang, Patrick J.
Rauss: The FERET database and evaluation procedure for face-recognition
algorithms. Image Vision Comput. 16(5): 295-306 (1998)

