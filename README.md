# Descriptions

### Knowledge-Driven Modulation of Neural Networks with Attention Mechanism for Next Activity Prediction (Preparing to submit)

This repository contains developed scripts for our proposed approach of predictive process monitoring.

We split the implementation into two folders for (i) synthetic and (ii) real-logs. For each folder, you can implement our approach by running 'run_experiments.py' file. 

You can also set your own parameters in 'log_utils.py' and 'shared_variables.py files located in the folder '~\src\commons'

We also provide a summarized result of our experiments in 'performance.csv' file, which is presented in our paper. 

## Input/Output files

The used event logs and BPMN models in the paper are located in the folder '~\data\input'.

Then, after running the script, trained models and results of predictive process monitoring will be saved in the folder '~\data\output'.

We also uploaded our trained models used in the paper in the folder '~\data\output_old'. The result files after evaluation which used in our paper are downloadable in this link: https://drive.google.com/file/d/12_qesWq_mu6i0Tb4ckZyzCHpJOwdMcP2/view?usp=sharing.


## How to train/evaluate

The main implementation script is the 'run_experiments.py' file. 

You can run the script in command line by 

(i) firstly changing to the working directory, e.g., using "cd C:\Users\ADMIN\~\KB-Modulation\implementation_real_logs", and then 

(ii) implementing the script like "python run_experiments.py --log 'sepsis_cases_1.csv' --train". Here, the option '--train' is only for training a model and, otherwise, you can select one among ['--train', '--evaluation' (for evaluation), '--full_run' (for both)]. 

In addition, an option 'weight' can allow you to configure the weight (importance) of BK in evaluation stage, e.g., "python run_experiments.py --log 'sepsis_cases_1.csv' --evaluation --weight '0.8'", otherwise, the default weight value 0 will be applied.

If you want to run the script using tools like Pycharm or VScode, in line 62-64 in 'run_experiments.py', you can configure (1) only training by setting 'default = True' in line 62, or (2) only evaluation by setting 'default = True' in line 63, or (3) both by setting 'default = True' in line 64.


## Visualization with graph

We also showed the trend graph by increasing weight value of BK. The all graphs can be seen in the folder '\plots_improvement'.

If you want to re-generate the graphs by yourself, you can simply do it by:

(i) installing R (https://cran.rstudio.com/) or Rstudio (https://posit.co/download/rstudio-desktop/) and,

(ii) then, just implementing 'Graph_reallogs.R' and 'Graph_synthetic.R' in the tool. In the R script, you only need to configure your working directory in the first line as "setwd("C:/Users/ADMIN/Desktop/~/KB_Modulation_results")", where KB_Modulation_results folder should contain the result files in this previous link: https://drive.google.com/file/d/12_qesWq_mu6i0Tb4ckZyzCHpJOwdMcP2/view?usp=sharing.
