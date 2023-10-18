# Introduction to Symbolic Music Alignment

This folder contains an introduction to symbolic music alignment using dynamic time warping.

To run the baseline submission for the challenge, run the following commands on the terminal:

```bash
cd PATH_TO_THE_REPO
cd alignment
conda activate miws23
python Baseline_Alignment.py -c -i PATH_TO_DATASET -o PATH_FOR_THE_OUTPUT
```

This will generate a compressed numpy file with the results. Please upload this file in the challenge server, which is located [here](https://challenges.cp.jku.at/challenge/14/).


## Score-to-Performance Alignment Task

This task is worth 67 points (out of a total of 200).

### Report

For this task, you are required to work in teams (you can select a team on the Moodle page) of up to 3 people. Teams of 1 person are allowed, but you have to select one of the (empty) teams on Moodle. Otherwise, you will **not** be allowed to submit your report!

For the project you will have to submit a report in the form of a blog post-like Jupyter Notebook.  The task will be graded based on the submitted report.

**Remember that there is only one report is for all three music analysis challenges (music alignment, key estimation, meter estimation)**.  The deadline for the complete report is on **January 10th, 2024 23:59**.

We will grade each task on 3 main aspects:

1. **Completeness** (30%): Did you do all tasks? Does the report include a detailed description of the problem an explanation of the methods used? Are the methods evaluated in some way? The part of the report for each task must include the following points (you can structure the report in any way you want, as long as these points are covered):

    * *Introduction*: What the specific task is about, why is it an interesting problem (think of musical and technical issues)
    * *Description of the method/methods used*: You don't need to include a full technical description of the methods, but you should describe why did you select your approach (how does the method address the particular musical problem), a brief description of the method, what are the parameters of the method (and how do they relate to the problem).
    * *Evaluation of the method(s)*: How do you evaluate the performance of the model? (i.e., which metrics do you use to assess the performance of the model). Include both your own evaluation using the training set, as well as the performance of the model on the leaderboard! Note that in many cases, the *loss function* used to train a model is not necessarily the best metric to compare models (e.g., a probabilistic classifier is trained to minimize the *cross entropy*, but the metric used to compare the models could be the *accuracy* or *F1-score*). Which datasets are you using (what information is contained in the datasets, size, etc.). If a method needs training, how was the method trained (including strategies for hyperparameter selection).
    * *Discussion of the results and your own conclusions*: Discuss what worked or did not work, which characteristics of the model lead to better performance. Do not be afraid to conduct ablation studies to see how different parts of the model contribute to the overall performance.

2. **Thoroughness and Correctness** (40%): You used appropriate methods to solve the problems, and the methods are used correctly. You should explain your assumptions on when/why to use a method/technique for solving the problems.
3. **Presentation and Clearness** (30%): Is the report well structured? Ideally, the report should not be only text or code, but it also should include some figures/images illustrating the methods. Imagine that you are writing a blog post for **non-experts**, and try to explain the methods in a simple and clear way, and do not be afraid to use some math! (You can use LaTeX/Markdown on Jupyter Notebooks)
4. **(Bonus) Creativity** (up to 30%): You can get extra points for creative solutions! Don’t be afraid to think outside the box, even if the results do not outperform other methods!

### Challenge

Each team should participate at **least once** in the challenge to get a grade in the reports. The deadline for submissions is **January 9th, 2024 23:59**! The winners of the challenge will be announced during the final concert/presentations on **January 10th**.

For this challenge, you will have to align a performance with its score, in a note-wise fashion and export your results as a compressed Numpy file. For convenience, we will provide both a training dataset consisting of performance, score and ground truth alignments in the CSV format used for [Parangonada](https://sildater.github.io/parangonada/), an interactive interface to compare and correct alignments.

You can use **any method that you want** (even if it is not one of the methods presented in this lecture).

For developing/evaluating/(and training, if you use a method that requires it), you will use the Vienna4x22 dataset, which is a dataset consisting of 4 piano pieces and 22 different performances of each piece (by different pianists). This is one of the standard datasets for analysis of expressive music performance.

For the challenge we will use an entirely different dataset!

For the challenge, your script submission should be executed in the following way:

```
python TeamName_Alignment.py -c -i path_to_the_data_directory -o output_directory
```

The file `TeamName_Alignment.py` should be as self-contained as possible and you can use third-party libraries that are not included in the conda environment for the course. You can use the methods defined in the class (and  available on the [GitHub repository](https://github.com/MusicalInformatics/miws23/tree/main)). Please upload a zip file with all of the files to run your submission, including the python script itself, the conda environment yaml file, any other helper files and trained model weights (if relevant).

Please follow the example in `Baseline_Alignment.py` in the `alignment` folder in the GitHub repository.
