# Contributing



## Tasks

Adding new tasks is straightforward. It is easiest to follow the scheme for existing tasks.

First, take a close look at the base class for tasks in `sbibm/tasks/task.py`: you will find a `_setup` method, that was not discussed in the guide for users. This method samples from the prior, generates observations, and finally calls `_sample_reference_posterior`, to generate samples from the reference posterior. All of these results are stored in csv files, and the generation of reference posterior samples happens in parallel.

If you take a look at `_sample_reference_posterior` in the base class, you will notice that this uses an MCMC sampling schemes. As long as the model is implemented using Pyro, the sampler should automatically generate reference posterior samples.

Some tasks, e.g., the `gaussian_linear` override `_sample_reference_posterior`. For this task, a closed form solution is available, which is used instead of MCMC, see `sbibm/tasks/gaussian_linear/task.py`.

Note also that each individual tasks ends with a `if __name__ == "__main__"` block at the end which calls `_setup`. This means that `_setup` is executed by calling `python sbibm/tasks/task_name/task.py`. This step overrides the existing reference posterior data, which is in the subfolder `sbibm/tasks/task_name/files/`. It should only be executed whenever a task is changed (and never by a user).


## Algorithms

To add new algorithms, take a look at the inferfaces to other third-party packages inside `sbibm/algorithms`. In general, each algorithm specifies a `run` function that gets `task` and `config` as arguments, and eventually returns the required `num_posterior_samples`. Using the task instance, all task-relevant functions and settings can be obtained, and `config` contains e.g. the algorithm hyperparameters. I'm glad to help with implementing new algorithms or adjusting configurations of exisiting ones.


### Code style

For docstrings and comments, we use [Google Style](http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings). We use automatic code formatters (use `pip install sbibm[dev]` to install them). In particular:

**[black](https://github.com/psf/black)**: Automatic code formatting for Python. You can run black manually from the console using `black .` in the top directory of the repository, which will format all files.

**[isort](https://github.com/timothycrosley/isort)**: Used to consistently order imports. You can run isort manually from the console using `isort -y` in the top directory.
