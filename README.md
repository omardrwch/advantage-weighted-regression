# advantage-weighted-regression

Implementation of Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning, by Peng et al. (2019) (https://arxiv.org/abs/1910.00177).

Uses the [rlberry](https://github.com/rlberry-py/rlberry) library.


Setup:

```bash
conda create -n awr python=3.8
conda activate awr
pip install gym[all]
pip install git+https://github.com/rlberry-py/rlberry.git@v0.2.1#egg=rlberry[torch_agents]
pip install tensorboard
```

Optional:

```
conda install -c conda-forge jupyterlab
```
