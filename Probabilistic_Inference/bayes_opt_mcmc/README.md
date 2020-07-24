# CO493 - Probabilistic Inference - Bayesian Optimisation Coursework

## PDF Coursework

You may find [here](https://gitlab.doc.ic.ac.uk/lg4615/co493-coursework-mcmc-vi-students/-/blob/master/C493_Coursework.pdf) the coursework description.

## Requirements

To install the requirements, use the following commands (with `python>=3.6` enabled by default):
```shell script
pip install matplotlib scipy numpy keras tensorflow jax jaxlib
```

## Launching a visualisation

You can visualise the results produced by your implementation 
by launching the python script contained in the corresponding file.

For example, if you want to visualise your predictions based on the Metropolis-Hastings samples,
in the Logistic Regression, you can execute the following command

```shell script
python -m distribution_prediction.metropolis_hastings.metropolis_hastings_logistic
```

## Remarks

* You are totally allowed to use all the functions available in `scipy.stats`

