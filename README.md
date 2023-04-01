# Description
To run the scripts, follow these steps:

1. Build a virtual enviornment with [Brian2](https://github.com/brian-team/brian2), [Scipy](https://scipy.org/), and [matplotlib](https://matplotlib.org/stable/index.html) installed.
2. Clone this repo.
3. In a terminal, cd to this directory, activate your environment, and enter `python sim.py`. 

This will perform a parameter sweep and stores visualiziations in a newly created `results` directory. By editing the content of `configs.py`. The defualt scenario models a charge-balanced biphasic stimulation protocole with a primary pulse-width of 70 $\mu s$ and a counterpulse duration of 10 times larger.
