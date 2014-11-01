pieflag
=======

![logo](./pieflag.jpg)

[CASA](http://casa.nrao.edu/) task to flag bad data by drawing upon statistics from reference channels in bandpass-calibrated data.

Let's face it, flagging is laborious and boring. You want a simple and effective tool to fight rfi like a boss. pieflag operates on a simple philosophy: you need to do some very basic pre-processing of your data so that the code can do its job robustly using a minimal set of assumptions.

pieflag works by comparing visibility amplitudes in each frequency channel to a 'reference' channel that is rfi-free, or manually ensured to be rfi-free. To operate effectively, pieflag must be supplied with bandpass-calibrated data (preliminary gain-calibration is also preferable).

pieflag has two core modes of operation -- static and dynamic flagging -- with an additional extend mode. Which mode you choose is largely dictated by the type of data you have. An extensive help file is included with pieflag that includes instructions for pre-processing your data and selecting the best mode of operation. You are encouraged to read the full documentation before running pieflag. Once you have carried out your pre-processing and selected the mode of operation, pieflag should work well 'out of the box' with its default parameters. By comparing to a clean reference channel, essentially all bad data will be identified and flagged by pieflag.

Lateset version: 2.0 ([download here](https://github.com/chrishales/pieflag/releases/latest))

Requires: CASA Version 4.3.0

pieflag originally written by Enno Middelberg 2005-2006 (Reference: [E. Middelberg, 2006, PASA, 23, 64](http://arxiv.org/abs/astro-ph/0603216)). Version 2.0 has been modified for use in CASA and updated to include wideband and SEFD effects by Christopher A. Hales (Reference: [C. A. Hales, E. Middelberg, 2014, Astrophysics Source Code Library, 1408.14](http://adsabs.harvard.edu/abs/2014ascl.soft08014H)).

pieflag Version 2.0 is released under a BSD 3-Clause Licence (open source, commercially useable); refer to the licence in this repository or the header of ```task_pieflag.py``` for details.

pieflag logo created by Chris Hales and the amazing graphic designer [Jasmin McDonald](http://www.theloop.com.au/JasminMcDonald/portfolio).

Correspondence regarding pieflag is always welcome.

Installation
======

Download the latest version of the source files from [here](https://github.com/chrishales/pieflag/releases/latest).

Place the source files into a directory containing your measurement set. Without changing directories, open CASA and type
```
os.system('buildmytasks')
```
then exit CASA. A number of files should have been produced, including ```mytasks.py```. Reopen CASA and type
```
execfile('mytasks.py')
```
To see the parameter listing, type
```
inp pieflag
```
For more details on how plot3d works, type
```
help pieflag
```
Now set some parameters and press go!

For a more permanent installation, go to the hidden directory ```.casa``` which resides in your home directory and create a file called ```init.py```. In this file, put the line
```
execfile('/<path_to_task_directory>/mytasks.py')
```

Acknowledging use of pieflag
======

We would appreciate your acknowledgement by citing Enno's PASA paper and Chris' ASCL entry (see above).