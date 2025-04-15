Defect μscopy toolkit
=====================

# Introduction

This is a toolkit for the analysis of defect μscopy images.
We will provide a brief discussion here of what the toolkit does, and how to get setup
if you're from another lab.
We will also detail where to get more information on the toolkit.
Note this toolkit is designed to be used for any spin defect, not just NV centres, and
in fact could be used for any quantitative (hyperspectral) microscopy technique.

Primarily this toolkit can read, process and analyse microscopy images.
I.e. it can load images, perform drift correction, crop, rebin, smooth, fit and plot.
It can also simulate images of magnetic flakes.
There is preliminary work towards supporting vector magnetometry and source 
reconstruction, but we haven't moved that over from the old 
[repo](https://github.com/casparvitch/qdmpy/) yet.

To install see INSTALL.md, and for more information on developing the toolkit see
DEVDOCS.md

# For users outside the Tetienne lab

We have implemented the toolkit in a modular way, with support for general 'inputs',
all you have to do is copy some of the example scripts and implement your own System
subclass - see dukit/systems.py for more information.

# API documentation

[Docs are here](https://qnslab.github.io/dukit/dukit/index.html)

# Normal usage

We usually use the toolkit in a Jupyter notebook, with a separate `.nb` for each
measurement - that way we keep the plot outputs in one place with the code that
created them. Good for reproducability!
