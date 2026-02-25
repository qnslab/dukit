Developer Documentation
-----------------------

# Overview

Thanks for coming here to try and improve the code.
You are probably here to see try and add new model functions to the fitting library,
scroll down for info.

If you're looking for help on how to *actually use* the library, see the `./examples` 
folder.

Before you get started, you will probably want to read the INSTALL.md file, which also
has some tips on how to get started with git & github, which should be helpful.

# State of the project

## Currently implemented

- Image stack loading and saving (`dukit.systems` and `dukit.io`)
  - Image stack registration (drift correction) (`dukit.driftcorrect`)
  - Cropping, rebinning & smoothing
- Fitting of the image stack (`dukit.pl`)
    - Lorentzians, damped circular functions (Rabi) and stretched exponentials (T1)
    - Extraction of the fit parameter uncertainty (named 'sigmas')
- Calculating local B field at defects (from fits) (`dukit.field`)
- A bunch of plotting utilities (`dukit.plot`)
- Simulation of magnetic flakes (`dukit.magsim`)

## TODO

- Vector magnetometry - see qdmpy.field
- Source reconstruction - see qdmpy.field
- If you add the two above, I recommend not adding the high-level interface functions, 
  and instead just the underlying more 'maths-y' functions - easier to implement and more
  modular.
- Aberration correction - see Sam Scholten thesis. Both for the field map and directly
  on the image stack.
- Go through docs carefully, e.g. read each file docstring.
- Test the asymmetric binning in fourier space.

# How to add a new scipyfit model function

Comparatively simple, just add a subclass of `dukit.pl.model.FitModel` in 
`dukit.pl.model` (or I guess in your script, that's fine too). You will need to implement
methods for evaluation (`dukit.pl.model._eval`), and optionally* for jacobian 
evaluation (`dukit.pl.model._jacobian`). Basically copy the other classes defined there
as a template. `get_param_defn` etc. methods also need to be defined. To add the model
to the library, add an import statement in the pl `__init__`, the dukit `__init__`,
and also please add to the docs at the top of the model file, and the docs at the top
of the dukit `__init__`.

*: Some of the functions assume you have a jacobian function and will fail if you don't
pass `sf_jac='2-point` or similar (to use the scipy built-in numeric jacobian 
approximation). So best to provide a jacobian, unless there is no analytic form.

# How to add a new cpufit/gpufit model function

It is a little trickier to add a model function to `cpufit`/`gpufit` as you'll need to 
recompile the library (see instructions below). We install `cpufit` and `gpufit` as pip
extensions (see INSTALL.md), so you will probably want to put any new output wheel files
in the `gpufit_wheels` directory.

First of all, you will want to work in our fork of 
[gpufit](https://github.com/gpufit/Gpufit), which can be accessed 
[here](https://github.com/casparvitch/Gpufit). Essentially we just add our fit functions.
We use the `master_QSL` branch & leave the `master` branch clean for merging with 
upstream. **So make sure you install from the `master_QSL` branch!**

For both `gpufit` and `cpufit` you want add to the `ModelID` enum in 
`Gpufit/Gpufit/constants.h` with a number to associate with your number. 
I've been using negative integers so that they don't clash  with the upstream 
(non-forked) project. The `ModelID` will also need to be added to the 
`pycpufit/cpufit.py` and `pygpufit/gpufit.py` files, respectively.

For `cpufit`, you will want to add two functions in `Gpufit/Cpufit/lm_fit_cpp.cpp`, e.g.
scroll down `calc_values_stretched_exp` and `calc_derivatives_stretched_exp` for
an example. Then add to `calc_curve_values` at the bottom.

For `gpufit`, add a new model `.cuh` file in `Gpufit/Gpufit/models`, and add to 
`Gpufit/Gpufit/models/model.cuh` (include at the top, and add to both switch statements).
You will need to specify the number of parameters and dimensions.

## Building from source

### Linux

Follow [instructions](
https://gpufit.readthedocs.io/en/latest/installation.html#compiling-gpufit-on-linux) - 
worked a lot easier than Windows.
Just swap the names depending on which version (`cpufit`/`gpufit`) you want to build.

You may want to read through the discussion below for an idea of some of the pitfalls.

### Windows

#### Install methodology

NOTE: the below has a lot of details. 
If your card is relatively modern it shouldn't be such a hastle, fortunately.

- check card details: what is it's compute capability? (nvidea website)
  - what cuda toolkits version to you need? (wikipedia calls this 'SDK')
- check compatibility of cuda toolkit with OS version.
- check compatibility with gpu driver.
- check compatibility with visual studio. 
- yes you do need overlap between all of these things. 
  Best to work it out before you start.
    - if you get it wrong, you'll need to uninstall everything and start again.
  
#### Where to find these details:

- is card cuda-compatible? also, what is the compute capability (roughly the cuda 
  generation/micro-architecture) of the graphics card?
   - nvidea has a website for this, [link](https://developer.nvidia.com/cuda-gpus)
- what cuda toolkits version can I use with my card/driver version?
   - most useful resource for me has been the cuda wiki page 
     [link](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)
   - 'SDK' is equivalent to 'cuda toolkits version'
   - this page also has a table you can use to determine the compute capability of your 
      graphics card
   - note gpufit was tested by the authors from 6.5-10.1 (but latest should be fine)
- cuda toolkits archive [link](https://developer.nvidia.com/cuda-toolkit-archive)
   - also links to documentation
   - section: 'installation guide windows' gives info on what OS version and visual 
     studio version will work with the selected toolkit version
- to find out about your nvidea driver:
   - in windows make sure you update it first (through device manager > display adapters)
   - also probably best to ensure the card isn't being used for graphics output!
   - right click on desktop > nvidea control panel > system information (bottom right), 
     will display driver version
- to find driver/cuda toolkits compatibility:
   - see nvidia cuda compatibility docs, I got this link at top of google 
     [link](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
   - go to section 5.1: Support > Hardware Support
   - will show a table: compatibility between hardware generation (Fermi, Kepler etc.), 
       compute capability and graphics driver version
       - this is an important one!!!

#### Installation Procedure

- Install/update gpu driver

- Install microsoft visual studio (MSVS)
   - *not to be confused with Visual Studio Code*
   - best to go with an older rather than newer version for compatibility with everything
     above
   - note 'community' version is free
   - where to get: [link](https://visualstudio.microsoft.com/vs/older-downloads/)
       - for older versions need a live account, I believe
   - pc will probably want to restart

- install cuda toolkits

- can then install [cmake](https://cmake.org/)
   - ensure it has been added to path

- useful to also install BOOST C++ library [link](https://www.boost.org/)
   - will take a little while to build itself (run bootstrap file)
   - this allows you to run tests after building gpufit source (useful!!)
     - update: you can just use the pybindings, faster than downloading and building
       BOOST.

- install python
   - I used windows msi install (>3.8)
   - ensure you add to path!
   - ensure pip installed!
   - make a folder at root (or C:) called 'src' -> will put all code here

- environment management:
   - either conda (miniconda) or pipenv (be nice on yourself and use conda)
   - miniconda install:
       - [link](https://docs.conda.io/en/latest/miniconda.html)
   - pipenv install:
       - `pip install --user pipx`
       - `pipx install pipenv`

- install git
   - [git install link](https://gitforwindows.org)
   - connect to gitlab -> follow notes in version control section
   - connecting git to sublime merge (in particular for gitlab)
       - **ensure you are using ssh links not https!**

#### Building GPUFit from source
 
- ENSURE `wheel` installed (`pip install wheel`) BEFORE compilation

- Grab latest QSL fork of gpufit from github
      - **MAKE SURE** you're on the master_qsl branch!!! -> this branch has our 
        additions/fixes!

- Can basically follow instruction in gpufit docs [link](https://gpufit.readthedocs.io/)
    - In general, the gpufit docs are quite good, but you need to fiddle around quite a 
      bit!

- compiler configuration (Cmake):
    -  First, identify the directory which contains the Gpufit source code 
       (for example, on a Windows computer the Gpufit source code may be stored in 
       `C:\src\gpufit`). Next, create a build directory outside the source code source 
       directory (e.g. `C:\src\gpufit-build`). Finally, run cmake to configure and 
       generate the compiler input files. The following commands, executed from the 
       command prompt, assume that the cmake executable (e.g. 
      `C:\Program Files\CMake\bin\cmake.exe`) is automatically found via the PATH 
       environment variable (if not, the full path to cmake.exe must be specified). This 
       example also assumes that the source and build directories have been set up as 
       specified above.
        - `cd C:\src\gpufit-build`
        - `cmake -G "Visual Studio 12 2013 Win64" C:\Sources\Gpufit`
    - I then open up the cmake gui (which will auto-populate fields from this previous 
       cmake run) to edit some more things:
        - set \_USE_CBLAS flag to be true (if you get errors when building try False -> 
          sometimes gpufit gets the name of the cuBLAS dll incorrect). Still not sure 
          how to fix this one, not a big issue though.
        - add BOOST_ROOT variable to wherever you installed/unpacked BOOST

- compiling (visual studio)
    - After configuring and generating the solution files using CMake, go to the desired
      build directory and open Gpufit.sln using Visual Studio. Select the “Debug” or 
      “Release” build options, as appropriate. Select the build target “ALL_BUILD”, and 
      build this target. If the build process completes without errors, the Gpufit binary
      files will be created in the corresponding “Debug” or “Release” folders in the 
      build directory.
    - The unit tests can be executed by building the target “RUN_TESTS” or by starting 
      the created executables in the output directory from the command line. (I RECOMMEND
      YOU RUN SOME TESTS!)

- Building python wheel file
    - ENSURE `wheel` installed (`pip install wheel`) BEFORE compilation
    - uninstall any previous version you installed (`pip uninstall pygpufit`)
    - `pip install C:\src\gpufit-build-Release\pyGpufit\dist\wheel_file_here.wh`
    - Add to `gpufit_wheels` folder in dukit repo, with appropriate arch (e.g. `win_amd64`).
