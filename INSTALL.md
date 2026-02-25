Install instructions
--------------------

# Basic install

(Assuming you know git - see below)

- [clone](https://github.com/casparvitch/dukit)
- install python3.11+ and [uv](https://docs.astral.sh/uv/)
- navigate to root-dir (containing 'src' etc.)

```bash
just install        # or: uv pip install -e .
```

or for an editable install: to edit library without reinstalling.
Note you will have to restart any jupyter kernel you have running to benefit from any updates.

# Installing extensions (cpufit/gpufit)

To install cpufit (Linux or Windows) or gpufit (Windows only):

```bash
just install-cpufit     # Install cpufit from local wheel
just install-gpufit     # Install gpufit from local wheel (Windows only)
just install-fit-backends # Install both (where available)
```

These libraries are installed from the wheel files in the `gpufit_wheels` folder.

# Documentation

`just docs`, or:

Navigate to root-dir (containing 'src' etc.)

```bash
uv pip install pdoc3
pdoc3 --output-dir docs/ --html --template-dir docs/ --force --skip-errors ./src/dukit/
```

# jupyter-lab widgets

- Install [nodejs](https://nodejs.org/en/) and [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/user_install.html#installing-the-jupyterlab-extension).

- For help, see [here](https://stackoverflow.com/questions/49542417/how-to-get-ipywidgets-working-in-jupyter-lab).  

- To save widget state: in jupyterlab > settings > advanced settings editor >
   jupyter widgets > saveState: true


# Version control / git

The project is housed on [GitHub](https://github.com/casparvitch/dukit), you will need to
be given access by an owner. To communicate with the Github server you will need to setup
an ssh key (on each device connected, there will need to be one on each lab computer as 
well). My installation instructions below are taken from the Gitlab Docs 
[here](https://docs.gitlab.com/ee/ssh/).

You can also use github in-browser, i.e. not using the git commands at all. 
This is not recommended but can be great in a pinch.


## git installation

If you're on windows you will need to download a Git/OpenSSH client (Unix systems have it
pre-installed). The simplest way to do this for integration with PyCharm is just to use 
[Git for Windows](https://gitforwindows.org/), even if you have WSL/Cygwin. 
Just do it, it isn't that big a package.

Open the Git Bash terminal (or Bash on Unix). Generate a new ED25519 SSH key pair:

```Bash
ssh-keygen -t ed25519 -C "<YOUR EMAIL HERE>"
```

You will be prompted to input a file path to save it to. Just click Enter to put it in 
the default \~/.ssh/config file. Once that's decided you will be prompted to input a 
password. To skip press Enter twice (make sure you don't do this for a shared PC such as
the Lab computers).

**Now add to your Gitlab account** (under settings in your browser)
To clip the public key to your clipboard for pasting, in Git Bash:

```Bash
cat ~/.ssh/id_ed25519.pub | clip
```

macOS:

```Bash
pbcopy < ~/.ssh/id_ed25519.pub
```

WSL/GNU/Linux (you may need to install xclip but it will give you instructions):

```Bash
xclip -sel clip < ~/.ssh/id_ed25519.pub
```

Now we can check whether that worked correctly:

```Bash
ssh -T git@git.unimelb.edu.au
```

If it's the first time you connect to Gitlab via SSH, you will be asked to verify it's 
authenticity (respond yes). You may also need to provide your password. 
If it responds with *Welcome to Github, @username!* then you're all setup. 

[//]: # (pip extras install [cpufit, gpufit])

[//]: # (see DEVDOCS.md for instructions on compiling cpufit, gpufit extensions)
