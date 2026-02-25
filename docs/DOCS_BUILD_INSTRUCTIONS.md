To build documentation:

- navigate to the root directory, e.g. /dukit/
    (which should contain the directory 'src')
- install pdoc: `pip3 install pdoc3` or similar 
    (see [pdoc3](https://pdoc3.github.io/pdoc/))
- cmd: `pdoc3 --output-dir docs/ --html --template-dir docs/ --force --skip-errors ./src/dukit/`
- (or similar, may be different on windows)
- OR you can run `just docs` from the root directory (requires [just](https://github.com/casey/just) and [uv](https://docs.astral.sh/uv/))

- the force option overwrites any existing docs
- skip-errors will avoid issues e.g. if you don't have pygpufit installed.

Note: we commit the docs, as they're built with github pages (not actions).
NB: the index.html file at this level is a redirect to dukit/index.html, leave that there.
