statistics-notebooks
===============================================================================

[![CI badge](https://github.com/rreece/statistics-notebooks/actions/workflows/ci.yml/badge.svg)](https://github.com/rreece/statistics-notebooks/actions)

Introduction
-------------------------------------------------------------------------------

This repo is for my statistics notes.


Launching a jupyter notebook server
-------------------------------------------------------------------------------

First, setup the virtualenv, then launch jupyter:

    source setup.sh
    cd notebooks/
    jupyter notebook --no-browser --port=8000

If you launched jupyter remotely, then on your local machine
you should port-forward:

    ssh -NfL 8000:localhost:8000 ryan@192.168.11.22


Docs
-------------------------------------------------------------------------------

-   [TODOs](docs/todos.md)

The notes and results in: [docs/](docs/)


Author
-------------------------------------------------------------------------------

Ryan Reece ([@rreece](https://github.com/rreece))            
Created: October 26, 2018

