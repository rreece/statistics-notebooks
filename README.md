statistics-notebooks
===============================================================================

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


See also
-------------------------------------------------------------------------------

-   [My statistics notes](http://rreece.github.io/outline-of-philosophy/statistics.html)
-   <https://scikit-hep.org/pyhf/>
-   <https://github.com/scikit-hep/pyhf>
-   <https://github.com/diana-hep/pyhf>
-   <https://github.com/matthewfeickert/Statistics-Notes>


Author
-------------------------------------------------------------------------------

Ryan Reece ([@rreece](https://github.com/rreece))            
Created: October 26, 2018
