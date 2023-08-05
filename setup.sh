# setup.sh
# Script for installing and/or activating a virtualenv


##-----------------------------------------------------------------------------
## pre-setup helpers, don't touch
##-----------------------------------------------------------------------------

path_of_this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
path_above_this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"

add_to_python_path()
{
    export PYTHONPATH=$1${PYTHONPATH:+:${PYTHONPATH}}
    echo "  Added $1 to your PYTHONPATH."
}

add_to_path()
{
    export PATH=$1${PATH:+:${PATH}}
    echo "  Added $1 to your PATH."
}

add_to_ld_library_path()
{
    export LD_LIBRARY_PATH=$1${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    echo "  Added $1 to your LD_LIBRARY_PATH."
}


##-----------------------------------------------------------------------------
## setup virtualenv
##-----------------------------------------------------------------------------

venv_name=".venv"

if [ -f ${venv_name}/bin/activate ]; then
    source ${venv_name}/bin/activate
else
    echo "  Setting up virtualenv ${venv_name}"
    python -m venv ${venv_name}
    source ${venv_name}/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi


##-----------------------------------------------------------------------------
## install other packages
##-----------------------------------------------------------------------------

if [ ! -d python ]; then
    echo "mkdir python"
    mkdir python
fi

# rreece/hepplot
if [ ! -d python/hepplot ]; then
    echo "Installing hepplot..."
    cd python
    git clone git@github.com:rreece/hepplot.git
    cd ..
fi


#-----------------------------------------------------------------------------
# setup paths
#-----------------------------------------------------------------------------

add_to_path ${path_of_this_dir}/scripts
add_to_python_path ${path_of_this_dir}/python

## cuda
#if [ -d /usr/local/cuda ]; then
#    add_to_ld_library_path /usr/local/cuda/lib64
#    add_to_ld_library_path /usr/local/cuda/extras/CUPTI/lib64
#fi

echo "  Done."


##-----------------------------------------------------------------------------

echo ""
echo "To start jupyter, do:"
echo "jupyter notebook --no-browser --port=7000"
echo ""
echo "On your local machine, you should port-forward:"
echo "ssh -NfL 7000:localhost:7000 ryan@192.168.50.77"
echo ""

