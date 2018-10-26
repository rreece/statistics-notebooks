#
# file: setup.sh
# author: Ryan Reece <ryan@cerebras.net>
# created: April 19, 2018
#
# Basic setup script
#
###############################################################################


##-----------------------------------------------------------------------------
## pre-setup helpers, don't touch
##-----------------------------------------------------------------------------

#SVN_USER=${SVN_USER:-$USER} # set SVN_USER to USER if not set

path_of_this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
path_above_this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
export MY_PROJECT=${path_of_this_dir}

add_to_python_path()
{
    export PYTHONPATH=$1:$PYTHONPATH
    echo "  Added $1 to your PYTHONPATH."
}

add_to_path()
{
    export PATH=$1:$PATH
    echo "  Added $1 to your PATH."
}

add_to_ld_library_path()
{
    export LD_LIBRARY_PATH=$1:$LD_LIBRARY_PATH
    echo "  Added $1 to your LD_LIBRARY_PATH."
}

##-----------------------------------------------------------------------------
## setup virtualenv
##-----------------------------------------------------------------------------

if [ -f env1/bin/activate ]; then
    source env1/bin/activate
else
    echo "  Setting up virtualenv env1"
    virtualenv -p python3 env1
    source env1/bin/activate
    pip3 install -r requirements.txt
fi


##-----------------------------------------------------------------------------
## install other packages
##-----------------------------------------------------------------------------

## tensorflow
#if [ ! -d tensorflow ]; then
#    echo "  Cloning tensorflow..."
#    git clone git@github.com:tensorflow/tensorflow.git
#    echo "  Configuring tensorflow build..."
#    export PYTHON_BIN_PATH=${path_of_this_dir}/env1/bin/python3
#    export TF_ENABLE_XLA=1
#    cd tensorflow
#    yes "" | $PYTHON_BIN_PATH configure.py
#    echo "  Building tensorflow..."
#    bazel build -c opt --copt=-march=native --copt=-mfpmath=both //tensorflow/tools/pip_package:build_pip_package
#    echo "  Building tensorflow pip packaage..."
#    bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp
#    echo "  Installing tensorflow pip whl..."
#    if [ -f /tmp/tensorflow-1.8.0-cp34-cp34m-linux_x86_64.whl ]; then
#        pip3 install /tmp/tensorflow-1.8.0-cp34-cp34m-linux_x86_64.whl
#    else
#        echo "  Error: cannot find /tmp/tensorflow-1.8.0-cp34-cp34m-linux_x86_64.whl"
#    fi
#    cd ${path_of_this_dir}
#fi

## tensorflow/models
#if [ ! -d models ]; then
#    git clone git@github.com:tensorflow/models.git
#    cd models
#    pip3 install -r official/requirements.txt
#    cd ${path_of_this_dir}
#fi

## tensorflow/benchmarks
#if [ ! -d benchmarks ]; then
#    git clone git@github.com:tensorflow/benchmarks.git
#fi

## cerebras/tf-models-private
#if [ ! -d tf-models-private ]; then
#    git clone git@github.com:Cerebras/tf-models-private.git
#fi


##-----------------------------------------------------------------------------
## setup PYTHONPATH
##-----------------------------------------------------------------------------

echo "  Setting up your PYTHONPATH."
add_to_python_path ${MY_PROJECT}/python
#add_to_python_path ${MY_PROJECT}/models
#add_to_python_path ${MY_PROJECT}/tf-models-private
#add_to_python_path ${MY_PROJECT}/tensorflow  ## don't need
echo "  done."


##-----------------------------------------------------------------------------
## setup PATH
##-----------------------------------------------------------------------------

#echo "  Setting up your PATH."
##add_to_path ${MY_PROJECT}
#add_to_path ${MY_PROJECT}/scripts
#echo "  done."


##-----------------------------------------------------------------------------
## setup LD_LIBRARY_PATH
##-----------------------------------------------------------------------------

## cuda
#if [ -d /usr/local/cuda ]; then
#    add_to_ld_library_path /usr/local/cuda/lib64
#    add_to_ld_library_path /usr/local/cuda/extras/CUPTI/lib64
#fi



