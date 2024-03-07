#!/bin/bash -i
# Want to add to .bashrc/.bash_profile
if [[ "$OSTYPE" == "darwin"* ]]; then
    BASH_FILE=$HOME"/.bash_profile"
else
    BASH_FILE=$HOME"/.bashrc"
fi

# Add an export to PYTHONPATH in bashrc/bash_profile
echo '#Added by refine plan installer (for specific install location)' >> $BASH_FILE
echo -e "export PYTHONPATH=$(pwd)/src:\$PYTHONPATH" >> $BASH_FILE
echo -e "export REFINE_PLAN_PATH=$(pwd)\n" >> $BASH_FILE

# Source the file to actually setup the new python path env var
export PYTHONPATH=`pwd`/src:$PYTHONPATH
export REFINE_PLAN_PATH=`pwd`
source $BASH_FILE
