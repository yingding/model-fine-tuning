# Windows
## (Optional) Create venv
```powershell
cd $env:USERPROFILE\Documents\VCS\llm-train;

$VERSION="3.12";
$ENV_NAME="azfdyft";
$ENV_SURFIX="pip";
$PM="pip";
$WORK_DIR="$env:USERPROFILE\Documents\VENV\";
.\envtools\create_env.ps1 -VERSION $VERSION -ENV_NAME $ENV_NAME -ENV_SURFIX $ENV_SURFIX -PM $PM -WORK_DIR $WORK_DIR;
```

# MacOSX
## (Optional) Create venv
```shell
pwd;
pushd $HOME/Code/VCS/ai/llm-train;

VERSION="3.12";
ENV_NAME="azfdyft";
ENV_SURFIX="pip";
ENV_FULL_NAME="${ENV_NAME}${VERSION}${ENV_SURFIX}";
ENV_DIR="$HOME/Code/VENV";
source ./envtools/create_env.sh -p "${ENV_DIR}/${ENV_FULL_NAME}" -v $VERSION;

popd;
pwd;
```

## Add a jupyter notebook kernel to VENV
```shell
VERSION="3.12";
ENV_NAME="azdfyft";
ENV_SURFIX="pip";

ENV_FULL_NAME="${ENV_NAME}${VERSION}${ENV_SURFIX}";
ENV_DIR="$HOME/Code/VENV";

source ${ENV_DIR}/${ENV_FULL_NAME}/bin/activate;

python3 -m pip install --upgrade pip;
python3 -m pip install ipykernel;

deactivate;
```

We need to reactivate the venv so that the ipython kernel is available after installation.
```shell
VERSION="3.12";
ENV_NAME="azfdyft";
ENV_SURFIX="pip";

ENV_FULL_NAME="${ENV_NAME}${VERSION}${ENV_SURFIX}";
ENV_DIR="$HOME/Code/VENV";

source ${ENV_DIR}/${ENV_FULL_NAME}/bin/activate;

python3 -m ipykernel install --user --name=${ENV_FULL_NAME} --display-name ${ENV_FULL_NAME};
```
Note: 
* restart the vs code, to select the venv as jupyter notebook kernel

Reference:
* https://ipython.readthedocs.io/en/stable/install/kernel_install.html
* https://anbasile.github.io/posts/2017-06-25-jupyter-venv/

## Remove ipykernel
```shell
VERSION="3.12";
ENV_NAME="azfdyft";
ENV_SURFIX="pip";

ENV_FULL_NAME="${ENV_NAME}${VERSION}${ENV_SURFIX}";
jupyter kernelspec uninstall -y ${ENV_FULL_NAME};
```