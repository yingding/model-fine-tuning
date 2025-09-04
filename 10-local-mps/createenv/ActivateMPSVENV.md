# MacOSX
```shell
VERSION="3.12";
ENV_NAME="mpsft";
ENV_SURFIX="pip";

ENV_FULL_NAME="${ENV_NAME}${VERSION}${ENV_SURFIX}";

ENV_DIR="$HOME/Code/VENV";
PROJ_DIR="$HOME/Code/VCS/ai/model-fine-tuning";

SUB_PROJ="10-local-mps";
PACKAGE_FILE="${PROJ_DIR}/${SUB_PROJ}/requirements.txt";

source ${ENV_DIR}/${ENV_FULL_NAME}/bin/activate;
which python3;

python3 -m pip install --upgrade pip;
python3 -m pip install -r ${PACKAGE_FILE} --no-cache;
```