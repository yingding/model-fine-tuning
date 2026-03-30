## Quick Start for Compute Instance
This is an example of building custom environment for Azure Machine Learning GPU Compute Instance.

> Disclaimer: This is a learning/sample artifact – not production hardened.

## 📚 Contents

- [Dragon Copilot Python Sample Extension](#dragon-copilot-python-sample-extension)
  - [1. Features](#1-features)
  - [2. Quick Start](#2-quick-start)
  	- [2.1 Quick Start for Linux and Mac](#21-quick-start-for-linux-and-mac)
  - [6. License](#6-license)

---
## 1. Features

---
## 2. Quick Start
Deploy a custom compute instance with Python SDK.

### 2.1 Quick Start for Linux and Mac
Ensure python3.14 is installed and can be executed from your cmd shell as `python3.14`.

Run the following cmds in bash/zsh to start server.
```shell
# 1. change to the azure machine learning compute instance directory
cd ./model-fine-tuning/02-aml-instance;

# 2. create venv, activate venv and install packages
python3.14 -m venv .venv && source .venv/bin/activate && python3.14 -m pip install --upgrade pip && python3.14 -m pip install -r deploy_requirements.txt;

# 3. create .venv kernel
python3.14 -m ipykernel install --user --name=.venv --display-name "Python (.venv)"

# 3. start server with uvicorn invocation
# python3.14 -m uvicorn app.main:app --host 0.0.0.0 --port 5181 --reload
```

### 2.2 Quick Start for Windows
Ensure `python3.14` is installed over Microsoft Store and can be executed from your powershell as `python3.14`.

Run the following cmds in the Powershell to start server.
```powershell
# 1. change to the pythonSampleExtension directory
cd .\samples\DragonCopilot\Workflow\pythonSampleExtension;

# 2. create venv, activate venv and install packages
python3.14 -m venv .venv && .\.venv\Scripts\activate && python3.12 -m pip install --upgrade pip && python3.12 -m pip install -r requirements.txt;

# 3. start server with uvicorn invocation
python3.12 -m uvicorn app.main:app --host 0.0.0.0 --port 5181 --reload
```


---
## 6. License
See root `LICENSE`.