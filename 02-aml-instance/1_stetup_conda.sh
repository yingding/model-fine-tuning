conda create -n llm_ft  --yes  python=3.12
conda run    -n llm_ft  --live-stream conda install pip
conda run    -n llm_ft  --live-stream pip install -r requirements.txt
conda run    -n llm_ft  --live-stream python -m ipykernel install --user --name llm_ft --display-name "Python (llm_ft)"