# Fine-tuning with GPU Computer Cluster in AML

## Building an Environment
Build a Customer Docker Container as Environment

MAR Microsoft Artifact Registry / MCR Microsoft Container Registry https://github.com/microsoft/ContainerRegistry#browsing-mcr-content

MAR: https://mcr.microsoft.com/
acpt: azure container for pytorch https://learn.microsoft.com/en-us/azure/machine-learning/resource-azure-container-for-pytorch?view=azureml-api-2

The ACPT Image list on MCR 
https://onnxruntime.ai/docs/ecosystem/ptca_image_list.html


custom ACPT (Azure Container Pytorch) https://learn.microsoft.com/en-us/azure/machine-learning/how-to-azure-container-for-pytorch-environment?view=azureml-api-2

Artifact cache in Azure Container Registry https://learn.microsoft.com/en-us/azure/container-registry/artifact-cache-overview

Pre-build Docker Images for Inference https://learn.microsoft.com/en-us/azure/machine-learning/concept-prebuilt-docker-images-inference?view=azureml-api-2


Sources:
* ACPT Versions https://learn.microsoft.com/en-us/azure/machine-learning/resource-azure-container-for-pytorch?view=azureml-api-2#supported-configurations-for-azure-container-for-pytorch-acpt
* Curated Env pytorc (registries, azureml, aifx) https://ml.azure.com/

mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:40
mcr.microsoft.com/aifx/acpt/stable-ubuntu2004-cu121-py310-torch22x:biweekly.202406.1
https://ml.azure.com/registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/version/40?tid=16b3c013-d300-468d-ac64-7eda0820b6d3

mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:biweekly.202508.1

https://onnxruntime.ai/docs/ecosystem/ptca_image_list.html


```shell
# MY_REGISTRY=<name of your registry>
# MY_REPOSITORY=<name of your repository>

# MY_REGISTRY="azureml"
# MY_REPOSITORY="curated"

MY_REGISTRY="aifx";
MY_REPOSITORY="acpt";

MY_REGISTRY="azureml";
MY_REPOSITORY="curated";

az acr manifest list-metadata \
  --registry $MY_REGISTRY \
  --name $MY_REPOSITORY \
  --query '[:].[digest, imageSize, tags[:]]' \
  -o table;
```

```powershell
# Define variables
# $MY_REGISTRY = "azureml"
# $MY_REPOSITORY = "curated"

$MY_REGISTRY = "aifx";
$MY_REPOSITORY = "acpt";

# Execute the Azure CLI command in PowerShell
az acr manifest list-metadata `
    --registry $MY_REGISTRY `
    --name $MY_REPOSITORY `
    --query '[:].[digest, imageSize, tags[:]]' `
    -o table
```

Reference:
* https://stackoverflow.com/questions/49351966/azure-container-registry-list-images-tags-programmatically#:~:text=This%20lists%20all%20the%20images%2C%20whether%20tagged%20or,associated%20with%20that%20image%2C%20whether%20tagged%20or%20not.


#### Create Environment and add dockerfile
Then you got a container building job running on Serverless VM (STANDARD_E4DS_V4)

https://ml.azure.com/experiments/id/prepare_image/runs/imgbldrun_2ff31ee?wsid=/subscriptions/92645b1b-8a8a-4693-b8e2-c214a523fe40/resourcegroups/general-aml/workspaces/general-aml&tid=16b3c013-d300-468d-ac64-7eda0820b6d3