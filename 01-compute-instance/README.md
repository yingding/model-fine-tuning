# Fine-tuning with GPU Compute Instance in AML

## Storage

Notebook partition are mounted in Computer Instance under the path 
`/home/azureuser/cloudfiles/code/Users/<user_alias>`

Home Dir `/home/azureuser`
Sub Dir
`~/cloudfiles/code/Users/<user_alias>`

In Jupyter notebook
default mount path: `/mnt/batch/tasks/shared/LS_root/mounts/clusters/t4sample`
user data home path: `code/Users/<user_alias>`
Full jupyter notebook path:
`/mnt/batch/tasks/shared/LS_root/mounts/clusters/t4sample/code/Users/<user_alias>`

## useful cmds
show the ssh pub key with `Get-Content`:
```PowerShell
 Get-Content C:\Users\<user_alias>\.ssh\id_rsa_xxx.pub
```

## Note
* currently, Compute Instance doesn't support identity-based storage account access at the time of creation. The AML Workspace shall be configured to use storage account key access instead, so that the Compute Instance can be created.
Otherwise you will receive an error `ein storage can't be connected, proxy error`, you will also see in the compute plane of the AML workspace, that a warning is show for your to login to connect storage.
First deploy with key access and then switch.
* Issue https://github.com/Azure-Samples/ai-studio-in-a-box/issues/7#issuecomment-2273346002
* Workaround in FAQ: https://github.com/Azure-Samples/ai-studio-in-a-box?tab=readme-ov-file#faq

## Example of setup a custom conda env in AML Compute Instance
* https://github.com/Azure/azureml-examples/blob/main/setup/setup-ci/setup-custom-conda-env.sh

## Reference

* AML Compute Instance https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-instance?view=azureml-api-2
* Create Compute Instance https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-compute-instance?view=azureml-api-2&tabs=python
* start up script location https://learn.microsoft.com/en-us/azure/machine-learning/how-to-customize-compute-instance?view=azureml-api-2#create-the-setup-script