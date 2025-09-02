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

## Reference