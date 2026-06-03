# Architecture — Compute Cluster fine-tuning workflow

End-to-end view of the network, identity, and data flows in this project.
Use it as a mental map when something fails so you know which layer to check.

---

## 1. End-to-end workflow

```mermaid
flowchart LR
  subgraph Local["💻 Local dev box (corp network)"]
    NB["Notebooks<br/>(00 → 01 → 02)"]
    SDK["azure-ai-ml SDK<br/>+ az CLI"]
    Smoke["smoke/verify_<br/>dataset_access.py"]
  end

  subgraph Azure["☁️ Azure subscription"]
    subgraph WS["AML Workspace (aml-ww-yw-dos)"]
      Ctrl["Control plane<br/>(jobs / data / models APIs)"]
      Reg["Model Registry"]
      MV["Managed VNet<br/>(allow_internet_outbound)"]
      Cluster["GPU Compute Cluster<br/>Cluster-A100-1GPU<br/>+ system-assigned MI"]
    end

    subgraph Net["Network gates"]
      NSP["Network Security<br/>Perimeter<br/>(IP /24 + sub rule)"]
      SAFW["Storage<br/>networkRuleSet<br/>(IP /24 + bypass)"]
    end

    subgraph Stg["Workspace storage (amlwwywdos…)"]
      Blob["Blob container<br/>azureml-blobstore-*"]
      File["File share<br/>workspacefilestore"]
    end

    ACR["ACR<br/>(custom env image)"]
    KV["Key Vault"]
  end

  NB -- "az login / SDK calls" --> Ctrl
  SDK -- "data.get / jobs.create" --> Ctrl
  Smoke -- "list_containers + blob read" --> Net
  Net -- "allow /24 from corp" --> Stg
  Ctrl -- "submit job" --> Cluster
  Cluster -- "MSI: Storage Blob Data Contributor" --> Blob
  Cluster -- "pull image" --> ACR
  Cluster -- "save ./outputs/" --> Blob
  Cluster -- "Run.register_model" --> Reg
  Reg -- "points at" --> Blob
  MV -. "managed PE" .-> Blob
  MV -. "managed PE" .-> File
  MV -. "managed PE" .-> KV
```

**Read this as**: your laptop talks to the AML control plane and (separately)
to storage's data plane through two independent firewalls (NSP + storage
`networkRuleSet`). The compute cluster lives inside the workspace's managed
VNet and reaches storage via system-installed private endpoints — it does
**not** depend on your laptop's IP at all.

---

## 2. Why both NSP and storage firewall?

```mermaid
flowchart TB
  Client["Client request<br/>(Studio preview, az CLI, SDK)"]
  PNA{"storage<br/>publicNetworkAccess?"}
  SAFW["storage networkRuleSet<br/>(ipRules + bypass)"]
  NSP["NSP inbound rules<br/>(per profile)"]
  Allow["✅ reach blob/file"]
  Deny["❌ 403 / connection refused"]

  Client --> PNA
  PNA -- "Enabled (default)" --> SAFW
  PNA -- "SecuredByPerimeter" --> NSP
  PNA -- "Disabled" --> Deny
  SAFW -- "match" --> Allow
  SAFW -- "no match" --> Deny
  NSP  -- "match" --> Allow
  NSP  -- "no match" --> Deny
```

The 00 prep notebook configures `publicNetworkAccess=Enabled`, which means
**the storage `networkRuleSet` is authoritative** for direct data-plane
traffic. NSP is still useful for NSP-gated paths and as documentation of
allowed networks, but **Step 2c (storage firewall sync) is what actually
unblocks the AML Studio preview and any laptop-direct blob access**.

---

## 3. Identity / RBAC matrix

```mermaid
flowchart LR
  subgraph Principals
    User["👤 You<br/>(AAD user)"]
    WSMSI["🤖 Workspace MSI"]
    ClusterMSI["🤖 Cluster MSI"]
  end

  subgraph Roles["RBAC on storage account"]
    Blob["Storage Blob Data<br/>Contributor"]
    File["Storage File Data<br/>Privileged Contributor"]
    Acct["Storage Account<br/>Contributor (control-plane)"]
  end

  subgraph Targets
    BlobC["Blob containers"]
    FileC["File share"]
    Listkeys["listKeys (shared-key)"]
  end

  User --> Blob
  User --> File
  User --> Acct
  WSMSI --> Blob
  WSMSI --> File
  ClusterMSI --> Blob

  Blob --> BlobC
  File --> FileC
  Acct --> Listkeys
```

Three principals, three roles. The most common gap is the **cluster MSI →
Storage Blob Data Contributor** assignment — Step 3 of the 00 prep notebook
fixes that.

---

## 4. Two-stage data flow at training time

```mermaid
sequenceDiagram
  participant L as 💻 Laptop
  participant Ctrl as AML control plane
  participant Stg as Workspace storage
  participant Cluster as GPU cluster (MI)
  participant Reg as Model registry

  Note over L: prepare_dataset.py (CPU job)
  L->>Ctrl: jobs.create (dataset prep)
  Ctrl->>Cluster: schedule CPU job
  Cluster->>Stg: write Arrow folder<br/>(datasets/.../v1.0.0)
  Cluster->>Reg: register Data asset

  Note over L: fine_tune_llama_3_doctor.py (GPU job)
  L->>Ctrl: jobs.create (training)
  Ctrl->>Cluster: schedule GPU job
  Cluster->>Stg: read Data asset (Arrow)
  Cluster->>Cluster: QLoRA SFT training
  Cluster->>Stg: write ./outputs/<br/>(adapter + merged + tokenizer)
  Cluster->>Cluster: in-memory inference test
  Cluster->>Reg: Run.register_model<br/>(adapter + merged)
```

The two jobs are intentionally split: dataset prep needs no GPU, training
needs an A100 — keeps cost down and the GPU job pure-training.

---

## 5. Storage layout

```
amlwwywdos1036066399.blob.core.windows.net/
└── azureml-blobstore-<workspace-id>/
    ├── datasets/
    │   └── ai-medical-chatbot-llama3/
    │       └── v1.0.0/                         ← Data asset (HF Arrow)
    │           ├── dataset_dict.json
    │           ├── train/data-00000-of-00001.arrow
    │           └── test/data-00000-of-00001.arrow
    └── ExperimentRun/dcid.<run-id>/outputs/
        ├── llama3-8b-chat-doctor/              ← LoRA adapter + tokenizer
        ├── llama3-8b-chat-doctor_config/       ← LoRA config
        ├── llama3-8b-chat-doctor_full/         ← Merged model + tokenizer
        └── sample_inference.json
```

Both `./outputs/` folders get auto-uploaded by AML, then `Run.register_model`
in `fine_tune_llama_3_doctor.py` creates registry entries pointing at the
uploaded paths.

---

## See also

- [Troubleshooting — Studio data preview](troubleshooting-studio-preview.md)
- [Prep notebook (`00_aml_cc_prepare_sub_environment.ipynb`)](../00_aml_cc_prepare_sub_environment.ipynb)
