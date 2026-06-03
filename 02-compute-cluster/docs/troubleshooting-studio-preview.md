# Troubleshooting — AML Studio data preview

## TL;DR

The AML Studio **Data → Preview** panel can fail with confusing errors that
sound like network/auth problems but are actually **UI-only format
limitations**. Before chasing firewalls and RBAC, run the smoke test:

```bash
cd 02-compute-cluster
python smoke/verify_dataset_access.py <data-asset-name>
```

If all four layers pass, your data is accessible — the Studio 400 is a UI bug.

---

## Symptoms

### Symptom 1 — `AxiosError: Request failed with status code 400`

```
AxiosError: Request failed with status code 400
    at Jt (https://ml.azure.com/assets/manualChunk_data-fetch-*.js:19:32069)
    at XMLHttpRequest.B (https://ml.azure.com/assets/manualChunk_data-fetch-*.js:20:2206)
```

Origin: the Studio frontend's data-fetch bundle. The 400 comes from Studio's
own `ml.azure.com` API rejecting the preview request — it never reached
storage.

### Symptom 2 — `PrivateEndpointResolutionFailureException`

```
StreamError(PermissionDenied(Some(AuthenticationError(
"Unable to get access token for resource named '<sa>' ...
PrivateEndpointResolutionFailureException was caused by
DataAccessPrivateEndpointResolutionException.
Cannot authenticate data access ... with Workspace system assigned identity.
Make sure to connect to the Workspace with a Private Endpoint or whitelist
your public ip address on storage."
))))
```

Origin: rslex (the dataprep stream library) running inside the Studio service.
The "private endpoint / whitelist" wording is **generic boilerplate** emitted
for any stream failure — it does not mean PE resolution is the actual cause.

---

## Diagnosis

The 4-layer smoke test in `smoke/verify_dataset_access.py` localizes the
fault:

| Layer | What it proves | If it fails… |
|---|---|---|
| 1. AAD token for `storage.azure.com` | `az login` is valid | `az logout && az login --tenant <tenant>` |
| 2. `BlobServiceClient.list_containers` | Storage firewall + RBAC OK | See [00 notebook](../00_aml_cc_prepare_sub_environment.ipynb) Steps 1–2c |
| 3. `ml_client.data.get` | AML control plane OK | Asset doesn't exist or workspace unreachable |
| 4. Direct blob read of the asset's files | Data plane fully OK | Same as layer 2 |

If **all four pass**, the data is accessible end-to-end. The Studio UI is
the issue.

---

## Common UI-only failure modes

| Asset content | What Studio's tabular preview does | Result |
|---|---|---|
| `*.csv`, `*.parquet`, `*.json`, `*.jsonl` | Renders rows | ✅ works |
| `*.arrow` (HuggingFace `datasets`) | No handler — throws 400 | ❌ AxiosError 400 |
| `*.pt`, `*.bin`, `*.safetensors` (model weights) | No handler — throws 400 | ❌ AxiosError 400 |
| Empty `uri_folder` | Nothing to preview | ❌ misleading 400 |
| Mixed binary + text in same folder | Picks the binary, fails | ❌ |

**This does not affect compute-cluster jobs.** Your training code reads the
folder with `datasets.load_from_disk(...)`, `torch.load(...)`, etc., which
work fine regardless of Studio's preview support.

---

## Workarounds

1. **Use the SDK / smoke test** to verify the asset content:
   ```bash
   python smoke/verify_dataset_access.py <name>
   ```
2. **Register a smaller CSV/Parquet "manifest"** alongside binary data if
   you need a Studio-visible summary.
3. **Hard refresh** the Studio tab (Cmd+Shift+R) to rule out stale auth tokens.
4. **Try incognito** to rule out browser extensions or cached service workers.

---

## When it really is a network / auth problem

If layer **2** (direct blob access) or layer **4** (asset file read) in the
smoke test fails, then it's genuinely infrastructure. Re-run the relevant
step in the [00 prep notebook](../00_aml_cc_prepare_sub_environment.ipynb):

| Failing layer | Likely fix |
|---|---|
| Layer 2, connection timeout / `AuthorizationFailure` | Step 2c — your IP rotated outside the `/24` |
| Layer 2, `403 AuthenticationFailed` with token present | Wait 5–10 min for RBAC propagation, or re-run Step 3 |
| Layer 4, "container not found" | The datastore points at a different storage account |
| Anything + GSA tunnel on | Disable Global Secure Access (IPv6 egress breaks NSP) |
