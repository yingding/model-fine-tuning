# Troubleshooting — AML asset registry gotchas

Lessons learned the hard way while wiring up the base-model download
pipeline. All apply to **Azure Machine Learning v2** (`azure-ai-ml` SDK,
`az ml` CLI).

---

## 1. Model versions must be positive integers — semver is rejected

```
HttpResponseError: (UserError) Model version must be a positive integer.
Code: ModelVersionOutOfRange
```

Different asset types accept different version formats:

| Asset type | Version format | Examples |
|---|---|---|
| `Data`        | Any string | `1.0.0`, `2024-06`, `my-run-3`, `1` |
| `Environment` | Any string | same |
| `Model`       | **Positive integer string only** | `1`, `2`, `42` |
| `Component`   | Any string | same as Data |

Plan accordingly — if you want consistent versioning, use plain integers
everywhere even though most asset types tolerate semver.

---

## 2. Model versions are immutable — only `archive`, no `delete` or repath

Once a model version is registered against a blob path, you **cannot**:
- Change its `path` (`A model with this name and version already exists.
  ... the existing model's asset path cannot be changed.`)
- Hard-delete it (`az ml model delete` doesn't exist — only `archive`)

If you registered against the wrong path, your only recovery is:

```bash
az ml model archive --name <name> --version <bad-version> -g <rg> --workspace-name <ws>
# Then re-register with a new version number pointing at the correct path
az ml model create --name <name> --version <next> --type custom_model \
  --path "azureml://datastores/workspaceblobstore/paths/<correct>/" ...
```

Archive only hides from default listings — the broken version still
exists in the registry, just not as `latest`.

**Lesson**: pick your blob path naming convention before the first
registration. Migrating later wastes a version number.

---

## 3. Sync blob copy is capped at 256 MB

```
The source request body for synchronous copy is too large and exceeds
the maximum permissible limit (256MB).
ErrorCode: CannotVerifyCopySource
```

`az storage blob copy start --requires-sync true` blocks the request until
the copy completes — works fine for small files but rejects anything ≥256 MB.

For large files (model weights, big datasets):

```bash
# Drop --requires-sync to use async server-side copy (no size limit).
az storage blob copy start \
  --account-name $SA --auth-mode login \
  --destination-container $CONTAINER \
  --destination-blob "$DST/big-file.safetensors" \
  --source-uri "https://${SA}.blob.core.windows.net/${CONTAINER}/${SRC}/big-file.safetensors"
```

The copy starts within seconds and runs server-side (no client bandwidth).
For intra-account copies on the same storage, even multi-GB shards
complete in seconds to a couple of minutes.

**Poll for completion**:

```bash
az storage blob show \
  --account-name $SA --auth-mode login \
  --container-name $CONTAINER --name "$DST/big-file.safetensors" \
  --query "properties.copy.status" -o tsv
# pending / success / failed / aborted
# After success, properties.copy.status becomes null again — verify by size.
```

---

## 4. HuggingFace downloads are slow without `HF_TOKEN`

```
Warning: You are sending unauthenticated requests to the HF Hub.
Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

Anonymous downloads of large files get **heavily throttled** — a 16 GB
model can take 30+ minutes. Setting an HF read token speeds this up
10–50×. Get one at <https://huggingface.co/settings/tokens>.

In the AML job, pass it via `environment_variables`:

```python
download_job = command(
    ...,
    environment_variables={"HF_TOKEN": "hf_xxxx..."},
)
```

⚠ **Don't commit tokens to git.** Use a Key Vault reference or an env var
loaded from `.env` (gitignored).

---

## 5. BlobFuse mounts report 0 bytes free (false alarm)

```
UserWarning: Not enough free disk space to download the file.
The expected file size is: 4999.80 MB.
The target location /mnt/azureml/.../model only has 0.00 MB free disk space.
```

`statvfs()` on BlobFuse returns `0` because BlobFuse can't predict free
blob storage. The warnings are **noise** — writes still succeed.

But there's a real issue lurking: `huggingface_hub.snapshot_download`
stages files under `<output_dir>/.cache/huggingface/download/` as
`.incomplete`, then renames into place. On BlobFuse the rename can be
slow or flaky.

**Fix**: stage to local node SSD first, then `shutil.copytree` to the
mount. See [`src/download_base_model.py`](../src/download_base_model.py)
for the pattern.

---

## 6. AML model registry path vs version label can diverge

This is a consequence of #2. If you ever migrate blob storage paths
(e.g. cleaning up old `v1.0.0/` blobs after switching to `v1/`), the
**registry entry's path is frozen** at registration time.

Two ways out:

**(a) Live with the divergence** — the training job mounts whatever path
the registry points at, so things still work. Just document the mapping
in the constants cell.

**(b) Realign by bumping the version** (recommended, what we ended up doing):
1. Copy blob `vN/` → `v(N+1)/` (server-side, async for files >256 MB).
2. Archive registry `@N`.
3. Register `@(N+1)` pointing at the new `v(N+1)/` path.
4. Delete the old `vN/` blobs.

End state: blob path version label matches registry version label, no
mental overhead. Cost: two version numbers burned (the broken `@N` plus
whatever earlier broken versions you accumulated). Archived versions
don't show up in default listings and don't consume blob storage —
they're just registry metadata.

**Lesson**: when bumping versions, the `BASE_MODEL_OUTPUT_URI` template
in the notebook should always use `v{VERSION}/` directly so future
downloads land in a path matching the registry version automatically.
Avoid one-off `v1/` overrides — they create the divergence in the
first place.

---

## See also

- [Troubleshooting — Studio data preview](troubleshooting-studio-preview.md)
- [Architecture diagrams](architecture.md)
- [`src/download_base_model.py`](../src/download_base_model.py) — CPU-side
  model snapshotting with local-SSD staging.
