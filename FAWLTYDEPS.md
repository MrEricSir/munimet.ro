# FawltyDeps Configuration

This project uses separate FawltyDeps configurations for the API and training subprojects.

## Project Structure

```
munimetro/
├── api/
│   ├── pyproject.toml       # FawltyDeps config for API
│   ├── requirements.txt
│   └── *.py
├── training/
│   ├── pyproject.toml       # FawltyDeps config for training
│   ├── requirements.txt
│   └── *.py
└── lib/
    └── muni_lib.py          # Shared library (used by both)
```

## Running FawltyDeps

### Check API dependencies

```bash
cd api
fawltydeps
```

This checks:
- Code: `api/*.py` + `lib/*.py`
- Dependencies: `api/requirements.txt`
- Custom mappings:
  - `PIL` → `pillow`
  - `google` → `google-cloud-storage`
- Ignores: `gunicorn` (CLI tool)

### Check Training dependencies

```bash
cd training
fawltydeps
```

This checks:
- Code: `training/*.py` + `lib/*.py`
- Dependencies: `training/requirements.txt`
- Custom mappings:
  - `PIL` → `pillow`

### Check Both

```bash
# From project root
(cd api && fawltydeps) && (cd training && fawltydeps)
```

## Why Separate Configs?

The API and training subprojects have:

1. **Different dependencies**
   - API: `gunicorn`, `falcon`, `whitenoise`, `google-cloud-storage`
   - Training: `torchvision`, `matplotlib`, data processing tools

2. **Different deployment targets**
   - API → Cloud Run (production)
   - Training → Local/VM (development)

3. **Different optimization goals**
   - API: Minimize image size, reduce dependencies
   - Training: Include all tools needed for experimentation

## Configuration Details

### API (`api/pyproject.toml`)

Key features:
- Maps `PIL` to `pillow` package (custom_mapping)
- Maps `google` to `google-cloud-storage` package (custom_mapping)
- Ignores `gunicorn` as unused (CLI tool, not imported)
- Ignores `numpy` as unused (lazy import in lib functions)
- Ignores `lib` as undeclared (local module, not a package)
- Includes shared `lib/` directory

### Training (`training/pyproject.toml`)

Key features:
- Maps `PIL` to `pillow` package (custom_mapping)
- Ignores `lib` as undeclared (local module, not a package)
- Ignores `google` as undeclared (only needed for API deployment)
- Includes shared `lib/` directory

## CI/CD Integration

The GitHub Actions workflow runs FawltyDeps in each subdirectory:

```yaml
- name: Check API dependencies
  run: fawltydeps --detailed
  working-directory: api

- name: Check Training dependencies
  run: fawltydeps --detailed
  working-directory: training
```

**Important**: Use step-level `working-directory`, not `--base-dir` flag. The `--base-dir` flag doesn't limit code scanning to that directory.

## Common Issues

### False Positive: "PIL undeclared"

**Solution**: Already mapped in both configs via `custom_mapping`

### False Positive: "google undeclared"

**Solution**: Already mapped in API config (`google.cloud.storage` → `google-cloud-storage`)

### False Positive: "gunicorn unused"

**Solution**: Already ignored in API config via `ignore_unused` (it's a CLI tool)

### False Positive: "numpy unused"

**Solution**: Already ignored in API config via `ignore_unused` (imported lazily inside lib functions)

### False Positive: "lib undeclared"

**Solution**: Already ignored in both configs via `ignore_undeclared` (it's a local module, not a package)

### False Positive: "google undeclared" (Training only)

**Solution**: Already ignored in training config via `ignore_undeclared` (google-cloud-storage only needed for API)

### True Positive: "torchvision unused"

**Solution**: Removed from both `api/requirements.txt` and `training/requirements.txt` (not actually used)

## Benefits

✅ **Accurate dependency checking** - Each subproject validated independently
✅ **Faster deployments** - API doesn't include training dependencies
✅ **Clear separation** - API and training concerns are isolated
✅ **CI/CD friendly** - Can check API deps before deployment
✅ **Smaller Docker images** - Only include what's needed
