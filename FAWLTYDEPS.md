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
- Maps `PIL` to `pillow` package
- Maps `google` to `google-cloud-storage` package
- Ignores `gunicorn` (WSGI server, not imported)
- Includes shared `lib/` directory

### Training (`training/pyproject.toml`)

Key features:
- Maps `PIL` to `pillow` package
- Includes shared `lib/` directory
- No special ignores (all packages should be imported)

## CI/CD Integration

The GitHub Actions workflow should check both:

```yaml
- name: Check API dependencies
  run: |
    cd api
    fawltydeps

- name: Check Training dependencies
  run: |
    cd training
    fawltydeps
```

## Common Issues

### False Positive: "PIL undeclared"

**Solution**: Already mapped in both configs via `custom_mapping`

### False Positive: "google undeclared"

**Solution**: Already mapped in API config (`google.cloud.storage` → `google-cloud-storage`)

### False Positive: "gunicorn unused"

**Solution**: Already ignored in API config (it's a CLI tool)

### True Positive: "torchvision unused" (API only)

**Solution**: Removed from `api/requirements.txt` (only needed for training)

## Benefits

✅ **Accurate dependency checking** - Each subproject validated independently
✅ **Faster deployments** - API doesn't include training dependencies
✅ **Clear separation** - API and training concerns are isolated
✅ **CI/CD friendly** - Can check API deps before deployment
✅ **Smaller Docker images** - Only include what's needed
