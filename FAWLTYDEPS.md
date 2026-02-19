# FawltyDeps Configuration

This project uses [FawltyDeps](https://github.com/tweag/FawltyDeps) to check for undeclared or unused Python dependencies.

## Project Structure

```
munimetro/
├── api/
│   ├── pyproject.toml       # FawltyDeps config
│   ├── requirements.txt
│   └── *.py
└── lib/
    └── *.py                 # Shared library
```

## Running FawltyDeps

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

## Configuration Details

### API (`api/pyproject.toml`)

Key features:
- Maps `PIL` to `pillow` package (custom_mapping)
- Maps `google` to `google-cloud-storage` package (custom_mapping)
- Ignores `gunicorn` as unused (CLI tool, not imported)
- Ignores `numpy` as unused (lazy import in lib functions)
- Ignores `lib` as undeclared (local module, not a package)
- Includes shared `lib/` directory

## CI/CD Integration

The GitHub Actions workflow runs FawltyDeps on push:

```yaml
- name: Check API dependencies
  run: fawltydeps --detailed
  working-directory: api
```

**Important**: Use step-level `working-directory`, not `--base-dir` flag. The `--base-dir` flag doesn't limit code scanning to that directory.

## Common Issues

### False Positive: "PIL undeclared"

**Solution**: Already mapped in config via `custom_mapping`

### False Positive: "google undeclared"

**Solution**: Already mapped in API config (`google.cloud.storage` → `google-cloud-storage`)

### False Positive: "gunicorn unused"

**Solution**: Already ignored in API config via `ignore_unused` (it's a CLI tool)

### False Positive: "numpy unused"

**Solution**: Already ignored in API config via `ignore_unused` (imported lazily inside lib functions)

### False Positive: "lib undeclared"

**Solution**: Already ignored in config via `ignore_undeclared` (it's a local module, not a package)
