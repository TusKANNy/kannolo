# Publishing Kannolo to PyPI

## Understanding the Workflows

There are **two separate GitHub workflows**:

1. **test-linux.yml + test-macos.yml** (automatic on every commit)
   - Run on each push to `main` or `develop` branch
   - Test that your code compiles on different Python versions (3.10-3.13)
   - **You don't need to do anything** — they run automatically

2. **publish-wheels.yml** (manual when releasing)
   - Only runs when you push a git tag (e.g., `git tag v0.5.0`)
   - Builds wheels for all platforms and publishes to PyPI
   - **You trigger it by tagging a release**

---

## One-time Setup (First Time Only)

1. Create a PyPI API token:
   - Go to https://pypi.org/manage/account/tokens/
   - Create a new token, select "Entire account" scope
   - Copy the token
   
2. Add it to GitHub:
   - Go to your repository → Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_PASSWORD`
   - Value: `pypi-...` (paste your token)

---

## Complete Command List

### Daily Development (Nothing Extra Needed)

```bash
# Just code normally
git add .
git commit -m "Your changes"
git push origin main
```

The test workflows run automatically.

### When Ready to Release (3 Commands)

**Before you do this:**
1. Update version in `pyproject.toml`
2. Make sure all tests pass (check GitHub Actions)
3. Decide on a version (e.g., `0.5.0`)

**Then:**

```bash
git add .
git commit -m "Release v0.5.0"
git tag v0.5.0
git push origin main --tags
```

**What happens next:**
- GitHub automatically builds wheels (~10-15 minutes)
- Publishes to PyPI
- Done! Users can now `pip install kannolo`

---

## Workflow Summary

| When | What Happens | Your Action |
|------|-------------|------------|
| Push to `main` | Tests run automatically | Nothing |
| Push to `develop` | Tests run automatically | Nothing |
| Time to release | Update version, tag, push | 3 git commands |

---

## What Gets Built When You Tag

For each tag, GitHub automatically:
- Builds Python 3.10, 3.11, 3.12, 3.13 wheels
- Builds for Linux, macOS (x86-64 + ARM64)
- Builds source distribution
- Publishes all to PyPI
