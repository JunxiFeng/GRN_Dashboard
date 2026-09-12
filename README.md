# SC.beta GRN dashboard — Streamlit Community Cloud

This directory is a self-contained, read-only deployment bundle. It contains
the dashboard and only the finalized precomputed tables used at runtime. It
does not run or rebuild the GRN, TF-gene correlation, pathway, or MAGMA analyses.

## Test locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app automatically reads `data/` beside `app.py`. To point the same app at
another compatible data bundle, set `SC_BETA_GRN_ROOT`.

## Publish on GitHub

Create a new GitHub repository from the contents of this directory:

```bash
cd streamlit_community_cloud
git init
git add app.py requirements.txt .streamlit data README.md
git commit -m "Deploy SC.beta GRN dashboard"
git branch -M main
git remote add origin https://github.com/ORG/REPOSITORY.git
git push -u origin main
```

No included file exceeds GitHub's 100 MB per-file limit. The repository is
data-heavy because the per-TF condition-association files are packaged for the
Target Genes tab.

## Deploy

1. Sign in at https://share.streamlit.io using GitHub.
2. Select **Create app**.
3. Choose the repository and `main` branch.
4. Set the entrypoint to `app.py`.
5. Choose a public `streamlit.app` subdomain and deploy.

The deployed URL is public. Do not publish the repository or app until every
included precomputed table is approved for public release.
