# Update the existing Read the Docs website

The Read the Docs project `eotlayer` is already connected to
[Sparky2076/eotlayer](https://github.com/Sparky2076/eotlayer). The requested website
is [the stable version](https://eotlayer.readthedocs.io/en/stable/). A new project
or a repository migration is not needed for this theme refresh.

## Deployment settings

The public project and version API records were checked on **2026-10-24**:

| Setting | Existing value |
| --- | --- |
| Project slug | `eotlayer` |
| Repository | `https://github.com/Sparky2076/eotlayer.git` |
| Default source branch | `codex/documentation` |
| Default documentation version | `latest` |
| Requested documentation version | `stable`, mapped to the Git branch `stable` |
| Configuration | `.readthedocs.yaml`, with Sphinx configuration `docs/conf.py` |
| Language and build Python | English, Python 3.12 |

Before the theme refresh, the successful `stable` deployment used commit
`8b1b42f25f779b0ece522c525e3cd7820a82e3ec`
([build 34660759](https://app.readthedocs.org/projects/eotlayer/builds/34660759/)).
The refresh preserves that commit's numerical implementation and public API.
The package remains `0.1.0.dev0`; a `stable` documentation branch does not declare
a stable software release or a PyPI publication.

The same documentation was also imported into `Percy-Deng/diff-sinkhorn` in
October 2026. That copy does not control this public site's deployment: the
existing Read the Docs source remains `Sparky2076/eotlayer`.

## Publish a reviewed documentation update

1. Validate the source-derived API signature and build the HTML locally:

   ```bash
   python -m pip install -r docs/requirements.txt
   python scripts/check_docs.py
   python -m sphinx -b html -W --keep-going docs docs/_build/html
   ```

2. Review the built homepage, navigation, API pages, mathematical rendering,
   search, and mobile menu. Numerical examples can be checked separately with
   `python scripts/check_docs.py --run-examples` after installing the included
   package and its runtime dependencies.
3. Push the reviewed documentation commit to `codex/documentation` and update
   `stable` to the same reviewed commit, preserving existing branch history.
   A push only to `codex/documentation` updates the source for `latest`; it does
   not move the independent `stable` Git branch.
4. The existing GitHub/Read the Docs webhook can start builds when these branches
   are updated. Check the
   [project builds](https://app.readthedocs.org/projects/eotlayer/builds/) for a
   successful **stable** build at the intended commit. If the webhook does not
   start it, the project owner can trigger that version's build in Read the Docs
   and inspect the integration settings.
5. Verify [the stable website](https://eotlayer.readthedocs.io/en/stable/) after
   the build completes. Checking the default `latest` website alone does not
   validate the URL requested here.

Changing the project's default documentation version is unnecessary. Local
file edits, a GitHub Actions artifact, and an update to the diff-sinkhorn copy
do not independently publish this hosted site.

## Validation

The GitHub Actions `Documentation` workflow validates the API signature and builds
HTML with all Sphinx warnings treated as errors. It uploads the HTML as a build
artifact. Read the Docs performs a separate hosted build from the selected branch
using `.readthedocs.yaml`; that build determines the public site's content.

## Optional API authorization

If automation is used, provide a Read the Docs API token through the local
`READTHEDOCS_TOKEN` environment variable. Do not commit a token or paste it into
the documentation, repository issues, or chat. GitHub authentication alone does
not grant access to Read the Docs project settings or its build API.
