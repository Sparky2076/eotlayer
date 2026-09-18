# Connect Read the Docs

Repository: https://github.com/Sparky2076/eotlayer

1. Sign in at https://app.readthedocs.org/ using the account that should own the
   EOTLayer documentation.
2. Choose **Add project**. Allow the Read the Docs GitHub App access to the
   `Sparky2076/eotlayer` repository, then select it. Alternatively choose
   **Configure manually** and enter `https://github.com/Sparky2076/eotlayer.git`.
3. Use project name **eotlayer** and confirm that the project slug is **eotlayer**.
   The public project lookup returned 404 during preparation, but this alone
   does not guarantee that the slug can be registered.
4. Configuration file: `.readthedocs.yaml`. Documentation language: English.
   Default source branch: `codex/documentation`.
5. Start the first build. In **Versions**, activate the **stable** branch and
   build it. Set the default documentation version to `stable` if desired.

Expected public URL after a successful stable build:
https://eotlayer.readthedocs.io/en/stable/

The `stable` documentation branch does not declare a stable software release;
the included package version remains `0.1.0.dev0`.

For a manual import, configure the repository webhook according to Read the Docs'
instructions to enable automatic rebuilds on pushes. The GitHub App integration
normally handles this automatically.

## Validation

The GitHub Actions `Documentation` workflow validates the API signature and builds
HTML with all Sphinx warnings treated as errors. It uploads the HTML as a build
artifact. Read the Docs separately hosts the site after its project is connected.

## Optional API authorization

If automation is used, provide a Read the Docs API token through the local
`READTHEDOCS_TOKEN` environment variable. Do not commit a token or paste it into
the documentation, repository issues, or chat. GitHub authentication alone does
not authorize creating a project in a Read the Docs account.
