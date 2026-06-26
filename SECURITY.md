# Security Policy

## Reporting a Vulnerability

Please report vulnerabilities **privately** via this repository's
**Security → Advisories → "Report a vulnerability"**. We aim to acknowledge within 72 hours.

## Supported Versions

The latest commit on `main` is the supported version.

## Notes

- This repo ships no published package or binary — it's a CI/eval harness plus config.
- No secrets are committed. `sk-hermes-local-dev` is a **non-secret placeholder** master key for
  a localhost proxy; real keys come only from environment variables / GitHub Actions secrets.
- GitHub Actions are pinned to commit SHAs; Dependabot keeps them current.
