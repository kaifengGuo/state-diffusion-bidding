# Security

This repository must remain free of credentials and internal infrastructure details.

Do not commit:

- API tokens, passwords, browser cookies, or authentication headers
- SSH/private keys or certificates
- internal hostnames, machine URLs, or user-specific server paths
- datasets, checkpoints, logs, or experiment caches

Use command-line arguments and local environment variables for paths. Report an accidental secret exposure by rotating the credential first, then removing it from Git history.
