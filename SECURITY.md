# Security Policy

StoMar is a self-hosted stock-market trading suite that touches real money,
broker credentials, and trading state. We treat security as a first-class
concern. A detailed description of the security model can be found in
[docs/Security-Model.md](docs/Security-Model.md).

## Supported Versions

Security fixes are applied to the latest release and, where the fix is
trivial to backport, to the current development branch.

| Version | Supported          |
| ------- | ------------------ |
| 0.0.x   | :white_check_mark: (latest) |
| < 0.0.x | :x:                |

Only the most recent tagged release receives security fixes. Upgrade to
the latest release to stay covered.

## Reporting a Vulnerability

**Do not open a public issue, PR, or discussion for a vulnerability.**

Report security concerns privately via the repository's
[Security tab](https://github.com/uttampaliwal/stomar/security)
(private vulnerability reporting), or email the maintainer directly.

Please include, when available:

- The affected file/endpoint and a minimal reproduction
- The impact (what an attacker could do)
- Whether the issue is publicly known or already exploited

You will receive an acknowledgment within **48 hours**, and a status update
within one week. Reports will be handled with discretion; we do not name
reporters without consent.

## Disclosure Policy

We follow a **coordinated disclosure** process:

1. Acknowledgment and triage within 48 hours.
2. Fix developed and validated, target timeline agreed with the reporter.
3. Fix shipped in the next release.
4. Public disclosure after the fix is released and users have had a chance
   to upgrade.

We will credit reporters (unless they prefer to stay anonymous) in the
release notes of the fixing version.

## Contact

- Private vulnerability reporting:
  https://github.com/uttampaliwal/stomar/security
- Maintainer: Uttam Paliwal `<uttam232002@gmail.com>`
- General issues: please open a regular issue per
  [CONTRIBUTING.md](CONTRIBUTING.md)

## Scope

The following are in scope for this policy:

- `api/` — HTTP API (auth, rate limiting, pipeline/order state)
- `services/` and `src/` — trading logic, artifact loading
  (`src/models/artifacts.py`), risk guard, kill switch
- `web/` — frontend session handling and API client
- Deployment guidance in `Dockerfile`, `docker-compose.yml`, and `.env.example`

Out of scope: third-party dependencies (report those to their own projects),
or environments the user has modified from the documented defaults.