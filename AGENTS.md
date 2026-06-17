# AGENTS.md

This file is for coding agents. It is laid out as organization wide rules followed by repo-specific information.

Current repo: tandemn-labs/tandemn-store

## Organization guide

### Overall coding style
- Avoid clever one-liners that hurt readability.
- Use comments only for non-obvious operational logic, failure modes, or cross-service contracts. Do not comment what the code already says.
- Follow the existing local patterns before inventing a new one.
- Simplicity first. No features beyond what was asked. No abstractions for single-use code. No "flexibility" or "configurability" that wasn't requested. No error handling for impossible scenarios. Do not add unnecessary complexity in order to attain goals like scalability and security.
- Make only surgical changes. Touch only what is needed, don't improve or refractor anything that is not absolutely necessary.
- Work backwards; Define the GOAL first (success criteria) then ASK QUESTIONS till verified. Your goal is to transform the goal into sub-tasks and verifiable goals. For multi-step taks, state a brief plan.

### Python rules
- Use PEP 8 as code style guide and PEP 257 as docstrings style guide.
- Ensure `pyproject.toml` exists with `ruff`, `mypy` rules
- Use the `./src/` layout for code
- Use `uv` for virtual environment
- Use the python stdlib `logging` library instead of `print()` in the codebase

### Testing Philosophy
- Integration tests should use local containers; never real cloud accounts.

### Repository Boundaries
- Do not commit credentials, .env files, generated caches, local Docker volumes, or large artifacts.

### YAML rules
- Use `.yaml` for new files.

### Other rules
- Ensure `.pre-commit-config.yaml` exists


## Repo-specific guide

This repository implements koi. TODO - add more stuff
