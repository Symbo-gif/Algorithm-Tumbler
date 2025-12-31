# Security Guidelines for Algorithm Tumbler

## Overview
Algorithm Tumbler is a Python-based DSL for algorithm manipulation. This document outlines security best practices for contributors and users to ensure the system remains secure.

## Secure Coding Practices
- **Avoid Dangerous Functions:** Do not use `eval()`, `exec()`, `pickle.loads()`, `os.system()`, `subprocess.call()`, or `input()` for user input. All code execution is handled through the controlled DSL interpreter.
- **Input Validation:** Always validate inputs using argparse for CLI and JSON schema validation for serialization.
- **Serialization:** Use JSON for data persistence. Avoid pickle to prevent deserialization attacks.
- **Error Handling:** Do not expose sensitive information in error messages. Use generic error responses.
- **Dependencies:** Regularly update dependencies and scan for vulnerabilities using tools like Bandit.

## Testing and Auditing
- Run the full test suite before committing changes.
- Perform security scans with Bandit on all Python code.
- Test edge cases, including malformed inputs and boundary conditions.
- Conduct regular security audits, especially after major changes.

## Reporting Vulnerabilities
If you discover a security vulnerability, please report it privately to the maintainers. Do not disclose publicly until a fix is available.

## CI/CD Security
- Automated security scans are run on every push and pull request via GitHub Actions.
- Bandit is used for static security analysis.
- Ensure all CI jobs pass before merging.

## Best Practices for Contributors
- Follow the principle of least privilege.
- Write unit tests for all new code.
- Review code changes for security implications.
- Use type hints and docstrings for clarity and maintainability.
