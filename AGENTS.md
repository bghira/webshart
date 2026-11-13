# Public text

Use repository-relative references and generic examples in publication payloads.
Run the client-side check on the exact staged changes, commit message, and final
publication text before committing, pushing, or publishing:

```bash
python scripts/check_public_text.py --staged --text-file commit-message.txt
python scripts/check_public_text.py --git-range origin/main..HEAD --text-file pr-title.txt --text-file pr-body.md
```

Apply the same check to subsequent edits and other publication payloads. Keep
machine-specific configuration local. Local policy can extend the check through
`WEBSHART_PRIVATE_TERMS`, with one value per line.

Publish only after the check passes. If it cannot run, stop and request guidance.
When content is blocked, report only:
`Blocked: local machine identity was found in public text.`
Never echo the matching content. A server-side check does not replace this gate.
