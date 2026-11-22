# Webshart agent instructions

## Worktree management

- At task start, run `git status --short --branch` and `git worktree list`.
  Use an existing checkout for reviews, CI checks, and other read-only work.
- Reuse one worktree for each active change, including follow-up edits, tests,
  and preparing a PR. Create a worktree when there is no suitable checkout or
  when unrelated work or a different base needs isolation. Fetch the target
  remote first, then branch from the current target outside the main checkout.
- Verify the path and branch before editing or running commands that write
  files. Keep task work in its worktree unless the user asks to update a
  specific checkout. Preserve pre-existing edits and untracked files.
- Clean up finished worktrees at handoff. Fetch the target and verify that a
  merged PR includes the work, including squash and rebase merges. Keep the
  worktree while local edits or tests are pending. A fully pushed PR can keep
  its branch after the checkout is removed. For completed work without a
  merge, confirm that unique commits remain on a local branch.
- Before removal, check staged, unstaged, untracked, and ignored files and
  confirm no process is using the checkout. Preserve any task artifacts needed
  later in a private archive outside the repository, verify the archive, and
  record its location. Leave unrelated files in place.
- Remove completed worktrees with `git worktree remove` without force. If files
  or an active process prevent safe removal, keep the worktree and report why.
  Preview stale records with `git worktree prune --dry-run --verbose` and prune
  once you confirm the checkouts are gone. Delete a local branch only after
  confirming its work is merged or it has no unique commits. Delete remote
  branches only with user authorization.
- At handoff, recheck `git worktree list` and the status of the main checkout and
  any remaining task worktrees. Report retained task worktrees and their reason.

## Public text

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
