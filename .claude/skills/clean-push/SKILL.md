---
name: clean-push
description: Run tests, lint, and formatting, then confirm branch and commit details before pushing to GitHub. Accepts an optional --pr flag to also open and merge a PR into main. Use when the user asks to "clean push", "push cleanly", or wants a verified push of the probably package.
---

# clean-push

Push changes to GitHub only after the codebase is verified clean and the user has
confirmed exactly what's being pushed where.

All commands run inside the `probably` conda environment (`conda activate probably`).

## Arguments

- `--pr` (optional, **default false**): when passed, after the push also open a PR
  into `main` and merge it. When absent, the skill only pushes the current branch —
  no PR is opened.

## Steps

1. **Run the checks, in order, and stop on the first failure:**
   - `pytest`
   - `ruff check .`
   - `black --check .`
     - If `black --check` reports files needing reformatting, run `black .` to fix
       them, then re-stage those files. Do not silently reformat and push without
       telling the user which files changed.

   If any check fails (tests, ruff, or an unfixable black issue), stop and report
   the failure — do not proceed to commit or push.

2. **Gather the git state:**
   - Current branch (`git branch --show-current`)
   - Upstream/target branch it pushes to (`git rev-parse --abbrev-ref --symbolic-full-name @{u}` if it exists, otherwise ask)
   - `git status` and `git diff` (staged + unstaged) to see what will be committed
   - A drafted commit message based on the actual changes

3. **Confirm with the user before pushing.** Explicitly show and ask them to confirm:
   - The branch commits are being made on, and the branch/remote being pushed to
   - The exact commit message
   - That checks passed (or what was auto-fixed by black)

   Do not push without an explicit go-ahead in this step, even if earlier steps all
   passed cleanly.

4. **Commit and push** only after confirmation, following the repo's standard
   commit-message attribution rules.

5. **If `--pr` was passed:** after the push succeeds, open a PR from the current
   branch into `main` (`gh pr create`), using the confirmed commit message (or a
   short summary of it) as the PR title/body. Before merging:
   - Show the user the PR URL, title, and body.
   - Explicitly confirm they want it merged into `main` now — this is a separate
     confirmation from step 3's push confirmation, since merging to `main` is a
     more consequential, harder-to-reverse action.
   - Only after that confirmation, merge it (`gh pr merge`). Do not use
     `--admin`/force-merge past failing required checks.
   - If the current branch already *is* `main`, skip PR creation entirely — there's
     nothing to open a PR against — and tell the user so.

## Notes

- Never use `--no-verify`, `--force`, or skip any of the checks above to "get past"
  a failure — fix the underlying issue or stop and report it.
- If the working tree has unrelated uncommitted changes the user didn't ask about,
  flag them rather than silently including or discarding them.
