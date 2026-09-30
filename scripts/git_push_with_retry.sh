#!/usr/bin/env bash
set -e

push_with_retry() {
  local attempt
  local stashed=0

  # Rebase requires a clean worktree. Other generated changes may
  # intentionally remain outside the commit being pushed.
  if [ -n "$(git status --porcelain)" ]; then
    echo
    echo "Arbetskopian innehåller ändringar som inte hör till denna push."
    echo "Stashar dem tillfälligt före rebase..."
    git stash push -u -m "blankdiss-push-with-retry"
    stashed=1
  fi

  restore_stash() {
    if [ "$stashed" -eq 1 ]; then
      echo
      echo "Återställer tillfälligt stashade ändringar..."
      git stash pop
      stashed=0
    fi
  }

  for attempt in 1 2 3 4 5; do
    echo
    echo "Push attempt $attempt..."
    git fetch origin main

    if git rebase origin/main; then
      if git push origin HEAD:main; then
        echo "Push lyckades."
        restore_stash
        return 0
      fi
    else
      echo "Rebase misslyckades."
      git rebase --abort || true
    fi

    if [ "$attempt" -lt 5 ]; then
      echo "Remote ändrades, försöker igen..."
      sleep 5
    fi
  done

  echo "Kunde inte pusha efter 5 försök."
  restore_stash
  return 1
}

push_with_retry
