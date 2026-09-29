#!/usr/bin/env bash
set -e
push_with_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    echo
    echo "Push attempt $attempt..."
    git fetch origin main
    if git rebase origin/main; then
      if git push origin HEAD:main; then
        echo "Push lyckades."
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
  return 1
}
push_with_retry
