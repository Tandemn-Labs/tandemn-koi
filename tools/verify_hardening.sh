#!/bin/bash
# verify_hardening.sh — check Phase 1-6 invariants after an E2E run.
#
# Exit code: 0 if all invariants held, non-zero otherwise. Safe to use
# in poll loops or as the last step of a scripted scenario.
#
# Env:
#   KOI_URL               default http://localhost:8090
#   ORCA_URL              required
#   ORCA_OUTBOX_DB_PATH   default ./state/outbox.db  (resolved from Orca's cwd)
#   KOI_RUNTIME_DB        default ./data/koi_runtime.db  (from Koi's cwd)
#   KOI_MEMORY_DB         default ./data/koi_memory.db   (from Koi's cwd)

set -u

KOI_URL="${KOI_URL:-http://localhost:8090}"
ORCA_URL="${ORCA_URL:?set ORCA_URL (e.g. https://xxx.trycloudflare.com)}"
OUTBOX_DB="${ORCA_OUTBOX_DB_PATH:-./state/outbox.db}"
KOI_RUNTIME_DB="${KOI_RUNTIME_DB:-./data/koi_runtime.db}"
KOI_MEMORY_DB="${KOI_MEMORY_DB:-./data/koi_memory.db}"

fail=0

bad() { echo "  ✗ $1"; fail=1; }
ok()  { echo "  ✓ $1"; }

# ----------------------------------------------------------------------
echo "=== [1/5] Koi /health ==="
if ! koi_json=$(curl -sf --max-time 5 "$KOI_URL/health"); then
  bad "Koi /health unreachable at $KOI_URL"
else
  status=$(echo "$koi_json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','missing'))")
  if [ "$status" = "ok" ]; then
    ok "Koi status=ok"
  else
    fatal=$(echo "$koi_json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('fatal','n/a'))")
    bad "Koi status=$status fatal=$fatal — a monitor task died"
  fi
  stale=$(echo "$koi_json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('stale_inbox_claims',0))")
  if [ "$stale" = "0" ]; then
    ok "stale_inbox_claims=0"
  else
    bad "stale_inbox_claims=$stale — a handler crashed mid-flight"
  fi
fi

# ----------------------------------------------------------------------
echo "=== [2/5] Orca /health ==="
if ! orca_json=$(curl -sf --max-time 5 "$ORCA_URL/health"); then
  bad "Orca /health unreachable at $ORCA_URL"
else
  enabled=$(echo "$orca_json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('outbox_enabled',False))")
  pending=$(echo "$orca_json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('outbox_pending',-1))")
  age=$(echo "$orca_json" | python3 -c "import sys,json; print(json.load(sys.stdin).get('outbox_oldest_undelivered_age_secs',-1))")
  if [ "$enabled" = "True" ]; then
    ok "outbox_enabled=true"
  else
    bad "outbox_enabled=$enabled — durable delivery is off"
  fi
  if [ "$pending" = "0" ]; then
    ok "outbox_pending=0"
  else
    bad "outbox_pending=$pending (oldest ${age}s) — events stuck"
  fi
fi

# ----------------------------------------------------------------------
echo "=== [3/5] Outbox SQLite ==="
if [ ! -f "$OUTBOX_DB" ]; then
  echo "  (skip) $OUTBOX_DB not found on this host — remote Orca?"
else
  undelivered=$(sqlite3 "$OUTBOX_DB" "SELECT COUNT(*) FROM outbox WHERE delivered_at IS NULL" 2>/dev/null || echo err)
  if [ "$undelivered" = "0" ]; then
    ok "no undelivered rows"
  else
    bad "$undelivered undelivered rows in outbox"
    sqlite3 "$OUTBOX_DB" \
      "SELECT event_id, event_type, attempts, last_status_code, last_error FROM outbox
       WHERE delivered_at IS NULL ORDER BY created_at DESC LIMIT 5"
  fi
fi

# ----------------------------------------------------------------------
echo "=== [4/5] Inbox SQLite ==="
if [ ! -f "$KOI_RUNTIME_DB" ]; then
  echo "  (skip) $KOI_RUNTIME_DB not found"
else
  stuck=$(sqlite3 "$KOI_RUNTIME_DB" "SELECT COUNT(*) FROM inbox WHERE status='processing'" 2>/dev/null || echo err)
  processed=$(sqlite3 "$KOI_RUNTIME_DB" "SELECT COUNT(*) FROM inbox WHERE status='processed'" 2>/dev/null || echo err)
  if [ "$stuck" = "0" ]; then
    ok "no stuck handlers (processed=$processed)"
  else
    bad "$stuck handlers stuck in 'processing'"
  fi
fi

# ----------------------------------------------------------------------
# Only check outcomes written after the hardening was deployed. Pre-hardening
# rows may have legacy duplicates; the unique index added in Phase 2d blocks
# NEW duplicates but doesn't retroactively dedup. Tunable via env.
echo "=== [5/5] Per-chain outcome uniqueness (recent only) ==="
OUTCOME_SINCE_DAYS="${OUTCOME_SINCE_DAYS:-1}"
if [ ! -f "$KOI_MEMORY_DB" ]; then
  echo "  (skip) $KOI_MEMORY_DB not found"
else
  dupes=$(sqlite3 "$KOI_MEMORY_DB" \
    "SELECT COUNT(*) FROM (
       SELECT decision_id, job_id, status, COUNT(*) n FROM outcomes
       WHERE timestamp > datetime('now', '-${OUTCOME_SINCE_DAYS} day')
       GROUP BY decision_id, job_id, status HAVING n > 1
     )" 2>/dev/null || echo err)
  recent=$(sqlite3 "$KOI_MEMORY_DB" \
    "SELECT COUNT(*) FROM outcomes
     WHERE timestamp > datetime('now', '-${OUTCOME_SINCE_DAYS} day')" 2>/dev/null || echo err)
  if [ "$dupes" = "0" ]; then
    ok "no duplicate outcomes in last ${OUTCOME_SINCE_DAYS}d (recent=$recent)"
  else
    bad "$dupes duplicate (decision_id, job_id, status) groups in last ${OUTCOME_SINCE_DAYS}d"
    sqlite3 "$KOI_MEMORY_DB" \
      "SELECT decision_id, job_id, status, COUNT(*) n FROM outcomes
       WHERE timestamp > datetime('now', '-${OUTCOME_SINCE_DAYS} day')
       GROUP BY decision_id, job_id, status HAVING n > 1 LIMIT 5"
  fi
fi

# ----------------------------------------------------------------------
echo
if [ "$fail" = "0" ]; then
  echo "✅ All Phase 1-6 invariants held."
  exit 0
else
  echo "❌ One or more invariants failed — see lines marked ✗ above."
  exit 1
fi
