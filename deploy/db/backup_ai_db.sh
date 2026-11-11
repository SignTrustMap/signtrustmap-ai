#!/bin/bash
set -euo pipefail

BACKUP_DIR="/media/taiduc_orico_ssd/DucPT/signtrustmap/signtrustmap-ai/working/backups"
mkdir -p "$BACKUP_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/signtrustmap_ai_${TIMESTAMP}.sql.gz"

echo "[AI DB Backup] Starting backup of signtrustmap_ai..."
docker exec stm-ai-db pg_dump -U stm_ai_user -d signtrustmap_ai | gzip > "$BACKUP_FILE"
echo "[AI DB Backup] Backup completed successfully: $BACKUP_FILE ($(du -h "$BACKUP_FILE" | cut -f1))"

# Keep last 7 backups
ls -dt "${BACKUP_DIR}"/signtrustmap_ai_*.sql.gz | tail -n +8 | xargs -r rm -f