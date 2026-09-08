# SignTrustMap AI inference engine

This service is the synchronous inference boundary used by the SignTrustMap Celery worker. Celery
owns queueing, retries, submission state, and database writes. This package downloads existing
submission media from MinIO, runs detector/tracker/classifier inference, writes candidate image
artifacts back to MinIO, and returns JSON results.

## Submission inference

`POST /api/v1/inference/submissions/{submission_id}` requires `X-Internal-Token` and a JSON body:

```json
{
  "media_objects": [
    {
      "media_type": "VIDEO",
      "bucket": "stm-raw-videos",
      "object_key": "uploads/user/submission/session/drive.mp4"
    },
    {
      "media_type": "GPX",
      "bucket": "stm-gpx-logs",
      "object_key": "uploads/user/submission/session/drive.gpx"
    }
  ]
}
```

The call is intentionally synchronous. Do not put another background-task or callback layer around
it; retry and delivery semantics belong to the calling Celery task. The former multipart image and
video inference endpoints have been retired.
