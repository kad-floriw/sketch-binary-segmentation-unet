#!/bin/bash
set -e

if [ "$1" = "run" ]; then
  WSGI_MODULE=wsgi
  WORKER_CLASS=gevent
  WORKER_CONNECTIONS=10
  MAX_REQUESTS_JITTER=10

  python get_weights.py

  exec gunicorn ${WSGI_MODULE}:app \
    --workers "$WORKERS" \
    --timeout "$TIMEOUT" \
    --bind=0.0.0.0:"$PORT" \
    --worker-class "$WORKER_CLASS" \
    --max-requests "$MAX_REQUESTS" \
    --worker-connections "$WORKER_CONNECTIONS" \
    --max-requests-jitter "$MAX_REQUESTS_JITTER"
else
  exec "$@"
fi
