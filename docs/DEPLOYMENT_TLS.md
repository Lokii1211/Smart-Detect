# Deployment: TLS termination and network exposure

SmartDetect serves biometric data — face embeddings, photographs of
identifiable people, and live camera video. **Uvicorn must never be exposed
directly.** It speaks plain HTTP: without TLS in front of it, every JWT,
password and video frame crosses the network in clear text, and anyone on
the path can replay a captured token for its full 8-hour life.

---

## 1. Reference topology

```
     internet / LAN
           │
           │  HTTPS 443  (TLS terminates here)
           ▼
   ┌────────────────────┐
   │ nginx / Caddy /    │  ← certificate, HSTS, security headers,
   │ ALB / Cloudflare   │    connection rate limiting
   └─────────┬──────────┘
             │  HTTP, loopback or private subnet only
             ▼
   ┌────────────────────┐
   │ uvicorn 127.0.0.1  │  ← bind to loopback, never 0.0.0.0
   │ backend.main:app   │
   └────────────────────┘
```

**Bind to loopback.** The README's `--port 8000` inherits uvicorn's default
host, which is `127.0.0.1` — keep it that way in production. Never
`--host 0.0.0.0` on a machine with a public interface.

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

---

## 2. nginx

MJPEG needs care: it is one infinitely long response. Buffering it stalls the
stream and eventually exhausts memory.

```nginx
server {
    listen 443 ssl http2;
    server_name smartdetect.example.org;

    ssl_certificate     /etc/letsencrypt/live/smartdetect.example.org/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/smartdetect.example.org/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers off;

    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options    "nosniff" always;
    add_header X-Frame-Options           "DENY"    always;
    add_header Referrer-Policy           "no-referrer" always;   # see §5

    client_max_body_size 500M;      # matches the video-upload cap

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
    }

    # MJPEG: never buffer, never time out mid-stream
    location /camera/stream/ {
        proxy_pass          http://127.0.0.1:8000;
        proxy_http_version  1.1;
        proxy_buffering     off;
        proxy_cache         off;
        proxy_read_timeout  24h;
        proxy_set_header    Connection "";
        proxy_set_header    X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Photographs of identifiable people. Auth is enforced by the app, but
    # caching must be suppressed at the edge too.
    location /snapshots/ {
        proxy_pass       http://127.0.0.1:8000;
        proxy_cache      off;
        add_header       Cache-Control "no-store, private" always;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}

server {                      # redirect plaintext
    listen 80;
    server_name smartdetect.example.org;
    return 301 https://$host$request_uri;
}
```

## 3. Caddy (automatic certificates)

```caddy
smartdetect.example.org {
    encode gzip
    header {
        Strict-Transport-Security "max-age=31536000; includeSubDomains"
        X-Content-Type-Options    "nosniff"
        X-Frame-Options           "DENY"
        Referrer-Policy           "no-referrer"
    }
    request_body { max_size 500MB }

    @stream path /camera/stream/*
    reverse_proxy @stream 127.0.0.1:8000 {
        flush_interval -1          # disable buffering for MJPEG
    }
    reverse_proxy 127.0.0.1:8000
}
```

---

## 4. Application configuration

```bash
python scripts/generate_env.py     # random JWT secret + passwords, mode 0600
```

The server **refuses to start** without `JWT_SECRET`, `ADMIN_PASSWORD` and
`OPERATOR_PASSWORD`, rejects a `JWT_SECRET` under 32 characters, and rejects
any value previously shipped as a default (they are in this repository's git
history).

| Variable | Purpose |
|---|---|
| `JWT_SECRET` | Token signing key. Rotating it invalidates all sessions. |
| `ADMIN_PASSWORD` / `OPERATOR_PASSWORD` | Account credentials. |
| `JWT_EXPIRE_HOURS` | Session lifetime (default 8). |
| `SMARTDETECT_CORS_ORIGINS` | Comma-separated allowed origins. Set to your dashboard's real origin; the default is localhost only. |
| `SMARTDETECT_PUBLIC_DOCS` | `1` exposes `/docs` without auth. **Leave unset in production.** |

---

## 5. Why `Referrer-Policy: no-referrer` matters here

`<img>` tags cannot send an `Authorization` header, so MJPEG streams and
snapshot images authenticate with `?token=` in the URL. That token therefore
appears in:

- browser history
- the `Referer` header of any outbound link
- proxy and web-server access logs

Mitigations already in place: those tokens are scoped `stream` (rejected by
every JSON endpoint) and expire in 60 minutes. `no-referrer` closes the
referrer leak. **Do not log full query strings** at the proxy.

---

## 6. Known gaps

These are real and not yet addressed. Do not deploy against members of the
public without resolving them.

1. **JWTs live in `sessionStorage`**, so any XSS in the dashboard can steal a
   session. The robust fix is an `httpOnly`, `Secure`, `SameSite=Strict`
   cookie plus CSRF tokens — that requires a same-site deployment and was not
   done here.
2. **No token revocation.** JWTs are stateless: a stolen token is valid until
   it expires. Rotating `JWT_SECRET` is the only kill switch and logs everyone
   out. A short `JWT_EXPIRE_HOURS` limits the damage window.
3. **Two shared accounts, no per-user identity.** There is no way to attribute
   a photo search to an individual operator, which a biometric system needs
   for accountability. Replace `_USERS` in `backend/auth.py` with a real user
   table with hashed passwords (argon2/bcrypt).
4. **No audit log of searches.** Nothing records who searched for whom.
5. **In-memory rate limiting.** `_rate_buckets` is per-process, so it resets on
   restart and does not coordinate across replicas. Use the proxy's rate
   limiting (`limit_req` in nginx) as the real control, or move to Redis.
6. **No retention or deletion policy.** Snapshots and embeddings accumulate
   indefinitely. Under GDPR Art. 9 / BIPA / India's DPDP Act this is the
   largest outstanding compliance gap — see `PROJECT_REVIEW.md` §4.2.

---

## 7. Pre-deployment checklist

- [ ] `scripts/generate_env.py` run; `.env` is mode 0600 and not committed
- [ ] uvicorn bound to `127.0.0.1`, not `0.0.0.0`
- [ ] TLS terminating at the proxy; HTTP redirects to HTTPS
- [ ] HSTS and `Referrer-Policy: no-referrer` set
- [ ] `SMARTDETECT_PUBLIC_DOCS` unset
- [ ] `SMARTDETECT_CORS_ORIGINS` set to the real dashboard origin
- [ ] `python scripts/audit_routes.py --check` passes
- [ ] Proxy access logs exclude query strings (stream tokens)
- [ ] `proxy_buffering off` verified on `/camera/stream/`
- [ ] Legal basis for processing biometric data established (§6.6)
