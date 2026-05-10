#!/usr/bin/env bash
# Generate self-signed CA + server certificate for AIIH TLS.
# Usage:  ./scripts/gen-certs.sh [output-dir]
# Default output:  ./certs/  (creates server.crt, server.key, ca.crt)

set -euo pipefail

OUT="${1:-certs}"
mkdir -p "$OUT"

DAYS="${AIIH_CERT_DAYS:-3650}"
CN="${AIIH_CERT_CN:-AetherMesh.local}"
SAN="${AIIH_CERT_SAN:-DNS:localhost,IP:127.0.0.1,IP:192.168.1.200,IP:192.168.1.123,DNS:*.local}"

echo "==> Generating CA key + cert ..."
openssl genrsa -out "$OUT/ca.key" 2048 2>/dev/null
openssl req -x509 -new -nodes -key "$OUT/ca.key" \
  -sha256 -days "$DAYS" \
  -subj "/CN=AIIH CA/O=AetherMesh/C=TW" \
  -out "$OUT/ca.crt"

echo "==> Generating server key + CSR ..."
openssl genrsa -out "$OUT/server.key" 2048 2>/dev/null
openssl req -new -key "$OUT/server.key" \
  -subj "/CN=$CN/O=AetherMesh/C=TW" \
  -out "$OUT/server.csr"

echo "==> Signing server cert with CA ..."
cat > "$OUT/san.cnf" <<EOF
[req]
distinguished_name = dn
x509_extensions = v3_req
prompt = no

[dn]
CN = $CN

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = $SAN
EOF

openssl x509 -req -in "$OUT/server.csr" \
  -CA "$OUT/ca.crt" -CAkey "$OUT/ca.key" -CAcreateserial \
  -sha256 -days "$DAYS" \
  -extfile "$OUT/san.cnf" -extensions v3_req \
  -out "$OUT/server.crt"

rm -f "$OUT/server.csr" "$OUT/san.cnf" "$OUT/ca.srl"

echo "==> Files created in $OUT/:"
ls -1 "$OUT/"*.crt "$OUT/"*.key

echo ""
echo "Next steps:"
echo "  1. Copy $OUT/server.crt and $OUT/server.key to your AIIH config dir"
echo "  2. In .env set:"
echo "       AIIH_SSL_CERTFILE=$OUT/server.crt"
echo "       AIIH_SSL_KEYFILE=$OUT/server.key"
echo "  3. Update internal URLs to https://"
echo "       AIIH_CONTROL_URL=https://127.0.0.1:9200"
echo "       AIIH_ROUTER_URL=https://127.0.0.1:8001"
echo "       AIIH_METRICS_URL=https://127.0.0.1:9100"
echo "  4. (Optional) Distribute $OUT/ca.crt to client machines so they"
echo "     trust the self-signed CA"
