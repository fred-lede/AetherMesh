from __future__ import annotations

import os
import tempfile

os.environ["AIIH_ADMIN_EMAIL"] = "admin@aethermesh.test"
os.environ["AIIH_ADMIN_PASSWORD"] = "AdminPass123!"
os.environ["AIIH_JWT_SECRET"] = "test-jwt-secret-not-for-production"

_db_path = tempfile.mktemp(suffix=".db")
os.environ["AIIH_DB_PATH"] = _db_path

import pytest
from fastapi.testclient import TestClient

from config.settings import settings
settings.dashboard_auth_enabled = True
settings.dashboard_auth_username = "testadmin"
settings.dashboard_auth_password = "testpass123"

from dashboard.dashboard_server import app, DASHBOARD_SESSION_COOKIE
from runtime.security.auth.password import hash_password, verify_password
from runtime.security.auth.jwt import create_access_token, decode_token
from runtime.security.auth.api_key import generate_api_key, hash_api_key
from runtime.security.database import SessionLocal
from runtime.security.models import User


DB_USER_EMAIL = "admin@aethermesh.test"
DB_USER_PASS = "AdminPass123!"
SETTINGS_USER = "testadmin"
SETTINGS_PASS = "testpass123"


class TestPassword:
    def test_hash_and_verify(self):
        h = hash_password("hello123")
        assert verify_password("hello123", h) is True

    def test_verify_wrong_password(self):
        h = hash_password("hello123")
        assert verify_password("wrong", h) is False

    def test_verify_malformed_hash(self):
        assert verify_password("x", "not-a-valid-format") is False

    def test_verify_empty_stored(self):
        assert verify_password("x", "") is False

    def test_empty_password(self):
        h = hash_password("")
        assert verify_password("", h) is True


class TestJWT:
    def test_create_and_decode(self):
        token = create_access_token(42, "admin")
        decoded = decode_token(token)
        assert decoded is not None
        assert decoded["sub"] == "42"
        assert decoded["role"] == "admin"
        assert decoded["type"] == "access"

    def test_decode_invalid(self):
        assert decode_token("invalid-token") is None

    def test_decode_empty(self):
        assert decode_token("") is None

    def test_decode_garbage(self):
        assert decode_token("not.a.token") is None

    def test_different_roles(self):
        admin = decode_token(create_access_token(1, "admin"))
        user = decode_token(create_access_token(2, "user"))
        assert admin["role"] == "admin"
        assert user["role"] == "user"


class TestApiKeyGeneration:
    def test_generate_format(self):
        raw, prefix, key_hash = generate_api_key()
        assert raw.startswith("ak_aiih_")
        assert len(prefix) == 20
        assert len(key_hash) == 64

    def test_hash_consistency(self):
        raw = "ak_aiih_test123"
        assert hash_api_key(raw) == hash_api_key(raw)

    def test_generate_unique(self):
        _, _, h1 = generate_api_key()
        _, _, h2 = generate_api_key()
        assert h1 != h2

    def test_prefix_integrity(self):
        raw, prefix, _ = generate_api_key()
        assert raw.startswith(prefix)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _admin_session(client) -> None:
    client.cookies.clear()
    resp = client.post("/login", data={"username": SETTINGS_USER, "password": SETTINGS_PASS}, follow_redirects=False)
    assert resp.status_code == 303


def _create_user(client, email: str, password: str, role: str = "user", display_name: str | None = None) -> dict:
    _admin_session(client)
    body = {"email": email, "password": password, "role": role}
    if display_name:
        body["display_name"] = display_name
    resp = client.post("/api/users", json=body)
    assert resp.status_code == 200, f"Failed to create user {email}: {resp.text}"
    return resp.json()


def _login_as(client, username: str, password: str) -> str:
    client.cookies.clear()
    resp = client.post("/login", data={"username": username, "password": password}, follow_redirects=False)
    assert resp.status_code == 303, f"Login failed for {username}: {resp.status_code}"
    return resp.cookies[DASHBOARD_SESSION_COOKIE]


class TestDashboardAuth:
    """Integration tests for dashboard auth routes."""

    def test_login_page_returns_html(self, client):
        resp = client.get("/login")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_login_page_unauthenticated_index_redirects(self, client):
        client.cookies.clear()
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/login"

    def test_form_login_settings_credentials(self, client):
        resp = client.post("/login", data={"username": SETTINGS_USER, "password": SETTINGS_PASS}, follow_redirects=False)
        assert resp.status_code == 303
        assert resp.headers.get("location") == "/"
        assert DASHBOARD_SESSION_COOKIE in resp.cookies

    def test_form_login_invalid_credentials(self, client):
        resp = client.post("/login", data={"username": SETTINGS_USER, "password": "wrong"}, follow_redirects=False)
        assert resp.status_code == 303
        assert "error=invalid_credentials" in resp.headers.get("location", "")

    def test_form_login_empty_fields(self, client):
        resp = client.post("/login", data={"username": "", "password": ""}, follow_redirects=False)
        assert resp.status_code == 303
        assert "error=invalid_credentials" in resp.headers.get("location", "")

    def test_jwt_login_created_user(self, client):
        _create_user(client, "jwtuser@test.local", "JwtUser123!")
        resp = client.post("/api/auth/login", json={"email": "jwtuser@test.local", "password": "JwtUser123!"})
        assert resp.status_code == 200
        data = resp.json()
        assert "token" in data
        assert data["user"]["role"] == "user"

    def test_jwt_login_bootstrap_admin_force_change(self, client):
        resp = client.post("/api/auth/login", json={"email": DB_USER_EMAIL, "password": DB_USER_PASS})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("force_password_change") is True

    def test_jwt_login_invalid_password(self, client):
        _create_user(client, "invalidpw@test.local", "ValidPass1!")
        resp = client.post("/api/auth/login", json={"email": "invalidpw@test.local", "password": "wrong"})
        assert resp.status_code == 401

    def test_jwt_login_nonexistent_email(self, client):
        resp = client.post("/api/auth/login", json={"email": "nobody@test.local", "password": "pass123"})
        assert resp.status_code == 401

    def test_jwt_login_missing_fields(self, client):
        resp = client.post("/api/auth/login", json={})
        assert resp.status_code == 400

    def test_jwt_login_with_display_name(self, client):
        _create_user(client, "displayname@test.local", "Pass1234!", role="user", display_name="TestDisplay")
        resp = client.post("/api/auth/login", json={"email": "displayname@test.local", "password": "Pass1234!"})
        assert resp.status_code == 200
        assert resp.json()["user"]["display_name"] == "TestDisplay"

    def test_jwt_force_change_password_flow(self, client):
        email = "forcechange2@test.local"
        _create_user(client, email, "OldPass123!")
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == email).first()
            u.must_change_password = True
            db.commit()
        finally:
            db.close()

        resp = client.post("/api/auth/login", json={"email": email, "password": "OldPass123!"})
        assert resp.status_code == 200
        assert resp.json().get("force_password_change") is True

        resp = client.post("/api/auth/change-password", json={"email": email, "old_password": "OldPass123!", "new_password": "NewPass456!"})
        assert resp.status_code == 200
        assert "token" in resp.json()

        resp = client.post("/api/auth/login", json={"email": email, "password": "NewPass456!"})
        assert resp.status_code == 200
        assert "token" in resp.json()
        assert resp.json()["user"]["role"] == "user"

        resp = client.post("/api/auth/login", json={"email": email, "password": "OldPass123!"})
        assert resp.status_code == 401

    def test_form_login_must_change_password_redirect(self, client):
        email = "mustchange@test.local"
        _create_user(client, email, "ChangeMe123!")
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.email == email).first()
            u.must_change_password = True
            db.commit()
        finally:
            db.close()

        login_resp = client.post("/login", data={"username": email, "password": "ChangeMe123!"}, follow_redirects=False)
        assert login_resp.status_code == 303
        assert "/change-password" in login_resp.headers.get("location", "")
        assert DASHBOARD_SESSION_COOKIE in login_resp.cookies

    def test_admin_list_users(self, client):
        _admin_session(client)
        resp = client.get("/api/users")
        assert resp.status_code == 200
        users = resp.json()
        assert isinstance(users, list)
        assert any(u["email"] == DB_USER_EMAIL for u in users)

    def test_list_users_without_session(self, client):
        client.cookies.clear()
        resp = client.get("/api/users")
        assert resp.status_code == 401

    def test_list_users_denied_for_regular_user(self, client):
        _create_user(client, "regularuser@test.local", "UserPass123!")
        _login_as(client, "regularuser@test.local", "UserPass123!")
        resp = client.get("/api/users")
        assert resp.status_code == 403

    def test_create_user_by_admin(self, client):
        _admin_session(client)
        resp = client.post("/api/users", json={"email": "newuser@test.local", "password": "NewUser123!", "display_name": "New User", "role": "user"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "newuser@test.local"
        assert data["role"] == "user"
        assert data["is_active"] is True

    def test_create_user_by_regular_user_denied(self, client):
        _create_user(client, "notadmin@test.local", "Pass1234!")
        _login_as(client, "notadmin@test.local", "Pass1234!")
        resp = client.post("/api/users", json={"email": "shouldfail@test.local", "password": "Pass1234!"})
        assert resp.status_code == 403

    def test_create_user_duplicate_email(self, client):
        _admin_session(client)
        resp = client.post("/api/users", json={"email": "duplicate@test.local", "password": "Pass1234!"})
        assert resp.status_code == 200
        resp2 = client.post("/api/users", json={"email": "duplicate@test.local", "password": "Pass1234!"})
        assert resp2.status_code == 409

    def test_create_user_invalid_role(self, client):
        _admin_session(client)
        resp = client.post("/api/users", json={"email": "badrole@test.local", "password": "Pass1234!", "role": "superadmin"})
        assert resp.status_code == 400

    def test_patch_user_by_admin(self, client):
        user = _create_user(client, "patchme@test.local", "PatchMe123!")
        _admin_session(client)
        resp = client.patch(f"/api/users/{user['id']}", json={"display_name": "Patched Name", "role": "admin"})
        assert resp.status_code == 200
        assert resp.json()["display_name"] == "Patched Name"
        assert resp.json()["role"] == "admin"

    def test_patch_user_by_regular_user_denied(self, client):
        user = _create_user(client, "cantpatch@test.local", "Pass1234!")
        _create_user(client, "otheruser@test.local", "Pass1234!")
        _login_as(client, "otheruser@test.local", "Pass1234!")
        resp = client.patch(f"/api/users/{user['id']}", json={"display_name": "Hacked"})
        assert resp.status_code == 403

    def test_patch_user_not_found(self, client):
        _admin_session(client)
        resp = client.patch("/api/users/99999", json={"display_name": "Nope"})
        assert resp.status_code == 404

    def test_delete_user_by_admin(self, client):
        user = _create_user(client, "deleteme@test.local", "DeleteMe123!")
        _admin_session(client)
        resp = client.delete(f"/api/users/{user['id']}")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_delete_user_by_regular_user_denied(self, client):
        user = _create_user(client, "cantdelete@test.local", "Pass1234!")
        _create_user(client, "regularuser2@test.local", "Pass1234!")
        _login_as(client, "regularuser2@test.local", "Pass1234!")
        resp = client.delete(f"/api/users/{user['id']}")
        assert resp.status_code == 403

    def test_delete_user_not_found(self, client):
        _admin_session(client)
        resp = client.delete("/api/users/99999")
        assert resp.status_code == 404

    def test_admin_api_keys_routes_denied_for_user(self, client):
        _create_user(client, "apikeyuser@test.local", "Pass1234!")
        _login_as(client, "apikeyuser@test.local", "Pass1234!")
        resp = client.get("/api/security/api-keys")
        assert resp.status_code == 403

    def test_self_service_get_me(self, client):
        _create_user(client, "meuser@test.local", "MePass123!")
        _login_as(client, "meuser@test.local", "MePass123!")
        resp = client.get("/api/auth/me")
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == "meuser@test.local"
        assert data["role"] == "user"

    def test_self_service_get_me_no_session(self, client):
        client.cookies.clear()
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401

    def test_self_service_get_me_invalid_session(self, client):
        client.cookies.clear()
        client.cookies.set(DASHBOARD_SESSION_COOKIE, "definitely-not-a-valid-token")
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401

    def test_self_service_change_password(self, client):
        _create_user(client, "changepw@test.local", "OldPass123!")
        _login_as(client, "changepw@test.local", "OldPass123!")
        resp = client.post("/api/auth/me/change-password", json={"old_password": "OldPass123!", "new_password": "NewPass456!"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        old_login = client.post("/api/auth/login", json={"email": "changepw@test.local", "password": "OldPass123!"})
        assert old_login.status_code == 401

        new_login = client.post("/api/auth/login", json={"email": "changepw@test.local", "password": "NewPass456!"})
        assert new_login.status_code == 200

    def test_self_service_change_password_wrong_old(self, client):
        _create_user(client, "wrongold@test.local", "CorrectPass1!")
        _login_as(client, "wrongold@test.local", "CorrectPass1!")
        resp = client.post("/api/auth/me/change-password", json={"old_password": "WrongPass1!", "new_password": "NewPass123!"})
        assert resp.status_code == 401

    def test_self_service_change_password_no_session(self, client):
        client.cookies.clear()
        resp = client.post("/api/auth/me/change-password", json={"old_password": "x", "new_password": "y"})
        assert resp.status_code == 401

    def test_self_service_change_password_short(self, client):
        _create_user(client, "shortpw@test.local", "LongEnough1!")
        _login_as(client, "shortpw@test.local", "LongEnough1!")
        resp = client.post("/api/auth/me/change-password", json={"old_password": "LongEnough1!", "new_password": "short"})
        assert resp.status_code == 400

    def test_self_service_list_api_keys(self, client):
        _create_user(client, "keylist@test.local", "KeyList123!")
        _login_as(client, "keylist@test.local", "KeyList123!")
        resp = client.get("/api/auth/me/api-keys")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_self_service_create_api_key(self, client):
        _create_user(client, "keycreate@test.local", "KeyCreate1!")
        _login_as(client, "keycreate@test.local", "KeyCreate1!")
        resp = client.post("/api/auth/me/api-keys", json={"name": "My Test Key"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "My Test Key"
        assert data["is_active"] is True
        assert "raw_key" in data
        assert data["raw_key"].startswith("ak_aiih_")

    def test_self_service_create_api_key_no_session(self, client):
        client.cookies.clear()
        resp = client.post("/api/auth/me/api-keys", json={"name": "test"})
        assert resp.status_code == 401

    def test_self_service_revoke_api_key(self, client):
        _create_user(client, "keyrevoke@test.local", "KeyRevoke1!")
        _login_as(client, "keyrevoke@test.local", "KeyRevoke1!")
        create = client.post("/api/auth/me/api-keys", json={"name": "To Revoke"})
        key_id = create.json()["id"]
        resp = client.delete(f"/api/auth/me/api-keys/{key_id}")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        list_resp = client.get("/api/auth/me/api-keys")
        keys = list_resp.json()
        assert not any(k["id"] == key_id and k["is_active"] for k in keys)

    def test_self_service_revoke_nonexistent_key(self, client):
        _create_user(client, "keybadrev@test.local", "KeyBadRev1!")
        _login_as(client, "keybadrev@test.local", "KeyBadRev1!")
        resp = client.delete("/api/auth/me/api-keys/99999")
        assert resp.status_code == 404

    def test_self_service_cannot_revoke_other_users_key(self, client):
        user1_email = "keyuser1x@test.local"
        user2_email = "keyuser2x@test.local"
        _create_user(client, user1_email, "KeyUser1A!")
        _create_user(client, user2_email, "KeyUser2A!")
        _login_as(client, user1_email, "KeyUser1A!")
        create = client.post("/api/auth/me/api-keys", json={"name": "User1 Key"})
        assert create.status_code == 200
        key_id = create.json()["id"]

        _login_as(client, user2_email, "KeyUser2A!")
        resp = client.delete(f"/api/auth/me/api-keys/{key_id}")
        assert resp.status_code == 404

    def test_db_user_deleted_still_has_session(self, client):
        user = _create_user(client, "deletedsession@test.local", "Deleted1!")
        saved_session = _login_as(client, "deletedsession@test.local", "Deleted1!")

        resp = client.get("/api/auth/me")
        assert resp.status_code == 200

        _admin_session(client)
        client.delete(f"/api/users/{user['id']}")

        client.cookies.clear()
        client.cookies.set(DASHBOARD_SESSION_COOKIE, saved_session, domain="testserver.local")
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401

    def test_session_still_valid_after_rotate(self, client):
        client.cookies.clear()
        resp1 = client.post("/login", data={"username": SETTINGS_USER, "password": SETTINGS_PASS}, follow_redirects=False)
        s1 = resp1.cookies[DASHBOARD_SESSION_COOKIE]
        client.cookies.clear()
        resp2 = client.post("/login", data={"username": SETTINGS_USER, "password": SETTINGS_PASS}, follow_redirects=False)
        s2 = resp2.cookies[DASHBOARD_SESSION_COOKIE]
        assert s1 != s2

        client.cookies.clear()
        client.cookies.set(DASHBOARD_SESSION_COOKIE, s1, domain="testserver.local")
        resp = client.get("/api/users")
        assert resp.status_code == 200

    def test_form_login_basic_auth_fallback(self, client):
        import base64
        creds = base64.b64encode(f"{SETTINGS_USER}:{SETTINGS_PASS}".encode()).decode()
        client.cookies.clear()
        resp = client.get("/api/health", headers={"Authorization": f"Basic {creds}"})
        assert resp.status_code == 200

    def test_unauthenticated_api_access(self, client):
        client.cookies.clear()
        resp = client.get("/api/overview")
        assert resp.status_code == 401

    def test_empty_post_login(self, client):
        resp = client.post("/login", data={}, follow_redirects=False)
        assert resp.status_code == 303
        assert "error=invalid_credentials" in resp.headers.get("location", "")


def test_cleanup():
    if os.path.exists(_db_path):
        os.unlink(_db_path)
