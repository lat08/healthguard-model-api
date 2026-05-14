"""Internal service authentication dependency (ADR-005 / D-013).

Grace period logic:
- HEALTHGUARD_INTERNAL_SECRET env unset -> log error + accept (grace).
- Header X-Internal-Secret missing -> log warning + accept (grace).
- Header X-Internal-Secret wrong value -> 401 Unauthorized (hard reject).

After grace period removal (Phase 5+), missing header will also 401.
"""

from __future__ import annotations

import hmac
import logging
from typing import Optional

from fastapi import Header, HTTPException, status

from app.config import settings

logger = logging.getLogger(__name__)

_GRACE_LOGGED = False


def verify_internal_secret(
    x_internal_secret: Optional[str] = Header(None),
) -> None:
    """Verify X-Internal-Secret header per ADR-005.

    Grace period: accept when secret not configured or header missing.
    Hard reject: header present but value mismatch.
    """
    global _GRACE_LOGGED

    configured_secret = settings.internal_secret

    # Grace: secret not configured in env.
    if not configured_secret:
        if not _GRACE_LOGGED:
            logger.error(
                "HEALTHGUARD_INTERNAL_SECRET is not set. "
                "All predict endpoints are unprotected (grace period). "
                "Set HEALTHGUARD_INTERNAL_SECRET in production."
            )
            _GRACE_LOGGED = True
        return

    # Grace: header not sent by caller (caller not yet updated).
    if x_internal_secret is None:
        logger.warning(
            "X-Internal-Secret header missing — accepting under grace period. "
            "Caller should add header."
        )
        return

    # Hard reject: header present but wrong value.
    if not hmac.compare_digest(x_internal_secret, configured_secret):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal service secret.",
        )
