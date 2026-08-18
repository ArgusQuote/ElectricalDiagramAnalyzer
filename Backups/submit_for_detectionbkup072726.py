import anvil.secrets
import anvil.stripe
import anvil.email
import anvil.users
import anvil.server
import json, time, io, re, traceback
from anvil.tables import app_tables

MAX_PDF_SIZE_MB = 50
MAX_PDF_SIZE_BYTES = MAX_PDF_SIZE_MB * 1024 * 1024
MAX_PDF_PAGES = 50

def _jsonify(x):
  return json.loads(json.dumps(x, default=str))

def _pdf_page_count(data: bytes) -> int:
  """
  Best-effort PDF page count.
  If page count cannot be read in Anvil, do not block upload.
  The VM/pipeline can still fail safely later if the PDF is bad.
  """
  try:
    try:
      from pypdf import PdfReader
    except Exception:
      from PyPDF2 import PdfReader

    reader = PdfReader(io.BytesIO(data))

    if getattr(reader, "is_encrypted", False):
      raise Exception("Encrypted PDFs are not supported.")

    return len(reader.pages)

  except Exception as e:
    if "Encrypted PDFs are not supported" in str(e):
      raise

    return 0

def _is_pdf(media):
  if not media:
    return False

  try:
    name = (getattr(media, "name", "") or "").lower()

    data = media.get_bytes()

    if not data:
      return False

    # Size limit
    if len(data) > MAX_PDF_SIZE_BYTES:
      raise Exception(
        f"PDF exceeds {MAX_PDF_SIZE_MB}MB limit."
      )

    # PDF magic bytes
    if not data.startswith(b"%PDF-"):
      return False

    # Extension sanity check
    if not name.endswith(".pdf"):
      return False

    # Page-count limit, best-effort only.
    # If Anvil cannot read page count, page_count returns 0 and we skip this check.
    page_count = _pdf_page_count(data)

    if page_count > MAX_PDF_PAGES:
      raise Exception(
        f"PDF has {page_count} pages. The current limit is {MAX_PDF_PAGES} pages."
      )

    return True

  except Exception as e:
    msg = str(e)
    if (
      "PDF exceeds" in msg
      or "PDF has" in msg
      or "Encrypted PDFs are not supported" in msg
    ):
      raise
    return False

def _owner_email() -> str:
  u = anvil.users.get_user()
  if not u:
    raise Exception("User not logged in")
  try:
    em = (u['email'] or "").strip().lower()
  except Exception:
    em = ""
  if not em:
    raise Exception("Your account has no email address.")
  return em

def _is_internal_admin_email(email: str) -> bool:
  email = str(email or "").strip().lower()
  if not email:
    return False

  # Supports either:
  # internal_admins.email = "you@email.com"
  # OR internal_admins.user = users row
  try:
    if app_tables.internal_admins.get(email=email):
      return True
  except Exception:
    pass

  try:
    u = anvil.users.get_user()
    if u and app_tables.internal_admins.get(user=u):
      return True
  except Exception:
    pass

  return False

def _require_entitled_owner_email() -> str:
  owner_email = _owner_email()

  # Internal admins may not be code_users.
  # Allow them through before requiring normal customer entitlement.
  try:
    if _is_internal_admin_email(owner_email):
      return owner_email
  except Exception:
    pass

  ent = anvil.server.call("is_current_user_entitled") or {}
  if not ent.get("ok"):
    raise Exception("Your access is not active. Please contact your admin.")

  return owner_email

def _safe_folder_key(value: str, fallback: str = "ungrouped") -> str:
  s = str(value or "").strip().lower()
  s = re.sub(r"[^a-z0-9._-]+", "_", s)
  s = s.strip("._-")
  return s or fallback

def _access_code_exclusion_default(ac) -> bool:
  """
  Group/account-level default exclusion setting.
  If True, every upload under this access code is forced into excluded retention.
  """
  if not ac:
    return False

  try:
    return bool(ac["exclude_from_improvement_default"])
  except Exception:
    return False

def _current_storage_context() -> dict:
  """
  Server-derived storage routing context.
  Never accept group_folder from the browser/client.

  Routing rule:
    1. internal_admins table wins first
    2. normal code_users/access_code next
    3. personal fallback last
  """
  u = anvil.users.get_user()
  if not u:
    raise Exception("User not logged in.")

  owner_email = _require_entitled_owner_email()

  # Internal admins always route to the admin folder.
  # This must happen before code_users lookup.
  if _is_internal_admin_email(owner_email):
    return {
      "owner_email": owner_email,
      "access_code_value": "INTERNAL_ADMIN",
      "plan_key": "internal_admin",
      "company_name": "Argus Internal Admins",
      "group_folder": "internal_admins",
      "exclude_from_improvement_default": False,
    }

  link = app_tables.code_users.get(user=u)
  ac = link["access_code"] if link and link["access_code"] else None

  # Personal fallback for any unusual entitled user without a code row.
  if not ac:
    return {
      "owner_email": owner_email,
      "access_code_value": "",
      "plan_key": "",
      "company_name": "",
      "group_folder": "personal",
      "exclude_from_improvement_default": False,
    }

  group_folder = _safe_folder_key(ac["group_folder"] or "", "")

  if not group_folder:
    raise Exception(
      "This account is missing a storage group folder. Please contact support."
    )

  company_name = ""
  try:
    company_name = (ac["company_name"] or "").strip()
  except Exception:
    company_name = ""

  return {
    "owner_email": owner_email,
    "access_code_value": (ac["code"] or "").strip(),
    "plan_key": (ac["plan_key"] or "").strip().lower(),
    "company_name": company_name,
    "group_folder": group_folder,
    "exclude_from_improvement_default": _access_code_exclusion_default(ac),
  }

@anvil.server.callable(require_user=True)
def get_upload_retention_policy():
  """
  Returns the current user's upload retention policy.
  Used by the Upload page to show/lock the exclusion checkbox when the group requires it.
  Server-side enforcement still happens in submit_for_detection().
  """
  storage_ctx = _current_storage_context()

  group_default = bool(storage_ctx.get("exclude_from_improvement_default", False))

  return _jsonify({
    "ok": True,
    "exclude_from_improvement_default": group_default,
    "retention_mode": "excluded_24h" if group_default else "standard",
    "group_folder": storage_ctx.get("group_folder") or "personal",
    "company_name": storage_ctx.get("company_name") or "",
    "plan_key": storage_ctx.get("plan_key") or "",
  })

@anvil.server.callable(require_user=True)
def submit_for_detection(file, ui_overrides=None, **kwargs):
  storage_ctx = _current_storage_context()
  owner_email = storage_ctx["owner_email"]

  if not _is_pdf(file):
    raise Exception("PDFs only")

  job_note = kwargs.get("job_note")

  user_requested_exclusion = bool(kwargs.get("exclude_from_improvement", False))
  group_default_exclusion = bool(storage_ctx.get("exclude_from_improvement_default", False))

  # Server-side enforcement:
  # If the access code/group requires exclusion, the browser cannot override it.
  exclude_from_improvement = bool(user_requested_exclusion or group_default_exclusion)

  out = anvil.server.call(
    "vm_submit_for_detection",
    media=file,
    ui_overrides=(ui_overrides or {}),
    job_note=job_note,
    owner_email=owner_email,
    group_folder=storage_ctx["group_folder"],
    access_code_value=storage_ctx["access_code_value"],
    plan_key=storage_ctx["plan_key"],
    company_name=storage_ctx["company_name"],
    exclude_from_improvement=exclude_from_improvement
  )

  out = dict(out or {})

  # Expected user-facing block: user already has an active queued/running job.
  # Return this normally instead of raising, so the client can show the real message.
  if out.get("ok") is False:
    state = str(out.get("state") or "").strip().lower()
    if state == "active_job_exists":
      return _jsonify({
        "ok": False,
        "state": "active_job_exists",
        "error": out.get("error") or (
          "You already have a job processing. Please wait for it to finish "
          "or cancel it from My Jobs."
        )
      })

    raise Exception(
      out.get("error") or "Could not submit job."
    )

  out["owner_email"] = owner_email
  out["owner_id"] = owner_email
  out["group_folder"] = storage_ctx["group_folder"]
  out["exclude_from_improvement"] = exclude_from_improvement
  out["exclude_from_improvement_default"] = group_default_exclusion
  return _jsonify(out)

@anvil.server.callable(require_user=True)
def get_job_status(job_id):
  """Resilient poll: auto-retry on Uplink disconnects; generic not_found for missing/stale jobs."""
  storage_ctx = _current_storage_context()
  owner_email = storage_ctx["owner_email"]
  group_folder = storage_ctx["group_folder"]

  job_id = str(job_id or "").strip()

  if not job_id:
    return {
      "state": "not_found",
      "error": "Job not found. Please resubmit your PDF."
    }

  # Security: do not pass path traversal or absolute-ish IDs to the VM.
  if "/" in job_id or "\\" in job_id or ".." in job_id:
    return {
      "state": "not_found",
      "error": "Job not found. Please resubmit your PDF."
    }

  for attempt in range(3):
    try:
      out = anvil.server.call("vm_get_job_status", job_id, owner_email, group_folder) or {}
      out = dict(out)

      # Backward-compatible safety in case the VM still returns the old message.
      err = str(out.get("error") or "")
      if (out.get("state") == "error") and ("Unknown job_id" in err):
        return {
          "state": "not_found",
          "error": "Job not found. Please resubmit your PDF."
        }

      return _jsonify(out)

    except anvil.server.UplinkDisconnectedError:
      if attempt < 2:
        time.sleep(0.75 * (attempt + 1))
        continue
      return {"state": "unknown", "error": "uplink_disconnected"}

    except Exception:
      return {
        "state": "unknown",
        "error": "Status unavailable. Please try again."
      }

@anvil.server.callable(require_user=True)
def rerun_rules_with_panel_edit(
  job_id: str,
  original_panel_name: str,
  edited_component: dict,
  original_source_path: str = None,
  target_email: str = ""
):
  """
  Client-safe wrapper for editing one panel and rerunning the rules engine only.

  Works for:
    - normal logged-in job owner
    - admin inspection mode when target_email is passed

  Returns:
    {"ok": True, "result": ...}
  or:
    {"ok": False, "error": "..."}
  """

  try:
    job_id = str(job_id or "").strip()
    original_panel_name = str(original_panel_name or "").strip()
    original_source_path = str(original_source_path or "").strip()
    target_email = str(target_email or "").strip().lower()

    if not job_id:
      return _jsonify({
        "ok": False,
        "error": "Missing job_id."
      })

    if "/" in job_id or "\\" in job_id or ".." in job_id:
      return _jsonify({
        "ok": False,
        "error": "Job not found."
      })

    if not original_panel_name and not original_source_path:
      return _jsonify({
        "ok": False,
        "error": "Missing original_panel_name or original_source_path."
      })

    if not isinstance(edited_component, dict):
      return _jsonify({
        "ok": False,
        "error": "edited_component must be a dictionary."
      })

    access = _resolve_image_access_context(job_id, target_email)
    owner_email = access["owner_email"]
    group_folder = access["group_folder"]

    last_error = None

    for attempt in range(3):
      try:
        out = anvil.server.call(
          "vm_rerun_rules_with_panel_edit",
          job_id,
          owner_email,
          group_folder,
          original_panel_name,
          edited_component,
          original_source_path
        ) or {}

        if not out.get("ok"):
          return _jsonify({
            "ok": False,
            "error": out.get("error") or "Could not rerun rules.",
            "raw": out
          })

        return _jsonify(out)

      except anvil.server.UplinkDisconnectedError:
        last_error = "Rules rerun unavailable. Please try again."
        if attempt < 2:
          time.sleep(0.5 * (attempt + 1))
          continue

        return _jsonify({
          "ok": False,
          "error": last_error
        })

      except Exception:
        return _jsonify({
          "ok": False,
          "error": "Rules rerun unavailable. Please try again."
        })

    return _jsonify({
      "ok": False,
      "error": last_error or "Rules rerun failed for an unknown reason."
    })

  except anvil.server.UnauthorizedError:
    return _jsonify({
      "ok": False,
      "error": "Unauthorized."
    })

  except Exception:
    return _jsonify({
      "ok": False,
      "error": "Rules rerun unavailable. Please try again."
    })

@anvil.server.callable(require_user=True)
def rerun_rules_with_global_defaults(job_id: str, updated_defaults: dict, target_email: str = ""):
  """
  Client-safe wrapper for editing whole-job panelboard defaults and rerunning
  the rules engine only.

  Works for:
    - normal logged-in job owner
    - admin inspection mode when target_email is passed

  Edits:
    - material / bussing material
    - enclosure / trim
    - rating type
    - breaker type
  """

  try:
    job_id = str(job_id or "").strip()
    target_email = str(target_email or "").strip().lower()

    if not job_id:
      return _jsonify({
        "ok": False,
        "error": "Missing job_id."
      })

    if "/" in job_id or "\\" in job_id or ".." in job_id:
      return _jsonify({
        "ok": False,
        "error": "Job not found."
      })

    if not isinstance(updated_defaults, dict):
      return _jsonify({
        "ok": False,
        "error": "updated_defaults must be a dictionary."
      })

    # Normal user:
    #   resolves to current user's storage context.
    #
    # Admin inspection:
    #   target_email routes to the inspected user's owner_email/group_folder.
    access = _resolve_image_access_context(job_id, target_email)
    owner_email = access["owner_email"]
    group_folder = access["group_folder"]

    allowed_materials = {"ALUMINUM", "COPPER"}
    allowed_ratings = {"FULLY_RATED", "SERIES_RATED"}
    allowed_enclosures = {"NEMA1", "NEMA3R"}
    allowed_trim_styles = {"FLUSH", "SURFACE"}
    allowed_breaker_types = {"BOLT_ON", "PLUG_ON"}

    material = str(
      updated_defaults.get("bussing_material")
      or updated_defaults.get("material")
      or ""
    ).strip().upper()

    rating_type = str(
      updated_defaults.get("rating_type")
      or updated_defaults.get("panel_rating_type")
      or ""
    ).strip().upper()

    enclosure = str(
      updated_defaults.get("enclosure")
      or ""
    ).strip().upper()

    trim_style = str(
      updated_defaults.get("default_trim_style")
      or updated_defaults.get("trim_style")
      or ""
    ).strip().upper()

    breaker_type = str(
      updated_defaults.get("breaker_type")
      or updated_defaults.get("branch_breaker_type")
      or updated_defaults.get("breaker_mounting")
      or ""
    ).strip().upper()

    if material not in allowed_materials:
      return _jsonify({
        "ok": False,
        "error": "Invalid material."
      })

    if rating_type not in allowed_ratings:
      return _jsonify({
        "ok": False,
        "error": "Invalid rating type."
      })

    if enclosure not in allowed_enclosures:
      return _jsonify({
        "ok": False,
        "error": "Invalid enclosure."
      })

    if trim_style not in allowed_trim_styles:
      return _jsonify({
        "ok": False,
        "error": "Invalid trim style."
      })

    if breaker_type not in allowed_breaker_types:
      return _jsonify({
        "ok": False,
        "error": "Invalid breaker type."
      })

    clean_defaults = {
      "bussing_material": material,
      "material": material,

      "rating_type": rating_type,
      "panel_rating_type": rating_type,

      "enclosure": enclosure,
      "default_trim_style": trim_style,
      "trim_style": trim_style,

      "breaker_type": breaker_type,
      "branch_breaker_type": breaker_type,
      "breaker_mounting": breaker_type,
    }

    last_error = None

    for attempt in range(3):
      try:
        out = anvil.server.call(
          "vm_rerun_rules_with_global_defaults",
          job_id,
          owner_email,
          group_folder,
          clean_defaults
        ) or {}

        if not out.get("ok"):
          return _jsonify({
            "ok": False,
            "error": out.get("error") or "Could not rerun rules.",
            "raw": out
          })

        return _jsonify(out)

      except anvil.server.UplinkDisconnectedError:
        last_error = "Rules rerun unavailable. Please try again."
        if attempt < 2:
          time.sleep(0.5 * (attempt + 1))
          continue

        return _jsonify({
          "ok": False,
          "error": last_error
        })

      except Exception as e:
        tb = traceback.format_exc()

        print(">>> GLOBAL DEFAULTS RERUN VM CALL FAILED")
        print(f">>> error_type={type(e).__name__}")
        print(f">>> error={e}")
        print(tb)

        return _jsonify({
          "ok": False,
          "error": f"VM call failed: {type(e).__name__}: {e}",
          "traceback": tb,
        })

    return _jsonify({
      "ok": False,
      "error": last_error or "Rules rerun failed for an unknown reason."
    })

  except anvil.server.UnauthorizedError:
    return _jsonify({
      "ok": False,
      "error": "Unauthorized."
    })

  except Exception as e:
    tb = traceback.format_exc()

    print(">>> GLOBAL DEFAULTS SERVER WRAPPER FAILED")
    print(f">>> error_type={type(e).__name__}")
    print(f">>> error={e}")
    print(tb)

    return _jsonify({
      "ok": False,
      "error": f"Server wrapper failed: {type(e).__name__}: {e}",
      "traceback": tb,
    })

@anvil.server.callable(require_user=True)
def get_job_result(job_id):
  """
  Fetch completed job result after polling confirms the job is done.
  Returns:
    {"ok": True, "result": {...}}
  or:
    {"ok": False, "error": "..."}
  """
  storage_ctx = _current_storage_context()
  owner_email = storage_ctx["owner_email"]
  group_folder = storage_ctx["group_folder"]

  job_id = str(job_id or "").strip()

  if not job_id:
    return {
      "ok": False,
      "error": "Job not found. Please resubmit your PDF."
    }

  if "/" in job_id or "\\" in job_id or ".." in job_id:
    return {
      "ok": False,
      "error": "Job not found. Please resubmit your PDF."
    }

  for attempt in range(3):
    try:
      out = anvil.server.call("vm_get_job_result", job_id, owner_email, group_folder) or {}
      out = dict(out or {})

      err = str(out.get("error") or "")

      if "Unknown job_id" in err:
        return {
          "ok": False,
          "error": "Job not found. Please resubmit your PDF."
        }

      if out.get("ok") is False:
        return {
          "ok": False,
          "error": out.get("error") or "Result unavailable. Please try again."
        }

      # Already wrapped correctly
      if isinstance(out.get("result"), dict):
        return _jsonify({
          "ok": True,
          "result": out.get("result")
        })

      # Raw result.json shape
      if isinstance(out, dict) and out:
        # Remove status/debug wrapper fields only if result is not present.
        # Otherwise just send whatever VM gave us to output2.
        return _jsonify({
          "ok": True,
          "result": out
        })

      return {
        "ok": False,
        "error": "Result was empty."
      }

    except anvil.server.UplinkDisconnectedError:
      if attempt < 2:
        time.sleep(0.75 * (attempt + 1))
        continue
      return {
        "ok": False,
        "error": "uplink_disconnected"
      }

    except Exception:
      return {
        "ok": False,
        "error": "Result unavailable. Please try again."
      }

def _resolve_image_access_context(job_id: str, target_email: str = "") -> dict:
  """
  Resolve image access for both:
    - normal user output/specs image calls
    - admin inspection image calls from accountmanagement -> output2

  Important:
  target_email means billing/admin inspection mode.
  Do NOT call _current_storage_context() first in that case, because a billing-only
  admin may not be the job owner.
  """
  job_id = str(job_id or "").strip()
  target_email = str(target_email or "").strip().lower()

  if not job_id:
    raise Exception("Missing job id.")

  if "/" in job_id or "\\" in job_id or ".." in job_id:
    raise Exception("Job not found.")

  # Admin inspection path
  if target_email:
    ctx = anvil.server.call("get_account_management_context") or {}

    users = ctx.get("users") or []
    team_emails = {
      str(u.get("email") or "").strip().lower()
      for u in users
      if str(u.get("email") or "").strip()
    }

    if target_email not in team_emails:
      raise anvil.server.UnauthorizedError("Unauthorized.")

    group_folder = _safe_folder_key(ctx.get("group_folder") or "", "")

    if not group_folder:
      raise Exception(
        "This account is missing a storage group folder. Please contact support."
      )

    return {
      "owner_email": target_email,
      "group_folder": group_folder,
      "admin_view": True,
    }

  # Normal user path
  storage_ctx = _current_storage_context()

  return {
    "owner_email": storage_ctx["owner_email"],
    "group_folder": storage_ctx["group_folder"],
    "admin_view": False,
  }
  


@anvil.server.callable(require_user=True)
def delete_job_for_user(job_id: str):
  """
  Secure user-facing delete endpoint.
  Deletes a completed/error/canceled job owned by the logged-in user.
  Active queued/running jobs should be canceled instead.
  """
  try:
    storage_ctx = _current_storage_context()
    owner_email = storage_ctx["owner_email"]
    group_folder = storage_ctx["group_folder"]

    job_id = str(job_id or "").strip()
    if not job_id:
      return {"ok": False, "error": "Missing job id."}

    if "/" in job_id or "\\" in job_id or ".." in job_id:
      return {"ok": False, "error": "Job not found."}

    out = anvil.server.call("vm_delete_job", job_id, owner_email, group_folder) or {}
    out = dict(out or {})

    if out.get("ok"):
      return {
        "ok": True,
        "state": out.get("state") or "deleted"
      }

    return {
      "ok": False,
      "error": out.get("error") or "Could not delete job."
    }

  except Exception:
    return {
      "ok": False,
      "error": "Delete unavailable. Please try again."
    }

@anvil.server.callable(require_user=True)
def cancel_job_for_user(job_id: str):
  """
  Secure user-facing cancel endpoint.
  Client provides only job_id. Owner identity comes from logged-in user.
  """
  try:
    storage_ctx = _current_storage_context()
    owner_email = storage_ctx["owner_email"]
    group_folder = storage_ctx["group_folder"]

    job_id = str(job_id or "").strip()
    if not job_id:
      return {"ok": False, "error": "Missing job id."}

    if "/" in job_id or "\\" in job_id or ".." in job_id:
      return {"ok": False, "error": "Job not found."}

    out = anvil.server.call("vm_cancel_job", job_id, owner_email, group_folder) or {}

    # Backward compatibility if VM returns True/False
    if isinstance(out, bool):
      return {
        "ok": bool(out),
        "state": "canceled" if out else "not_canceled",
        "error": "" if out else "Job could not be canceled."
      }

    out = dict(out or {})

    if out.get("ok"):
      return {
        "ok": True,
        "state": out.get("state") or "canceled",
        "deleted": bool(out.get("deleted"))
      }

    return {
      "ok": False,
      "error": out.get("error") or "Job could not be canceled."
    }

  except Exception:
    return {
      "ok": False,
      "error": "Cancel unavailable. Please try again."
    }

@anvil.server.callable(require_user=True)
def list_jobs(limit: int = 50):
  storage_ctx = _current_storage_context()
  owner_email = storage_ctx["owner_email"]
  group_folder = storage_ctx["group_folder"]

  out = anvil.server.call("vm_list_jobs", owner_email, int(limit), group_folder)
  return _jsonify(out)

# --------------------------------------------------
# Compatibility test
# --------------------------------------------------

@anvil.server.callable(require_user=True)
def cancel_temporary_analysis():
  """
  Cancel/remove the current user's temporary
  Specs and Compatibility analysis jobs.
  """
  storage_ctx = _current_storage_context()

  return _jsonify(
    anvil.server.call(
      "vm_clear_temporary_analysis_jobs",
      storage_ctx["owner_email"],
      storage_ctx["group_folder"],
    ) or {}
  )


@anvil.server.background_task
def _bg_submit_compatibility_test(
  file,
  owner_email,
  group_folder,
  access_code_value,
  plan_key,
  company_name,
):
  """
  Upload the selected drawings PDF to the
  compatibility-only VM endpoint.
  """
  if not _is_pdf(file):
    raise Exception("PDFs only")

  result = anvil.server.call(
    "vm_submit_compatibility_test",
    media=file,
    owner_email=owner_email,
    group_folder=group_folder,
    access_code_value=access_code_value,
    plan_key=plan_key,
    company_name=company_name,
  ) or {}

  if not result.get("ok"):
    raise Exception(
      result.get("error")
      or (
        "Could not start the "
        "compatibility test."
      )
    )

  job_id = result.get("job_id")

  if not job_id:
    raise Exception(
      "No compatibility job id "
      "was returned."
    )

  return {
    "ok": True,
    "job_id": job_id,
  }


@anvil.server.callable(require_user=True)
def start_compatibility_upload_task(file):
  """
  Start the PDF upload as an Anvil
  background task.
  """
  storage_ctx = _current_storage_context()

  if not _is_pdf(file):
    raise Exception("PDFs only")

  cleanup = anvil.server.call(
    "vm_clear_temporary_analysis_jobs",
    storage_ctx["owner_email"],
    storage_ctx["group_folder"],
  ) or {}

  if not cleanup.get("ok"):
    raise Exception(
      cleanup.get("error")
      or "Could not prepare the compatibility test."
    )

  task = anvil.server.launch_background_task(
    "_bg_submit_compatibility_test",
    file,
    storage_ctx["owner_email"],
    storage_ctx["group_folder"],
    storage_ctx["access_code_value"],
    storage_ctx["plan_key"],
    storage_ctx["company_name"],
  )

  return {
    "ok": True,
    "task_id": task.get_id(),
  }


@anvil.server.callable(require_user=True)
def get_compatibility_upload_task_status(
  task_id,
):
  """
  Poll the Anvil background upload task.
  """
  task = anvil.server.get_background_task(
    task_id
  )

  if not task:
    return {
      "state": "error",
      "error": (
        "Compatibility upload task "
        "was not found."
      ),
    }

  termination = (
    task.get_termination_status()
  )

  if termination is None:
    return {
      "state": "running",
    }

  if termination == "completed":
    try:
      result = (
        task.get_return_value()
        or {}
      )

      return {
        "state": "done",
        "result": _jsonify(result),
      }

    except Exception as e:
      return {
        "state": "error",
        "error": (
          f"{type(e).__name__}: {e}"
        ),
      }

  if termination in {
    "failed",
    "killed",
    "missing",
  }:
    try:
      task.get_return_value()

    except Exception as e:
      return {
        "state": "error",
        "error": (
          str(e)
          or (
            "Compatibility upload "
            "failed."
          )
        ),
      }

  return {
    "state": "error",
    "error": (
      "Compatibility upload failed."
    ),
  }


@anvil.server.callable(require_user=True)
def get_compatibility_status(job_id):
  """
  Secure compatibility status/result poll.
  """
  storage_ctx = _current_storage_context()

  job_id = str(job_id or "").strip()

  if (
    not job_id
    or "/" in job_id
    or "\\" in job_id
    or ".." in job_id
  ):
    return {
      "state": "not_found",
      "error": (
        "Compatibility test not found."
      ),
    }

  for attempt in range(3):
    try:
      result = anvil.server.call(
        "vm_get_compatibility_status",
        job_id,
        storage_ctx["owner_email"],
        storage_ctx["group_folder"],
      ) or {}

      # Do not pass this response through _jsonify().
      # Completed compatibility results may include
      # BlobMedia for the analyzed panel overlay.
      return dict(result)

    except (
      anvil.server.UplinkDisconnectedError
    ):
      if attempt < 2:
        time.sleep(
          0.75 * (attempt + 1)
        )
        continue

      return {
        "state": "unknown",
        "error": "uplink_disconnected",
      }

    except Exception:
      return {
        "state": "error",
        "error": (
          "Compatibility status is "
          "currently unavailable."
        ),
      }


@anvil.server.callable(require_user=True)
def delete_compatibility_job(job_id):
  """
  Delete the temporary compatibility job
  after the result modal is shown.
  """
  storage_ctx = _current_storage_context()

  job_id = str(job_id or "").strip()

  if (
    not job_id
    or not job_id.startswith(
      "compatibility__"
    )
    or "/" in job_id
    or "\\" in job_id
    or ".." in job_id
  ):
    return False

  try:
    return bool(
      anvil.server.call(
        "vm_delete_compatibility_job",
        job_id,
        storage_ctx["owner_email"],
        storage_ctx["group_folder"],
      )
    )

  except Exception:
    return False

# Specs analysis
@anvil.server.callable(require_user=True)
def upload_specs_pdf(file, job_name=None, exclude_from_improvement=False):
  storage_ctx = _current_storage_context()
  owner_email = storage_ctx["owner_email"]

  if not _is_pdf(file):
    raise Exception("PDFs only")

  user_requested_exclusion = bool(exclude_from_improvement)
  group_default_exclusion = bool(storage_ctx.get("exclude_from_improvement_default", False))
  final_exclusion = bool(user_requested_exclusion or group_default_exclusion)

  out = anvil.server.call(
    "vm_upload_specs_pdf",
    media=file,
    owner_email=owner_email,
    group_folder=storage_ctx["group_folder"],
    access_code_value=storage_ctx["access_code_value"],
    plan_key=storage_ctx["plan_key"],
    company_name=storage_ctx["company_name"],
    job_name=(job_name or ""),
    exclude_from_improvement=final_exclusion
  )

  return _jsonify(out or {})

@anvil.server.callable(require_user=True)
def start_specs_analysis(job_id):
  storage_ctx = _current_storage_context()
  return anvil.server.call(
    "vm_start_specs_analysis",
    job_id,
    storage_ctx["owner_email"],
    storage_ctx["group_folder"]
  ) or {}


@anvil.server.callable(require_user=True)
def get_specs_status(job_id):
  storage_ctx = _current_storage_context()

  for attempt in range(3):
    try:
      out = anvil.server.call(
        "vm_get_specs_status",
        job_id,
        storage_ctx["owner_email"],
        storage_ctx["group_folder"]
      )
      return _jsonify(out)

    except anvil.server.UplinkDisconnectedError:
      if attempt < 2:
        time.sleep(0.75 * (attempt + 1))
        continue
      return {"state": "unknown", "error": "uplink_disconnected"}

    except Exception:
      return {"state": "error", "error": "Status unavailable. Please try again."}



@anvil.server.callable(require_user=True)
def delete_specs_job(job_id):
  storage_ctx = _current_storage_context()

  ok = bool(anvil.server.call(
    "vm_delete_specs_job",
    job_id,
    storage_ctx["owner_email"],
    storage_ctx["group_folder"]
  ))

  if not ok:
    raise Exception(f"Could not delete specs temp job: {job_id}")

  return True

@anvil.server.background_task
def _bg_upload_and_start_specs(file, owner_email, group_folder, access_code_value, plan_key, company_name, job_name="", exclude_from_improvement=False):
  if not _is_pdf(file):
    raise Exception("PDFs only")

  upload_out = anvil.server.call(
    "vm_upload_specs_pdf",
    media=file,
    owner_email=owner_email,
    group_folder=group_folder,
    access_code_value=access_code_value,
    plan_key=plan_key,
    company_name=company_name,
    job_name=(job_name or ""),
    exclude_from_improvement=bool(exclude_from_improvement)
  ) or {}

  job_id = upload_out.get("job_id")
  if not job_id:
    raise Exception("No specs job_id returned after upload.")

  start_out = anvil.server.call(
    "vm_start_specs_analysis",
    job_id,
    owner_email,
    group_folder
  ) or {}

  if not start_out.get("ok"):
    raise Exception("Could not start specs analysis.")

  return {
    "ok": True,
    "job_id": job_id,
    "upload": upload_out,
    "start": start_out
  }

@anvil.server.callable(require_user=True)
def start_specs_upload_task(file, job_name=None, exclude_from_improvement=False):
  storage_ctx = _current_storage_context()
  owner_email = storage_ctx["owner_email"]

  if not _is_pdf(file):
    raise Exception("PDFs only")

  final_exclusion = bool(
    exclude_from_improvement
    or storage_ctx.get("exclude_from_improvement_default", False)
  )

  cleanup = anvil.server.call(
    "vm_clear_temporary_analysis_jobs",
    owner_email,
    storage_ctx["group_folder"],
  ) or {}

  if not cleanup.get("ok"):
    raise Exception(
      cleanup.get("error")
      or "Could not prepare Specs analysis."
    )

  task = anvil.server.launch_background_task(
    "_bg_upload_and_start_specs",
    file,
    owner_email,
    storage_ctx["group_folder"],
    storage_ctx["access_code_value"],
    storage_ctx["plan_key"],
    storage_ctx["company_name"],
    (job_name or ""),
    final_exclusion
  )

  return {
    "ok": True,
    "task_id": task.get_id()
  }

@anvil.server.callable(require_user=True)
def get_specs_upload_task_status(task_id):
  task = anvil.server.get_background_task(task_id)
  if not task:
    return {"state": "error", "error": "Upload task not found."}

  term = task.get_termination_status()

  if term is None:
    return {"state": "running"}

  if term == "completed":
    try:
      result = task.get_return_value() or {}
      return {"state": "done", "result": result}
    except Exception:
      return {"state": "error", "error": "Upload task status unavailable. Please try again."}

  if term in ("failed", "killed", "missing"):
    try:
      task.get_return_value()
      return {"state": "error", "error": f"Upload task {term}."}
    except Exception:
      return {"state": "error", "error": "Upload task status unavailable. Please try again."}

  return {"state": "error", "error": "Upload task status unavailable. Please try again."}
