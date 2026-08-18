from ._anvil_designer import uploadTemplate
from anvil import *
import anvil.server, anvil.users
from anvil import js
from .. import router_state
from ..AccessCodeDialog import AccessCodeDialog
from ..SpecsUploadDialog import SpecsUploadDialog
from ..SpecsResultsDialog import SpecsResultsDialog

USE_UI_OVERRIDES = True

class upload(uploadTemplate):
  def __init__(self, **properties):
    self.init_components(**properties)

    # FULL WIDTH
    self.set_event_handler("show", self._on_show)

    if not anvil.users.get_user():
      js.window.location.hash = "landing"
      return
    # --- Entitlement gate (access code) ---
    if not self._ensure_entitled():
      # _ensure_entitled() may already have redirected to account management
      if not getattr(self, "_entitlement_redirected", False):
        js.window.location.hash = "landing"
      return

    self._group_exclude_from_improvement = False
    self._load_upload_retention_policy()

    self._job_name = None

    if hasattr(self, "go_btn"):
      self.go_btn.text = "Generate BOM"

    if hasattr(self, "instructions_btn"):
      self.instructions_btn.set_event_handler('click', self.instructions_btn_click)

    if hasattr(self, "jobs_btn"):
      self.jobs_btn.set_event_handler('click', self.jobs_btn_click)

    if hasattr(self, "logout_btn"):
      self.logout_btn.set_event_handler('click', self.logout_btn_click)

    if hasattr(self, "enclosure_style_dd"):
      self.enclosure_style_dd.items = [
        ("NEMA 1 — Flush", "NEMA1_FLUSH"),
        ("NEMA 1 — Surface", "NEMA1_SURFACE"),
        ("NEMA 3R (Outdoor)", "NEMA3R"),
      ]
      self.enclosure_style_dd.selected_value = "NEMA1_FLUSH"

    if hasattr(self, "file_loader"):
      self.file_loader.accept = "application/pdf,.pdf"
      self.file_loader.set_event_handler('change', self.file_loader_change)
      self._install_upload_drop_zone()
  
    if not hasattr(self, "timer_1"):
      self.timer_1 = Timer()
      self.timer_1.interval = 0
      self.timer_1.set_event_handler('tick', self.timer_1_tick)
      self.add_component(self.timer_1)

    self._specs_upload_task_id = None
    self._specs_job_id = None
    self._specs_poll_phase = None
    self._specs_result = None

    self._compatibility_upload_task_id = None
    self._compatibility_job_id = None
    self._compatibility_poll_phase = None
    self._compatibility_result = None

    # Files/options held while the processing modal is open.
    self._specs_pending_file = None
    self._specs_pending_exclude = False
    self._compatibility_pending_file = None

    # Shared processing-modal state.
    self._analysis_modal = None
    self._analysis_modal_flow = None
    self._analysis_cancel_requested = False
    self._analysis_error = None
    self._analysis_started_ms = None
    self._analysis_tick_count = 0

    # Existing page lock remains active behind the modal.
    self._upload_page_busy = False
    self._upload_page_control_states = {}
  
  def _load_upload_retention_policy(self):
    """
    Loads account/group-level retention policy.
    If the group requires excluded retention, lock the exclusion checkbox on.
    """
    self._group_exclude_from_improvement = False

    try:
      policy = anvil.server.call("get_upload_retention_policy") or {}
      self._group_exclude_from_improvement = bool(
        policy.get("exclude_from_improvement_default", False)
      )
    except Exception:
      self._group_exclude_from_improvement = False

    box = getattr(self, "exclude_box", None)
    if not box:
      return

    if self._group_exclude_from_improvement:
      try:
        box.checked = True
        box.enabled = False
      except Exception:
        pass

      try:
        box.text = "Exclude from improvement review (required by your organization)"
      except Exception:
        pass

      try:
        box.tooltip = (
          "Your organization requires uploaded jobs to be excluded from Argus "
          "improvement review and retained only for the short excluded-job window."
        )
      except Exception:
        pass
    else:
      try:
        box.enabled = True
      except Exception:
        pass
    
  def _ensure_entitled(self) -> bool:
    """
    Returns True if user is allowed to use the app.
    If not entitled, prompts for code redemption.
    """
    # 1) Are we already entitled?
    try:
      status = anvil.server.call("is_current_user_entitled") or {}
    except Exception as e:
      alert(f"Server error checking access:\n{e}")
      return False

    if status.get("ok"):
      return True

    # 2) Not entitled -> prompt for access code
    msg = status.get("message") or "Not entitled."
    Notification(msg, style="warning").show()

    code = self._prompt_for_access_code()
    if not code:
      return False

    if code == "__MANAGE_ACCOUNT__":
      self._entitlement_redirected = True
      js.window.location.hash = "account"
      return False
    
    # 3) Try redeem
    try:
      res = anvil.server.call("redeem_access_code", code) or {}
    except Exception as e:
      alert(f"Could not redeem access code:\n{e}")
      return False

    if not res.get("ok"):
      alert(res.get("message") or "Access code rejected.")
      return False

    # 4) Re-check entitlement (should now pass)
    try:
      status2 = anvil.server.call("is_current_user_entitled") or {}
    except Exception:
      status2 = {}

    if status2.get("ok"):
      Notification("Access code accepted.", style="success").show()
      return True

    alert(status2.get("message") or "Access still not active.")
    return False

  def _prompt_for_access_code(self) -> str:
    code = alert(
      content=AccessCodeDialog(),
      title="Access Code Required",
      buttons=[],
      large=False,
      dismissible=False
    )
    return (code or "").strip()

  def _force_full_width_everywhere(self, **e):
    try:
      js.window.setTimeout(self._force_full_width_now, 0)
    except Exception:
      self._force_full_width_now()

  def _on_show(self, **e):
    self._force_full_width_everywhere()
  
    try:
      js.window.setTimeout(self._install_upload_drop_zone, 0)
      js.window.setTimeout(self._install_upload_drop_zone, 150)
    except Exception:
      self._install_upload_drop_zone()
      
  def _force_full_width_now(self):
    try:
      node = js.get_dom_node(self)

      # Walk up and kill max-width on every ancestor
      el = node
      hops = 0
      while el is not None and hops < 50:
        try:
          s = el.style
          s.setProperty("max-width", "none", "important")
          s.setProperty("width", "100%", "important")
          s.setProperty("margin-left", "0", "important")
          s.setProperty("margin-right", "0", "important")
        except Exception:
          pass

        try:
          el = el.parentElement
        except Exception:
          el = None

        hops += 1

      # Also force root elements
      for root in [js.document.documentElement, js.document.body]:
        try:
          root.style.setProperty("max-width", "none", "important")
          root.style.setProperty("width", "100%", "important")
          root.style.setProperty("margin", "0", "important")
        except Exception:
          pass

      # Common Anvil mount point
      try:
        agh = js.document.getElementById("appGoesHere")
        if agh:
          agh.style.setProperty("max-width", "none", "important")
          agh.style.setProperty("width", "100%", "important")
      except Exception:
        pass

    except Exception:
      pass

  # --- make the job note string ---
  def _build_job_note(self, file_media):
    if hasattr(self, "text_box_1") and self.text_box_1.text:
      job_name = self.text_box_1.text.strip()
    else:
      fname = (getattr(file_media, "name", "") or "").strip()
      job_name = fname.rsplit(".", 1)[0] if fname else "untitled"

    job_name = job_name.replace("|", "/").replace("\n", " ").replace("\r", " ")
    if len(job_name) > 80:
      job_name = job_name[:80]
    self._job_name = job_name

    try:
      submitted_at_utc = js.window.Date().toISOString()
      submitted_local  = js.window.Date().toLocaleString()
      tz_offset_min    = int(js.window.Date().getTimezoneOffset())
    except Exception:
      from datetime import datetime
      submitted_at_utc = datetime.utcnow().isoformat() + "Z"
      submitted_local  = submitted_at_utc
      tz_offset_min    = 0

    user = anvil.users.get_user()
    user_email = ""
    if user:
      try:
        user_email = user['email'] or ""
      except Exception:
        pass

    return (
      f"job_name={job_name} | submitted_at_utc={submitted_at_utc} | "
      f"submitted_local={submitted_local} | tz_offset_min={tz_offset_min} | user={user_email}"
    )

  # --- helper: ensure uploaded file is a PDF ---
  def _is_pdf(self, media):
    if not media:
      return False
    ct = (getattr(media, "content_type", "") or "").lower()
    name = (getattr(media, "name", "") or "").lower()
    return ("pdf" in ct) or name.endswith(".pdf")

  def _set_upload_status(self, text: str):
    """
    Updates the small status text under the upload PDF control.
    """
    try:
      root = js.get_dom_node(self)
      if not root:
        return
  
      status = root.querySelector("#upload_file_status")
      if status:
        status.textContent = str(text or "No PDF selected")
    except Exception:
      pass
  
  def _install_upload_drop_zone(self):
    """
    Installs drag/drop for the upload page.
  
    Firefox/Linux note:
    Browsers will open a dropped PDF by default unless dragover/drop are blocked
    at the document/window level. This method blocks that default behavior first,
    then only processes the file when it is dropped on the Argus drop zone.
    """
    try:
      root = js.get_dom_node(self)
      if not root:
        return
  
      drop_zone = root.querySelector(".argus-file-dropzone")
      if not drop_zone:
        return
  
      fl_node = js.get_dom_node(self.file_loader)
      if not fl_node:
        return
  
      input_el = fl_node.querySelector("input[type='file']")
      if not input_el:
        return
  
      # Avoid double-installing listeners if the form re-shows.
      if drop_zone.getAttribute("data-drop-installed") == "1":
        return
  
      drop_zone.setAttribute("data-drop-installed", "1")
  
      def _is_inside_drop_zone(target):
        try:
          return bool(target and drop_zone.contains(target))
        except Exception:
          return False
  
      def _stop(e):
        try:
          e.preventDefault()
          e.stopPropagation()
        except Exception:
          pass
  
      def _clear_drag_state():
        try:
          drop_zone.classList.remove("is-dragging")
        except Exception:
          pass
  
      def _set_drag_state(e):
        try:
          if _is_inside_drop_zone(e.target):
            drop_zone.classList.add("is-dragging")
          else:
            drop_zone.classList.remove("is-dragging")
        except Exception:
          pass
  
      def _handle_drop(e):
        _stop(e)
        _clear_drag_state()

        if getattr(
          self,
          "_upload_page_busy",
          False
        ):
          Notification(
            (
              "Document processing is already "
              "in progress."
            ),
            style="warning",
            timeout=2
          ).show()
          return
  
        try:
          if not _is_inside_drop_zone(e.target):
            return
  
          files = e.dataTransfer.files
          if not files or files.length < 1:
            return
  
          first_file = files.item(0)
          file_name = str(first_file.name or "").strip()
          file_name_l = file_name.lower()
  
          if not file_name_l.endswith(".pdf"):
            self._set_upload_status("No PDF selected")
            Notification("PDFs only (.pdf).", style="warning").show()
            return
  
          # Put the dropped file into the native input behind Anvil's FileLoader.
          # This lets the normal FileLoader change handler run.
          try:
            data_transfer = js.window.DataTransfer.new()
          except Exception:
            data_transfer = js.window.DataTransfer()
  
          data_transfer.items.add(first_file)
          input_el.files = data_transfer.files
  
          try:
            change_event = js.window.Event.new("change", {"bubbles": True})
          except Exception:
            change_event = js.window.Event("change", {"bubbles": True})
  
          input_el.dispatchEvent(change_event)
  
          self._set_upload_status(f"Selected: {file_name}")
          Notification("PDF selected.", style="success", timeout=1.5).show()
  
        except Exception as err:
          self._set_upload_status("No PDF selected")
          Notification(
            "Drag/drop was blocked by the browser. Click Upload PDF instead.",
            style="warning",
            timeout=3
          ).show()
  
      # Drop-zone visual listeners.
      drop_zone.addEventListener("dragenter", lambda e: (_stop(e), _set_drag_state(e)))
      drop_zone.addEventListener("dragover", lambda e: (_stop(e), _set_drag_state(e)))
      drop_zone.addEventListener("dragleave", lambda e: (_stop(e), _clear_drag_state()))
      drop_zone.addEventListener("drop", _handle_drop)
  
      # CRITICAL:
      # Firefox/Linux will open PDFs unless the page-level default is blocked.
      def _document_dragover(e):
        _stop(e)
        _set_drag_state(e)
  
        try:
          e.dataTransfer.dropEffect = "copy"
        except Exception:
          pass
  
      def _document_drop(e):
        _handle_drop(e)
  
      js.document.addEventListener("dragover", _document_dragover)
      js.document.addEventListener("drop", _document_drop)
  
      try:
        js.window.addEventListener("dragover", _document_dragover)
        js.window.addEventListener("drop", _document_drop)
      except Exception:
        pass
  
    except Exception:
      pass
  
  def file_loader_change(self, file=None, **event_args):
    media = file or getattr(self.file_loader, "file", None)
  
    if not self._is_pdf(media):
      try:
        self.file_loader.file = None
      except Exception:
        pass
  
      self._set_upload_status("No PDF selected")
      Notification("PDFs only (.pdf).", style="warning").show()
      return
  
    name = str(getattr(media, "name", "") or "PDF selected").strip()
    self._set_upload_status(f"Selected: {name}")

  def build_ui_overrides(self):
    overrides = {"panelboards": {}, "transformers": {}, "disconnects": {}}
    if getattr(self, "check_box_1", None) and self.check_box_1.checked:
      overrides["panelboards"]["bussing_material"] = "COPPER"

    sel = getattr(self, "enclosure_style_dd", None)
    choice = sel.selected_value if sel else "NEMA1_FLUSH"
    if choice == "NEMA3R":
      overrides["panelboards"]["enclosure"] = "NEMA3R"
    elif choice == "NEMA1_SURFACE":
      overrides["panelboards"]["enclosure"] = "NEMA1"
      overrides["panelboards"]["default_trim_style"] = "SURFACE"
    else:
      overrides["panelboards"]["enclosure"] = "NEMA1"
      overrides["panelboards"]["default_trim_style"] = "FLUSH"

    if getattr(self, "check_box_4", None) and self.check_box_4.checked:
      overrides["panelboards"]["rating_type"] = "SERIES_RATED"
    if getattr(self, "check_box_5", None) and self.check_box_5.checked:
      overrides["panelboards"]["allow_plug_on_breakers"] = False
    if getattr(self, "check_box_6", None) and (not self.check_box_6.checked):
      overrides["panelboards"]["allow_square_d_spd"] = False
    if getattr(self, "check_box_10", None) and (not self.check_box_10.checked):
      overrides["disconnects"]["allow_littlefuse"] = False

    return {k: v for k, v in overrides.items() if v}

  def go_btn_click(self, **event_args):
    if not anvil.users.get_user():
      js.window.location.hash = "landing"
      return
  
    f = getattr(self.file_loader, "file", None)
    if not self._is_pdf(f):
      Notification("Please choose a PDF.", style="warning").show()
      return
  
    try:
      # Build the job name + note (same as before)
      job_note = self._build_job_note(f)
      ui_overrides = self.build_ui_overrides() if USE_UI_OVERRIDES else {}
    
      # Optional: stash job_name for the downstream form
      _ = self._job_name  # set by _build_job_note
    
      # Exclude-from-improvement option
      user_checked_exclusion = bool(
        getattr(self, "exclude_box", None) and self.exclude_box.checked
      )

      # Group policy wins. If the access code requires exclusion, every job is excluded.
      exclude_from_improvement = bool(
        user_checked_exclusion or getattr(self, "_group_exclude_from_improvement", False)
      )
    
      if exclude_from_improvement:
        if getattr(self, "_group_exclude_from_improvement", False):
          confirm_title = "Organization exclusion policy"
          confirm_text = (
            "Your organization requires jobs to be excluded from Argus improvement review.\n\n"
            "The raw uploaded PDF will be deleted immediately after processing.\n\n"
            "Processed artifacts and outputs, including overlays and editable results, "
            "will remain available for up to 24 hours so you can review, edit, regenerate, "
            "and download the job.\n\n"
            "After the 24-hour retention window expires, the full job is eligible "
            "for automatic deletion by Argus's retention cleanup process.\n\n"
            "Continue?"
          )
        else:
          confirm_title = "Confirm excluded job"
          confirm_text = (
            "This job will be excluded from Argus improvement review.\n\n"
            "The raw uploaded PDF will be deleted immediately after processing.\n\n"
            "Processed artifacts and outputs, including overlays and editable results, "
            "will remain available for up to 24 hours so you can review, edit, regenerate, "
            "and download the job.\n\n"
            "After the 24-hour retention window expires, the full job is eligible "
            "for automatic deletion by Argus's retention cleanup process.\n\n"
            "Continue?"
          )

        confirmed = alert(
          content=confirm_text,
          title=confirm_title,
          buttons=[
            ("Cancel", False),
            ("Continue", True),
          ],
          large=True,
          dismissible=False
        )
    
        if not confirmed:
          return
    
      # Hand off to the dedicated full-screen processing form.
      # That form will:
      #   - Upload/submit asynchronously
      #   - Show progress and elapsed time
      #   - Poll until done
      #   - Navigate to output on completion
      # Router navigation (do NOT open_form)
      router_state.processing_args = {
        "file": f,
        "ui_overrides": ui_overrides,
        "job_note": job_note,
        "job_name": self._job_name,
        "exclude_from_improvement": exclude_from_improvement
      }
      js.window.location.hash = "processing"
      return

  
    except Exception as e:
      msg = (getattr(e, "args", [""]) or [""])[0] or str(e) or type(e).__name__
      alert(f"❌ Failed to start job:\n{msg}")
  
  def timer_1_tick(self, **event_args):
    """
    Shared polling timer for the Specs Analyzer and
    Quick Compatibility Test processing modals.
    """
    try:
      self._refresh_analysis_processing_modal()

      if getattr(
        self,
        "_compatibility_poll_phase",
        None
      ):
        self._compatibility_timer_tick()
        return

      if getattr(
        self,
        "_specs_poll_phase",
        None
      ):
        self._specs_timer_tick()
        return

      self.timer_1.interval = 0

    except Exception as e:
      flow = (
        "compatibility"
        if getattr(
          self,
          "_compatibility_poll_phase",
          None
        )
        else "specs"
      )

      self._fail_active_analysis(
        flow,
        e
      )

  def logout_btn_click(self, **event_args):
    anvil.users.logout()
    Notification("Logged out", timeout=1.5).show()
    js.window.location.hash = "landing"

  def instructions_btn_click(self, **event_args):
    js.window.location.hash = "instructions"

  def jobs_btn_click(self, **event_args):
    js.window.location.hash = "myjobs"

  @handle("button_1", "click")
  def button_1_click(self, **event_args):
    js.window.location.hash = "coveragemap"

  @handle("specs_btn", "click")
  def specs_btn_click(self, **event_args):
    self._run_specs_modal_flow()

  def _set_upload_page_busy(
    self,
    busy: bool,
    active_flow: str = None,
    status_text: str = None
  ):
    """
    Lock the Upload page while Specs or Compatibility
    analysis is running. Only the matching Cancel button
    remains available after the VM job has been created.
    """
    control_names = (
      "instructions_btn",
      "button_1",
      "jobs_btn",
      "logout_btn",
      "text_box_1",
      "file_loader",
      "Checklist",
      "compatibility_test_btn",
      "specs_btn",
      "check_box_1",
      "enclosure_style_dd",
      "check_box_4",
      "check_box_5",
      "check_box_6",
      "go_btn",
      "exclude_box",
    )

    cancel_names = (
      "specs_cancel_btn",
      "compatibility_cancel_btn",
    )

    if busy:
      if not self._upload_page_busy:
        self._upload_page_control_states = {}

        for name in control_names:
          component = getattr(
            self,
            name,
            None
          )

          if component is None:
            continue

          state = {}

          try:
            state["enabled"] = bool(
              component.enabled
            )
          except Exception:
            pass

          try:
            state["text"] = component.text
          except Exception:
            pass

          self._upload_page_control_states[
            name
          ] = state

      self._upload_page_busy = True

      for name in control_names:
        component = getattr(
          self,
          name,
          None
        )

        if component is None:
          continue

        try:
          component.enabled = False
        except Exception:
          pass

      # Hide both cancel buttons first.
      for name in cancel_names:
        button = getattr(
          self,
          name,
          None
        )

        if button is None:
          continue

        try:
          button.visible = False
          button.enabled = False
          button.text = "Cancel"
        except Exception:
          pass

      cancel_button = None
      job_id = None

      if active_flow == "compatibility":
        try:
          self.compatibility_test_btn.text = (
            status_text or "Processing..."
          )
        except Exception:
          pass

        cancel_button = getattr(
          self,
          "compatibility_cancel_btn",
          None
        )

        job_id = self._compatibility_job_id

      elif active_flow == "specs":
        try:
          self.specs_btn.text = (
            status_text or "Processing..."
          )
        except Exception:
          pass

        cancel_button = getattr(
          self,
          "specs_cancel_btn",
          None
        )

        job_id = self._specs_job_id

      # The cancel button appears only after the
      # actual VM job has been created.
      if cancel_button is not None and job_id:
        try:
          cancel_button.visible = True
          cancel_button.enabled = True
        except Exception:
          pass

      return

    saved_states = (
      self._upload_page_control_states
      or {}
    )

    for name, state in saved_states.items():
      component = getattr(
        self,
        name,
        None
      )

      if component is None:
        continue

      if "enabled" in state:
        try:
          component.enabled = state[
            "enabled"
          ]
        except Exception:
          pass

      if "text" in state:
        try:
          component.text = state[
            "text"
          ]
        except Exception:
          pass

    for name in cancel_names:
      button = getattr(
        self,
        name,
        None
      )

      if button is None:
        continue

      try:
        button.visible = False
        button.enabled = False
        button.text = "Cancel"
      except Exception:
        pass

    self._upload_page_control_states = {}
    self._upload_page_busy = False

  def _cancel_temporary_analysis(self):
    confirm_panel = ColumnPanel(
      spacing="medium"
    )

    confirm_panel.add_component(
      Label(
        text=(
          "Are you sure you want to cancel?"
        ),
        bold=True,
        font_size=16,
        align="center"
      )
    )

    button_grid = GridPanel()

    no_btn = Button(
      text="No",
      icon="fa:times",
      background="#b00020",
      foreground="#ffffff",
      bold=True
    )

    yes_btn = Button(
      text="Yes",
      icon="fa:check",
      role="primary-color",
      bold=True
    )

    def no_click(**event_args):
      confirm_panel.raise_event(
        "x-close-alert",
        value=False
      )

    def yes_click(**event_args):
      confirm_panel.raise_event(
        "x-close-alert",
        value=True
      )

    no_btn.set_event_handler(
      "click",
      no_click
    )

    yes_btn.set_event_handler(
      "click",
      yes_click
    )

    # GridPanel guarantees the buttons remain
    # beside each other instead of stacking.
    button_grid.add_component(
      no_btn,
      row="cancel_buttons",
      col_xs=0,
      width_xs=6
    )

    button_grid.add_component(
      yes_btn,
      row="cancel_buttons",
      col_xs=6,
      width_xs=6
    )

    confirm_panel.add_component(
      button_grid
    )

    confirmed = alert(
      content=confirm_panel,
      title="Cancel Analysis",
      buttons=[],
      large=False,
      dismissible=False
    )

    if not confirmed:
      return

    for name in (
      "specs_cancel_btn",
      "compatibility_cancel_btn",
    ):
      button = getattr(
        self,
        name,
        None
      )

      if button is None:
        continue

      try:
        if button.visible:
          button.enabled = False
          button.text = "Canceling..."
      except Exception:
        pass

    try:
      result = anvil.server.call(
        "cancel_temporary_analysis"
      ) or {}

      if not result.get("ok"):
        raise Exception(
          result.get("error")
          or "Could not cancel the analysis."
        )

      Notification(
        "Analysis canceled.",
        style="success",
        timeout=2
      ).show()

    except Exception as e:
      alert(
        f"Could not cancel analysis:\n{e}"
      )

    finally:
      self.timer_1.interval = 0

      self._specs_upload_task_id = None
      self._specs_job_id = None
      self._specs_poll_phase = None
      self._specs_result = None

      self._compatibility_upload_task_id = None
      self._compatibility_job_id = None
      self._compatibility_poll_phase = None

      self._set_upload_page_busy(False)

  @handle(
    "specs_cancel_btn",
    "click"
  )
  def specs_cancel_btn_click(
    self,
    **event_args
  ):
    self._cancel_temporary_analysis()

  @handle(
    "compatibility_cancel_btn",
    "click"
  )
  def compatibility_cancel_btn_click(
    self,
    **event_args
  ):
    self._cancel_temporary_analysis()

  
  # specs helpers
  def _apply_specs_defaults_to_page(self, detected_overrides: dict):
    """
    Push detected spec defaults back into the page controls.
    """
    detected_overrides = detected_overrides or {}
    pb = detected_overrides.get("panelboards") or {}

    # Bussing material
    bussing = pb.get("bussing_material")
    if hasattr(self, "check_box_1") and bussing is not None:
      self.check_box_1.checked = (str(bussing).upper() == "COPPER")

    # Rating type
    rating_type = pb.get("rating_type")
    if hasattr(self, "check_box_4") and rating_type is not None:
      self.check_box_4.checked = (str(rating_type).upper() == "SERIES_RATED")

    # Plug-on breakers
    # Your UI override means:
    # check_box_5.checked == True  -> allow_plug_on_breakers = False
    allow_plug_on = pb.get("allow_plug_on_breakers")
    if hasattr(self, "check_box_5") and allow_plug_on is not None:
      self.check_box_5.checked = (allow_plug_on is False)

    Notification("Specs defaults applied to this page.", style="success").show()

  def _run_specs_modal_flow(self):
    if getattr(
      self,
      "_upload_page_busy",
      False
    ):
      Notification(
        (
          "Document processing is already "
          "in progress."
        ),
        style="warning"
      ).show()
      return

    upload_result = alert(
      content=SpecsUploadDialog(
        force_excluded=getattr(
          self,
          "_group_exclude_from_improvement",
          False
        )
      ),
      title="Specs PDF Analyzer",
      foreground="white",
      buttons=[],
      large=False,
      dismissible=True
    )

    if (
      not upload_result
      or not upload_result.get("file")
    ):
      return

    specs_pdf = upload_result["file"]

    exclude_specs_from_improvement = bool(
      upload_result.get(
        "exclude_from_improvement"
      )
      or getattr(
        self,
        "_group_exclude_from_improvement",
        False
      )
    )

    try:
      self._set_upload_page_busy(
        True,
        active_flow="specs",
        status_text="Uploading..."
      )

      start_task_resp = anvil.server.call(
        "start_specs_upload_task",
        specs_pdf,
        self._job_name or "",
        exclude_specs_from_improvement
      ) or {}

      self._specs_upload_task_id = (
        start_task_resp.get("task_id")
      )

      if not self._specs_upload_task_id:
        raise Exception(
          "Could not start specs upload task."
        )

      self._specs_job_id = None
      self._specs_poll_phase = (
        "upload_task"
      )

      self.timer_1.interval = 0.75

    except Exception as e:
      self._specs_upload_task_id = None
      self._specs_job_id = None
      self._specs_poll_phase = None

      self._set_upload_page_busy(False)

      alert(
        f"❌ Specs analysis failed:\n{e}"
      )

  def _specs_timer_tick(self):
    try:
      # ------------------------------------------
      # Phase 1: upload the Specs PDF
      # ------------------------------------------
      if (
        self._specs_poll_phase
        == "upload_task"
      ):
        task_status = anvil.server.call(
          "get_specs_upload_task_status",
          self._specs_upload_task_id
        ) or {}

        task_state = str(
          task_status.get("state") or ""
        ).strip().lower()

        if task_state == "done":
          task_result = (
            task_status.get("result")
            or {}
          )

          self._specs_job_id = (
            task_result.get("job_id")
          )

          if not self._specs_job_id:
            raise Exception(
              "Upload completed, but no Specs "
              "job ID was returned."
            )

          self._specs_poll_phase = (
            "specs_job"
          )

          self._set_upload_page_busy(
            True,
            active_flow="specs",
            status_text="Processing..."
          )

          return

        if task_state == "error":
          raise Exception(
            task_status.get("error")
            or "Specs upload failed."
          )

        self._set_upload_page_busy(
          True,
          active_flow="specs",
          status_text="Uploading..."
        )

        return

      # ------------------------------------------
      # Phase 2: Specs processing
      # ------------------------------------------
      if (
        self._specs_poll_phase
        == "specs_job"
      ):
        status = anvil.server.call(
          "get_specs_status",
          self._specs_job_id
        ) or {}

        state = str(
          status.get("state") or ""
        ).strip().lower()

        if (
          state == "unknown"
          and status.get("error")
          == "uplink_disconnected"
        ):
          return

        self._set_upload_page_busy(
          True,
          active_flow="specs",
          status_text="Processing..."
        )

        if state == "done":
          self.timer_1.interval = 0
          self._specs_poll_phase = None

          result = (
            status.get("result")
            or {}
          )

          try:
            modal_result = alert(
              content=SpecsResultsDialog(
                result
              ),
              title="Specs Analysis Results",
              buttons=[],
              large=True,
              dismissible=True
            )

            if (
              modal_result
              and modal_result.get("apply")
            ):
              self._apply_specs_defaults_to_page(
                modal_result.get(
                  "detected_overrides"
                ) or {}
              )

          finally:
            self._specs_result = None
            self._specs_upload_task_id = None
            self._specs_job_id = None
            self._specs_poll_phase = None

            self._set_upload_page_busy(
              False
            )

          return

        if state == "error":
          raise Exception(
            status.get("error")
            or "Specs analysis failed."
          )

        if state == "not_found":
          raise Exception(
            "Specs job was not found."
          )

        return

    except Exception as e:
      self.timer_1.interval = 0

      self._specs_result = None
      self._specs_upload_task_id = None
      self._specs_job_id = None
      self._specs_poll_phase = None

      self._set_upload_page_busy(False)

      alert(
        f"❌ Specs analysis failed:\n"
        f"{type(e).__name__}: {e}"
      )

  @handle("compatibility_test_btn", "click")
  def compatibility_test_btn_click(
    self,
    **event_args
  ):
    self._run_compatibility_test()
  
  
  def _set_compatibility_busy(
    self,
    busy: bool,
    text: str = None
  ):
    """
    Use the shared Upload-page lock for the
    compatibility test.
    """
    if busy:
      self._set_upload_page_busy(
        True,
        active_flow="compatibility",
        status_text=(
          text or "Processing..."
        )
      )
    else:
      self._set_upload_page_busy(False)
  
  
  def _reset_compatibility_test_ui(self):
    self._compatibility_upload_task_id = None
    self._compatibility_job_id = None
    self._compatibility_poll_phase = None
  
    self._set_compatibility_busy(
      False,
      "Run Quick Test"
    )
  
  
  def _run_compatibility_test(self):
    drawings_pdf = getattr(
      self.file_loader,
      "file",
      None
    )
  
    if not self._is_pdf(drawings_pdf):
      Notification(
        (
          "Upload a drawings PDF above before "
          "running the quick compatibility test."
        ),
        style="warning"
      ).show()
      return
  
    if (
      getattr(
        self,
        "_compatibility_poll_phase",
        None
      )
      or getattr(
        self,
        "_specs_poll_phase",
        None
      )
    ):
      Notification(
        (
          "Another document analysis is already "
          "running. Let it finish before starting "
          "the compatibility test."
        ),
        style="warning"
      ).show()
      return
  
    try:
      self._set_compatibility_busy(
        True,
        "Uploading..."
      )
  
      response = anvil.server.call(
        "start_compatibility_upload_task",
        drawings_pdf
      ) or {}
  
      task_id = response.get("task_id")
  
      if not task_id:
        raise Exception(
          "Could not start the compatibility upload."
        )
  
      self._compatibility_upload_task_id = (
        task_id
      )
  
      self._compatibility_job_id = None
  
      self._compatibility_poll_phase = (
        "upload_task"
      )
  
      self.timer_1.interval = 0.75
  
    except Exception as e:
      self._reset_compatibility_test_ui()
  
      alert(
        f"❌ Compatibility test failed:\n{e}"
      )
  
  
  def _delete_compatibility_job_safely(
    self,
    job_id
  ):
    if not job_id:
      return
  
    try:
      anvil.server.call(
        "delete_compatibility_job",
        job_id
      )
    except Exception:
      # The VM fallback retention cleanup will
      # remove abandoned tests after 20 minutes.
      pass
  
  
  def _compatibility_timer_tick(self):
    try:
      # ------------------------------------------------
      # Phase 1: Anvil background upload
      # ------------------------------------------------
      if (
        self._compatibility_poll_phase
        == "upload_task"
      ):
        task_status = anvil.server.call(
          "get_compatibility_upload_task_status",
          self._compatibility_upload_task_id
        ) or {}

        task_state = str(
          task_status.get("state") or ""
        ).strip().lower()

        if task_state == "done":
          task_result = (
            task_status.get("result")
            or {}
          )

          job_id = task_result.get("job_id")

          if not job_id:
            raise Exception(
              "Upload completed, but no "
              "compatibility job ID was returned."
            )

          self._compatibility_job_id = job_id

          self._compatibility_poll_phase = (
            "compatibility_job"
          )

          self._set_compatibility_busy(
            True,
            "Processing..."
          )

          return

        if task_state == "error":
          raise Exception(
            task_status.get("error")
            or "Compatibility upload failed."
          )

        self._set_compatibility_busy(
          True,
          "Uploading..."
        )

        return

      # ------------------------------------------------
      # Phase 2: VM compatibility processing
      # ------------------------------------------------
      if (
        self._compatibility_poll_phase
        == "compatibility_job"
      ):
        status = anvil.server.call(
          "get_compatibility_status",
          self._compatibility_job_id
        ) or {}

        state = str(
          status.get("state") or ""
        ).strip().lower()

        if (
          state == "unknown"
          and status.get("error")
          == "uplink_disconnected"
        ):
          return

        self._set_compatibility_busy(
          True,
          "Processing..."
        )

        if state == "done":
          self.timer_1.interval = 0

          job_id = self._compatibility_job_id

          result = (
            status.get("result")
            or {}
          )

          modal_action = None

          try:
            modal_action = (
              self._show_compatibility_result(
                result
              )
            )

          finally:
            self._delete_compatibility_job_safely(
              job_id
            )

            self._reset_compatibility_test_ui()

          if modal_action == "checklist":
            self.Checklist_click()

          return

        if state == "error":
          raise Exception(
            status.get("error")
            or "Compatibility test failed."
          )

        if state in {
          "not_found",
          "canceled",
          "cancelled"
        }:
          raise Exception(
            status.get("error")
            or "Compatibility test was not found."
          )

        return

    except Exception as e:
      self.timer_1.interval = 0

      job_id = self._compatibility_job_id

      self._delete_compatibility_job_safely(
        job_id
      )

      self._reset_compatibility_test_ui()

      alert(
        f"❌ Compatibility test failed:\n"
        f"{type(e).__name__}: {e}"
      )
  
  def _show_compatibility_result(
    self,
    result: dict
  ):
    """
    Show the three checklist-style result sections.
    """
    result = result or {}
    report = result.get("report") or {}
  
    grade = str(
      report.get("grade") or "unconfirmed"
    ).strip().lower()
  
    if grade == "likely":
      overall_icon = "✅"
      overall_color = "#2e7d32"
  
    elif grade == "questionable":
      overall_icon = "⚠️"
      overall_color = "#b26a00"
  
    else:
      overall_icon = "❌"
      overall_color = "#b00020"
  
    body = ColumnPanel(
      spacing="medium"
    )
  
    body.add_component(
      Label(
        text=(
          f"{overall_icon} "
          f"{report.get('headline') or 'Compatibility Result'}"
        ),
        bold=True,
        font_size=22,
        foreground=overall_color
      )
    )
  
    summary = str(
      report.get("summary") or ""
    ).strip()
  
    if summary:
      body.add_component(
        Label(
          text=summary,
          font_size=15
        )
      )
  
    panel_name = str(
      report.get("panel_name") or ""
    ).strip()
  
    if panel_name:
      body.add_component(
        Label(
          text=f"Representative panel: {panel_name}",
          bold=True,
          font_size=14
        )
      )
  
    for section in (
      report.get("sections") or []
    ):
      section_status = str(
        section.get("status") or ""
      ).strip().lower()
  
      if section_status == "good":
        section_icon = "✓"
        section_color = "#2e7d32"
  
      elif section_status == "questionable":
        section_icon = "!"
        section_color = "#b26a00"
  
      elif section_status == "problem":
        section_icon = "×"
        section_color = "#b00020"
  
      else:
        section_icon = "—"
        section_color = "#666666"
  
      body.add_component(
        Label(
          text=(
            f"{section_icon} "
            f"{section.get('title') or ''}"
          ),
          bold=True,
          font_size=17,
          foreground=section_color
        )
      )
  
      body.add_component(
        Label(
          text=str(
            section.get("message") or ""
          ),
          font_size=14
        )
      )
  
    notes = report.get("notes") or []
  
    if notes:
      body.add_component(
        Label(
          text="Additional Notes",
          bold=True,
          font_size=16
        )
      )
  
      for note in notes:
        body.add_component(
          Label(
            text=f"• {note}",
            font_size=13
          )
        )
  
    body.add_component(
      Label(
        text=(
          "This test analyzes one representative panel "
          "from the PDF. It does not guarantee that every "
          "panel or drawing style in the job will process "
          "successfully."
        ),
        font_size=13,
        italic=True,
        foreground="#666666"
      )
    )
  
    return alert(
      content=body,
      title="Quick Compatibility Test",
      buttons=[
        ("Compatibility Checklist", "checklist"),
        ("Close", "close"),
      ],
      large=True,
      dismissible=True
    )
  
  @handle("Checklist", "click")
  def Checklist_click(self, **event_args):
    """Will It Run checklist modal"""

    body_panel = ColumnPanel(spacing="medium")
    checklist_boxes = {}

    checklist_sections = [
      {
        "title": "Panel Identification",
        "rows": [
          {
            "key": "trimmed_pdf",
            "text": "PDF file trimmed to only relevant pages.",
            "weight": 2,
            "note": (
              "PDFs should be trimmed to only the relevant pages to reduce "
              "junk detections and speed up processing."
            ),
          },
          {
            "key": "schedules_not_touching",
            "text": "Panel schedules do not touch other schedules, boundaries, or unrelated items.",
            "weight": 10,
            "cap": 30,
            "note": (
              "Panel schedules that touch other schedules, boundaries, or unrelated items "
              "are very likely to be missed. A person can visually separate these areas, "
              "but computer vision may see them as one large blob of information."
            ),
          },
        ],
      },
      {
        "title": "General Structure",
        "rows": [
          {
            "key": "header_above_breakers",
            "text": "Header section is above the breaker section.",
            "weight": 10,
            "cap": 45,
            "note": (
              "Currently Argus can only process panels with the header section "
              "above the breaker section. These panels may be found, but they are unlikely "
              "to be confidently analyzed."
            ),
          },
          {
            "key": "single_grid",
            "text": "Breaker section uses a single clear grid/table layout.",
            "weight": 4,
            "note": (
              "Argus performs best when the breaker section uses one clear grid/table layout "
              "with visible grid lines. Non-standard layouts may reduce breaker detection confidence."
            ),
          },
          {
            "key": "readable_font",
            "text": "Text is readable with a simple, standard font.",
            "weight": 3,
            "note": (
              "Argus can handle some variation in fonts, but text that is unclear, distorted, "
              "or highly non-standard may cause missed or incorrect detections."
            ),
          },
        ],
      },
      {
        "title": "Header Section",
        "rows": [
          {
            "key": "strong_label_value",
            "text": "Strong Label - Value association. (higher confidence)",
            "weight": 3,
            "note": (
              "Argus performs better when values are clearly tied to labels, such as "
              "“Bus Amps: 225A” or “Volts: 480Y/277V”. It can sometimes handle values alone, "
              "but unclear associations may cause missed attributes."
            ),
          },
          {
            "kind": "note",
            "text": "Example: “Bus Amps: 225A” or “Volts: 480Y/277V”",
          },
          {
            "key": "traditional_wording",
            "text": "Traditional wording or common abbreviation. (higher confidence)",
            "weight": 6,
            "note": (
              "Unusual wording or uncommon abbreviations may cause Argus to miss important "
              "panel attributes. Traditional terms like Bus Amps, AIC, kAIC, Volts, and BRKR "
              "are more reliable."
            ),
          },
          {
            "kind": "note",
            "text": "Example: Bus Amps, AIC, kAIC, Volts, BRKR, etc.",
          },
          {
            "kind": "plain",
            "text": "Required panel attributes are included:",
          },
          {
            "key": "bus_amps",
            "text": "Bus Amps.",
            "weight": 10,
            "cap": 50,
            "note": (
              "Bus Amps is required. If it is missing from the drawings, Argus will move "
              "the panel to the problem panel section and the user will need to add it manually."
            ),
          },
          {
            "key": "main_breaker_amps",
            "text": "Main Breaker Amps, if applicable.",
            "weight": 0,
            "note": "",
          },
          {
            "key": "voltage",
            "text": "Voltage.",
            "weight": 10,
            "cap": 50,
            "note": (
              "Voltage is required. If it is missing from the drawings, the panel will need "
              "to be rescued manually before Argus can produce a confident result."
            ),
          },
          {
            "key": "kaic",
            "text": "kAIC Rating. (default is 22k if not found)",
            "weight": 0,
            "note": (
              "If kAIC is not found, Argus will default to 22k. This is not usually a major "
              "confidence issue, but the output should be reviewed."
            ),
          },
        ],
      },
      {
        "title": "Breaker Section",
        "rows": [
          {
            "key": "breaker_column_headers",
            "text": "Clear header labels for all columns.",
            "weight": 4,
            "note": (
              "Clear breaker column labels are required for confident breaker analysis. "
              "The panel part numbers may still be produced, but breakers may be missing."
            ),
          },
          {
            "key": "odd_even_mirrored",
            "text": "Odd and even circuits are mirrored left / right, and labeled.",
            "weight": 4,
            "note": (
              "Argus expects a standard odd/even breaker section layout. If the circuits "
              "are not mirrored left and right, breaker detection may be incomplete."
            ),
          },
          {
            "key": "amps_poles_one_row",
            "text": "Amps and poles are listed on one row for each breaker.",
            "weight": 4,
            "note": (
              "Breaker analysis is more reliable when amps and poles are listed on one row "
              "for each breaker. If not, panel results may still be created, but breaker "
              "outputs may be missing or incomplete."
            ),
          },
          {
            "key": "minimal_revision_clouds",
            "text": "Minimal revision clouds/obscured information.",
            "weight": 3,
            "note": (
              "Panels containing revision clouds or other markings that obscure information "
              "may have values that Argus cannot clearly read and might miss."
            ),
          },
        ],
      },
    ]

    def add_section(title, rows):
      body_panel.add_component(
        Label(
          text=title,
          bold=True,
          font_size=18,
          foreground="#117bc3"
        )
      )

      for row in rows:
        kind = row.get("kind", "check")

        if kind == "note":
          body_panel.add_component(
            Label(
              text=row.get("text", ""),
              font_size=13,
              italic=True,
              foreground="#666666"
            )
          )
        elif kind == "plain":
          body_panel.add_component(
            Label(
              text=row.get("text", ""),
              font_size=14,
            )
          )
        else:
          cb = CheckBox(
            text=row.get("text", ""),
            checked=False,
            font_size=14
          )
          checklist_boxes[row["key"]] = {
            "checkbox": cb,
            "text": row.get("text", ""),
            "weight": row.get("weight", 0),
            "note": row.get("note", ""),
            "cap": row.get("cap", None),
          }
          body_panel.add_component(cb)

    def calculate_confidence():
      score = 100
      warnings = []
      unchecked_keys = set()

      # Cumulative penalties.
      # These are intentionally direct percent drops instead of relative weights.
      scoring = {
        # Base / detection / layout foundation
        "trimmed_pdf": {
          "penalty": 3,
          "group": "base",
        },
        "schedules_not_touching": {
          "penalty": 55,
          "group": "base",
          "cap": 30,
        },
        "header_above_breakers": {
          "penalty": 50,
          "group": "base",
          "cap": 45,
        },
        "single_grid": {
          "penalty": 12,
          "group": "base",
        },
        "readable_font": {
          "penalty": 8,
          "group": "base",
        },

        # Header / base panel information
        "strong_label_value": {
          "penalty": 10,
          "group": "header",
        },
        "traditional_wording": {
          "penalty": 17,
          "group": "header",
        },
        "bus_amps": {
          "penalty": 35,
          "group": "header",
          "cap": 55,
        },
        "main_breaker_amps": {
          "penalty": 0,
          "group": "header",
        },
        "voltage": {
          "penalty": 35,
          "group": "header",
          "cap": 55,
        },
        "kaic": {
          "penalty": 0,
          "group": "header",
        },

        # Breaker extraction only
        "breaker_column_headers": {
          "penalty": 28,
          "group": "breaker",
        },
        "odd_even_mirrored": {
          "penalty": 18,
          "group": "breaker",
        },
        "amps_poles_one_row": {
          "penalty": 10,
          "group": "breaker",
        },
        "minimal_revision_clouds": {
          "penalty": 4,
          "group": "breaker",
        },
      }

      base_failures = 0
      header_failures = 0
      breaker_failures = 0
      score_caps = []

      for key, item in checklist_boxes.items():
        checked = bool(item["checkbox"].checked)

        if checked:
          continue

        unchecked_keys.add(key)

        note = item.get("note", "")
        if note:
          warnings.append(note)

        rule = scoring.get(key, {})
        penalty = int(rule.get("penalty", 0) or 0)
        group = rule.get("group", "")
        cap = rule.get("cap", None)

        score -= penalty

        if cap is not None:
          score_caps.append(cap)

        if group == "base" and penalty > 0:
          base_failures += 1
        elif group == "header" and penalty > 0:
          header_failures += 1
        elif group == "breaker" and penalty > 0:
          breaker_failures += 1

      # Hard structural logic.
      # These prevent a bad foundation from still looking decent.
      if (
        "schedules_not_touching" in unchecked_keys
        and "header_above_breakers" in unchecked_keys
      ):
        score_caps.append(10)

      # If multiple base/layout items fail, confidence should collapse fast.
      if base_failures >= 3:
        score_caps.append(20)
      elif base_failures >= 2:
        score_caps.append(35)

      # Missing both required panel attributes is worse than either one alone.
      if "bus_amps" in unchecked_keys and "voltage" in unchecked_keys:
        score_caps.append(35)

      # Breaker-only failures should not imply the whole panel is unusable.
      # It may still identify panels and build base panel part numbers,
      # but breaker output confidence should be capped.
      major_breaker_keys = {
        "breaker_column_headers",
        "odd_even_mirrored",
        "amps_poles_one_row",
      }
      major_breaker_failures = len(unchecked_keys.intersection(major_breaker_keys))

      if major_breaker_failures >= 3:
        score_caps.append(75)
      elif major_breaker_failures >= 2:
        score_caps.append(82)

      if score_caps:
        score = min(score, min(score_caps))

      score = max(0, min(100, int(round(score))))

      return score, warnings

    def show_confidence_modal(score, warnings):
      if score >= 80:
        icon = "✅"
        headline = "Likely high confidence outputs"
        color = "#2e7d32"
      elif score >= 55:
        icon = "⚠️"
        headline = "Medium confidence — review recommended"
        color = "#b26a00"
      else:
        icon = "❌"
        headline = "Low confidence — manual rescue likely"
        color = "#b00020"

      filled = int(round(score / 5.0))  # 20 blocks total
      empty = 20 - filled
      bar = ("█" * filled) + ("░" * empty)

      result_panel = ColumnPanel(spacing="medium")

      result_panel.add_component(
        Label(
          text=f"{icon} {headline}",
          bold=True,
          font_size=22,
          foreground=color
        )
      )

      result_panel.add_component(
        Label(
          text=f"{bar}  {score}%",
          font_size=20,
          foreground=color
        )
      )

      if warnings:
        result_panel.add_component(
          Label(
            text="Items to review:",
            bold=True,
            font_size=16
          )
        )

        for warning in warnings:
          result_panel.add_component(
            Label(
              text=f"• {warning}",
              font_size=14
            )
          )
      else:
        result_panel.add_component(
          Label(
            text="No compatibility warnings were found based on the selected checklist items.",
            font_size=14
          )
        )

      result_panel.add_component(
        Label(
          text=(
            "This is a compatibility estimate only. Final results may still vary based on "
            "drawing quality, formatting, and extracted information."
          ),
          font_size=13,
          italic=True,
          foreground="#666666"
        )
      )

      alert(
        content=result_panel,
        title="Will It Run?",
        buttons=[("Exit", True)],
        large=True,
        dismissible=True
      )

    def will_it_run_click(sender, **event_args):
      score, warnings = calculate_confidence()
      show_confidence_modal(score, warnings)

    def download_checklist_click(sender, **event_args):
      try:
        checklist_pdf = anvil.server.call("get_will_it_run_checklist_pdf")
        download(checklist_pdf)
      except Exception as e:
        alert(f"Could not download checklist:\n{e}")

    for section in checklist_sections:
      add_section(section["title"], section["rows"])

    body_panel.add_component(
      Label(
        text="*If the breaker analysis cannot be completed, you may still receive panel part numbers.",
        font_size=13,
        italic=True
      )
    )

    body_panel.add_component(
      Label(
        text='Review the “Coverage Map” for more information and deeper explanation.',
        font_size=13,
        italic=True
      )
    )

    action_panel = FlowPanel(spacing="medium")

    will_it_run_btn = Button(
      text="Will it run?",
      role="primary-color",
      icon="fa:check-circle"
    )
    will_it_run_btn.set_event_handler("click", will_it_run_click)
    action_panel.add_component(will_it_run_btn)

    download_btn = Button(
      text="Download Checklist",
      role="primary-color",
      icon="fa:download"
    )
    download_btn.set_event_handler("click", download_checklist_click)
    action_panel.add_component(download_btn)

    body_panel.add_component(action_panel)

    alert(
      content=body_panel,
      title="Quick Compatibility Checklist",
      buttons=[("Exit", True)],
      large=True,
      dismissible=True,
    )

  @handle("exclude_box", "change")
  def exclude_box_change(self, **event_args):
    """This method is called when this checkbox is checked or unchecked"""
    pass
