"""Desktop notifications and the GitHub-token flag file (plan §9.2)."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("gnss.adsblol")

APP_ID = "GNSS pipeline"


def toast(title: str, message: str) -> bool:
    """Windows toast via winotify, then a PowerShell fallback, then log only."""
    if sys.platform != "win32":
        log.warning("notification: %s - %s", title, message)
        return False
    try:
        from winotify import Notification
        Notification(app_id=APP_ID, title=title, msg=message, duration="long").show()
        return True
    except Exception as e:
        log.info("winotify failed (%s); trying PowerShell toast", e)
    ps = (
        "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] > $null;"
        "$t = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent("
        "[Windows.UI.Notifications.ToastTemplateType]::ToastText02);"
        "$x = $t.GetElementsByTagName('text');"
        f"$x.Item(0).AppendChild($t.CreateTextNode('{_ps(title)}')) > $null;"
        f"$x.Item(1).AppendChild($t.CreateTextNode('{_ps(message)}')) > $null;"
        "$n = [Windows.UI.Notifications.ToastNotification]::new($t);"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('GNSS pipeline').Show($n)"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=20,
                       capture_output=True, check=True)
        return True
    except Exception as e:
        log.warning("toast failed (%s): %s - %s", e, title, message)
        return False


def _ps(s: str) -> str:
    return s.replace("'", "''")


TOKEN_HELP = """\
GNSS pipeline: a GitHub token is needed.

Reason:      {reason}
Raised at:   {raised}
Limit resets {reset} (the pipeline keeps working and retries API calls after the reset)

The pipeline keeps downloading and processing days whose files are already resolved. API calls
(release asset listing) wait for the hourly reset. To lift the limit to 5,000 requests/hour:

  1. https://github.com/settings/personal-access-tokens/new
     Fine-grained token, "Public repositories (read-only)", no extra permissions.
  2. Set it as a user environment variable (PowerShell):
         [Environment]::SetEnvironmentVariable("GITHUB_TOKEN", "<token>", "User")
  3. Nothing else: the running pipeline re-reads it at the next catalog pass and deletes this file
     once API calls succeed. The token is never logged, written to config, or committed.
"""


class TokenNotifier:
    def __init__(self, flag_path: Path, events=None, status=None):
        self.flag_path = flag_path
        self.events = events
        self.status = status
        self._notified_window: float | None = None

    def raise_needed(self, reason: str, reset_ts: float | None) -> None:
        reset = datetime.fromtimestamp(reset_ts).strftime("%Y-%m-%d %H:%M") if reset_ts else "unknown"
        self.flag_path.parent.mkdir(parents=True, exist_ok=True)
        self.flag_path.write_text(TOKEN_HELP.format(
            reason=reason, raised=datetime.now().strftime("%Y-%m-%d %H:%M"), reset=reset),
            encoding="utf-8")
        if self.status:
            self.status.flag("TOKEN NEEDED", f"resets {reset[-5:]}")
        window = reset_ts or time.time() // 3600
        if self._notified_window == window:
            return
        self._notified_window = window
        if self.events:
            self.events.warn(f"GitHub API rate limit reached; token needed (see {self.flag_path})")
        toast("GNSS pipeline: GitHub token needed", f"See {self.flag_path.name}")

    def clear(self) -> None:
        if self.flag_path.exists():
            self.flag_path.unlink()
            if self.events:
                self.events.info("GitHub API calls succeeding again; token flag cleared", True)
        if self.status:
            self.status.unflag("TOKEN NEEDED")
