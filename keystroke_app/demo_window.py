from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .capture import RunCapture
from .profile_manager import ProfileManager
from .prompts import PromptQueue
from .touch_id import HAS_NATIVE_TOUCH_ID, request_touch_id


WINDOW_BG = "#0d0d0d"
PANEL_BG = "#141414"
ACCENT_GREEN = "#00ff88"
ACCENT_RED = "#ff3355"
ACCENT_YELLOW = "#ffcc00"
ACCENT_BLUE = "#4488ff"
ACCENT_GRAY = "#555555"
FONT_MONO = ("Menlo", 13)
FONT_MONO_LARGE = ("Menlo", 18, "bold")
FONT_MONO_HUGE = ("Menlo", 28, "bold")
FONT_UI = ("Helvetica Neue", 12)
FONT_UI_BOLD = ("Helvetica Neue", 13, "bold")
HYSTERESIS_THRESHOLD = 3
TOUCH_ID_TRIGGER_CONSECUTIVE = 3
SCORE_HISTORY_MAX = 30
LIVE_WINDOW_INTERVAL_MS = 6000
LIVE_WINDOW_MIN_PRINTABLE = 30


class DemoWindow(tk.Toplevel):
    def __init__(
        self,
        parent: tk.Tk,
        profile_manager: ProfileManager,
        sentence_bank: List[str],
        get_feature_fn: Callable[[Dict[str, Any]], np.ndarray],
    ):
        super().__init__(parent)
        self.profile_manager = profile_manager
        self.sentence_bank = sentence_bank
        self.get_feature_fn = get_feature_fn

        self.title("Behavioral Identity Firewall — Live Demo")
        self.configure(bg=WINDOW_BG)
        self.geometry("1200x800")
        self.minsize(1000, 700)
        self.resizable(True, True)

        self.capture = RunCapture()
        self.prompt_queue = PromptQueue(sentence_bank, 3)
        self.live_window_capture = RunCapture()

        self.demo_running = False
        self.phase = "idle"

        self.current_identity: Optional[str] = None
        self.current_closest: Optional[str] = None
        self.current_dist: float = 0.0
        self.all_distances: Dict[str, float] = {}

        self.hysteresis_state = "unknown"
        self.hysteresis_consecutive = 0
        self.displayed_state = "unknown"

        self.touch_id_triggered = False
        self.touch_id_in_progress = False
        self.consecutive_unknown_count = 0
        self.reauth_result: Optional[bool] = None

        self.sentence_count = 0
        self.session_votes: Dict[str, int] = {}
        self.score_history: List[Tuple[int, str, float, bool]] = []
        self.analysis_window_count = 0
        self.timed_window_count = 0
        self.live_window_job: Optional[str] = None

        self._profile_bar_widgets: Dict[str, Tuple[tk.Label, tk.Canvas, tk.Label]] = {}

        self._build_ui()
        self._bind_keys()
        self._update_idle_ui()
        self._poll_touch_id_result()

    def _build_ui(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        left = tk.Frame(self, bg=WINDOW_BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=(16, 8))
        left.rowconfigure(4, weight=1)

        hdr = tk.Label(
            left,
            text="BEHAVIORAL IDENTITY FIREWALL",
            bg=WINDOW_BG,
            fg=ACCENT_BLUE,
            font=("Helvetica Neue", 11, "bold"),
            anchor="w",
        )
        hdr.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        self.lbl_identity = tk.Label(
            left,
            text="● SYSTEM IDLE",
            bg=WINDOW_BG,
            fg=ACCENT_GRAY,
            font=FONT_MONO_HUGE,
            anchor="w",
            justify="left",
        )
        self.lbl_identity.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        self.lbl_subtitle = tk.Label(
            left,
            text="Load profiles and start demo to begin",
            bg=WINDOW_BG,
            fg=ACCENT_GRAY,
            font=FONT_MONO,
            anchor="w",
            justify="left",
        )
        self.lbl_subtitle.grid(row=2, column=0, sticky="ew", pady=(0, 16))

        bars_outer = tk.Frame(left, bg=PANEL_BG, bd=0)
        bars_outer.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        bars_outer.columnconfigure(0, weight=1)

        bars_header = tk.Label(
            bars_outer,
            text="PROFILE DISTANCES",
            bg=PANEL_BG,
            fg=ACCENT_GRAY,
            font=("Helvetica Neue", 10, "bold"),
            anchor="w",
        )
        bars_header.pack(anchor="w", padx=10, pady=(8, 4))

        self.bars_frame = tk.Frame(bars_outer, bg=PANEL_BG)
        self.bars_frame.pack(fill="x", padx=10, pady=(0, 8))

        graph_outer = tk.Frame(left, bg=PANEL_BG, bd=0)
        graph_outer.grid(row=4, column=0, sticky="nsew")
        graph_outer.rowconfigure(1, weight=1)
        graph_outer.columnconfigure(0, weight=1)

        graph_header = tk.Label(
            graph_outer,
            text="CONFIDENCE HISTORY",
            bg=PANEL_BG,
            fg=ACCENT_GRAY,
            font=("Helvetica Neue", 10, "bold"),
            anchor="w",
        )
        graph_header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))

        self.graph_canvas = tk.Canvas(graph_outer, bg=PANEL_BG, highlightthickness=0, height=140)
        self.graph_canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        self.graph_canvas.bind("<Configure>", lambda e: self._redraw_graph())

        right = tk.Frame(self, bg=WINDOW_BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=(16, 8))
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        ctrl = tk.Frame(right, bg=WINDOW_BG)
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        self.btn_start = tk.Button(
            ctrl,
            text="▶  START DEMO",
            bg=ACCENT_GREEN,
            fg="#000000",
            font=("Helvetica Neue", 12, "bold"),
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
            command=self.start_demo,
        )
        self.btn_start.pack(side="left", padx=(0, 8))

        self.btn_stop = tk.Button(
            ctrl,
            text="■  STOP",
            bg="#333333",
            fg="#ffffff",
            font=("Helvetica Neue", 12, "bold"),
            relief="flat",
            padx=16,
            pady=6,
            cursor="hand2",
            state="disabled",
            command=self.stop_demo,
        )
        self.btn_stop.pack(side="left")

        self.lbl_sentence_count = tk.Label(
            ctrl,
            text="sentences: 0",
            bg=WINDOW_BG,
            fg=ACCENT_GRAY,
            font=FONT_MONO,
        )
        self.lbl_sentence_count.pack(side="right")

        prompt_outer = tk.Frame(right, bg=PANEL_BG)
        prompt_outer.grid(row=1, column=0, sticky="ew", pady=(0, 8))

        prompt_lbl = tk.Label(
            prompt_outer,
            text="TYPE:",
            bg=PANEL_BG,
            fg=ACCENT_GRAY,
            font=("Helvetica Neue", 9, "bold"),
            anchor="w",
        )
        prompt_lbl.pack(anchor="w", padx=10, pady=(6, 0))

        self.lbl_prompt = tk.Label(
            prompt_outer,
            text="",
            bg=PANEL_BG,
            fg="#cccccc",
            font=("Menlo", 12),
            anchor="w",
            justify="left",
            wraplength=420,
        )
        self.lbl_prompt.pack(anchor="w", padx=10, pady=(2, 8))

        input_outer = tk.Frame(right, bg=PANEL_BG)
        input_outer.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
        input_outer.rowconfigure(1, weight=1)
        input_outer.columnconfigure(0, weight=1)

        input_lbl = tk.Label(
            input_outer,
            text="INPUT:",
            bg=PANEL_BG,
            fg=ACCENT_GRAY,
            font=("Helvetica Neue", 9, "bold"),
            anchor="w",
        )
        input_lbl.grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 0))

        self.input_box = tk.Text(
            input_outer,
            bg="#1a1a1a",
            fg="#ffffff",
            insertbackground="#ffffff",
            font=("Menlo", 13),
            relief="flat",
            padx=10,
            pady=8,
            wrap="word",
            height=5,
        )
        self.input_box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 8))
        self.input_box.tag_configure("mismatch", foreground=ACCENT_RED)
        self.input_box.focus_set()

        stats_outer = tk.Frame(right, bg=PANEL_BG)
        stats_outer.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        stats_lbl = tk.Label(
            stats_outer,
            text="SESSION VOTES",
            bg=PANEL_BG,
            fg=ACCENT_GRAY,
            font=("Helvetica Neue", 9, "bold"),
            anchor="w",
        )
        stats_lbl.pack(anchor="w", padx=10, pady=(6, 2))

        self.lbl_votes = tk.Label(
            stats_outer,
            text="—",
            bg=PANEL_BG,
            fg="#aaaaaa",
            font=FONT_MONO,
            anchor="w",
            justify="left",
        )
        self.lbl_votes.pack(anchor="w", padx=10, pady=(0, 8))

        profiles_outer = tk.Frame(right, bg=PANEL_BG)
        profiles_outer.grid(row=4, column=0, sticky="ew")

        profiles_lbl = tk.Label(
            profiles_outer,
            text="LOADED PROFILES",
            bg=PANEL_BG,
            fg=ACCENT_GRAY,
            font=("Helvetica Neue", 9, "bold"),
            anchor="w",
        )
        profiles_lbl.pack(anchor="w", padx=10, pady=(6, 2))

        self.lbl_profiles_list = tk.Label(
            profiles_outer,
            text="none",
            bg=PANEL_BG,
            fg="#aaaaaa",
            font=FONT_MONO,
            anchor="w",
            justify="left",
        )
        self.lbl_profiles_list.pack(anchor="w", padx=10, pady=(0, 8))

        status_bar = tk.Frame(self, bg="#111111", height=32)
        status_bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        status_bar.columnconfigure(0, weight=1)

        self.lbl_status_bar = tk.Label(
            status_bar,
            text="IDLE — load profiles in main window then click START DEMO",
            bg="#111111",
            fg=ACCENT_GRAY,
            font=("Helvetica Neue", 10),
            anchor="w",
        )
        self.lbl_status_bar.pack(side="left", padx=12, pady=6)

        self.lbl_backend = tk.Label(
            status_bar,
            text="backend: handcrafted",
            bg="#111111",
            fg="#444444",
            font=("Helvetica Neue", 10),
            anchor="e",
        )
        self.lbl_backend.pack(side="right", padx=12, pady=6)

    def _bind_keys(self):
        self.input_box.bind("<KeyPress>", self._on_key_press)
        self.input_box.bind("<KeyRelease>", self._on_key_release)

    def _set_status(self, msg: str):
        self.lbl_status_bar.configure(text=msg)

    def set_backend_label(self, backend_name: str):
        self.lbl_backend.configure(text=f"backend: {backend_name}")

    def _update_idle_ui(self):
        profiles = self.profile_manager.list_profiles()
        if profiles:
            self.lbl_profiles_list.configure(text="  ".join(profiles))
        else:
            self.lbl_profiles_list.configure(text="none — load profiles in main window")

    def start_demo(self):
        profiles = self.profile_manager.list_profiles()
        if not profiles:
            self._set_status("ERROR: No profiles loaded. Load profiles in the main window first.")
            return

        self.demo_running = True
        self.phase = "running"
        self.sentence_count = 0
        self.session_votes.clear()
        self.score_history.clear()
        self.analysis_window_count = 0
        self.timed_window_count = 0
        self.hysteresis_state = "unknown"
        self.hysteresis_consecutive = 0
        self.displayed_state = "unknown"
        self.consecutive_unknown_count = 0
        self.touch_id_triggered = False
        self.touch_id_in_progress = False
        self.reauth_result = None
        self.current_identity = None
        self.current_closest = None
        self.all_distances = {}
        self.live_window_capture.reset()

        self.prompt_queue.reset()
        self._update_prompt_label()
        self.input_box.delete("1.0", "end")
        self.capture.reset()

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")

        self.lbl_profiles_list.configure(text="  ".join(profiles))

        self._set_identity_display("running_no_data")
        self._set_status("DEMO RUNNING — type the prompt sentences naturally")
        self._update_profile_bars_init()
        self._start_live_window_loop()

    def stop_demo(self):
        self.demo_running = False
        self.phase = "idle"
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.input_box.delete("1.0", "end")
        self.capture.reset()
        self._cancel_live_window_loop()
        self.live_window_capture.reset()
        self._set_status(f"Demo stopped. {self.sentence_count} sentence(s) typed.")
        self._set_identity_display("idle")

    def _on_key_press(self, event):
        if not self.demo_running:
            return
        if self.phase != "running":
            self._start_live_window_loop()
            return
        self.capture.on_key_press(event)
        self.live_window_capture.on_key_press(event)

        if event.keysym in {
            "Shift_L",
            "Shift_R",
            "Control_L",
            "Control_R",
            "Alt_L",
            "Alt_R",
            "Caps_Lock",
        }:
            return

        if event.keysym == "BackSpace":
            self.after(1, self._maybe_accept_run)
            return

        if event.keysym in {
            "Delete",
            "Left",
            "Right",
            "Up",
            "Down",
            "Home",
            "End",
            "Return",
        }:
            return "break"

        ch = event.char
        if ch == "":
            return "break"

        self.input_box.tag_remove("sel", "1.0", "end")
        self.input_box.mark_set("insert", "end-1c")
        self.after(1, self._maybe_accept_run)

    def _on_key_release(self, event):
        if not self.demo_running or self.phase != "running":
            return
        self.capture.on_key_release(event)
        self.live_window_capture.on_key_release(event)

    def _maybe_accept_run(self):
        if not self.demo_running or self.phase != "running":
            return

        typed = self.input_box.get("1.0", "end-1c")
        target = self.prompt_queue.current()
        self._update_mismatch_highlight(typed, target)

        if typed == target:
            raw_run = self.capture.build_raw_run()
            if raw_run is not None:
                try:
                    feat = self.get_feature_fn(raw_run)
                except Exception as ex:
                    self._set_status(f"Scoring error: {ex}")
                else:
                    self._process_score(feat, source="sentence")

            self.prompt_queue.advance()
            self._update_prompt_label()
            self.input_box.delete("1.0", "end")
            self.input_box.tag_remove("mismatch", "1.0", "end")
            self.capture.reset()
            self.live_window_capture.reset()

    def _update_prompt_label(self):
        if self.demo_running:
            self.lbl_prompt.configure(text=self.prompt_queue.current())
        else:
            self.lbl_prompt.configure(text="")

    def _update_mismatch_highlight(self, typed: str, target: str):
        self.input_box.tag_remove("mismatch", "1.0", "end")
        limit = min(len(typed), len(target))
        mismatch_idx = None
        for i in range(limit):
            if typed[i] != target[i]:
                mismatch_idx = i
                break
        if mismatch_idx is None and len(typed) > len(target):
            mismatch_idx = len(target)
        if mismatch_idx is not None and mismatch_idx < len(typed):
            start = f"1.0+{mismatch_idx}c"
            end = f"1.0+{len(typed)}c"
            self.input_box.tag_add("mismatch", start, end)

    @staticmethod
    def _count_printable_events(events: List[Any]) -> int:
        count = 0
        for ev in events:
            if getattr(ev, "kind", "") != "keydown":
                continue
            ch = str(getattr(ev, "char", ""))
            if ch == " " or (ch and ch.isprintable()):
                count += 1
        return count

    def _start_live_window_loop(self):
        self._cancel_live_window_loop()
        self.live_window_job = self.after(LIVE_WINDOW_INTERVAL_MS, self._live_window_tick)

    def _cancel_live_window_loop(self):
        if self.live_window_job is not None:
            try:
                self.after_cancel(self.live_window_job)
            except Exception:
                pass
            self.live_window_job = None

    def _live_window_tick(self):
        self.live_window_job = None
        if not self.demo_running or self.phase != "running":
            return

        printable = self._count_printable_events(self.live_window_capture.events)
        if printable >= LIVE_WINDOW_MIN_PRINTABLE:
            raw_run = self.live_window_capture.build_raw_run()
            if raw_run is not None:
                try:
                    feat = self.get_feature_fn(raw_run)
                except Exception as ex:
                    self._set_status(f"Live window scoring error: {ex}")
                else:
                    self._process_score(feat, source="window")
            self.live_window_capture.reset()

        self._start_live_window_loop()

    def _process_score(self, feat: np.ndarray, source: str):
        X = feat.reshape(1, -1)
        try:
            best_name, best_dist, all_dists = self.profile_manager.identify(X)
        except Exception as ex:
            self._set_status(f"Identify error: {ex}")
            return

        if not all_dists:
            return

        closest_name = min(all_dists, key=all_dists.get)
        closest_dist = float(all_dists[closest_name])

        if source == "sentence":
            self.sentence_count += 1
            log_label = f"sentence {self.sentence_count}"
        else:
            self.timed_window_count += 1
            log_label = f"window {self.timed_window_count}"
        self.analysis_window_count += 1

        self.current_closest = closest_name
        self.current_dist = closest_dist
        self.all_distances = dict(all_dists)

        if best_name is not None:
            self.current_identity = best_name
            if source == "sentence":
                self.session_votes[best_name] = self.session_votes.get(best_name, 0) + 1
        else:
            self.current_identity = None

        is_identified = best_name is not None
        self.score_history.append((self.analysis_window_count, closest_name, closest_dist, is_identified))
        if len(self.score_history) > SCORE_HISTORY_MAX:
            self.score_history.pop(0)

        new_state = "identified" if is_identified else "unknown"
        if new_state == self.hysteresis_state:
            self.hysteresis_consecutive += 1
        else:
            self.hysteresis_state = new_state
            self.hysteresis_consecutive = 1

        if self.hysteresis_consecutive >= HYSTERESIS_THRESHOLD or self.analysis_window_count == 1:
            self.displayed_state = self.hysteresis_state

        if self.displayed_state == "unknown":
            self.consecutive_unknown_count += 1
        else:
            self.consecutive_unknown_count = 0

        if (
            self.consecutive_unknown_count >= TOUCH_ID_TRIGGER_CONSECUTIVE
            and not self.touch_id_in_progress
            and not self.touch_id_triggered
        ):
            self._trigger_touch_id()

        self._update_identity_display()
        self._update_profile_bars()
        if source == "sentence":
            self._update_votes_label()
            self._update_sentence_count_label()
        self._redraw_graph()

        all_dists_str = " | ".join(f"{k}: {v:.2f}" for k, v in sorted(all_dists.items(), key=lambda x: x[1]))
        verdict = f"IDENTIFIED: {best_name}" if best_name is not None else f"UNKNOWN (closest: {closest_name})"
        self._log(f"[{log_label}]  {all_dists_str}  →  {verdict}")

    def _set_identity_display(self, mode: str):
        if mode == "idle":
            self.lbl_identity.configure(text="● SYSTEM IDLE", fg=ACCENT_GRAY)
            self.lbl_subtitle.configure(text="Load profiles and start demo to begin", fg=ACCENT_GRAY)
        elif mode == "running_no_data":
            self.lbl_identity.configure(text="◌ CALIBRATING...", fg=ACCENT_YELLOW)
            self.lbl_subtitle.configure(text="Type a few sentences to build confidence", fg=ACCENT_GRAY)
        elif mode == "verifying":
            self.lbl_identity.configure(text="⟳ VERIFYING IDENTITY", fg=ACCENT_YELLOW)
            self.lbl_subtitle.configure(text="Touch ID authentication requested", fg=ACCENT_YELLOW)
        elif mode == "auth_success":
            self.lbl_identity.configure(text="✓ IDENTITY CONFIRMED", fg=ACCENT_GREEN)
            self.lbl_subtitle.configure(text="Touch ID authentication successful", fg=ACCENT_GREEN)
        elif mode == "auth_failed":
            self.lbl_identity.configure(text="✗ AUTHENTICATION FAILED", fg=ACCENT_RED)
            self.lbl_subtitle.configure(text="Touch ID failed — session flagged", fg=ACCENT_RED)

    def _update_identity_display(self):
        if self.analysis_window_count == 0:
            self._set_identity_display("running_no_data")
            return
        if self.touch_id_in_progress:
            return

        if self.displayed_state == "identified" and self.current_identity is not None:
            self.lbl_identity.configure(
                text=f"● IDENTIFIED: {self.current_identity.upper()}",
                fg=ACCENT_GREEN,
            )
            summary = f"{self.analysis_window_count} window(s) analyzed"
            if self.sentence_count:
                summary += f", {self.sentence_count} sentence(s)"
            self.lbl_subtitle.configure(text=f"dist {self.current_dist:.2f} — {summary}", fg="#888888")
        else:
            subtitle_name = self.current_closest or "?"
            self.lbl_identity.configure(text="● UNKNOWN TYPIST", fg=ACCENT_RED)
            summary = f"{self.analysis_window_count} window(s)"
            if self.sentence_count:
                summary += f", {self.sentence_count} sentence(s)"
            self.lbl_subtitle.configure(
                text=f"closest: {subtitle_name} (dist {self.current_dist:.2f}) — {summary}",
                fg=ACCENT_RED,
            )

    def _update_profile_bars_init(self):
        for widget in self.bars_frame.winfo_children():
            widget.destroy()
        self._profile_bar_widgets.clear()

        for name in self.profile_manager.list_profiles():
            row_frame = tk.Frame(self.bars_frame, bg=PANEL_BG)
            row_frame.pack(fill="x", pady=2)
            row_frame.columnconfigure(1, weight=1)

            name_lbl = tk.Label(
                row_frame,
                text=name[:12].ljust(12),
                bg=PANEL_BG,
                fg="#aaaaaa",
                font=("Menlo", 11),
                width=12,
                anchor="w",
            )
            name_lbl.grid(row=0, column=0, sticky="w", padx=(0, 8))

            bar_canvas = tk.Canvas(row_frame, bg="#222222", highlightthickness=0, height=16)
            bar_canvas.grid(row=0, column=1, sticky="ew")

            dist_lbl = tk.Label(
                row_frame,
                text="—",
                bg=PANEL_BG,
                fg="#888888",
                font=("Menlo", 11),
                width=8,
                anchor="e",
            )
            dist_lbl.grid(row=0, column=2, sticky="e", padx=(8, 0))

            self._profile_bar_widgets[name] = (name_lbl, bar_canvas, dist_lbl)

    def _update_profile_bars(self):
        if not self.all_distances:
            return
        max_display = 15.0
        threshold = 4.5

        for name, widgets in self._profile_bar_widgets.items():
            name_lbl, bar_canvas, dist_lbl = widgets
            dist = self.all_distances.get(name, 0.0)
            dist_lbl.configure(text=f"{dist:.2f}")

            bar_canvas.update_idletasks()
            width = max(2, bar_canvas.winfo_width())
            bar_canvas.delete("all")

            bar_canvas.create_rectangle(0, 0, width, 16, fill="#222222", outline="")
            fill_frac = min(1.0, dist / max_display)
            fill_w = int(fill_frac * width)
            is_closest = name == self.current_closest
            color = ACCENT_GREEN if dist <= threshold else (ACCENT_RED if is_closest else "#884444")
            if fill_w > 0:
                bar_canvas.create_rectangle(0, 2, fill_w, 14, fill=color, outline="")
            thresh_x = int((threshold / max_display) * width)
            bar_canvas.create_line(thresh_x, 0, thresh_x, 16, fill=ACCENT_YELLOW, width=1)
            name_lbl.configure(fg="#ffffff" if is_closest else "#666666")

    def _redraw_graph(self):
        canvas = self.graph_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width < 10 or height < 10:
            return

        if not self.score_history:
            canvas.create_text(
                width // 2,
                height // 2,
                text="no data yet",
                fill=ACCENT_GRAY,
                font=("Menlo", 11),
            )
            return

        max_display = 12.0
        threshold = 4.5
        thresh_y = height - int((threshold / max_display) * height)
        canvas.create_line(0, thresh_y, width, thresh_y, fill=ACCENT_YELLOW, width=1, dash=(4, 4))
        canvas.create_text(
            width - 4,
            thresh_y - 6,
            text="threshold",
            fill=ACCENT_YELLOW,
            font=("Menlo", 8),
            anchor="e",
        )

        points: List[Tuple[int, int]] = []
        n = len(self.score_history)
        for idx, (_, _, dist, _) in enumerate(self.score_history):
            x = int(idx / max(1, n - 1) * width)
            y = height - int(min(1.0, dist / max_display) * height)
            y = max(2, min(height - 2, y))
            points.append((x, y))

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            color = ACCENT_GREEN if self.score_history[i][3] else ACCENT_RED
            canvas.create_line(x1, y1, x2, y2, fill=color, width=2)

        for idx, (x, y) in enumerate(points):
            color = ACCENT_GREEN if self.score_history[idx][3] else ACCENT_RED
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")

        last_x, last_y = points[-1]
        last_dist = self.score_history[-1][2]
        canvas.create_text(
            last_x,
            last_y - 12,
            text=f"{last_dist:.2f}",
            fill="#ffffff",
            font=("Menlo", 9),
            anchor="s",
        )

    def _update_votes_label(self):
        if not self.session_votes:
            self.lbl_votes.configure(text="—")
            return
        total = self.sentence_count or 1
        parts = []
        for name, count in sorted(self.session_votes.items(), key=lambda x: x[1], reverse=True):
            pct = int(count / total * 100)
            parts.append(f"{name}: {count}/{total} ({pct}%)")
        self.lbl_votes.configure(text="   ".join(parts))

    def _update_sentence_count_label(self):
        self.lbl_sentence_count.configure(text=f"sentences: {self.sentence_count}")

    def _trigger_touch_id(self):
        self.touch_id_in_progress = True
        self.touch_id_triggered = True
        self.phase = "reauth"
        self._set_identity_display("verifying")
        self._set_status("SECURITY ALERT: Unknown typist detected — Touch ID required")

        if HAS_NATIVE_TOUCH_ID:
            thread = threading.Thread(target=self._run_touch_id_request, daemon=True)
            thread.start()
        else:
            # Run synchronously on the Tk thread so fallback dialogs are safe.
            self.after(50, self._run_touch_id_request)

    def _run_touch_id_request(self):
        try:
            result = request_touch_id("Behavioral Identity Firewall: re-authenticate to continue")
        except Exception as ex:
            self._set_status(f"Touch ID error: {ex}")
            result = False
        self.reauth_result = result

    def _poll_touch_id_result(self):
        if self.reauth_result is not None:
            success = self.reauth_result
            self.reauth_result = None
            self.touch_id_in_progress = False

            if success:
                self._set_identity_display("auth_success")
                self._set_status("Touch ID successful — session resumed")
                self.phase = "running"
                self.hysteresis_state = "identified"
                self.hysteresis_consecutive = HYSTERESIS_THRESHOLD
                self.displayed_state = "identified"
                self.consecutive_unknown_count = 0
                self.touch_id_triggered = False
                self.after(2000, self._update_identity_display)
            else:
                self._set_identity_display("auth_failed")
                self._set_status("Touch ID FAILED — session flagged as unauthorized")
                self.phase = "running"
                self.after(5000, self._reset_touch_id_trigger)

        self.after(200, self._poll_touch_id_result)

    def _reset_touch_id_trigger(self):
        self.touch_id_triggered = False
        self.consecutive_unknown_count = 0

    def destroy(self):
        self._cancel_live_window_loop()
        super().destroy()
