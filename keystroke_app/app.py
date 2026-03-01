from copy import deepcopy
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .capture import RunCapture, build_feature_vector_from_raw_run
from .config import (
    DATASET_VERSION,
    DEFAULT_DATASET_PATH,
    DEFAULT_DISTANCE_THRESHOLD,
    PROMPT_QUEUE_SIZE,
    SENTENCE_DATASET_PATH,
)
from .embedder_runtime import EmbedderRuntime, try_load_default_embedder_runtime
from .profile_manager import ProfileManager
from .prompts import PromptQueue, load_sentence_bank
from .storage import (
    SessionData,
    load_session_data,
    merge_session_data_file,
    save_session_data,
    set_feature_builder,
)
from .verifier import Verifier


SENTENCE_BANK = load_sentence_bank(SENTENCE_DATASET_PATH)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Keystroke Dynamics (Sentence Dataset) - Same vs Different")
        self.geometry("980x680")

        self.phase = "idle"  # idle | enroll | test | live_id | done
        self.phase_target_runs: Optional[int] = None

        self.prompt_queue = PromptQueue(SENTENCE_BANK, PROMPT_QUEUE_SIZE)
        self.capture = RunCapture()
        self.verifier = Verifier()
        self.profile_manager = ProfileManager()
        self.embedder_runtime: Optional[EmbedderRuntime] = None
        self.feature_backend_name: str = "handcrafted"
        self._embedder_startup_message: str = ""

        runtime, runtime_msg = try_load_default_embedder_runtime()
        if runtime is not None:
            self.embedder_runtime = runtime
            self.feature_backend_name = "embedder"
            self._embedder_startup_message = runtime_msg
            set_feature_builder(runtime.build_feature_vector_from_raw_run)
        else:
            self._embedder_startup_message = runtime_msg
            set_feature_builder(None)

        self.enroll_samples: List[np.ndarray] = []
        self.enroll_raw_runs: List[Dict[str, Any]] = []
        self.test_samples: List[np.ndarray] = []  # cumulative, persisted with dataset save
        self.test_raw_runs: List[Dict[str, Any]] = []
        self.test_run_samples: List[np.ndarray] = []  # current test window only
        self.test_run_raw_runs: List[Dict[str, Any]] = []  # current test window only

        self.latest_phase_name: Optional[str] = None  # "enroll" | "test"
        self.latest_phase_data: Optional[SessionData] = None

        self.profile_status_var = tk.StringVar(value="Loaded profiles: 0")

        self.live_id_state = "unknown"
        self.live_id_consecutive = 0
        self.live_id_sentence_count = 0
        self.live_id_votes: Dict[str, int] = {}
        self.live_id_unknown_closest: Dict[str, int] = {}

        self._build_ui()
        self._render_prompt_queue()
        self._set_status(
            f"Feature backend: {self.feature_backend_name}. Click 'Start Enrollment' or load a dataset."
        )
        if self._embedder_startup_message:
            self._log(self._embedder_startup_message)
        self._update_progress_ui()
        self._update_runs_label()
        self._set_idle_controls()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(
            frm,
            text="Type the top sentence exactly (including capitals and punctuation):",
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")

        self.target_box = tk.Text(frm, height=10, wrap="word", font=("Consolas", 12))
        self.target_box.pack(fill="x", pady=(6, 10))
        self.target_box.configure(state="disabled")

        ttk.Label(frm, text="Typing area:", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.input_box = tk.Text(frm, height=7, wrap="word", font=("Consolas", 12))
        self.input_box.pack(fill="both", expand=False, pady=(6, 10))
        self.input_box.tag_configure("mismatch", foreground="#c1121f")
        self.input_box.focus_set()

        self.input_box.bind("<KeyPress>", self._on_key_press)
        self.input_box.bind("<KeyRelease>", self._on_key_release)

        ctrl = ttk.Frame(frm)
        ctrl.pack(fill="x", pady=(4, 10))

        ttk.Label(ctrl, text="Enroll sentences:").pack(side="left")
        self.enroll_target_var = tk.StringVar(value="20")
        self.spn_enroll_target = ttk.Spinbox(
            ctrl,
            from_=1,
            to=500,
            increment=1,
            width=5,
            textvariable=self.enroll_target_var,
        )
        self.spn_enroll_target.pack(side="left", padx=(6, 10))

        self.btn_enroll = ttk.Button(ctrl, text="Start Enrollment", command=self.start_enroll)
        self.btn_enroll.pack(side="left")

        ttk.Label(ctrl, text="Test sentences:").pack(side="left", padx=(12, 0))
        self.test_target_var = tk.StringVar(value="10")
        self.spn_test_target = ttk.Spinbox(
            ctrl,
            from_=1,
            to=500,
            increment=1,
            width=5,
            textvariable=self.test_target_var,
        )
        self.spn_test_target.pack(side="left", padx=(6, 10))

        self.btn_test = ttk.Button(ctrl, text="Start Test", command=self.start_test, state="disabled")
        self.btn_test.pack(side="left", padx=(8, 0))

        self.btn_save_dataset = ttk.Button(
            ctrl,
            text="Save Dataset...",
            command=self.save_dataset,
            state="disabled",
        )
        self.btn_save_dataset.pack(side="left", padx=(8, 0))

        self.btn_load_dataset = ttk.Button(
            ctrl,
            text="Load Dataset...",
            command=self.load_dataset,
        )
        self.btn_load_dataset.pack(side="left", padx=(8, 0))

        self.btn_merge_dataset = ttk.Button(
            ctrl,
            text="Merge Dataset...",
            command=self.merge_dataset,
            state="disabled",
        )
        self.btn_merge_dataset.pack(side="left", padx=(8, 0))

        self.btn_reset = ttk.Button(ctrl, text="Reset", command=self.reset_all)
        self.btn_reset.pack(side="left", padx=(8, 0))

        multi = ttk.LabelFrame(frm, text="Multi-Profile Identification", padding=8)
        multi.pack(fill="both", expand=False, pady=(0, 10))

        ttk.Label(multi, textvariable=self.profile_status_var).pack(anchor="w")

        list_frame = ttk.Frame(multi)
        list_frame.pack(fill="x", pady=(6, 6))

        self.lst_profiles = tk.Listbox(list_frame, height=5, exportselection=False)
        self.lst_profiles.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.lst_profiles.yview)
        scrollbar.pack(side="left", fill="y")
        self.lst_profiles.configure(yscrollcommand=scrollbar.set)

        multi_btns = ttk.Frame(multi)
        multi_btns.pack(fill="x")

        self.btn_load_profiles = ttk.Button(multi_btns, text="Load Profiles...", command=self.load_profiles_dialog)
        self.btn_load_profiles.pack(side="left", padx=(0, 8))

        self.btn_clear_profiles = ttk.Button(multi_btns, text="Clear Profiles", command=self.clear_profiles)
        self.btn_clear_profiles.pack(side="left", padx=(0, 8))

        cfg = ttk.Frame(multi)
        cfg.pack(fill="x", pady=(4, 6))
        ttk.Label(cfg, text="Sentences:").pack(side="left")
        self.live_id_target_var = tk.StringVar(value="10")
        self.spn_live_id_target = ttk.Spinbox(
            cfg,
            from_=0,
            to=500,
            increment=1,
            width=5,
            textvariable=self.live_id_target_var,
        )
        self.spn_live_id_target.pack(side="left", padx=(6, 6))
        ttk.Label(cfg, text="(0 = unlimited)").pack(side="left")

        live_btns = ttk.Frame(multi)
        live_btns.pack(fill="x", pady=(4, 6))
        self.btn_live_id_start = ttk.Button(live_btns, text="Start Live ID", command=self.start_live_id, state="disabled")
        self.btn_live_id_start.pack(side="left", padx=(0, 8))
        self.btn_live_id_stop = ttk.Button(live_btns, text="Stop Live ID", command=self.stop_live_id, state="disabled")
        self.btn_live_id_stop.pack(side="left")

        self.lbl_live_id = ttk.Label(multi, text="Live: --", foreground="#555")
        self.lbl_live_id.pack(anchor="w")
        self.lbl_live_id_overall = ttk.Label(multi, text="Overall: --", foreground="#555")
        self.lbl_live_id_overall.pack(anchor="w")

        stats = ttk.Frame(frm)
        stats.pack(fill="x", pady=(6, 0))

        self.lbl_progress = ttk.Label(stats, text="Progress: --", font=("Segoe UI", 11, "bold"))
        self.lbl_progress.pack(side="left")

        self.lbl_runs = ttk.Label(stats, text="Runs: 0 enroll / 0 test", font=("Segoe UI", 11))
        self.lbl_runs.pack(side="left", padx=(16, 0))

        self.lbl_status = ttk.Label(frm, text="", font=("Segoe UI", 11))
        self.lbl_status.pack(fill="x", pady=(10, 0))

        self.output = tk.Text(frm, height=10, wrap="word", font=("Consolas", 11))
        self.output.pack(fill="both", expand=True, pady=(10, 0))
        self.output.configure(state="disabled")

    def _log(self, msg: str):
        self.output.configure(state="normal")
        self.output.insert("end", msg + "\n")
        self.output.see("end")
        self.output.configure(state="disabled")


    def _set_status(self, msg: str):
        self.lbl_status.configure(text=msg)

    def _build_run_feature(self, raw_run: Dict[str, Any]) -> np.ndarray:
        if self.embedder_runtime is not None:
            return self.embedder_runtime.build_feature_vector_from_raw_run(raw_run)
        return build_feature_vector_from_raw_run(raw_run)

    @staticmethod
    def _first_mismatch_index(typed: str, target: str) -> Optional[int]:
        limit = min(len(typed), len(target))
        for i in range(limit):
            if typed[i] != target[i]:
                return i
        if len(typed) > len(target):
            return len(target)
        return None

    def _update_input_mismatch_highlight(self, typed: str, target: str):
        self.input_box.tag_remove("mismatch", "1.0", "end")
        mismatch_idx = self._first_mismatch_index(typed, target)
        if mismatch_idx is None or mismatch_idx >= len(typed):
            return
        start = f"1.0+{mismatch_idx}c"
        end = f"1.0+{len(typed)}c"
        self.input_box.tag_add("mismatch", start, end)

    def _update_runs_label(self):
        self.lbl_runs.configure(text=f"Runs: {len(self.enroll_samples)} enroll / {len(self.test_samples)} test")

    def _current_phase_run_count(self) -> int:
        if self.phase == "enroll":
            return len(self.enroll_samples)
        if self.phase == "test":
            return len(self.test_run_samples)
        if self.phase == "live_id":
            return self.live_id_sentence_count
        return 0

    def _update_progress_ui(self):
        if self.phase in ("enroll", "test", "live_id") and self.phase_target_runs is not None:
            self.lbl_progress.configure(text=f"Progress: {self._current_phase_run_count()}/{self.phase_target_runs}")
        else:
            self.lbl_progress.configure(text="Progress: --")

    def _update_profile_listbox(self):
        if not hasattr(self, "lst_profiles"):
            return
        names = self.profile_manager.list_profiles()
        self.lst_profiles.delete(0, "end")
        for name in names:
            self.lst_profiles.insert("end", name)
        self.profile_status_var.set(f"Loaded profiles: {len(names)}")
        self._update_live_id_button_state()

    def _update_live_id_button_state(self):
        if not hasattr(self, "btn_live_id_start"):
            return
        can_start = self.phase in ("idle", "done") and len(self.profile_manager.list_profiles()) >= 1
        self.btn_live_id_start.configure(state="normal" if can_start else "disabled")
        self.btn_live_id_stop.configure(state="normal" if self.phase == "live_id" else "disabled")

    @staticmethod
    def _parse_target_runs(raw_value: str, label: str) -> int:
        try:
            target = int(raw_value)
        except Exception as ex:
            raise ValueError(f"{label} must be an integer.") from ex
        if target < 1:
            raise ValueError(f"{label} must be at least 1.")
        return target

    def _has_session_data(self) -> bool:
        if self.latest_phase_data is None:
            return False
        return bool(self.latest_phase_data.enrollment_samples) or bool(self.latest_phase_data.test_samples)

    def _clear_latest_phase_snapshot(self):
        self.latest_phase_name = None
        self.latest_phase_data = None

    def _set_latest_phase_snapshot(self, phase_name: str):
        mean = None
        inv_cov = None
        if self.verifier.mean is not None:
            mean = np.array(self.verifier.mean, dtype=np.float32, copy=True)
        if self.verifier.inv_cov is not None:
            inv_cov = np.array(self.verifier.inv_cov, dtype=np.float32, copy=True)
        distance_threshold = float(self.verifier.distance_threshold)

        if phase_name == "enroll":
            data = SessionData(
                enrollment_samples=[np.array(s, dtype=np.float32, copy=True) for s in self.enroll_samples],
                test_samples=[],
                enrollment_raw_runs=[deepcopy(run) for run in self.enroll_raw_runs],
                test_raw_runs=[],
                enrollment_mean=mean,
                enrollment_inv_cov=inv_cov,
                distance_threshold=distance_threshold,
            )
        elif phase_name == "test":
            data = SessionData(
                enrollment_samples=[],
                test_samples=[np.array(s, dtype=np.float32, copy=True) for s in self.test_run_samples],
                enrollment_raw_runs=[],
                test_raw_runs=[deepcopy(run) for run in self.test_run_raw_runs],
                enrollment_mean=mean,
                enrollment_inv_cov=inv_cov,
                distance_threshold=distance_threshold,
            )
        else:
            raise ValueError(f"Unsupported phase snapshot: {phase_name}")

        self.latest_phase_name = phase_name
        self.latest_phase_data = data

    def _set_running_controls(self):
        self.btn_enroll.configure(state="disabled")
        self.btn_test.configure(state="disabled")
        self.spn_enroll_target.configure(state="disabled")
        self.spn_test_target.configure(state="disabled")
        self.btn_save_dataset.configure(state="disabled")
        self.btn_load_dataset.configure(state="disabled")
        self.btn_merge_dataset.configure(state="disabled")
        self.btn_reset.configure(state="disabled")
        self.btn_load_profiles.configure(state="disabled")
        self.btn_clear_profiles.configure(state="disabled")
        self.btn_live_id_start.configure(state="disabled")
        self.btn_live_id_stop.configure(state="disabled")

    def _set_idle_controls(self):
        self.btn_enroll.configure(state="normal")
        self.btn_test.configure(state="normal" if self.verifier.has_reference() else "disabled")
        self.spn_enroll_target.configure(state="normal")
        self.spn_test_target.configure(state="normal")
        has_data = self._has_session_data()
        self.btn_save_dataset.configure(state="normal" if has_data else "disabled")
        self.btn_load_dataset.configure(state="normal")
        self.btn_merge_dataset.configure(state="normal" if has_data else "disabled")
        self.btn_reset.configure(state="normal")
        self.btn_load_profiles.configure(state="normal")
        self.btn_clear_profiles.configure(state="normal")
        self._update_profile_listbox()
        self._update_live_id_button_state()

    def _render_prompt_queue(self):
        self.target_box.configure(state="normal")
        self.target_box.delete("1.0", "end")
        self.target_box.insert("1.0", self.prompt_queue.as_text())
        self.target_box.configure(state="disabled")

    def _reset_prompt_queue(self):
        self.prompt_queue.reset()
        self._render_prompt_queue()

    def _clear_input(self):
        self.input_box.delete("1.0", "end")
        self.input_box.tag_remove("mismatch", "1.0", "end")
        self.capture.reset()

    def load_profiles_dialog(self):
        paths = filedialog.askopenfilenames(
            title="Load Profile Dataset(s)",
            initialdir=str(DEFAULT_DATASET_PATH.parent),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not paths:
            return

        loaded = []
        errors = []
        existing_names = set(self.profile_manager.list_profiles())

        for raw_path in paths:
            path = Path(raw_path)
            name = path.stem
            suffix = 2
            while name in existing_names:
                name = f"{path.stem}_{suffix}"
                suffix += 1
            try:
                self.profile_manager.load_profile(name, path)
                existing_names.add(name)
                loaded.append((name, path))
            except Exception as ex:
                errors.append(f"{path.name}: {ex}")

        self._update_profile_listbox()
        for name, path in loaded:
            self._log(f"Loaded profile '{name}' from {path.name}")
        if loaded:
            self._set_status(f"Loaded {len(loaded)} profile(s) for identification.")
        if errors:
            messagebox.showerror("Some profiles failed", "\n".join(errors))

    def clear_profiles(self):
        self.profile_manager.clear()
        self._update_profile_listbox()
        self._log("Cleared all profiles.")
        self._update_live_id_button_state()

    def _reset_live_id_state(self):
        self.live_id_state = "unknown"
        self.live_id_consecutive = 0
        self.live_id_sentence_count = 0
        self.live_id_votes.clear()
        self.live_id_unknown_closest.clear()
        self.lbl_live_id.configure(text="Live: --", foreground="#555")
        self.lbl_live_id_overall.configure(text="Overall: --", foreground="#555")

    def start_live_id(self):
        if self.phase not in ("idle", "done"):
            return
        if len(self.profile_manager.list_profiles()) < 1:
            messagebox.showwarning("Need profiles", "Load at least one profile before Live ID.")
            return
        try:
            target = int(self.live_id_target_var.get())
        except Exception as ex:
            messagebox.showwarning("Invalid setting", f"Sentences must be an integer.\n{ex}")
            return
        if target < 0:
            messagebox.showwarning("Invalid setting", "Sentences must be >= 0.")
            return

        self.phase = "live_id"
        self.phase_target_runs = target if target > 0 else None
        self._reset_live_id_state()
        self._reset_prompt_queue()
        self._set_running_controls()
        self.btn_live_id_stop.configure(state="normal")
        self._update_progress_ui()
        self._set_status("LIVE ID: Type sentences to identify typist. Click 'Stop Live ID' when done.")
        profiles = ", ".join(self.profile_manager.list_profiles())
        self._log(f"\n[Live ID started] profiles: {profiles if profiles else 'none'}")
        self._clear_input()
        self._update_live_id_button_state()

    def stop_live_id(self):
        if self.phase != "live_id":
            return
        self._finish_live_id(stopped=True)

    def _finish_live_id(self, stopped: bool):
        if self.phase != "live_id":
            return
        total = self.live_id_sentence_count
        self.phase = "idle"
        self.phase_target_runs = None
        overall_text, overall_color = self._compute_live_id_overall()
        self.lbl_live_id_overall.configure(text=overall_text, foreground=overall_color)
        reason = "stopped" if stopped else "finished"
        self._log(f"[Live ID {reason}] {total} sentence(s) typed")
        if total > 0:
            summary = overall_text.replace("Overall:  ", "")
            self._log(f"Final verdict: {summary}")
            self._set_status(f"Live ID result: {summary}")
        else:
            self._set_status("Live ID session ended (no sentences).")
        self._render_prompt_queue()
        self._clear_input()
        self._update_progress_ui()
        self._set_idle_controls()

    def _compute_live_id_overall(self) -> Tuple[str, str]:
        total = self.live_id_sentence_count
        if total == 0:
            return "Overall: --", "#555"

        total_identified = sum(self.live_id_votes.values())
        total_unknown = sum(self.live_id_unknown_closest.values())

        if total_identified > total_unknown and self.live_id_votes:
            overall_name = max(self.live_id_votes, key=self.live_id_votes.get)
            overall_count = self.live_id_votes[overall_name]
            return (
                f"Overall:  IDENTIFIED: {overall_name}  ({overall_count}/{total} sentences)",
                "#1b9c85",
            )

        if self.live_id_unknown_closest:
            overall_closest = max(self.live_id_unknown_closest, key=self.live_id_unknown_closest.get)
        else:
            overall_closest = "n/a"
        overall_count = total_unknown
        return (
            f"Overall:  UNKNOWN  —  closest: {overall_closest}  ({overall_count}/{total} sentences)",
            "#c1121f",
        )

    def _live_id_score(self, feat: np.ndarray):
        expected_dim = self.profile_manager.expected_feature_dim()
        if expected_dim is not None and feat.shape[0] != expected_dim:
            self._log("Rejected LIVE ID run: feature dimension mismatch with loaded profiles.")
            return

        X = feat.reshape(1, -1)
        try:
            best_name, best_distance, distances = self.profile_manager.identify(X)
        except Exception as ex:
            self._log(f"Live ID identify failed: {ex}")
            self._finish_live_id(stopped=True)
            return

        if not distances:
            return

        closest_name = min(distances, key=distances.get)
        closest_dist = float(distances[closest_name])

        self.live_id_sentence_count += 1
        sentence_idx = self.live_id_sentence_count

        if best_name is not None:
            self.live_id_votes[best_name] = self.live_id_votes.get(best_name, 0) + 1
        else:
            self.live_id_unknown_closest[closest_name] = self.live_id_unknown_closest.get(closest_name, 0) + 1

        new_state = "identified" if best_name is not None else "unknown"
        if new_state == self.live_id_state:
            self.live_id_consecutive += 1
        else:
            self.live_id_state = new_state
            self.live_id_consecutive = 1

        if self.live_id_consecutive >= 3 or sentence_idx == 1:
            display_dist = float(best_distance) if best_name is not None else closest_dist
            if self.live_id_state == "identified":
                label_text = f"Live:  IDENTIFIED: {best_name}  (dist: {display_dist:.2f})"
                label_color = "#1b9c85"
            else:
                label_text = f"Live:  UNKNOWN  —  closest: {closest_name} (dist: {closest_dist:.2f})"
                label_color = "#c1121f"
            self.lbl_live_id.configure(text=label_text, foreground=label_color)

        overall_text, overall_color = self._compute_live_id_overall()
        self.lbl_live_id_overall.configure(text=overall_text, foreground=overall_color)

        all_dists_str = " | ".join(f"{k}: {v:.2f}" for k, v in sorted(distances.items(), key=lambda x: x[1]))
        verdict = f"IDENTIFIED: {best_name}" if best_name is not None else f"UNKNOWN (closest: {closest_name})"
        self._log(f"[sentence {sentence_idx}]  {all_dists_str}  →  {verdict}")

        if self.phase_target_runs is not None and sentence_idx >= self.phase_target_runs:
            self._finish_live_id(stopped=False)

    def _apply_session_data(self, data: SessionData):
        self.enroll_samples = [np.array(s, dtype=np.float32, copy=True) for s in data.enrollment_samples]
        self.test_samples = [np.array(s, dtype=np.float32, copy=True) for s in data.test_samples]
        self.enroll_raw_runs = [deepcopy(run) for run in data.enrollment_raw_runs]
        self.test_raw_runs = [deepcopy(run) for run in data.test_raw_runs]
        self.test_run_samples.clear()
        self.test_run_raw_runs.clear()

        self.verifier.clear()
        self.verifier.distance_threshold = float(DEFAULT_DISTANCE_THRESHOLD)

        if len(self.enroll_samples) >= 3:
            try:
                X = np.stack(self.enroll_samples, axis=0)
                self.verifier.fit(X)
            except ValueError as ex:
                self._log(f"Verifier fit failed while loading dataset: {ex}")
        elif data.enrollment_mean is not None and data.enrollment_inv_cov is not None:
            # Fall back to stored stats if available (e.g., test-only snapshot).
            self.verifier.mean = np.array(data.enrollment_mean, dtype=np.float32, copy=True)
            self.verifier.inv_cov = np.array(data.enrollment_inv_cov, dtype=np.float32, copy=True)
            self.verifier._n_enrollment_runs = len(self.enroll_samples)

        self._update_runs_label()
        self._update_progress_ui()
        self._set_idle_controls()

    def reset_all(self):
        self.phase = "idle"
        self.phase_target_runs = None
        self.enroll_samples.clear()
        self.enroll_raw_runs.clear()
        self.test_samples.clear()
        self.test_raw_runs.clear()
        self.test_run_samples.clear()
        self.test_run_raw_runs.clear()
        self._clear_latest_phase_snapshot()
        self.capture.reset()
        self.verifier.clear()
        self.verifier.distance_threshold = float(DEFAULT_DISTANCE_THRESHOLD)
        self._reset_prompt_queue()
        self._clear_input()
        self._update_runs_label()
        self._update_progress_ui()
        self._set_idle_controls()
        self._set_status("Reset. Click 'Start Enrollment' or load a dataset.")
        self._log("\n[Reset]\n")

    def load_dataset(self):
        if self.phase in ("enroll", "test", "live_id"):
            messagebox.showwarning("Busy", "Stop the current run before loading a dataset.")
            return

        path_str = filedialog.askopenfilename(
            title="Load Dataset",
            initialdir=str(DEFAULT_DATASET_PATH.parent),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path_str:
            return

        path = Path(path_str)
        try:
            data = load_session_data(path)
        except Exception as ex:
            messagebox.showerror("Load failed", f"Could not load dataset:\n{ex}")
            return

        self._clear_latest_phase_snapshot()
        self._apply_session_data(data)
        self._reset_prompt_queue()
        self._clear_input()
        self._log(f"Loaded dataset from: {path}")
        self._set_status(f"Dataset loaded: {path.name}. Save/Merge now use next completed phase only.")

    def save_dataset(self):
        if not self._has_session_data():
            messagebox.showwarning("No data", "No recent phase data is available to save. Complete enrollment or test.")
            return

        path_str = filedialog.asksaveasfilename(
            title="Save Dataset",
            defaultextension=".json",
            initialdir=str(DEFAULT_DATASET_PATH.parent),
            initialfile=DEFAULT_DATASET_PATH.name,
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path_str:
            return

        path = Path(path_str)
        try:
            assert self.latest_phase_data is not None
            save_session_data(path, self.latest_phase_data, DATASET_VERSION)
        except Exception as ex:
            messagebox.showerror("Save failed", f"Could not save dataset:\n{ex}")
            return

        phase_label = self.latest_phase_name if self.latest_phase_name is not None else "phase"
        self._log(f"Saved latest {phase_label} dataset to: {path}")
        self._set_status(f"Dataset saved: {path.name}")
        self._set_idle_controls()

    def merge_dataset(self):
        if not self._has_session_data():
            messagebox.showwarning("No data", "No recent phase data is available to merge. Complete enrollment or test.")
            return

        path_str = filedialog.askopenfilename(
            title="Select Dataset JSON to Merge Into",
            initialdir=str(DEFAULT_DATASET_PATH.parent),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path_str:
            return

        path = Path(path_str)
        try:
            assert self.latest_phase_data is not None
            merged = merge_session_data_file(path, self.latest_phase_data, DATASET_VERSION)
            self._apply_session_data(merged)
        except Exception as ex:
            messagebox.showerror("Merge failed", f"Could not merge dataset:\n{ex}")
            return

        phase_label = self.latest_phase_name if self.latest_phase_name is not None else "phase"
        self._log(f"Merged latest {phase_label} dataset into: {path}")
        self._set_status(f"Merged into dataset: {path.name}")

    def start_enroll(self):
        try:
            target_runs = self._parse_target_runs(self.enroll_target_var.get(), "Enroll sentences")
        except Exception as ex:
            messagebox.showwarning("Invalid setting", str(ex))
            return

        self.reset_all()
        self.phase = "enroll"
        self.phase_target_runs = target_runs
        self._reset_prompt_queue()
        self._set_running_controls()
        self._update_progress_ui()
        self._set_status(f"ENROLLMENT: Type {target_runs} sentence(s) exactly.")
        self._log(f"[Enrollment started] target runs = {target_runs}")

    def start_test(self):
        if self.phase not in ("idle", "done"):
            return
        if not self.verifier.has_reference():
            messagebox.showwarning("Not enough enrollment", "Enroll first or load a dataset with enrollment runs.")
            return

        try:
            target_runs = self._parse_target_runs(self.test_target_var.get(), "Test sentences")
        except Exception as ex:
            messagebox.showwarning("Invalid setting", str(ex))
            return

        self.phase = "test"
        self.phase_target_runs = target_runs
        self.test_run_samples.clear()
        self.test_run_raw_runs.clear()
        self._reset_prompt_queue()
        self._set_running_controls()
        self._update_progress_ui()
        self._set_status(f"TEST: Type {target_runs} sentence(s) exactly; completed lines will rotate.")
        self._log(f"\n[Test started] target runs = {target_runs}")
        self._clear_input()

    def _finish_phase(self):
        if self.phase == "enroll":
            accepted = len(self.enroll_samples)
            self._log(f"[Enrollment finished] accepted runs = {accepted}")
            if accepted < 3:
                self._log("Need at least 3 enrollment runs to build a profile. Try again.")
                self._set_status("Enrollment needs at least 3 completed runs. Click 'Start Enrollment' and try again.")
                self.phase = "idle"
                self.phase_target_runs = None
                self._set_idle_controls()
                self._update_progress_ui()
                return

            X = np.stack(self.enroll_samples, axis=0)
            self.verifier.fit(X)
            self._set_latest_phase_snapshot("enroll")
            self._log(f"Enrollment profile built from {accepted} run(s).")
            self._set_status(f"Enrollment complete. Built profile from {accepted} runs.")
            self.phase = "idle"
            self._clear_input()
            self._set_idle_controls()

        elif self.phase == "test":
            self._log(f"[Test finished] accepted runs = {len(self.test_run_samples)}")
            if len(self.test_run_samples) < 1:
                self._log("No test runs captured. Try again and complete the prompt.")
                self._set_status("No test runs captured. Click 'Start Test' again and type continuously.")
                self.phase = "idle"
                self.phase_target_runs = None
                self._set_idle_controls()
                self._update_progress_ui()
                return

            Xtest = np.stack(self.test_run_samples, axis=0)
            try:
                distances, inlier = self.verifier.score(Xtest)
            except Exception as ex:
                self._log(f"Scoring failed: {ex}")
                self._set_status("Scoring failed due to reference/data mismatch. Re-enroll or load another dataset.")
                self.phase = "idle"
                self.phase_target_runs = None
                self._set_idle_controls()
                self._update_progress_ui()
                return

            inlier_frac = float(inlier.mean())
            avg_distance = float(distances.mean())
            min_distance = float(distances.min())
            max_distance = float(distances.max())

            self._log("\n--- RESULT ---")
            self._log(f"Avg Mahalanobis distance: {avg_distance:.4f} (lower = more like enrolled user)")
            self._log(
                f"Distance range: [{min_distance:.4f}, {max_distance:.4f}] "
                f"vs threshold {self.verifier.distance_threshold:.2f}"
            )
            self._log(f"Inlier fraction:    {inlier_frac:.2%} (runs classified as 'User A-like')")

            same = inlier_frac >= 0.60
            self._log("VERDICT: " + ("SAME PERSON (likely)" if same else "DIFFERENT PERSON (likely)"))

            self._set_latest_phase_snapshot("test")
            self._set_status("Test done. Save/Merge now use this test snapshot.")
            self.phase = "idle"
            self._set_idle_controls()

        self.phase_target_runs = None
        self._update_progress_ui()
        self._update_runs_label()

    def _maybe_accept_run(self):
        if self.phase not in ("enroll", "test", "live_id"):
            return
        typed = self.input_box.get("1.0", "end-1c")
        target = self.prompt_queue.current()
        self._update_input_mismatch_highlight(typed, target)

        if typed == target:
            raw_run = self.capture.build_raw_run()
            if raw_run is None:
                self._log("Run matched text but no usable timing data was captured; avoid paste and type the prompt.")
            else:
                feat = self._build_run_feature(raw_run)
                if self.phase == "enroll":
                    self.enroll_samples.append(feat)
                    self.enroll_raw_runs.append(deepcopy(raw_run))
                    self._log(f"Accepted ENROLL run #{len(self.enroll_samples)}")
                elif self.phase == "test":
                    if self.verifier.mean is not None and feat.shape[0] != self.verifier.mean.shape[0]:
                        self._log("Rejected TEST run: feature dimension mismatch with loaded reference.")
                    else:
                        self.test_run_samples.append(feat)
                        self.test_samples.append(feat)
                        self.test_run_raw_runs.append(deepcopy(raw_run))
                        self.test_raw_runs.append(deepcopy(raw_run))
                        self._log(
                            f"Accepted TEST run #{len(self.test_run_samples)} "
                            f"(total saved tests: {len(self.test_samples)})"
                        )
                elif self.phase == "live_id":
                    self._live_id_score(feat)

                self._update_runs_label()
                self._update_progress_ui()

            # Consume top line and append a new sentence at the bottom.
            self.prompt_queue.advance()
            self._render_prompt_queue()
            self._clear_input()
            if self.phase_target_runs is not None and self._current_phase_run_count() >= self.phase_target_runs:
                self._finish_phase()
            return

        if len(typed) > len(target) + 5:
            self._set_status("You overshot the prompt. Use Reset or backspace; aim to match exactly.")

    def _on_key_press(self, event):
        if self.phase not in ("enroll", "test", "live_id"):
            return

        # Capture all keydown events (including modifiers) for raw dataset logging.
        self.capture.on_key_press(event)

        if event.keysym in {"Shift_L", "Shift_R", "Control_L", "Control_R", "Alt_L", "Alt_R", "Caps_Lock"}:
            return

        if event.keysym == "BackSpace":
            # Allow corrections while still re-evaluating the run afterward.
            self.after(1, self._maybe_accept_run)
            return

        if event.keysym in {"Delete", "Left", "Right", "Up", "Down", "Home", "End", "Return"}:
            return "break"

        ch = event.char
        if ch == "":
            return "break"

        # Force append-at-end behavior even if the caret was moved manually.
        self.input_box.tag_remove("sel", "1.0", "end")
        self.input_box.mark_set("insert", "end-1c")
        self.after(1, self._maybe_accept_run)

    def _on_key_release(self, event):
        if self.phase not in ("enroll", "test", "live_id"):
            return
        self.capture.on_key_release(event)


def main():
    app = App()
    app.mainloop()
