from functools import partial
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

from train import TrainingSession


def make_session(run_path: Path, rounds: int, checkpoint_period: int) -> TrainingSession:
    session = TrainingSession(
        args=SimpleNamespace(profile=False),
        config=SimpleNamespace(
            knot_episodes=[0, rounds * 2],
            checkpoint_period=checkpoint_period,
            visualize=0,
            generation=SimpleNamespace(num_environments=2, num_iterations=1, num_devices=1),
            train=SimpleNamespace(fresh_pass_factor=1.0, batch_size=2),
        ),
        run_path=str(run_path),
        rooms=[],
        device=Mock(),
        generation_devices=[],
        engine=Mock(),
        train_batch_envs=[],
        main_model=Mock(),
        ema_model=Mock(),
        balance_model=Mock(),
        main_optimizer=Mock(),
        balance_optimizer=Mock(),
        loss_config=Mock(),
        experience=Mock(),
        train_batch_prefetcher=Mock(),
        generation_executors=[Mock()],
    )
    session.aim_run = Mock()
    data = Mock()
    data.to.return_value = data
    outcomes = Mock()
    outcomes.to.return_value = outcomes
    session.generate_round = Mock(return_value=(data, outcomes, Mock(), Mock(), Mock(), {}, []))
    session.train_round = Mock(return_value=(0.0, 0.0))
    session.log_outcomes = Mock()
    session.save_checkpoint = Mock()
    return session


def run_session(session: TrainingSession) -> None:
    step_config = SimpleNamespace(
        balance_train=SimpleNamespace(batch_size=2),
        train=SimpleNamespace(proposal_target_temperature=1.0),
    )
    with (
        patch("train.instantiate_scheduleable_config", return_value=step_config),
        patch("train.compute_balance_metric_values"),
        patch("train.compute_candidate_diagnostics"),
    ):
        session.run()


def stop_during_training(session: TrainingSession, signum: int, *_args) -> tuple[float, float]:
    session.request_stop(signum, None)
    return 0.0, 0.0


def wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + 30
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.01)


def wait_during_round(session: TrainingSession, directory: Path, result: tuple) -> tuple:
    (directory / "ready").touch()
    while not session.stop_requested:
        time.sleep(0.01)
    (directory / "stopping").touch()
    wait_for_file(directory / "finish-round")
    return result


def record_checkpoint(directory: Path, stage: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if stage == "save":
        (directory / "saving").touch()
        wait_for_file(directory / "finish-save")
    path.write_text("checkpoint completed")


def wait_in_worker(directory: Path, finished) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    (directory / "worker-ready").touch()
    finished.wait()


def run_signal_child(directory: Path, stage: str) -> None:
    session = make_session(directory, rounds=3, checkpoint_period=100)
    session.generate_round.side_effect = partial(
        wait_during_round, session, directory, session.generate_round.return_value
    )
    session.save_checkpoint.side_effect = partial(record_checkpoint, directory, stage)
    signal.signal(signal.SIGINT, session.request_stop)
    signal.signal(signal.SIGTERM, session.request_stop)
    context = multiprocessing.get_context("spawn")
    finished = context.Event()
    worker = context.Process(target=wait_in_worker, args=(directory, finished))
    worker.start()
    try:
        wait_for_file(directory / "worker-ready")
        run_session(session)
    finally:
        (directory / "cleanup").touch()
        finished.set()
        worker.join()


class TrainingShutdownTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.session = make_session(self.directory, rounds=3, checkpoint_period=100)

    def assert_resources_closed(self) -> None:
        self.session.train_batch_prefetcher.close.assert_called_once_with()
        self.session.generation_executors[0].shutdown.assert_called_once_with()
        self.session.aim_run.close.assert_called_once_with()

    def test_normal_exit_saves_latest_completed_round(self) -> None:
        run_session(self.session)
        self.assertEqual(self.session.num_episodes, 6)
        self.session.save_checkpoint.assert_called_once_with(self.session.checkpoint_path(3))
        self.assert_resources_closed()

    def test_periodic_checkpoint_is_not_repeated_on_exit(self) -> None:
        self.session.config.checkpoint_period = 1
        run_session(self.session)
        self.assertEqual(
            self.session.save_checkpoint.call_args_list,
            [call(self.session.checkpoint_path(round_idx)) for round_idx in (1, 2, 3)],
        )

    def test_stop_finishes_round_and_saves(self) -> None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signum=signum):
                session = make_session(self.directory, rounds=3, checkpoint_period=100)
                session.train_round.side_effect = partial(stop_during_training, session, signum)
                run_session(session)
                self.assertEqual(session.num_episodes, 2)
                session.generate_round.assert_called_once_with()
                session.experience.store.assert_called_once()
                session.save_checkpoint.assert_called_once_with(session.checkpoint_path(1))

    def test_stop_before_round_saves_without_starting_more_work(self) -> None:
        self.session.num_episodes = 2
        self.session.request_stop(signal.SIGINT, None)
        run_session(self.session)
        self.session.generate_round.assert_not_called()
        self.session.save_checkpoint.assert_called_once_with(self.session.checkpoint_path(1))

    def test_failed_partial_round_is_not_checkpointed(self) -> None:
        self.session.train_round.side_effect = RuntimeError("training failed")
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            run_session(self.session)
        self.session.save_checkpoint.assert_not_called()
        self.assert_resources_closed()

    def test_save_failure_still_closes_resources(self) -> None:
        self.session.save_checkpoint.side_effect = OSError("disk full")
        with self.assertRaisesRegex(OSError, "disk full"):
            run_session(self.session)
        self.assert_resources_closed()

    @unittest.skipUnless(os.name == "posix", "Requires POSIX process-group signals")
    def test_real_terminal_interrupts(self) -> None:
        for stage in ("single", "round", "save"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                process = subprocess.Popen(
                    [sys.executable, __file__, "--signal-child", str(directory), stage],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                try:
                    wait_for_file(directory / "ready")
                    os.killpg(process.pid, signal.SIGINT)
                    wait_for_file(directory / "stopping")
                    self.assertIsNone(process.poll())
                    if stage != "round":
                        (directory / "finish-round").touch()
                    if stage == "save":
                        wait_for_file(directory / "saving")
                    if stage != "single":
                        os.killpg(process.pid, signal.SIGINT)
                    # An orphan worker would keep these pipes open and cause a timeout.
                    stdout, stderr = process.communicate(timeout=10)
                    self.assertEqual(
                        process.returncode,
                        0 if stage == "single" else 130,
                        stdout + stderr,
                    )
                    self.assertEqual(
                        (directory / "checkpoints" / "round_1.safetensors").exists(),
                        stage == "single",
                    )
                    self.assertEqual((directory / "cleanup").exists(), stage == "single")
                finally:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.communicate(timeout=10)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--signal-child":
        run_signal_child(Path(sys.argv[2]), sys.argv[3])
    else:
        unittest.main()
