"""Reinforcement learning training - migrated from run_reinforcement_learning.py"""
import base64
import cProfile
import glob
import io
import os.path
import pstats
import signal
import subprocess
import time
from datetime import datetime

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from autogamen.ai.mlp import Net, VectorMLPPlayer
from autogamen.game import vector_game as vg
from autogamen.game.board import Board
from autogamen.game.game_types import Color, Dice, Point


def _fmt_percent(p: float) -> str:
    return f"{p:.1%}"


def get_checkpoint_schedule(max_games: int) -> list[int]:
    """generate checkpoint schedule: 1k, 2k, 4k, 10k, 20k, 50k, 100k, 150k, 200k, ..."""
    schedule = []
    # pattern: 1, 2, 5 (within each magnitude), then step up
    for magnitude in [1_000, 10_000, 100_000, 1_000_000]:
        for multiplier in [1, 2, 5]:
            checkpoint = magnitude * multiplier
            if checkpoint <= max_games:
                schedule.append(checkpoint)
            else:
                return schedule
    return schedule


def find_latest_training_run() -> str | None:
    """scan out/train/ and return the latest tag (alphabetically sorted) that has checkpoints"""
    train_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'out', 'train')
    if not os.path.exists(train_base):
        return None

    # get all subdirectories
    tags = [d for d in os.listdir(train_base) if os.path.isdir(os.path.join(train_base, d))]
    if not tags:
        return None

    # sort alphabetically (timestamps will sort correctly)
    tags.sort()

    # return the latest tag that has at least one .torch file
    for tag in reversed(tags):
        tag_dir = os.path.join(train_base, tag)
        if glob.glob(os.path.join(tag_dir, "*.torch")):
            return tag

    return None


def net_directory(tag: str) -> str:
    """Get the directory for storing network files for a specific training run"""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'out', 'train', tag)


def train_directory(tag: str) -> str:
    """Get the directory for storing training reports for a specific training run"""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'out', 'train', tag)


def clean_offcycle_checkpoints(run_dir: str, checkpoint_schedule: list[int]) -> None:
    """delete off-cycle checkpoints (not in the schedule)"""
    all_checkpoints = glob.glob(os.path.join(run_dir, "net-*.torch"))
    schedule_set = set(checkpoint_schedule)

    for checkpoint_path in all_checkpoints:
        # extract game count from filename: net-0001337.torch -> 1337
        basename = os.path.basename(checkpoint_path)
        if basename.startswith("net-") and basename.endswith(".torch"):
            game_count_str = basename[4:-6]  # strip "net-" and ".torch"
            try:
                game_count = int(game_count_str)
                if game_count not in schedule_set:
                    os.remove(checkpoint_path)
            except ValueError:
                # malformed filename, skip it
                pass


def write_checkpoint(path: str, net: "Net", game_count: int, run_dir: str, checkpoint_schedule: list[int]) -> None:
    """Save training checkpoint and clean up off-cycle checkpoints"""
    torch.save({
        'model_state_dict': net.state_dict(),
        'eligibility': net.eligibility_traces,
        'game_count': game_count,
        'time': datetime.now().isoformat()
    }, path)

    # clean up any off-cycle checkpoints
    clean_offcycle_checkpoints(run_dir, checkpoint_schedule)


def load_checkpoint(path: str) -> tuple["Net", dict]:
    """Load checkpoint. Returns (Net, checkpoint_raw_info)"""
    checkpoint = torch.load(path)
    n = Net()
    n.eligibility_traces = checkpoint['eligibility']
    n.load_state_dict(checkpoint['model_state_dict'])
    return n, checkpoint


def _setup_board() -> Board:
    """create standard backgammon starting position"""
    return Board([
        Point(2, Color.White),
        Point(),
        Point(),
        Point(),
        Point(),
        Point(5, Color.Black),
        Point(),
        Point(3, Color.Black),
        Point(),
        Point(),
        Point(),
        Point(5, Color.White),
        Point(5, Color.Black),
        Point(),
        Point(),
        Point(),
        Point(3, Color.White),
        Point(),
        Point(5, Color.White),
        Point(),
        Point(),
        Point(),
        Point(),
        Point(2, Color.Black),
    ])


class GameMetrics:
    """Metrics collected during a single game"""
    def __init__(self) -> None:
        self.turn_count = 0
        self.td_errors: list[torch.Tensor] = []

    def avg_td_error(self) -> float:
        return sum(abs(e.item()) for e in self.td_errors) / len(self.td_errors) if self.td_errors else 0.0


def run_game(white: "VectorMLPPlayer", black: "VectorMLPPlayer", net: "Net") -> GameMetrics:
    """run a single training game using vector_game operations.

    both players share the same net and use batched evaluation.
    """
    metrics = GameMetrics()

    # intercept TD errors for metrics
    original_update = net.update_weights
    def wrapped_update(p: torch.Tensor, p_next: torch.Tensor) -> torch.Tensor:
        td_error = original_update(p, p_next)
        metrics.td_errors.append(td_error)
        return td_error

    net.update_weights = wrapped_update  # type: ignore[method-assign]

    # reset eligibility traces at start of game
    white.reset_eligibility_traces()
    black.reset_eligibility_traces()

    # initialize board vector
    vec = vg.from_board(_setup_board())

    # determine starting player
    starting_dice = Dice()
    while starting_dice.roll[0] == starting_dice.roll[1]:
        starting_dice = Dice()

    active_color = Color.White if starting_dice.roll[0] > starting_dice.roll[1] else Color.Black
    active_player = white if active_color == Color.White else black

    turn_number = 0

    while True:
        # switch to next player
        active_color = active_color.opponent()
        active_player = white if active_color == Color.White else black

        # roll dice
        dice = Dice()
        turn_number += 1

        # player chooses and applies best move
        _, vec = active_player.choose_move(vec, active_color, dice.effective_roll())

        # check for winner
        winner = vg.winner(vec)
        if winner is not None:
            metrics.turn_count = turn_number
            break

    # restore original method
    net.update_weights = original_update  # type: ignore[method-assign]
    return metrics


class TrainingMetrics:
    """Metrics collected during training"""
    def __init__(self) -> None:
        self.game_lengths: list[int] = []
        self.td_errors: list[float] = []
        self.weight_norms: list[float] = []
        self.timestamps: list[float] = []


def generate_html_report(tag: str, metrics: TrainingMetrics, games: int) -> str:
    """Generate HTML report with matplotlib charts (black background).

    Returns the path to the generated report.
    """
    # Set matplotlib style for black background
    plt.style.use('dark_background')

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    fig.patch.set_facecolor('black')

    game_numbers = list(range(1, len(metrics.game_lengths) + 1))

    # Chart 1: Game Length over Time
    ax = axes[0]
    ax.set_facecolor('black')
    ax.plot(game_numbers, metrics.game_lengths, color='cyan', linewidth=0.5, alpha=0.5)
    # Add rolling average
    window = min(50, len(metrics.game_lengths) // 10)
    if window > 1:
        rolling_avg = [sum(metrics.game_lengths[max(0, i-window):i+1]) / min(window, i+1)
                      for i in range(len(metrics.game_lengths))]
        ax.plot(game_numbers, rolling_avg, color='yellow', linewidth=2, label=f'{window}-game avg')
        ax.legend()
    ax.set_xlabel('Game Number', color='white')
    ax.set_ylabel('Game Length (turns)', color='white')
    ax.set_title('Game Length Over Time', color='white', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)

    # Chart 2: TD Error over Time
    ax = axes[1]
    ax.set_facecolor('black')
    ax.plot(game_numbers, metrics.td_errors, color='lime', linewidth=0.5, alpha=0.5)
    # Add rolling average
    if window > 1:
        rolling_avg = [sum(metrics.td_errors[max(0, i-window):i+1]) / min(window, i+1)
                      for i in range(len(metrics.td_errors))]
        ax.plot(game_numbers, rolling_avg, color='orange', linewidth=2, label=f'{window}-game avg')
        ax.legend()
    ax.set_xlabel('Game Number', color='white')
    ax.set_ylabel('Average TD Error', color='white')
    ax.set_title('TD Error Over Time', color='white', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)

    # Chart 3: Weight Norm over Time
    ax = axes[2]
    ax.set_facecolor('black')
    ax.plot(game_numbers, metrics.weight_norms, color='magenta', linewidth=1)
    ax.set_xlabel('Game Number', color='white')
    ax.set_ylabel('Total Weight Norm', color='white')
    ax.set_title('Network Weight Norm Over Time', color='white', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='black', edgecolor='none', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>TD-GAMMON Training Report - {tag}</title>
    <style>
        body {{
            background-color: #000000;
            color: #ffffff;
            font-family: 'Courier New', monospace;
            margin: 0;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            border-bottom: 2px solid #00ff00;
            margin-bottom: 30px;
        }}
        h1 {{
            color: #00ff00;
            margin: 0;
        }}
        .timestamp {{
            color: #888888;
            font-size: 14px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-box {{
            background-color: #1a1a1a;
            border: 1px solid #333333;
            padding: 15px;
            border-radius: 5px;
        }}
        .stat-label {{
            color: #888888;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .stat-value {{
            color: #00ff00;
            font-size: 24px;
            font-weight: bold;
            margin-top: 5px;
        }}
        .charts {{
            margin-top: 30px;
        }}
        img {{
            width: 100%;
            height: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>TD-GAMMON Training Report</h1>
        <div class="timestamp">Run: {tag}</div>
    </div>

    <div class="stats">
        <div class="stat-box">
            <div class="stat-label">Total Games</div>
            <div class="stat-value">{games:,}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Avg Game Length</div>
            <div class="stat-value">{sum(metrics.game_lengths) / len(metrics.game_lengths):.1f}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Final Game Length</div>
            <div class="stat-value">{metrics.game_lengths[-1]}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Avg TD Error</div>
            <div class="stat-value">{sum(metrics.td_errors) / len(metrics.td_errors):.4f}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Total Time</div>
            <div class="stat-value">{metrics.timestamps[-1]:.1f}s</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Time per Game</div>
            <div class="stat-value">{metrics.timestamps[-1] / games:.3f}s</div>
        </div>
    </div>

    <div class="charts">
        <img src="data:image/png;base64,{img_base64}" alt="Training Metrics">
    </div>
</body>
</html>
"""

    # Write to file (overwrite previous report with latest data)
    report_path = os.path.join(train_directory(tag), "report.html")
    with open(report_path, 'w') as f:
        f.write(html)

    return report_path


def run_reinforcement_learning(
    games: int,
    tag: str | None,
    alpha: float,
    profile: bool,
) -> tuple[str, TrainingMetrics, "Net"]:
    """Main entry point for reinforcement learning training.

    Returns (tag, metrics, net) for report generation.
    """
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    # determine tag: explicit, or auto-resume from latest, or create new
    if tag is None:
        latest_tag = find_latest_training_run()
        if latest_tag:
            tag = latest_tag
            print(f"ℹ resuming from latest training run: {tag}")
        else:
            # create new tag: timestamp_githash
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                git_hash = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    stderr=subprocess.DEVNULL
                ).decode().strip()
                tag = f"{timestamp}_{git_hash}"
            except (subprocess.CalledProcessError, FileNotFoundError):
                tag = timestamp
            print(f"ℹ starting new training run: {tag}")
    else:
        print(f"ℹ using tag: {tag}")

    # ensure directories exist
    run_dir = net_directory(tag)
    os.makedirs(run_dir, exist_ok=True)

    # generate checkpoint schedule
    checkpoint_schedule = get_checkpoint_schedule(games)
    print(f"⏱ training {games} games with checkpoints at: {checkpoint_schedule}")

    # try to find a checkpoint file to load
    checkpoint_paths = sorted(glob.glob(os.path.join(run_dir, "*.torch")))
    if len(checkpoint_paths):
        latest_checkpoint = checkpoint_paths[-1]
        net, checkpoint = load_checkpoint(latest_checkpoint)
        starting_game = checkpoint['game_count']
        print(f"✔ loaded checkpoint from {os.path.basename(latest_checkpoint)} (game {starting_game})")
    else:
        net = Net()
        starting_game = 0
        print("✔ initialized new network")

    # set learning rate if provided
    if alpha:
        net.learning_rate = alpha

    # metrics tracking
    metrics = TrainingMetrics()
    start_time = time.perf_counter()

    # setup ctrl-c handler for graceful checkpoint on interrupt
    interrupted = False
    def signal_handler(signum: int, frame: object) -> None:
        nonlocal interrupted
        interrupted = True
        print("\n⚠ interrupt received, checkpointing before exit...")

    signal.signal(signal.SIGINT, signal_handler)

    white, black = VectorMLPPlayer(Color.White, net, True), VectorMLPPlayer(Color.Black, net, True)

    pbar = tqdm(range(1, games + 1), desc="training", unit="game")
    for i in pbar:
        # run training game
        game_metrics = run_game(white, black, net)

        # record metrics
        metrics.game_lengths.append(game_metrics.turn_count)
        metrics.td_errors.append(game_metrics.avg_td_error())
        metrics.timestamps.append(time.perf_counter() - start_time)

        # calculate weight norm
        weight_norm = sum(torch.norm(p).item() for p in net.parameters())
        metrics.weight_norms.append(weight_norm)

        # update progress bar with recent metrics
        if i >= 10:
            avg_len = sum(metrics.game_lengths[-10:]) / 10
            avg_td = sum(metrics.td_errors[-10:]) / 10
            pbar.set_postfix({
                'len': f'{avg_len:.1f}',
                'td': f'{avg_td:.4f}',
                'w': f'{weight_norm:.2f}'
            })

        # checkpointing at scheduled intervals
        current_total_games = starting_game + i
        if current_total_games in checkpoint_schedule:
            checkpoint_path = os.path.join(run_dir, f"net-{current_total_games:07d}.torch")
            write_checkpoint(checkpoint_path, net, current_total_games, run_dir, checkpoint_schedule)

            # generate incremental report with full history
            generate_html_report(tag, metrics, i)

        # check for interrupt after each game
        if interrupted:
            current_total_games = starting_game + i
            checkpoint_path = os.path.join(run_dir, f"net-{current_total_games:07d}.torch")
            write_checkpoint(checkpoint_path, net, current_total_games, run_dir, checkpoint_schedule)
            print(f"✔ saved interrupt checkpoint at game {current_total_games}")
            generate_html_report(tag, metrics, i)
            print(f"⚠ training interrupted after {i} games")
            return tag, metrics, net

    elapsed = time.perf_counter() - start_time
    print(f"✔ finished! {elapsed:.1f}s total, {elapsed / games:.3f}s per game")

    if profile:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME)
        ps.print_stats(0.2)
        print(s.getvalue())

    return tag, metrics, net
