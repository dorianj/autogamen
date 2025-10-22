"""Reinforcement learning training - migrated from run_reinforcement_learning.py"""
import base64
import cProfile
import glob
import io
import os.path
import pstats
import time
from datetime import datetime

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch

from autogamen.ai.mlp import MLPPlayer, Net, VectorMLPPlayer
from autogamen.ai.players import BozoPlayer, DeltaPlayer
from autogamen.game import vector_game as vg
from autogamen.game.board import Board
from autogamen.game.game_types import Color, Dice, Point
from autogamen.game.match import Match


def _fmt_percent(p: float) -> str:
    return f"{p:.1%}"


def net_directory() -> str:
    """Get the directory for storing network files"""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'out', 'models')


def train_directory() -> str:
    """Get the directory for storing training reports"""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'out', 'train')


def write_checkpoint(path: str, net: "Net", game_count: int) -> None:
    """Save training checkpoint"""
    torch.save({
        'model_state_dict': net.state_dict(),
        'eligibility': net.eligibility_traces,
        'game_count': game_count,
        'time': datetime.now().isoformat()
    }, path)


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


def run_exhib_match(net: "Net", cls: type) -> tuple[bool, int, int]:
    """Run exhibition match for evaluation. Returns (won, mlp_points, opponent_points)"""
    match = Match([MLPPlayer(Color.White, net, False), cls(Color.Black)], 15)

    while match.winner is None:
        match.tick()

    won = match.winner.color == Color.White
    print("{} exhibition vs {}, {} to {} in {} turns".format(
        "WON" if won else "Lost",
        cls.__name__,
        match.points[Color.White],
        match.points[Color.Black],
        match.turn_count,
    ))
    return won, match.points[Color.White], match.points[Color.Black]


class TrainingMetrics:
    """Metrics collected during training"""
    def __init__(self) -> None:
        self.game_lengths: list[int] = []
        self.td_errors: list[float] = []
        self.weight_norms: list[float] = []
        self.eval_games: list[int] = []  # Game numbers where evaluations occurred
        self.bozo_wins: list[bool] = []
        self.delta_wins: list[bool] = []
        self.timestamps: list[float] = []  # Time elapsed for each game


def generate_html_report(timestamp: str, metrics: TrainingMetrics, games: int) -> str:
    """Generate HTML report with matplotlib charts (black background).

    Returns the path to the generated report.
    """
    # Set matplotlib style for black background
    plt.style.use('dark_background')

    # Create figure with subplots
    num_charts = 4 if metrics.eval_games else 3
    fig, axes = plt.subplots(num_charts, 1, figsize=(12, 4 * num_charts))
    fig.patch.set_facecolor('black')

    if num_charts == 3:
        axes = list(axes)

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

    # Chart 4: Win Rate (if available)
    if metrics.eval_games:
        ax = axes[3]
        ax.set_facecolor('black')
        # Calculate cumulative win rates
        bozo_rate = [sum(metrics.bozo_wins[:i+1]) / (i+1) for i in range(len(metrics.bozo_wins))]
        delta_rate = [sum(metrics.delta_wins[:i+1]) / (i+1) for i in range(len(metrics.delta_wins))]
        ax.plot(metrics.eval_games, bozo_rate, color='red', linewidth=2, marker='o', label='vs BozoPlayer')
        ax.plot(metrics.eval_games, delta_rate, color='blue', linewidth=2, marker='s', label='vs DeltaPlayer')
        ax.set_xlabel('Game Number', color='white')
        ax.set_ylabel('Win Rate', color='white')
        ax.set_title('Win Rate in Exhibition Matches', color='white', fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
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
    <title>TD-GAMMON Training Report - {timestamp}</title>
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
        <div class="timestamp">Run: {timestamp}</div>
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

    # Write to file
    report_path = os.path.join(train_directory(), f"{timestamp}_report.html")
    with open(report_path, 'w') as f:
        f.write(html)

    return report_path


def run_reinforcement_learning(
    games: int,
    checkpoint_interval: int,
    alpha: float,
    profile: bool,
    exhibition: bool,
) -> tuple[str, TrainingMetrics, "Net"]:
    """Main entry point for reinforcement learning training.

    Returns (timestamp, metrics, net) for report generation.
    """
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Running learning for {games} games, checkpointing every "
          f"{checkpoint_interval} games to timestamp {run_timestamp}")

    # Ensure directories exist
    net_dir = net_directory()
    train_dir = train_directory()
    os.makedirs(net_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    # Try to find a checkpoint file to load
    checkpoint_paths = sorted(glob.glob(os.path.join(net_dir, "*.torch")))
    if len(checkpoint_paths):
        net, checkpoint = load_checkpoint(checkpoint_paths[-1])
        gen = checkpoint['game_count']
        print(f"Loaded checkpoint file with {checkpoint['game_count']} games from {checkpoint['time']}")
    else:
        print("Training net from scratch...")
        net = Net()
        gen = 0

    # Set learning rate if provided
    if alpha:
        net.learning_rate = alpha

    # Metrics tracking
    metrics = TrainingMetrics()
    start_time = time.perf_counter()

    white, black = VectorMLPPlayer(Color.White, net, True), VectorMLPPlayer(Color.Black, net, True)

    for i in range(1, games + 1):
        # Run training game
        game_metrics = run_game(white, black, net)

        # Record metrics
        metrics.game_lengths.append(game_metrics.turn_count)
        metrics.td_errors.append(game_metrics.avg_td_error())
        metrics.timestamps.append(time.perf_counter() - start_time)

        # Calculate weight norm
        weight_norm = sum(torch.norm(p).item() for p in net.parameters())
        metrics.weight_norms.append(weight_norm)

        # Progress logging
        if i % 10 == 0:
            avg_len = sum(metrics.game_lengths[-10:]) / 10
            avg_td = sum(metrics.td_errors[-10:]) / 10
            print(f"Game {i}/{games}: avg_len={avg_len:.1f}, avg_td_error={avg_td:.4f}, weight_norm={weight_norm:.2f}")

        # Checkpointing
        if i % checkpoint_interval == 0:
            print(f"Checkpointing at {i}")
            checkpoint_path = os.path.join(net_dir, f"net-{run_timestamp}-{i+gen:07d}.torch")
            write_checkpoint(checkpoint_path, net, i + gen)

            # Exhibition matches for evaluation
            if exhibition:
                metrics.eval_games.append(i)
                bozo_won, _, _ = run_exhib_match(net, BozoPlayer)
                delta_won, _, _ = run_exhib_match(net, DeltaPlayer)
                metrics.bozo_wins.append(bozo_won)
                metrics.delta_wins.append(delta_won)

    elapsed = time.perf_counter() - start_time
    print(f"Finished! {elapsed:.1f}s total, {elapsed / games:.3f}s per game")

    if profile:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME)
        ps.print_stats(0.2)
        print(s.getvalue())

    return run_timestamp, metrics, net
