##########################################################################
# VOCAB FIXO UNIVERSAL PARA LANCES DE XADREZ (ULTRA RÁPIDO, ZERO SCAN)
##########################################################################
import argparse
import glob
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import numpy as np
import chess
from chess import pgn, Board
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import Conv2D, Flatten, Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import tensorflow as tf  # type: ignore
import tkinter as tk
from tkinter import ttk, messagebox

FILES = [f"data/{x}" for x in os.listdir("data") if x.endswith(".pgn")]

# ---------------------------------------------
# 1. GERAR TODAS AS POSSÍVEIS CLASSES DE LANCES
# ---------------------------------------------
move_to_int = {}
int_to_move = {}

index = 0
for orig in range(64):
    for dest in range(64):
        for promo in [None, "q", "r", "b", "n"]:
            key = f"{orig}-{dest}-{promo}"
            move_to_int[key] = index
            int_to_move[index] = key
            index += 1

NUM_CLASSES = len(move_to_int)


# ---------------------------------------------
# 2. FUNÇÕES DE CONVERSÃO
# ---------------------------------------------

def configure_gpu() -> bool:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return False
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return True


def gpu_status() -> Tuple[bool, List[str], bool]:
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = [gpu.name for gpu in gpus]
    return bool(gpus), gpu_names, tf.test.is_built_with_cuda()


def board_to_matrix(board: Board) -> np.ndarray:
    m = np.zeros((8, 8, 12), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        ptype = piece.piece_type - 1
        color = 0 if piece.color else 6
        m[row, col, ptype + color] = 1
    return m


def encode_move(move: chess.Move) -> int:
    orig = move.from_square
    dest = move.to_square
    promo = move.promotion

    if promo == chess.QUEEN:
        promo = "q"
    elif promo == chess.ROOK:
        promo = "r"
    elif promo == chess.BISHOP:
        promo = "b"
    elif promo == chess.KNIGHT:
        promo = "n"
    else:
        promo = None

    key = f"{orig}-{dest}-{promo}"
    return move_to_int[key]


def decode_move(key: str) -> chess.Move:
    orig, dest, promo = key.split("-")
    orig = int(orig)
    dest = int(dest)
    promo_piece = None
    if promo == "q":
        promo_piece = chess.QUEEN
    elif promo == "r":
        promo_piece = chess.ROOK
    elif promo == "b":
        promo_piece = chess.BISHOP
    elif promo == "n":
        promo_piece = chess.KNIGHT
    return chess.Move(orig, dest, promotion=promo_piece)


# ---------------------------------------------
# 3. GENERATOR SEM SCAN, SEM RAM, SEM DOR
# ---------------------------------------------

def data_generator(pgn_files: Iterable[str], batch_size: int = 512):
    X_batch, Y_batch = [], []

    for file in pgn_files:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                game = pgn.read_game(f)
                if game is None:
                    break

                board = game.board()
                for move in game.mainline_moves():
                    X_batch.append(board_to_matrix(board))
                    Y_batch.append(encode_move(move))
                    board.push(move)

                    if len(X_batch) == batch_size:
                        yield np.array(X_batch), to_categorical(Y_batch, NUM_CLASSES)
                        X_batch, Y_batch = [], []


# ---------------------------------------------
# 4. MODELO
# ---------------------------------------------

def build_model(num_classes: int) -> Sequential:
    model = Sequential(
        [
            Conv2D(64, (3, 3), activation="relu", input_shape=(8, 8, 12)),
            Conv2D(128, (3, 3), activation="relu"),
            Flatten(),
            Dense(256, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def find_latest_model(models_dir: str = "models") -> Optional[str]:
    candidates = glob.glob(os.path.join(models_dir, "**", "*.keras"), recursive=True)
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def predict_next_move(board: Board, model: Sequential) -> Optional[chess.Move]:
    board_matrix = board_to_matrix(board).reshape(1, 8, 8, 12)
    predictions = model.predict(board_matrix, verbose=0)[0]
    sorted_indices = np.argsort(predictions)[::-1]
    for idx in sorted_indices:
        idx = int(idx)
        if idx in int_to_move:
            candidate = decode_move(int_to_move[idx])
            if candidate in board.legal_moves:
                return candidate
    return None


def rank_legal_moves(board: Board, model: Sequential) -> List[Tuple[chess.Move, float]]:
    board_matrix = board_to_matrix(board).reshape(1, 8, 8, 12)
    predictions = model.predict(board_matrix, verbose=0)[0]
    ranked: List[Tuple[chess.Move, float]] = []
    for move in board.legal_moves:
        try:
            idx = encode_move(move)
        except KeyError:
            continue
        ranked.append((move, float(predictions[idx])))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def aggregate_input_weights(weights: np.ndarray, hidden_indices: np.ndarray) -> np.ndarray:
    if weights.shape[0] % 12 != 0:
        plane_means = np.mean(weights, axis=0)
        return np.tile(plane_means, (12, 1))[:, hidden_indices]
    plane_size = weights.shape[0] // 12
    reshaped = weights.reshape(12, plane_size, -1)
    plane_means = reshaped.mean(axis=1)
    return plane_means[:, hidden_indices]


class TrainingGUI(tk.Tk):
    def __init__(self, model: Sequential, epochs: int, steps_per_epoch: int):
        super().__init__()
        self.title("Treino Neural Chess")
        self.geometry("900x600")
        self.model = model
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.current_epoch = 0
        self.current_step = 0
        self.best_loss = None
        self.best_weights = None

        self.progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Iniciando treino...")
        self.loss_var = tk.StringVar(value="Loss: --")
        self.best_var = tk.StringVar(value="Melhor loss: --")

        self._build_ui()

    def _build_ui(self):
        top_frame = ttk.Frame(self)
        top_frame.pack(fill=tk.X, padx=12, pady=12)

        ttk.Label(top_frame, textvariable=self.status_var, font=("Arial", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(top_frame, textvariable=self.loss_var).pack(anchor=tk.W, pady=(4, 0))
        ttk.Label(top_frame, textvariable=self.best_var).pack(anchor=tk.W)

        self.progress = ttk.Progressbar(top_frame, variable=self.progress_var, maximum=100)
        self.progress.pack(fill=tk.X, pady=(8, 0))

        ttk.Button(top_frame, text="Ver rede neural (melhor)", command=self._show_network).pack(
            side=tk.RIGHT, padx=4
        )

        self.network_window = NetworkWindow(self, self.model)

    def update_status(self, epoch: int, step: int, loss: Optional[float]):
        self.current_epoch = epoch
        self.current_step = step
        total_steps = self.epochs * self.steps_per_epoch
        overall_step = (epoch - 1) * self.steps_per_epoch + step
        if total_steps > 0:
            self.progress_var.set(100 * overall_step / total_steps)
        self.status_var.set(f"Treinando... Época {epoch}/{self.epochs} | Passo {step}/{self.steps_per_epoch}")
        if loss is not None:
            self.loss_var.set(f"Loss: {loss:.4f}")
        self._refresh()

    def update_best(self, best_loss: float):
        self.best_loss = best_loss
        self.best_var.set(f"Melhor loss: {best_loss:.4f}")
        if self.model is not None:
            self.best_weights = self.model.get_weights()
        self._refresh_network()

    def _refresh(self):
        self.update_idletasks()
        self.update()

    def _show_network(self):
        if self.network_window.winfo_exists():
            self.network_window.lift()
        else:
            self.network_window = NetworkWindow(self, self.model)

    def _refresh_network(self):
        if self.network_window.winfo_exists():
            self.network_window.refresh(self.best_weights)


class TrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, gui: TrainingGUI, steps_per_epoch: int):
        super().__init__()
        self.gui = gui
        self.steps_per_epoch = steps_per_epoch
        self.best_loss = None

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        self.gui.update_status(self.gui.current_epoch, batch + 1, loss)

    def on_epoch_begin(self, epoch, logs=None):
        self.gui.current_epoch = epoch + 1
        self.gui.update_status(epoch + 1, 0, None)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss is None:
            return
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = float(loss)
            self.gui.update_best(self.best_loss)


def train_model(
    pgn_files: List[str],
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    output_dir: str,
    use_gui: bool,
) -> str:
    model = build_model(NUM_CLASSES)
    model.summary()
    callbacks: List[tf.keras.callbacks.Callback] = []
    gui = None
    if use_gui:
        gui = TrainingGUI(model, epochs, steps_per_epoch)
        callbacks.append(TrainingCallback(gui, steps_per_epoch))
    model.fit(
        data_generator(pgn_files, batch_size=batch_size),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
    )
    if gui:
        gui.update_status(epochs, steps_per_epoch, None)

    fol = os.path.join(output_dir, f"{datetime.now().year}_{datetime.now().month}_{datetime.now().day}")
    os.makedirs(fol, exist_ok=True)
    mod = f"model_{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}.keras"
    model_path = os.path.join(fol, mod)
    model.save(model_path)
    print("Saved:", model_path)
    return model_path


@dataclass
class GameInfo:
    title: str
    moves: List[chess.Move]
    san_moves: List[str]


class ChessGUI(tk.Tk):
    def __init__(self, games: List[GameInfo], model: Optional[Sequential]):
        super().__init__()
        self.title("Neural Chess Viewer")
        self.geometry("1100x700")

        self.games = games
        self.model = model
        self.current_game: Optional[GameInfo] = None
        self.current_board = Board()
        self.move_index = 0
        self.square_size = 60

        self.range_size_var = tk.StringVar(value="10")
        self.range_var = tk.StringVar()

        self._build_ui()
        self._populate_ranges()

    def _build_ui(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Label(control_frame, text="Tamanho do grupo:").pack(side=tk.LEFT)
        range_size = ttk.Combobox(control_frame, textvariable=self.range_size_var, values=["10", "25"], width=5)
        range_size.pack(side=tk.LEFT, padx=4)
        range_size.bind("<<ComboboxSelected>>", lambda _e: self._populate_ranges())

        ttk.Label(control_frame, text="Intervalo:").pack(side=tk.LEFT, padx=(16, 0))
        self.range_combo = ttk.Combobox(control_frame, textvariable=self.range_var, width=12, state="readonly")
        self.range_combo.pack(side=tk.LEFT, padx=4)
        self.range_combo.bind("<<ComboboxSelected>>", lambda _e: self._populate_games())

        ttk.Button(control_frame, text="Ver rede neural", command=self._show_network).pack(
            side=tk.RIGHT, padx=4
        )

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left_frame, text="Jogos:").pack(anchor=tk.W)
        self.games_listbox = tk.Listbox(left_frame, width=40, height=25)
        self.games_listbox.pack(fill=tk.Y, expand=True)
        self.games_listbox.bind("<<ListboxSelect>>", self._on_game_select)

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))

        board_frame = ttk.Frame(right_frame)
        board_frame.pack(side=tk.LEFT, padx=(0, 12))

        self.board_canvas = tk.Canvas(
            board_frame,
            width=self.square_size * 8,
            height=self.square_size * 8,
            bg="#1e1e1e",
            highlightthickness=0,
        )
        self.board_canvas.pack()

        nav_frame = ttk.Frame(board_frame)
        nav_frame.pack(pady=8)

        ttk.Button(nav_frame, text="Anterior", command=self._prev_move).pack(side=tk.LEFT, padx=4)
        ttk.Button(nav_frame, text="Próximo", command=self._next_move).pack(side=tk.LEFT, padx=4)

        ai_frame = ttk.Frame(board_frame)
        ai_frame.pack(pady=4)
        ttk.Button(ai_frame, text="Sugestão IA", command=self._suggest_move).pack(side=tk.LEFT, padx=4)
        ttk.Button(ai_frame, text="IA joga", command=self._ai_move).pack(side=tk.LEFT, padx=4)
        self.ai_label = ttk.Label(ai_frame, text="")
        self.ai_label.pack(side=tk.LEFT, padx=4)

        moves_frame = ttk.Frame(right_frame)
        moves_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(moves_frame, text="Lances:").pack(anchor=tk.W)
        self.moves_text = tk.Text(moves_frame, height=30, width=40, wrap=tk.WORD)
        self.moves_text.pack(fill=tk.BOTH, expand=True)
        self.moves_text.configure(state=tk.DISABLED)

        ranking_frame = ttk.Frame(right_frame)
        ranking_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 0))
        ttk.Label(ranking_frame, text="Ranking IA (melhor → pior):").pack(anchor=tk.W)
        self.ranking_list = tk.Listbox(ranking_frame, width=30, height=25)
        self.ranking_list.pack(fill=tk.BOTH, expand=True)

        self._draw_board()
        self._update_ranking()

    def _populate_ranges(self):
        range_size = int(self.range_size_var.get())
        total_games = len(self.games)
        ranges = []
        for start in range(1, total_games + 1, range_size):
            end = min(start + range_size - 1, total_games)
            ranges.append(f"{start}-{end}")
        if not ranges:
            ranges = ["0-0"]
        self.range_combo["values"] = ranges
        self.range_var.set(ranges[0])
        self._populate_games()

    def _populate_games(self):
        self.games_listbox.delete(0, tk.END)
        if not self.games:
            return
        start, end = self._parse_range(self.range_var.get())
        for idx in range(start - 1, end):
            if 0 <= idx < len(self.games):
                self.games_listbox.insert(tk.END, f"{idx + 1}. {self.games[idx].title}")
        if self.games_listbox.size():
            self.games_listbox.selection_set(0)
            self._on_game_select(None)

    def _parse_range(self, range_text: str) -> Tuple[int, int]:
        try:
            start_str, end_str = range_text.split("-")
            return int(start_str), int(end_str)
        except ValueError:
            return (1, len(self.games))

    def _on_game_select(self, _event):
        selection = self.games_listbox.curselection()
        if not selection:
            return
        start, _ = self._parse_range(self.range_var.get())
        index = start - 1 + selection[0]
        if index < 0 or index >= len(self.games):
            return
        self.current_game = self.games[index]
        self.current_board.reset()
        self.move_index = 0
        self._render_moves()
        self._draw_board()
        self.ai_label.configure(text="")
        self._update_ranking()

    def _render_moves(self):
        if not self.current_game:
            return
        self.moves_text.configure(state=tk.NORMAL)
        self.moves_text.delete("1.0", tk.END)
        for idx, san in enumerate(self.current_game.san_moves, start=1):
            move_label = f"{idx}. {san} "
            self.moves_text.insert(tk.END, move_label)
        self.moves_text.configure(state=tk.DISABLED)

    def _draw_board(self):
        self.board_canvas.delete("all")
        colors = ["#F0D9B5", "#B58863"]
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                self.board_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

        for square, piece in self.current_board.piece_map().items():
            row, col = divmod(square, 8)
            x = col * self.square_size + self.square_size // 2
            y = row * self.square_size + self.square_size // 2
            symbol = self._piece_symbol(piece)
            self.board_canvas.create_text(x, y, text=symbol, font=("Arial", 28))

    def _piece_symbol(self, piece: chess.Piece) -> str:
        symbols = {
            "P": "♙",
            "N": "♘",
            "B": "♗",
            "R": "♖",
            "Q": "♕",
            "K": "♔",
            "p": "♟",
            "n": "♞",
            "b": "♝",
            "r": "♜",
            "q": "♛",
            "k": "♚",
        }
        return symbols[piece.symbol()]

    def _next_move(self):
        if not self.current_game:
            return
        if self.move_index >= len(self.current_game.moves):
            return
        self.current_board.push(self.current_game.moves[self.move_index])
        self.move_index += 1
        self._draw_board()
        self.ai_label.configure(text="")
        self._update_ranking()

    def _prev_move(self):
        if not self.current_game:
            return
        if self.move_index == 0:
            return
        self.current_board.pop()
        self.move_index -= 1
        self._draw_board()
        self.ai_label.configure(text="")
        self._update_ranking()

    def _suggest_move(self):
        if self.model is None:
            messagebox.showwarning("Modelo", "Nenhum modelo carregado para sugerir lances.")
            return
        move = predict_next_move(self.current_board, self.model)
        if move is None:
            self.ai_label.configure(text="Sem lance legal")
        else:
            self.ai_label.configure(text=f"IA sugere: {move.uci()}")
        self._update_ranking()

    def _ai_move(self):
        if self.model is None:
            messagebox.showwarning("Modelo", "Nenhum modelo carregado para jogar.")
            return
        move = predict_next_move(self.current_board, self.model)
        if move is None:
            self.ai_label.configure(text="Sem lance legal")
            return
        self.current_board.push(move)
        self.move_index = min(self.move_index + 1, len(self.current_game.moves) if self.current_game else 0)
        self._draw_board()
        self.ai_label.configure(text=f"IA jogou: {move.uci()}")
        self._update_ranking()

    def _update_ranking(self):
        self.ranking_list.delete(0, tk.END)
        if self.model is None:
            self.ranking_list.insert(tk.END, "Modelo não carregado.")
            return
        rankings = rank_legal_moves(self.current_board, self.model)
        if not rankings:
            self.ranking_list.insert(tk.END, "Sem lances legais.")
            return
        for move, score in rankings:
            self.ranking_list.insert(tk.END, f"{move.uci()}  {score:.3f}")

    def _show_network(self):
        if self.model is None:
            messagebox.showwarning("Modelo", "Nenhum modelo carregado para visualizar.")
            return
        NetworkWindow(self, self.model)


class NetworkWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, model: Sequential):
        super().__init__(parent)
        self.title("Rede Neural")
        self.geometry("450x750")
        self.canvas = tk.Canvas(self, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.model = model
        self._draw_network()

    def refresh(self, weights: Optional[List[np.ndarray]] = None):
        self._draw_network(weights)

    def _draw_network(self, weights: Optional[List[np.ndarray]] = None):
        self.canvas.delete("all")
        width = 450
        height = 750
        padding_x = 80
        padding_y = 50

        input_labels = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]
        input_nodes = len(input_labels)
        hidden_nodes = 6
        output_nodes = 8

        input_x = padding_x
        hidden_x = width // 2
        output_x = width - padding_x

        input_positions = self._vertical_positions(input_nodes, height, padding_y)
        hidden_positions = self._vertical_positions(hidden_nodes, height, padding_y + 100)
        output_positions = self._vertical_positions(output_nodes, height, padding_y + 100)

        if weights is not None:
            dense1_weights = weights[-4]
            dense2_weights = weights[-2]
        else:
            dense1_weights = self.model.layers[-2].get_weights()[0]
            dense2_weights = self.model.layers[-1].get_weights()[0]
        hidden_indices = np.linspace(0, dense1_weights.shape[1] - 1, hidden_nodes, dtype=int)
        output_indices = np.linspace(0, dense2_weights.shape[1] - 1, output_nodes, dtype=int)

        input_weights = self._aggregate_input_weights(dense1_weights, hidden_indices)
        output_weights = dense2_weights[hidden_indices][:, output_indices]

        self._draw_connections(input_x, hidden_x, input_positions, hidden_positions, input_weights)
        self._draw_connections(hidden_x, output_x, hidden_positions, output_positions, output_weights)

        for idx, y in enumerate(input_positions):
            self._draw_node(input_x, y, input_labels[idx])
        for idx, y in enumerate(hidden_positions):
            self._draw_node(hidden_x, y, f"H{idx + 1}")
        for idx, y in enumerate(output_positions):
            self._draw_node(output_x, y, f"O{idx + 1}")

        self.canvas.create_text(
            width // 2,
            height - 20,
            text=f"Hidden nodes: {hidden_nodes}  |  Connections: {input_nodes * hidden_nodes + hidden_nodes * output_nodes}",
            fill="#d0d0d0",
            font=("Arial", 12),
        )

    def _aggregate_input_weights(self, weights: np.ndarray, hidden_indices: np.ndarray) -> np.ndarray:
        return aggregate_input_weights(weights, hidden_indices)

    def _draw_connections(
        self,
        x1: int,
        x2: int,
        y_positions_1: List[int],
        y_positions_2: List[int],
        weight_matrix: np.ndarray,
    ):
        if weight_matrix.size == 0:
            return
        max_abs = np.max(np.abs(weight_matrix)) + 1e-6
        for i, y1 in enumerate(y_positions_1):
            for j, y2 in enumerate(y_positions_2):
                weight = weight_matrix[i, j]
                color = self._weight_color(weight / max_abs)
                thickness = 1 + int(3 * abs(weight) / max_abs)
                self.canvas.create_line(x1 + 15, y1, x2 - 15, y2, fill=color, width=thickness)

    def _draw_node(self, x: int, y: int, label: str):
        r = 12
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="#111", outline="#d0d0d0", width=2)
        self.canvas.create_text(x, y, text=label, fill="#e6e6e6", font=("Arial", 10))

    def _weight_color(self, value: float) -> str:
        if value >= 0:
            green = int(120 + 100 * min(value, 1))
            return f"#3d{green:02x}3d"
        red = int(120 + 100 * min(abs(value), 1))
        return f"#{red:02x}3d3d"

    def _vertical_positions(self, count: int, height: int, padding: int) -> List[int]:
        available = height - 2 * padding
        step = available // max(count - 1, 1)
        return [padding + i * step for i in range(count)]


def load_games(pgn_files: Iterable[str]) -> List[GameInfo]:
    games: List[GameInfo] = []
    for file in pgn_files:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                game = pgn.read_game(f)
                if game is None:
                    break
                board = game.board()
                moves: List[chess.Move] = []
                san_moves: List[str] = []
                for move in game.mainline_moves():
                    san_moves.append(board.san(move))
                    moves.append(move)
                    board.push(move)
                headers = game.headers
                title = f"{headers.get('White', 'White')} vs {headers.get('Black', 'Black')} ({headers.get('Date', '????')})"
                games.append(GameInfo(title=title, moves=moves, san_moves=san_moves))
    return games


def run_gui(model_path: Optional[str]):
    model = None
    if model_path:
        model = load_model(model_path)
    games = load_games(FILES)
    app = ChessGUI(games, model)
    app.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="Neural Chess")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Treinar modelo")
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--steps", type=int, default=20000)
    train_parser.add_argument("--batch", type=int, default=512)
    train_parser.add_argument("--output", type=str, default="models")
    train_parser.add_argument("--gui", dest="gui", action="store_true", default=True)
    train_parser.add_argument("--no-gui", dest="gui", action="store_false")

    gui_parser = subparsers.add_parser("gui", help="Abrir GUI")
    gui_parser.add_argument("--model", type=str, default=None)

    return parser.parse_args()


def main():
    gpu_enabled = configure_gpu()
    has_gpu, gpu_names, built_with_cuda = gpu_status()
    if gpu_enabled:
        print("GPU detectada: usando RTX para treino/inferência.")
        print("GPUs disponíveis:", ", ".join(gpu_names))
    else:
        print("GPU não detectada: usando CPU.")
        if not built_with_cuda:
            print("TensorFlow sem suporte CUDA. Instale tensorflow-gpu/CUDA compatível.")

    args = parse_args()
    if args.command == "train":
        train_model(FILES, args.epochs, args.steps, args.batch, args.output, args.gui)
        return

    if args.command == "gui":
        model_path = args.model or find_latest_model()
        if model_path is None:
            print("Nenhum modelo encontrado. Treine primeiro com o comando train.")
        run_gui(model_path)
        return

    print("Escolha um comando: train ou gui")


if __name__ == "__main__":
    main()
