##########################################################################
# VOCAB FIXO UNIVERSAL PARA LANCES DE XADREZ (ULTRA RÁPIDO, ZERO SCAN)
##########################################################################
import os
import numpy as np
import chess
from datetime import datetime
from chess import pgn
from chess import Board
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, Flatten, Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tqdm import tqdm

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
def board_to_matrix(board):
    m = np.zeros((8, 8, 12))
    for square, piece in board.piece_map().items():
        row, col = divmod(square, 8)
        ptype = piece.piece_type - 1
        color = 0 if piece.color else 6
        m[row, col, ptype + color] = 1
    return m

def encode_move(move):
    orig = move.from_square
    dest = move.to_square
    promo = move.promotion

    if promo == chess.QUEEN: promo = "q"
    elif promo == chess.ROOK: promo = "r"
    elif promo == chess.BISHOP: promo = "b"
    elif promo == chess.KNIGHT: promo = "n"
    else: promo = None

    key = f"{orig}-{dest}-{promo}"
    return move_to_int[key]


# ---------------------------------------------
# 3. GENERATOR SEM SCAN, SEM RAM, SEM DOR
# ---------------------------------------------
def data_generator(pgn_files, batch_size=512):
    X_batch, Y_batch = [], []

    for file in pgn_files:
        with open(file, "r") as f:
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
model = Sequential([
    Conv2D(64, (3,3), activation="relu", input_shape=(8,8,12)),
    Conv2D(128, (3,3), activation="relu"),
    Flatten(),
    Dense(256, activation="relu"),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


# ---------------------------------------------
# 5. TREINAR (SEM CONTAR POSIÇÕES, SEM SCAN)
# ---------------------------------------------
model.fit(
    data_generator(FILES, batch_size=512),
    epochs=50,
    steps_per_epoch=20000     # valor arbitrário, funciona perfeito
)


# ---------------------------------------------
# 6. SALVAR
# ---------------------------------------------
fol = f"models/{datetime.now().year}_{datetime.now().month}_{datetime.now().day}"
os.makedirs(fol, exist_ok=True)
mod = f"model_{datetime.now().hour}_{datetime.now().minute}_{datetime.now().second}.keras"

model.save(f"{fol}/{mod}")
print("Saved:", f"{fol}/{mod}")

# 7) Reconstrói mapeamento pra inferência
int_to_move = {v: k for k, v in move_to_int.items()}

def predict_next_move(board):
    board_matrix = board_to_matrix(board).reshape(1, 8, 8, 12)
    predictions = model.predict(board_matrix)[0]
    legal_moves = [m.uci() for m in board.legal_moves]
    sorted_indices = np.argsort(predictions)[::-1]
    for idx in sorted_indices:
        idx = int(idx)
        if idx in int_to_move:
            candidate = int_to_move[idx]
            if candidate in legal_moves:
                return candidate
    return None

# Interface simples (opcional)
if __name__ == "__main__":
    board = Board()
    choice = input("White or black? (w/b): ").lower()
    choice = chess.BLACK if choice == 'b' else chess.WHITE
    while not board.is_game_over():
        if board.turn != choice:
            next_move = predict_next_move(board)
            if next_move is None:
                print("Modelo não achou movimento legal — empate técnico.")
                break
            board.push_uci(next_move)
        else:
            move = input("Your move (san format): ")
            board.push_san(move)
        print(board)
