"""
FEN <-> compact board encoding.

Board encoding used everywhere in this project (68-byte training record):
  bytes[0:64] : one byte per square, square index = rank*8 + file,
                rank 0 = the '1' rank (a1..h1), file 0 = 'a' file.
                Piece codes:
                  0  = empty
                  1..6  = white P, N, B, R, Q, K
                  7..12 = black P, N, B, R, Q, K
  byte[64]    : side to move, 0 = white, 1 = black
  bytes[65:67]: int16 little-endian centipawn eval, ALWAYS from White's
                point of view (positive = good for White), clipped to
                [-3000, 3000]
  byte[67]    : flags, bit0 = 1 if this position is a "mate" score
                (the int16 eval field then holds +/-3000 as a saturated
                stand-in and should usually be skipped or handled specially
                during training)

This mirrors the fields in the Lichess/chess-position-evaluations dataset:
  fen, line, depth, knodes, cp, mate
We only use fen, cp and mate here.
"""

import struct

PIECE_CODE = {
    "P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6,
    "p": 7, "n": 8, "b": 9, "r": 10, "q": 11, "k": 12,
}

RECORD_STRUCT = struct.Struct("<64sBhB")  # 64 bytes, 1 byte, int16, 1 byte
RECORD_SIZE = RECORD_STRUCT.size  # 68


def fen_to_board_bytes(fen: str) -> bytes:
    """Convert the piece-placement + side-to-move part of a FEN into the
    64-byte board array described above. Only needs the first two
    space-separated fields of the FEN."""
    parts = fen.split()
    placement = parts[0]
    stm = parts[1] if len(parts) > 1 else "w"

    board = bytearray(64)
    rank = 7  # FEN starts at rank 8 (index 7)
    file = 0
    for ch in placement:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            sq = rank * 8 + file
            board[sq] = PIECE_CODE[ch]
            file += 1

    board.append(0 if stm == "w" else 1)
    return bytes(board)


def pack_record(fen: str, cp, mate) -> bytes:
    """Pack a dataset row into a fixed 68-byte record.
    cp: int or None. mate: int or None (plies to mate, sign = side that mates).
    Eval is always stored from WHITE's perspective.
    """
    board_and_stm = fen_to_board_bytes(fen)
    board = board_and_stm[:64]
    stm = board_and_stm[64]

    is_mate = 0
    if mate is not None:
        is_mate = 1
        eval_cp = 3000 if mate > 0 else -3000
    else:
        eval_cp = int(cp)
        if eval_cp > 3000:
            eval_cp = 3000
        elif eval_cp < -3000:
            eval_cp = -3000

    return RECORD_STRUCT.pack(board, stm, eval_cp, is_mate)


def unpack_record(record: bytes):
    board, stm, eval_cp, is_mate = RECORD_STRUCT.unpack(record)
    return board, stm, eval_cp, bool(is_mate)
