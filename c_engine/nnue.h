#ifndef NNUE_H
#define NNUE_H

#include <stdint.h>

/*
 * Board convention (must match fen_utils.py / nnue_dataset.py exactly):
 *   square index = rank*8 + file, a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63
 *   piece codes:
 *     0        = empty
 *     1..6     = white P, N, B, R, Q, K
 *     7..12    = black P, N, B, R, Q, K
 *
 * side_to_move: 0 = white to move, 1 = black to move
 */

typedef struct {
    int input_size;   /* 768 */
    int l1_size;       /* 256 */
    int l2_size;       /* 32 */
    int l3_size;       /* 32 */
    float cp_scale;    /* eval scale used at export time, default 410.0 */

    float *fc1_w; /* [l1_size][input_size] */
    float *fc1_b; /* [l1_size] */
    float *fc2_w; /* [l2_size][l1_size] */
    float *fc2_b; /* [l2_size] */
    float *fc3_w; /* [l3_size][l2_size] */
    float *fc3_b; /* [l3_size] */
    float *fc4_w; /* [1][l3_size] */
    float *fc4_b; /* [1] */
} NNUEModel;

/* Loads a weight file exported by export_weights.py.
 * Returns 0 on success, nonzero on failure. */
int nnue_load(NNUEModel *m, const char *path);

/* Frees buffers allocated by nnue_load. */
void nnue_free(NNUEModel *m);

/* Evaluates a position. `board` must be 64 entries using the piece codes
 * above. Returns an approximate centipawn score from the perspective of
 * the side to move (positive = good for the side to move). */
float nnue_evaluate(const NNUEModel *m, const int board[64], int side_to_move);

#endif /* NNUE_H */
