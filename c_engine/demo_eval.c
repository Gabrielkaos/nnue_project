#include <stdio.h>
#include "nnue.h"

/* Starting position, White to move. Board convention: see nnue.h */
static int START_BOARD[64] = {
    4, 2, 3, 5, 6, 3, 2, 4,   /* rank 1: R N B Q K B N R */
    1, 1, 1, 1, 1, 1, 1, 1,   /* rank 2: pawns */
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    7, 7, 7, 7, 7, 7, 7, 7,   /* rank 7: black pawns */
    10, 8, 9, 11, 12, 9, 8, 10 /* rank 8: r n b q k b n r */
};

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s weights.bin\n", argv[0]);
        return 1;
    }
    NNUEModel m;
    int rc = nnue_load(&m, argv[1]);
    if (rc != 0) {
        fprintf(stderr, "failed to load weights (code %d)\n", rc);
        return 1;
    }
    printf("loaded: input=%d l1=%d l2=%d l3=%d cp_scale=%.2f\n",
           m.input_size, m.l1_size, m.l2_size, m.l3_size, m.cp_scale);

    float eval_white = nnue_evaluate(&m, START_BOARD, 0);
    printf("start position eval (white to move): %.2f cp\n", eval_white);

    nnue_free(&m);
    return 0;
}
