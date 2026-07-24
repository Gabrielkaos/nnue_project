#include "nnue.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float *read_floats(FILE *f, int count) {
    float *buf = (float *)malloc(sizeof(float) * (size_t)count);
    if (!buf) return NULL;
    if (fread(buf, sizeof(float), (size_t)count, f) != (size_t)count) {
        free(buf);
        return NULL;
    }
    return buf;
}

int nnue_load(NNUEModel *m, const char *path) {
    memset(m, 0, sizeof(*m));

    FILE *f = fopen(path, "rb");
    if (!f) return 1;

    char magic[4];
    int32_t version;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "NNUE", 4) != 0) {
        fclose(f);
        return 2;
    }
    if (fread(&version, sizeof(version), 1, f) != 1) { fclose(f); return 3; }

    int32_t input_size, l1, l2, l3;
    float cp_scale;
    if (fread(&input_size, sizeof(int32_t), 1, f) != 1) { fclose(f); return 4; }
    if (fread(&l1, sizeof(int32_t), 1, f) != 1) { fclose(f); return 4; }
    if (fread(&l2, sizeof(int32_t), 1, f) != 1) { fclose(f); return 4; }
    if (fread(&l3, sizeof(int32_t), 1, f) != 1) { fclose(f); return 4; }
    if (fread(&cp_scale, sizeof(float), 1, f) != 1) { fclose(f); return 4; }

    m->input_size = input_size;
    m->l1_size = l1;
    m->l2_size = l2;
    m->l3_size = l3;
    m->cp_scale = cp_scale;

    m->fc1_w = read_floats(f, l1 * input_size);
    m->fc1_b = read_floats(f, l1);
    m->fc2_w = read_floats(f, l2 * l1);
    m->fc2_b = read_floats(f, l2);
    m->fc3_w = read_floats(f, l3 * l2);
    m->fc3_b = read_floats(f, l3);
    m->fc4_w = read_floats(f, 1 * l3);
    m->fc4_b = read_floats(f, 1);

    fclose(f);

    if (!m->fc1_w || !m->fc1_b || !m->fc2_w || !m->fc2_b ||
        !m->fc3_w || !m->fc3_b || !m->fc4_w || !m->fc4_b) {
        nnue_free(m);
        return 5;
    }
    return 0;
}

void nnue_free(NNUEModel *m) {
    free(m->fc1_w); free(m->fc1_b);
    free(m->fc2_w); free(m->fc2_b);
    free(m->fc3_w); free(m->fc3_b);
    free(m->fc4_w); free(m->fc4_b);
    memset(m, 0, sizeof(*m));
}

static inline float clipped_relu(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

/* out[i] = clipped_relu( bias[i] + sum_j in[j] * w[i*in_size + j] ) */
static void linear_clipped(const float *in, int in_size,
                            const float *w, const float *b, int out_size,
                            float *out) {
    for (int i = 0; i < out_size; i++) {
        float acc = b[i];
        const float *row = w + (size_t)i * in_size;
        for (int j = 0; j < in_size; j++) {
            acc += in[j] * row[j];
        }
        out[i] = clipped_relu(acc);
    }
}

float nnue_evaluate(const NNUEModel *m, const int board[64], int side_to_move) {
    /* Build the 768-dim sparse feature vector, mirroring the board when
     * it's Black to move so the net always sees "side to move" features.
     * This MUST match nnue_dataset.py exactly. */
    float features[768];
    memset(features, 0, sizeof(features));

    for (int sq = 0; sq < 64; sq++) {
        int piece;
        int feat_sq;

        if (side_to_move == 1) {
            int mirrored_sq = sq ^ 56;
            int p = board[mirrored_sq];
            if (p == 0) continue;
            piece = (p <= 6) ? (p + 6) : (p - 6); /* swap color */
            feat_sq = sq;
        } else {
            piece = board[sq];
            if (piece == 0) continue;
            feat_sq = sq;
        }

        int plane = piece - 1; /* 0..11 */
        features[plane * 64 + feat_sq] = 1.0f;
    }

    float h1[256], h2[64], h3[64]; /* sized for the max l1/l2/l3 we expect;
                                       real sizes come from m->l*_size */
    /* Guard against overly large configured sizes overflowing these
     * fixed buffers. If you train with l1 > 256 or l2/l3 > 64, bump these. */
    if (m->l1_size > 256 || m->l2_size > 64 || m->l3_size > 64) {
        return 0.0f; /* misconfigured; caller should check sizes up front */
    }

    linear_clipped(features, m->input_size, m->fc1_w, m->fc1_b, m->l1_size, h1);
    linear_clipped(h1, m->l1_size, m->fc2_w, m->fc2_b, m->l2_size, h2);
    linear_clipped(h2, m->l2_size, m->fc3_w, m->fc3_b, m->l3_size, h3);

    float out = m->fc4_b[0];
    for (int j = 0; j < m->l3_size; j++) {
        out += h3[j] * m->fc4_w[j];
    }

    /* out is a logit; multiply by cp_scale to get an approximate
     * centipawn score, matching how targets were built in training
     * (target = sigmoid(cp / cp_scale)). */
    return out * m->cp_scale;
}
