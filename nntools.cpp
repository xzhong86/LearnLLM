
#include <cmath>
#include "arrays.hpp"

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

namespace nn {

void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    //#pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}


void rmsnorm(Array1D<float> o, Array1D<float> x, Array1D<float> w) {
    assert_msg(o.size() == x.size() && x.size() == w.size(), "mismatch");
    rmsnorm(o.data(), x.data(), w.data(), w.size());
}

void softmax(Array1D<float> x, int size) {
    assert_msg(size <= x.size(), "overflow");
    softmax(x.data(), size);
}

// W (d,n) @ X (n,) -> XOUT (d,)
//void matmul(float* xout, float* x, float* w, int n, int d)
void matmul(Array1D<float> xout, Array1D<float> x, Array2D<float> w) {
    int n = w.d1size();
    int d = w.d2size();
    assert_msg(x.size()    == n, "mismatch");
    assert_msg(xout.size() == d, "mismatch");
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w.raw()[i * n + j] * x[j];
            //val += w[i,j] * x[j];
        }
        xout[i] = val;
    }
}

} // namespace nn
