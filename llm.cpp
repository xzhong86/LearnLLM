// start with llama.c project, with tinystories model.

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <map>
#include <format>

#include "llm-utils.hpp"
#include "arrays.hpp"
#include "nntools.hpp"

using namespace llm;
using namespace nn;

using std::string;
using std::vector;


// llm config struct, also is file header. copy from llama.c/run.c
struct Config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
};
struct TransformerWeights {
    // token embedding table
    Array2D<float> token_embedding_table; // (vocab_size, dim)
    // weights for rmsnorms
    Array2D<float> rms_att_weight; // (layer, dim) rmsnorm weights
    Array2D<float> rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    Array3D<float> wq; // (layer, dim, n_heads * head_size)
    Array3D<float> wk; // (layer, dim, n_kv_heads * head_size)
    Array3D<float> wv; // (layer, dim, n_kv_heads * head_size)
    Array3D<float> wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    Array3D<float> w1; // (layer, hidden_dim, dim)
    Array3D<float> w2; // (layer, dim, hidden_dim)
    Array3D<float> w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    Array1D<float> rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    //float* wcls;
    Array2D<float> wcls;  // (vocab_size, dim)?
};


// from llama2.c/run.c, but C++ version.
struct RunState {
    // current wave of activations
    Array1D<float> x;      // activation at current time stamp (dim,)
    Array1D<float> xb;     // same, but inside a residual branch (dim,)
    Array1D<float> xb2;    // an additional buffer just for convenience (dim,)
    Array1D<float> hb;     // buffer for hidden dimension in the ffn (hidden_dim,)
    Array1D<float> hb2;    // buffer for hidden dimension in the ffn (hidden_dim,)
    Array1D<float> q;      // query (dim,)
    Array1D<float> k;      // key (dim,), just pointer
    Array1D<float> v;      // value (dim,), just pointer
    Array2D<float> att;    // buffer for scores/attention values (n_heads, seq_len)
    Array1D<float> logits; // output logits
    // kv cache
    Array3D<float> key_cache;   // (layer, seq_len, dim)
    Array3D<float> value_cache; // (layer, seq_len, dim)

    ArrayPool<float> pool_;

    RunState(Config *p) {
        int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        pool_.alloc(p->dim*4 + p->hidden_dim*2 + p->n_layers * p->seq_len * kv_dim*2 +
                    p->n_heads * p->seq_len + p->vocab_size);
        x   = pool_.alloc1D(p->dim);
        xb  = pool_.alloc1D(p->dim);
        xb2 = pool_.alloc1D(p->dim);
        hb  = pool_.alloc1D(p->hidden_dim);
        hb2 = pool_.alloc1D(p->hidden_dim);
        q   = pool_.alloc1D(p->dim);
        key_cache   = pool_.alloc3D(p->n_layers, p->seq_len, kv_dim);
        value_cache = pool_.alloc3D(p->n_layers, p->seq_len, kv_dim);
        att    = pool_.alloc2D(p->n_heads, p->seq_len);
        logits = pool_.alloc1D(p->vocab_size);
    }
    ~RunState() {}
};


struct TokenStr {
    const char *str;
    TokenStr(const char *s) : str(s) { }
    bool operator<(const TokenStr &rhl) const {
        return std::strcmp(str, rhl.str) < 0;
    }
};

struct Tokenizer {
    vector<char*> vocab;
    vector<float> vocab_scores;
    std::map<TokenStr, int> vocab_map;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings

    Tokenizer(string tokenizer_path, int vocab_size)
        : vocab(vocab_size, nullptr), vocab_scores(vocab_size), vocab_size(vocab_size) {
        for (int i = 0; i < 256; i++) {
            byte_pieces[i * 2] = (unsigned char)i;
            byte_pieces[i * 2 + 1] = '\0';
        }
        std::ifstream ifs(tokenizer_path, std::ios_base::binary);
        if (!ifs.good()) throw_error("open tokenizer failed.");
        read_from(&max_token_length, ifs);
        int len;
        for (int i = 0; i < vocab_size; i++) {
            read_from(&vocab_scores[i], ifs);
            read_from(&len, ifs);
            vocab[i] = new char[len];
            read_from(vocab[i], len, ifs);
            vocab[i][len] = '\0';
        }
        ifs.close();
    }
    ~Tokenizer() {
        for (int i = 0; i < vocab_size; i++)
            delete [] vocab[i];
    }

    // relate to str_lookup
    int lookupToken(const string &str) {
        auto it = vocab_map.find(TokenStr(str.c_str()));
        return it == vocab_map.end() ? -1 : it->second;
    }

    vector<int> encode(string &text, int8_t bos, int8_t eos) {
        if (vocab_map.empty()) {
            for (int i = 0; i < vocab_size; i++)
                vocab_map.insert({vocab[i], i});
        }

        vector<int> tokens;
        if (bos) tokens.push_back(1);
        if (!text.empty()) // for some bug, refer to llama.c/run.c
            tokens.push_back(lookupToken(" ")); // append prefix

        // work with UTF-8
        string str_buffer;
        for (auto it = text.begin(); it != text.end(); ) {
            str_buffer.clear();
            str_buffer.push_back(*it);
            if ((*it & 0xc0) == 0xc0) {  // start utf-8 sequence
                for (++it ; (*it & 0xc0) == 0x80; ++it)
                    str_buffer.push_back(*it);
            }
            else
                ++it;
            int id = lookupToken(str_buffer);
            if (id != -1)
                tokens.push_back(id);
            else  // byte encode fallback, see llama2.c/run.c
                for (char c: str_buffer)
                    tokens.push_back(int(c) + 3);
        }

        // try merge pair with best score
        while (tokens.size() > 1) {
            //bool  has_best = false;
            float best_score = -1e10;
            int   best_id  = -1;
            auto  best_it  = tokens.end();
            for (auto it = tokens.begin(); it + 1 != tokens.end(); ++it) {
                auto nit = std::next(it);
                string str = std::format("{}{}", vocab[*it], vocab[*nit]);
                int id = lookupToken(str);
                if (id != -1 && vocab_scores[id] > best_score) {
                    best_it = it;
                    best_id = id;
                    best_score = vocab_scores[id];
                }
            }
            if (best_it == tokens.end())  // no best/better
                break;
            *best_it = best_id;
            tokens.erase(std::next(best_it));
        }

        if (eos) tokens.push_back(2);
        return tokens;
    }

    char* decode(int prev_token, int token) {
        char *piece = vocab[token];
        if (prev_token == 1 && piece[0] == ' ') { piece++; }
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        // parse this and convert and return the actual byte
        unsigned char byte_val;
        if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
            piece = (char*)byte_pieces + byte_val * 2;
        }
        return piece;
    }


};


// ----------------------------------------------------------------------------
// Sampler


int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

#if 0
int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}
#endif

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}


struct ProbIndex {
    float prob;
    int index;
}; // struct used when sorting probabilities during top-p sampling
struct Sampler {
    int vocab_size;
    float temperature;
    float topp;
    unsigned long rng_state;
    vector<ProbIndex> probindex; // buffer used in top-p sampling

    Sampler(int size, float temp, float topp, unsigned long seed)
        : vocab_size(size), temperature(temp), topp(topp), rng_state(seed), probindex(size) {
    }
    ~Sampler() {}

    int sample(float* logits) {
        // sample the token given the logits and some hyperparameters
        int next;
        if (temperature == 0.0f) {
            // greedy argmax sampling: take the token with the highest probability
            next = sample_argmax(logits, vocab_size);
        } else {
            //// apply the temperature to the logits
            //for (int q = 0; q < vocab_size; q++) { logits[q] /= sampler->temperature; }
            //// apply softmax to the logits to get the probabilities for next token
            //softmax(logits, vocab_size);
            //// flip a (float) coin (this is our source of entropy for sampling)
            //float coin = random_f32(&rng_state);
            //// we sample from this distribution to get the next token
            //if (topp <= 0 || topp >= 1) {
            //    // simply sample from the predicted probability distribution
            //    next = sample_mult(logits, vocab_size, coin);
            //} else {
            //    // top-p (nucleus) sampling, clamping the least likely tokens to zero
            //    next = sample_topp(logits, vocab_size, topp, probindex, coin);
            //}
            next = 0;
            throw_error("not support");
        }
        return next;
    }

};

class LLM {
    string model_path_;
    Config cfg_;
    OSMemMap mmap_;
    TransformerWeights *weights_ = nullptr;
    RunState *run_state_ = nullptr;
    Tokenizer *tokenizer_ = nullptr;
    Sampler *sampler_ = nullptr;

public:
    LLM() { }
    LLM(string path) : model_path_(path) {
        loadModel(path);
    }
    void loadModel(string path) {
        if (model_path_.empty()) model_path_ = path;
        else if (run_state_) throw_error("load model twice!");
        std::ifstream ifs(model_path_, std::ios::binary);
        if (!ifs.good()) throw_error("open file failed.");
        read_from(&cfg_, ifs);
        ifs.close();
        dumpConfig();
        loadParametersMMap();
        run_state_ = new RunState(&cfg_);
    }
    void loadParametersMMap() {
        mmap_.load(model_path_.c_str());
        float *ptr = mmap_.getPtr<float>(sizeof(Config));
        long  size = mmap_.getMemSize() / sizeof(float);
        ArrayPool<float> pool;
        pool.useSpace(ptr, size);

        weights_ = new TransformerWeights;
        auto *w = weights_;
        auto *p = &cfg_;
        bool shared_weights = p->vocab_size > 0;
        int head_size = p->dim / p->n_heads;
        long n_layers = p->n_layers;

#define mark_a1d(M, D1)         do { w->M = pool.alloc1D(D1);         } while (0)
#define mark_a2d(M, D1, D2)     do { w->M = pool.alloc2D(D1, D2);     } while (0)
#define mark_a3d(M, D1, D2, D3) do { w->M = pool.alloc3D(D1, D2, D3); } while (0)

        mark_a2d(token_embedding_table, p->vocab_size, p->dim);
        mark_a2d(rms_att_weight, n_layers, p->dim);

        mark_a3d(wq, n_layers, p->dim, (p->n_heads * head_size));
        mark_a3d(wk, n_layers, p->dim, (p->n_kv_heads * head_size));
        mark_a3d(wv, n_layers, p->dim, (p->n_kv_heads * head_size));

        mark_a3d(wo, n_layers, (p->n_heads * head_size), p->dim);
        mark_a2d(rms_ffn_weight, n_layers, p->dim);

        mark_a3d(w1, n_layers, p->dim, p->hidden_dim);
        mark_a3d(w2, n_layers, p->hidden_dim, p->dim);
        mark_a3d(w3, n_layers, p->dim, p->hidden_dim);

        mark_a1d(rms_final_weight, p->dim);

        ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
        ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
        if (shared_weights)
            w->wcls = w->token_embedding_table;
        else
            w->wcls = pool.alloc2D(p->vocab_size, p->dim);
    }
    void dumpConfig() {
        std::cout << "Model Config:\ndim = " << cfg_.dim
                  << "\nhidden_dim = " << cfg_.hidden_dim
                  << "\nn_layers = " << cfg_.n_layers
                  << "\nn_heads = " << cfg_.n_heads
                  << "\nn_kv_heads = " << cfg_.n_kv_heads
                  << "\nvocab_size = " << cfg_.vocab_size
                  << "\nseq_len = " << cfg_.seq_len
                  << std::endl;
    }

    void loadTokenizer(string path) {
        std::cout << "load tokenizer " << path << " ...\n";
        tokenizer_ = new Tokenizer(path, cfg_.vocab_size);
    }

    void initSampler(float temperature, float topp, unsigned long seed) {
        sampler_ = new Sampler(cfg_.vocab_size, temperature, topp, seed);
    }

    void forwardLayer(int pos, int i_layer) {
        Config* p = &cfg_;
        TransformerWeights* w = weights_;
        RunState *s = run_state_;

        int dim    = p->dim;
        int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
        int hidden_dim =  p->hidden_dim;
        int head_size  = dim / p->n_heads;

        // attention rmsnorm
        rmsnorm(s->xb, s->x, w->rms_att_weight[i_layer]);

        // key and value point to the kv cache
        s->k = s->key_cache[i_layer, pos];
        s->v = s->value_cache[i_layer, pos];

        // qkv matmuls for this position
        matmul(s->q, s->xb, w->wq[i_layer]);
        matmul(s->k, s->xb, w->wk[i_layer]);
        matmul(s->v, s->xb, w->wv[i_layer]);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                auto vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        int h;
        //#pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q.data() + h * head_size;
            // attention scores for this head
            auto att = s->att[h];
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = s->key_cache[i_layer, t].data() + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb.data() + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = s->value_cache[i_layer, t].data() + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo[i_layer]);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, s->x, w->rms_ffn_weight[i_layer]);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(s->hb,  s->xb, w->w1[i_layer]);
        matmul(s->hb2, s->xb, w->w3[i_layer]);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2[i_layer]);

        // residual connection
        for (int i = 0; i < dim; i++) {
            s->x[i] += s->xb[i];
        }
    }

    float* forward(int token, int pos) {
        // a few convenience variables
        Config* p = &cfg_;
        TransformerWeights* w = weights_;
        RunState *s = run_state_;
        //float    *x = s->x.data();  // activation at current time stamp (dim,)

        //int dim    = p->dim;

        // copy the token embedding into x
        auto content_row = w->token_embedding_table[token];
        //memcpy(s->x, content_row, dim*sizeof(*x));
        s->x.copy(content_row);

        // forward all the layers
        for(int l = 0; l < p->n_layers; l++)
            forwardLayer(pos, l);

        // final rmsnorm
        rmsnorm(s->x, s->x, w->rms_final_weight);

        // classifier into logits
        matmul(s->logits.data(), s->x.data(), w->wcls.raw(), p->dim, p->vocab_size); // fixme
        //matmul(s->logits, s->x, w->wcls);
        return s->logits.data();
    }

    void printPiece(char *piece) {
        // piece might be a raw byte token, and we only want to print printable chars or whitespace
        // because some of the other bytes can be various control codes, backspace, etc.
        if (piece == NULL) { return; }
        if (piece[0] == '\0') { return; }
        if (piece[1] == '\0') {
            unsigned char byte_val = piece[0];
            if (!(isprint(byte_val) || isspace(byte_val))) {
                return; // bad byte, don't print it
            }
        }
        std::cout << piece;
        std::cout.flush();
    }

    void generate(string prompt, int steps) {
        vector<int> prompt_tokens = tokenizer_->encode(prompt, 1, 0);
        if (prompt_tokens.size() == 0)
            throw_error("fail on prompt_tokens");

        // main loop
        //long start = 0;  // used to time our code, only initialized after first iteration
        int next;        // will store the next token in the sequence
        int token = prompt_tokens[0]; // kick off with the first token in the prompt
        int pos = 0;     // position in the sequence
        while (pos < steps) {
            float* logits = forward(token, pos);
            if (pos < prompt_tokens.size() - 1)
                next = prompt_tokens[pos + 1];
            else
                next = sampler_->sample(logits);
            pos ++;

            if (next == 1) break;

            // print the token as string, decode it with the Tokenizer object
            char* piece = tokenizer_->decode(token, next);
            printPiece(piece); // print string, but skips "unsafe" bytes

            token = next;
        }
        std::cout << std::endl;
    }
};


int main(int ac, char *av[])
{
    if (ac < 3) {
        std::cerr << "error: need model and tokenizer!\n";
        std::cout << "usage: run_llm path/to/model.bin path/to/tokenizer.bin\n";
        return -1;
    }

    LLM llm;
    llm.loadModel(av[1]);
    llm.loadTokenizer(av[2]);

    float temperature = 0.0f;
    float topp = 0.9f;
    unsigned long rng_seed = 1234;
    llm.initSampler(temperature, topp, rng_seed);

    //string prompt("She is beautiful");
    string prompt("");
    int steps = 256;
    llm.generate(prompt, steps);
}
