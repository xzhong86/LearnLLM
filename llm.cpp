// start with llama.c project, with tinystories model.

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
//#include <stdexcept>
#include <map>
#include <format>

#include <fcntl.h>     // for O_RDONLY
#include <unistd.h>    // for poxis open
#include <sys/mman.h>  // for poxis mmap

#define throw_error(MSG) do { throw std::runtime_error(MSG); } while (0)

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
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
};


// from llama2.c/run.c, but C++ version.
struct RunState {
    // current wave of activations
    vector<float> x; // activation at current time stamp (dim,)
    vector<float> xb; // same, but inside a residual branch (dim,)
    vector<float> xb2; // an additional buffer just for convenience (dim,)
    vector<float> hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    vector<float> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    vector<float> q; // query (dim,)
    vector<float> k; // key (dim,)
    vector<float> v; // value (dim,)
    vector<float> att; // buffer for scores/attention values (n_heads, seq_len)
    vector<float> logits; // output logits
    // kv cache
    vector<float> key_cache;   // (layer, seq_len, dim)
    vector<float> value_cache; // (layer, seq_len, dim)

    RunState(Config *p) {
        int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
        x.resize  (p->dim, 0);
        xb.resize (p->dim, 0);
        xb2.resize(p->dim, 0);
        hb.resize (p->hidden_dim, 0);
        hb2.resize(p->hidden_dim, 0);
        q.resize  (p->dim, 0);
        key_cache.resize  (p->n_layers * p->seq_len * kv_dim, 0);
        value_cache.resize(p->n_layers * p->seq_len * kv_dim, 0);
        att.resize(p->n_heads * p->seq_len, 0);
        logits.resize(p->vocab_size, 0);
    }
    ~RunState() {}
};

template <typename T>
inline void isread(T *p, std::istream &is) {
    is.read(reinterpret_cast<char*>(p), sizeof(T));
}
template <typename T>
inline void isread(T *p, int n, std::istream &is) {
    is.read(reinterpret_cast<char*>(p), sizeof(T) * n);
}

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
        isread(&max_token_length, ifs);
        int len;
        for (int i = 0; i < vocab_size; i++) {
            isread(&vocab_scores[i], ifs);
            isread(&len, ifs);
            vocab[i] = new char[len];
            isread(vocab[i], len, ifs);
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

};



// ----------------------------------------------------------------------------
// Sampler
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
};


class OSMemMap {
    int fd = -1;
    size_t file_size;
    void *ptr;
public:
    OSMemMap() {}
    OSMemMap(const char *path) { load(path); }
    ~OSMemMap() {
        if (fd != -1) {
            if (ptr != MAP_FAILED) { munmap(ptr, file_size); }
            close(fd);
        }
    }
    void load(const char *path) {
        if (fd != -1) throw_error("bad load twice.");
        fd = open(path, O_RDONLY);
        if (fd == -1) throw_error("open file failed.");
        ptr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) throw_error("mmap failed!\n");
    }
    template <typename T>
    T *getPtr() const { return static_cast<T*>(ptr); }
};

class LLM {
    string model_path_;
    Config cfg_;
    OSMemMap mmap_;
    TransformerWeights *weights_ = nullptr;
    RunState *run_state_ = nullptr;
    Tokenizer *tokenizer_ = nullptr;

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
        isread(&cfg_, ifs);
        ifs.close();
        dumpConfig();
        loadParametersMMap();
        run_state_ = new RunState(&cfg_);
    }
    void loadParametersMMap() {
        mmap_.load(model_path_.c_str());
        float *ptr = mmap_.getPtr<float>();
        weights_ = new TransformerWeights;
        auto *w = weights_;
        auto *p = &cfg_;
        bool shared_weights = p->vocab_size > 0;
        int head_size = p->dim / p->n_heads;
        long n_layers = p->n_layers;

        // from llama2.c/run.c
        w->token_embedding_table = ptr;
        ptr += p->vocab_size * p->dim;
        w->rms_att_weight = ptr;
        ptr += n_layers * p->dim;
        w->wq = ptr;
        ptr += n_layers * p->dim * (p->n_heads * head_size);
        w->wk = ptr;
        ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
        w->wv = ptr;
        ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
        w->wo = ptr;
        ptr += n_layers * (p->n_heads * head_size) * p->dim;
        w->rms_ffn_weight = ptr;
        ptr += n_layers * p->dim;
        w->w1 = ptr;
        ptr += n_layers * p->dim * p->hidden_dim;
        w->w2 = ptr;
        ptr += n_layers * p->hidden_dim * p->dim;
        w->w3 = ptr;
        ptr += n_layers * p->dim * p->hidden_dim;
        w->rms_final_weight = ptr;
        ptr += p->dim;
        ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
        ptr += p->seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
        w->wcls = shared_weights ? w->token_embedding_table : ptr;
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

    void generate(string prompt, int steps) {
        vector<int> prompt_tokens = tokenizer_->encode(prompt, 1, 0);
        if (prompt_tokens.size() == 0)
            throw_error("fail on prompt_tokens");

        // main loop
        ; // notice, continue here
    }
};


int main(int ac, char *av[])
{
    if (ac < 2) {
        std::cout << "error: need model!\n";
        return -1;
    }
    LLM llm;
    llm.loadModel(av[1]);
    llm.loadTokenizer("../llama2-c/tokenizer.bin");

    string prompt;
    int steps = 256;
    llm.generate(prompt, steps);
}
