// start with llama.c project, with tinystories model.

#include <string>
#include <fstream>
#include <iostream>
//#include <stdexcept>

#include <fcntl.h>     // for O_RDONLY
#include <unistd.h>    // for poxis open
#include <sys/mman.h>  // for poxis mmap


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


using std::string;

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
        if (fd != -1) throw std::runtime_error("bad load twice.");
        fd = open(path, O_RDONLY);
        if (fd == -1) throw std::runtime_error("open file failed.");
        ptr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (ptr == MAP_FAILED) throw std::runtime_error("mmap failed!\n");
    }
    template <typename T>
    T *getPtr() const { return static_cast<T*>(ptr); }
};

class LLM {
    string model_path_;
    Config cfg_;
    OSMemMap mmap_;
    TransformerWeights *weights_ = nullptr;

public:
    LLM(string path) : model_path_(path) {
        loadModel();
    }
    void loadModel() {
        std::ifstream ifs(model_path_, std::ios::binary);
        if (!ifs.good()) throw std::runtime_error("open file failed.");
        ifs.read(reinterpret_cast<char*>(&cfg_), sizeof(cfg_));
        ifs.close();
        dumpConfig();

        loadParametersMMap();
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
};

int main(int ac, char *av[])
{
    if (ac < 2) {
        std::cout << "error: need model!\n";
        return -1;
    }
    LLM llm(av[1]);
}
