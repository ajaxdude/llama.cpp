#include "llama-ple-disk.h"

#include "llama-bounded-file.h"
#include "llama-impl.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

struct llama_ple_disk::impl {
    std::string fname;
    bool        direct = false;
    size_t      offs   = 0;
    ggml_type   type   = GGML_TYPE_COUNT;
    int64_t     ne0    = 0;
    int64_t     nrows  = 0;
    size_t      rs     = 0;     // bytes per row
    ggml_to_float_t to_float = nullptr;
    std::unique_ptr<llama_bounded_file> file;

    // direct-mapped cache of raw rows: slot = row & mask, tag = row
    size_t               n_slots = 0;
    size_t               mask    = 0;
    std::vector<int64_t> tags;
    std::vector<uint8_t> slab;

    // per-gather scratch, reused
    std::vector<int32_t>                       uniq;
    std::vector<uint8_t>                       raw;      // [uniq.size(), rs]
    std::vector<std::pair<int32_t, uint32_t>>  misses;   // (row, index into uniq)
    llama_bounded_file::buffer                 bounce0;

    std::mutex mtx; // one gather at a time; a model is shared by every context built on it

    // reader pool, started on first use
    int32_t                  n_threads = 1;
    std::vector<std::thread> workers;
    std::mutex               pm;
    std::condition_variable  cv_work;
    std::condition_variable  cv_done;
    uint64_t                 gen     = 0;
    size_t                   pending = 0;
    std::atomic<size_t>      next{0};
    bool                     stop    = false;

    uint64_t st_calls = 0, st_rows = 0, st_uniq = 0, st_hits = 0, st_reads = 0, st_bytes = 0;
    double   st_ms = 0;

    impl(const std::string & fname, size_t offs, ggml_type type, int64_t ne0, int64_t nrows, const params & p)
        : fname(fname), offs(offs), type(type), ne0(ne0), nrows(nrows) {
#if defined(_WIN32)
        throw std::runtime_error("llama_ple_disk: not supported on Windows");
#else
        if (ne0 <= 0 || nrows <= 0) {
            throw std::runtime_error("llama_ple_disk: invalid table shape");
        }
        if (ggml_is_quantized(type) && ne0 % ggml_blck_size(type) != 0) {
            throw std::runtime_error(format("llama_ple_disk: row of %lld %s elements is not a whole number of blocks",
                                            (long long) ne0, ggml_type_name(type)));
        }
        rs = ggml_row_size(type, ne0);
        if (type != GGML_TYPE_F32) {
            to_float = ggml_get_type_traits(type)->to_float;
            if (to_float == nullptr) {
                throw std::runtime_error(format("llama_ple_disk: no dequantizer for %s", ggml_type_name(type)));
            }
        }

        llama_bounded_file::params fp;
        fp.direct_io = p.direct_io;
        file = std::make_unique<llama_bounded_file>(fname, fp);
        direct = file->direct_io();
        if (rs == 0 || (uint64_t) nrows > UINT64_MAX / rs) {
            throw std::runtime_error("llama_ple_disk: table extent overflow");
        }
        const uint64_t bytes = (uint64_t) nrows * rs;
        if (offs > file->size() || bytes > file->size() - offs) {
            throw std::runtime_error("llama_ple_disk: table extent is outside the file");
        }

        n_threads = std::max<int32_t>(1, p.n_threads);

        if (p.cache_bytes >= rs) {
            size_t n = p.cache_bytes / rs;
            n_slots = 1;
            while (n_slots <= n / 2) {
                n_slots *= 2;
            }
            mask = n_slots - 1;
            tags.assign(n_slots, -1);
            slab.resize(n_slots * rs);
        }
#endif
    }

    ~impl() {
#if !defined(_WIN32)
        {
            std::lock_guard<std::mutex> lk(pm);
            stop = true;
        }
        cv_work.notify_all();
        for (auto & w : workers) {
            w.join();
        }
#endif
    }

#if !defined(_WIN32)
    void read_row(int64_t row, uint8_t * dst, llama_bounded_file::buffer & bounce) const {
        const uint64_t off = (uint64_t) offs + (uint64_t) row * rs;
        file->read(off, dst, rs, bounce);
    }

    void worker() {
        llama_bounded_file::buffer bounce = file->make_buffer(rs);
        uint64_t  seen   = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lk(pm);
                cv_work.wait(lk, [&] { return stop || gen != seen; });
                if (stop) {
                    break;
                }
                seen = gen;
            }
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= misses.size()) {
                    break;
                }
                read_row(misses[i].first, raw.data() + (size_t) misses[i].second * rs, bounce);
            }
            {
                std::lock_guard<std::mutex> lk(pm);
                if (--pending == 0) {
                    cv_done.notify_one();
                }
            }
        }
    }

    void run_misses() {
        if (n_threads <= 1 || misses.size() <= 2) {
            if (direct && bounce0.empty()) {
                bounce0 = file->make_buffer(rs);
            }
            for (const auto & m : misses) {
                read_row(m.first, raw.data() + (size_t) m.second * rs, bounce0);
            }
            return;
        }
        if (workers.empty()) {
            workers.reserve(n_threads);
            for (int32_t i = 0; i < n_threads; ++i) {
                workers.emplace_back([this] { worker(); });
            }
        }
        {
            std::lock_guard<std::mutex> lk(pm);
            next    = 0;
            pending = workers.size();
            ++gen;
        }
        cv_work.notify_all();
        std::unique_lock<std::mutex> lk(pm);
        cv_done.wait(lk, [&] { return pending == 0; });
    }

    void gather(const int32_t * idx, size_t n, float * dst) {
        std::lock_guard<std::mutex> lk(mtx);
        const auto t0 = std::chrono::steady_clock::now();

        uniq.assign(idx, idx + n);
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        if (!uniq.empty() && (uniq.front() < 0 || (int64_t) uniq.back() >= nrows)) {
            GGML_ABORT("llama_ple_disk: row index out of range (%d..%d of %lld rows)",
                       uniq.front(), uniq.back(), (long long) nrows);
        }

        raw.resize(uniq.size() * rs);
        misses.clear();
        for (size_t i = 0; i < uniq.size(); ++i) {
            if (n_slots) {
                const size_t slot = (size_t) uniq[i] & mask;
                if (tags[slot] == uniq[i]) {
                    memcpy(raw.data() + i * rs, slab.data() + slot * rs, rs);
                    continue;
                }
            }
            misses.emplace_back(uniq[i], (uint32_t) i);
        }

        if (!misses.empty()) {
            run_misses();
            if (n_slots) {
                for (const auto & m : misses) {
                    const size_t slot = (size_t) m.first & mask;
                    tags[slot] = m.first;
                    memcpy(slab.data() + slot * rs, raw.data() + (size_t) m.second * rs, rs);
                }
            }
        }

        for (size_t k = 0; k < n; ++k) {
            const size_t    i   = (size_t) (std::lower_bound(uniq.begin(), uniq.end(), idx[k]) - uniq.begin());
            const uint8_t * src = raw.data() + i * rs;
            float *         out = dst + k * (size_t) ne0;
            if (to_float) {
                to_float(src, out, ne0);
            } else {
                memcpy(out, src, rs);
            }
        }

        st_calls += 1;
        st_rows  += n;
        st_uniq  += uniq.size();
        st_hits  += uniq.size() - misses.size();
        st_reads += misses.size();
        for (const auto & miss : misses) {
            st_bytes += file->read_size((uint64_t) offs + (uint64_t) miss.first * rs, rs);
        }
        st_ms    += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
#endif
};

llama_ple_disk::llama_ple_disk(const std::string & fname, size_t offs, ggml_type type, int64_t ne0, int64_t nrows, const params & p)
    : pimpl(std::make_unique<impl>(fname, offs, type, ne0, nrows, p)) {}

llama_ple_disk::~llama_ple_disk() {
    if (pimpl->st_calls) {
        LLAMA_LOG_INFO("%s: %llu batches, %llu rows gathered (%llu unique): %llu cache hits, %llu disk reads (%.1f MiB), %.1f ms total\n",
                       __func__, (unsigned long long) pimpl->st_calls, (unsigned long long) pimpl->st_rows,
                       (unsigned long long) pimpl->st_uniq, (unsigned long long) pimpl->st_hits,
                       (unsigned long long) pimpl->st_reads, pimpl->st_bytes / (1024.0 * 1024.0), pimpl->st_ms);
    }
}

void llama_ple_disk::gather(const int32_t * idx, size_t n, float * dst) {
#if defined(_WIN32)
    GGML_UNUSED(idx); GGML_UNUSED(n); GGML_UNUSED(dst);
    GGML_ABORT("llama_ple_disk: not supported on Windows");
#else
    pimpl->gather(idx, n, dst);
#endif
}

int64_t llama_ple_disk::n_rows()   const { return pimpl->nrows; }
int64_t llama_ple_disk::ne0()      const { return pimpl->ne0;   }
size_t  llama_ple_disk::row_size() const { return pimpl->rs;    }

std::string llama_ple_disk::describe() const {
    return format("%s @ %zu: %lld rows x %lld %s (%zu B/row, %.2f GiB), %s I/O, %d threads, row cache %zu MiB",
                  pimpl->fname.c_str(), pimpl->offs, (long long) pimpl->nrows, (long long) pimpl->ne0,
                  ggml_type_name(pimpl->type), pimpl->rs, (double) pimpl->nrows * pimpl->rs / (1024.0 * 1024.0 * 1024.0),
                  pimpl->direct ? "direct" : "buffered", pimpl->n_threads,
                  pimpl->n_slots * pimpl->rs / (1024 * 1024));
}
