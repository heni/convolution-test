#include "traits.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <functional>
#include <cmath>

struct RecursiveTensorVisitor {
    std::vector<std::pair<size_t, size_t>> bounds;

    mutable struct {
        std::vector<size_t> pos;
        std::function<void(const std::vector<size_t>& pos)> fn;
    } iter_state;

    void iterate(size_t depth) const {
        if (depth == bounds.size()) {
            iter_state.fn(iter_state.pos);
        } else {
            for (size_t v = bounds[depth].first; v < bounds[depth].second; ++v) {
                iter_state.pos[depth] = v;
                iterate(depth+1);
            }
        }
    }

    void operator()(const std::function<void(const std::vector<size_t>& pos)>& fn) const {
        iter_state.fn = fn;
        iter_state.pos = std::vector<size_t>(bounds.size());

        iterate(0);
    }
};

struct TensorF32 {
    std::vector<size_t> dims;
    std::vector<float> data;

    static size_t capacity(const std::vector<size_t>& dims) {
        return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
    }

    TensorF32(const std::vector<size_t>&dims)
        : dims(dims),
        data(capacity(dims))
    {}

    size_t index(const std::vector<size_t>& indices) const {
        assert(indices.size() <= dims.size());

        size_t acc = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            acc = acc * dims[i] + indices[i];
        }
        return acc;
    }

    float& at(const std::vector<size_t>& indices) {
        return data[index(indices)];
    }

    const float& at(const std::vector<size_t>& indices) const {
        return data[index(indices)];
    }

    template<typename... Args,
        typename = std::enable_if_t<are_all_convertible<size_t, Args...>::value>>
    size_t index(Args&& ...args) const {
        return index({static_cast<size_t>(args)...});
    }

    template<typename... Args,
        typename = std::enable_if_t<are_all_convertible<size_t, Args...>::value>>
    float& at(Args&& ...args) {
        return data[index({static_cast<size_t>(args)...})];
    }

    template<typename... Args,
        typename = std::enable_if_t<are_all_convertible<size_t, Args...>::value>>
    const float& at(Args&& ...args) const {
        return data[index({static_cast<size_t>(args)...})];
    }

    TensorF32& copy_from(const TensorF32& src, const std::vector<size_t>& offsets) {
        const size_t D = dims.size();
        assert(src.dims.size() == D && offsets.size() == D);
        assert(offsets[D-1] + src.dims[D-1] <= dims[D-1]);

        std::vector<std::pair<size_t, size_t>> bounds(D - 1);
        std::generate(bounds.begin(), bounds.end(), [n=0, this, &src, &offsets]() mutable {
            const size_t i = n++;
            assert(offsets[i] + src.dims[i] <= dims[i]);
            return std::pair<size_t, size_t>{offsets[i], offsets[i] + src.dims[i]};
        });

        RecursiveTensorVisitor{.bounds = bounds}([this, &src, &offsets, D](const std::vector<size_t>& pos){
            std::vector<size_t> src_pos(pos.size());
            std::transform(pos.begin(), pos.end(), offsets.begin(), src_pos.begin(), std::minus<size_t>{});

            const size_t src_index = src.index(src_pos) * src.dims[D-1];
            const size_t dst_index = index(pos) * dims[D-1] + offsets[D-1];

            std::copy_n(src.data.begin() + src_index, src.dims[D-1], data.begin() + dst_index);
        });

        return *this;
    }

    TensorF32 pad(const std::vector<std::pair<size_t, size_t>>& paddings) const {
        assert(paddings.size() == dims.size());

        std::vector<size_t> pdims(dims.size());
        std::generate(pdims.begin(), pdims.end(), [n=0, this, &paddings] () mutable {
            const size_t i = n++;
            return dims[i] + paddings[i].first + paddings[i].second;
        });

        std::vector<size_t> offsets(dims.size());
        std::transform(paddings.begin(), paddings.end(), offsets.begin(), [](const auto& val) {
            return val.first;
        });

        return TensorF32(pdims).copy_from(*this, offsets);
    }

    TensorF32& transform(const std::function<float(float)>& f) {
        for (auto& v: data) {
            v = f(v);
        }
        return *this;
    }

    float compare(const TensorF32& other) const {
        assert(data.size() == other.data.size() && dims == other.dims);
        float cmp_res = 0.0;
        for (size_t i = 0; i < data.size(); ++i) {
            cmp_res += (std::abs(data[i] - other.data[i]) - cmp_res) / (i+1);
        }
        return cmp_res;
    }
};


TensorF32 load_tensor_from_file(const std::string& fpath, const std::vector<size_t>& dims) {
    TensorF32 res(dims);

    std::ifstream in(fpath);
    assert(in);

    assert(in.seekg(128, std::ios_base::cur));
    assert(in.read(
        const_cast<char*>(reinterpret_cast<const char*>(res.data.data())),
        res.data.size() * sizeof(res.data[0])
    ));

    return res;
}

float silu(float val) {
    val = std::max<float>(-15.0, val);
    return val / (1 + exp(-val));
}

TensorF32 conv2d(const TensorF32& maps, const TensorF32& filters, size_t stride=1) {
    const auto convolve_r = [](const float* m_ptr, const float* f_ptr, size_t count) {
#if defined(STL_CONVOLVE) && (STL_CONVOLVE)
        return std::inner_product(m_ptr, m_ptr + count, f_ptr, 0.0);
#else
        float res = 0.0;
        for (size_t i = 0; i < count; ++i) {
            res += *m_ptr * *f_ptr;
            ++m_ptr;
            ++f_ptr;
        }
        return res;
#endif
    };

    assert(maps.dims.size() == 3 && filters.dims.size() == 4);
    assert(maps.dims[0] == maps.dims[1]);
    assert(filters.dims[1] == filters.dims[2]);
    assert(maps.dims[2] == filters.dims[3]);

    const size_t N = maps.dims[0], F = filters.dims[0], K = filters.dims[1], D = maps.dims[2];
    const size_t p = K / 2;

    const auto st_time = std::chrono::steady_clock::now();
    TensorF32 padded_maps = maps.pad({{p, p-1}, {p, p-1}, {0,0}});
    const auto end_time = std::chrono::steady_clock::now();
    std::cout << "padding time: " << std::chrono::duration<double>(end_time - st_time).count() << std::endl;

    TensorF32 res({N/stride, N/stride, F});

    size_t out_index = 0;
    for (size_t i = 0; i < N; i += stride) {
        for (size_t j = 0; j < N; j += stride) {
            for(size_t f = 0; f < F; ++f) {
                float conv_val = 0.0;
                for (size_t k = 0; k < K; ++k) {
                    conv_val += convolve_r(&padded_maps.at(i+k, j, 0), &filters.at(f, k, 0, 0), K * D);
                }
                res.data[out_index++] = conv_val;
            }
        }
    }

    return res;
}

int main() {
    const auto start_tm = std::chrono::steady_clock::now();
    TensorF32 image = load_tensor_from_file("tank.npy", {640, 640, 3});
    TensorF32 filters = load_tensor_from_file("filters.npy", {64, 3, 3, 3});
    TensorF32 expected = load_tensor_from_file("expected.npy", {320, 320, 64});

    const auto read_tm = std::chrono::steady_clock::now();
    std::cout << "read time: " << std::chrono::duration<double>(read_tm - start_tm).count() << std::endl;

    /* Index tests:
    std::cout << image.at(320, 320, 0) << "," << image.at(320, 320, 1) << "," << image.at(320, 320, 2) << std::endl;
    std::cout << image.index(320, 320, 0) << std::endl;
    */

    TensorF32 res = conv2d(image, filters, 2);
    res.transform(silu);
    const auto conv_tm = std::chrono::steady_clock::now();
    std::cout << "conv2d time: " << std::chrono::duration<double>(conv_tm - read_tm).count() << std::endl;

    std::cout << "conv2d results for [0,0]: ";
    for (size_t i = 0; i < 64; ++i) {
        std::cout << res.data[i] << ",";
    }
    std::cout << std::endl;

    std::cout << "Difference with expected: " << res.compare(expected) << std::endl;

    return 0;
}

