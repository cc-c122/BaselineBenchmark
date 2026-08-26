#include <torch/extension.h>

#include <ATen/ATen.h>
#include <gcu/gcu_stream.h>
#include <gcu/gcu_utils.h>
#include <topsaten/topsaten_ops.h>

#include <tuple>
#include <vector>

namespace {

topsStream_t current_stream(const at::Tensor& tensor) {
  return torch_gcu::getCurrentGCUStream(tensor.get_device());
}

at::Tensor tops_cumsum(const at::Tensor& input, int64_t dim,
                       topsStream_t stream) {
  auto output = at::empty_like(input);
  auto output_t = torch_gcu::createTopsatenTensor(output);
  auto input_t = torch_gcu::createTopsatenTensor(input);
  torch_gcu::CHECK_TOPSATEN_CALL(topsaten::topsatenCumsum(
      output_t, input_t, static_cast<int32_t>(dim), TOPSATEN_DATA_FP32,
      stream));
  return output;
}

at::Tensor tops_exp(const at::Tensor& input, topsStream_t stream) {
  auto output = at::empty_like(input);
  auto output_t = torch_gcu::createTopsatenTensor(output);
  auto input_t = torch_gcu::createTopsatenTensor(input);
  torch_gcu::CHECK_TOPSATEN_CALL(
      topsaten::topsatenExp(output_t, input_t, stream));
  return output;
}

at::Tensor tops_bmm(const at::Tensor& lhs, const at::Tensor& rhs,
                    topsStream_t stream) {
  TORCH_CHECK(lhs.dim() == 3 && rhs.dim() == 3,
              "TopsAten BMM expects rank-3 tensors");
  auto output = at::empty(
      {lhs.size(0), lhs.size(1), rhs.size(2)}, lhs.options());
  auto output_t = torch_gcu::createTopsatenTensor(output);
  auto lhs_t = torch_gcu::createTopsatenTensor(lhs);
  auto rhs_t = torch_gcu::createTopsatenTensor(rhs);
  torch_gcu::CHECK_TOPSATEN_CALL(
      topsaten::topsatenBmm(output_t, lhs_t, rhs_t, stream));
  return output;
}

at::Tensor pad_sequence(const at::Tensor& input, int64_t padded_seqlen) {
  if (input.size(1) == padded_seqlen) {
    return input;
  }
  auto output = at::zeros(
      {input.size(0), padded_seqlen, input.size(2)}, input.options());
  output.slice(1, 0, input.size(1)).copy_(input);
  return output;
}

}  // namespace

std::tuple<at::Tensor, c10::optional<at::Tensor>> chunk_gla_tops_forward(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& g,
    double scale,
    const c10::optional<at::Tensor>& initial_state,
    const at::Tensor& causal_mask,
    bool output_final_state,
    bool state_v_first,
    int64_t chunk_size) {
  TORCH_CHECK(torch_gcu::is_gcu(q), "q must be a GCU tensor");
  TORCH_CHECK(q.dim() == 4 && k.sizes() == q.sizes() &&
                  g.sizes() == q.sizes(),
              "q, k and g must have shape [B, T, H, K]");
  TORCH_CHECK(v.dim() == 4 && v.size(0) == q.size(0) &&
                  v.size(1) == q.size(1) && v.size(2) == q.size(2),
              "v must have shape [B, T, H, V]");
  TORCH_CHECK(chunk_size > 0, "chunk_size must be positive");

  const int64_t batch = q.size(0);
  const int64_t seqlen = q.size(1);
  const int64_t heads = q.size(2);
  const int64_t key_dim = q.size(3);
  const int64_t value_dim = v.size(3);
  const int64_t batch_heads = batch * heads;
  const int64_t num_chunks =
      (seqlen + chunk_size - 1) / chunk_size;
  const int64_t padded_seqlen = num_chunks * chunk_size;
  const auto stream = current_stream(q);

  auto qf = q.transpose(1, 2).contiguous().to(at::kFloat)
                .reshape({batch_heads, seqlen, key_dim});
  auto kf = k.transpose(1, 2).contiguous().to(at::kFloat)
                .reshape({batch_heads, seqlen, key_dim});
  auto vf = v.transpose(1, 2).contiguous().to(at::kFloat)
                .reshape({batch_heads, seqlen, value_dim});
  auto gf = g.transpose(1, 2).contiguous().to(at::kFloat)
                .reshape({batch_heads, seqlen, key_dim});

  // For wide keys, materializing factors for every chunk increases memory
  // traffic more than it saves launches. Keep the lower-memory sequential
  // path; K<=64 uses the batched path below.
  if (key_dim > 64) {
    at::Tensor state;
    if (initial_state.has_value()) {
      state = initial_state.value().to(at::kFloat);
      if (state_v_first) {
        state = state.transpose(-1, -2);
      }
      state = state.contiguous().reshape(
          {batch_heads, key_dim, value_dim});
    } else {
      state = at::zeros(
          {batch_heads, key_dim, value_dim},
          q.options().dtype(at::kFloat));
    }

    std::vector<at::Tensor> outputs;
    outputs.reserve(num_chunks);
    for (int64_t start = 0; start < seqlen; start += chunk_size) {
      const int64_t stop = std::min(start + chunk_size, seqlen);
      const int64_t length = stop - start;
      auto q_chunk = qf.slice(1, start, stop);
      auto k_chunk = kf.slice(1, start, stop);
      auto v_chunk = vf.slice(1, start, stop);
      auto g_chunk = gf.slice(1, start, stop);

      auto g_cumsum = tops_cumsum(g_chunk, 1, stream);
      auto g_first = g_cumsum.select(1, 0);
      auto g_last = g_cumsum.select(1, length - 1);
      auto g_mid = (g_first + g_last) * 0.5;

      auto q_factor = q_chunk * tops_exp(
          g_cumsum - g_mid.unsqueeze(1), stream);
      q_factor.mul_(scale);
      auto k_factor = k_chunk * tops_exp(
          g_mid.unsqueeze(1) - g_cumsum, stream);

      auto scores = tops_bmm(
          q_factor, k_factor.transpose(1, 2), stream);
      auto mask = causal_mask.slice(0, 0, length).slice(1, 0, length);
      scores.masked_fill_(mask.logical_not().unsqueeze(0), 0.0);
      auto chunk_output = tops_bmm(scores, v_chunk, stream);

      auto q_state = q_chunk * tops_exp(g_cumsum, stream);
      q_state.mul_(scale);
      chunk_output.add_(tops_bmm(q_state, state, stream));
      outputs.push_back(chunk_output);

      auto decay = tops_exp(g_last, stream);
      auto update = tops_bmm(
          k_factor.transpose(1, 2), v_chunk, stream);
      update.mul_(
          tops_exp(g_last - g_mid, stream).unsqueeze(-1));
      state = state * decay.unsqueeze(-1) + update;
    }

    auto output = at::cat(outputs, 1)
                      .reshape({batch, heads, seqlen, value_dim})
                      .transpose(1, 2)
                      .to(q.scalar_type());
    if (!output_final_state) {
      return std::make_tuple(output, c10::nullopt);
    }
    auto final_state =
        state.reshape({batch, heads, key_dim, value_dim});
    if (state_v_first) {
      final_state =
          final_state.transpose(-1, -2).contiguous();
    }
    return std::make_tuple(output, final_state);
  }

  qf = pad_sequence(qf, padded_seqlen);
  kf = pad_sequence(kf, padded_seqlen);
  vf = pad_sequence(vf, padded_seqlen);
  gf = pad_sequence(gf, padded_seqlen);

  const auto gate_shape =
      std::vector<int64_t>{batch_heads, num_chunks, chunk_size, key_dim};
  auto q_blocks = qf.reshape(gate_shape);
  auto k_blocks = kf.reshape(gate_shape);
  auto g_blocks = gf.reshape(gate_shape);

  // All chunks are independent until the state recurrence. Hoisting these
  // operations replaces O(num_chunks) cumsum/exp launches with six launches.
  auto g_cumsum = tops_cumsum(g_blocks, 2, stream);
  auto g_first = g_cumsum.select(2, 0);
  auto g_last = g_cumsum.select(2, chunk_size - 1);
  auto g_mid = (g_first + g_last) * 0.5;

  auto q_factor = q_blocks * tops_exp(
      g_cumsum - g_mid.unsqueeze(2), stream);
  q_factor.mul_(scale);
  auto k_factor = k_blocks * tops_exp(
      g_mid.unsqueeze(2) - g_cumsum, stream);
  auto q_state = q_blocks * tops_exp(g_cumsum, stream);
  q_state.mul_(scale);
  auto decay = tops_exp(g_last, stream);
  auto update_scale = tops_exp(g_last - g_mid, stream);

  q_factor = q_factor.reshape({batch_heads, padded_seqlen, key_dim});
  k_factor = k_factor.reshape({batch_heads, padded_seqlen, key_dim});
  q_state = q_state.reshape({batch_heads, padded_seqlen, key_dim});

  at::Tensor state;
  if (initial_state.has_value()) {
    state = initial_state.value().to(at::kFloat);
    if (state_v_first) {
      state = state.transpose(-1, -2);
    }
    state = state.contiguous().reshape({batch_heads, key_dim, value_dim});
  } else {
    state = at::zeros(
        {batch_heads, key_dim, value_dim}, q.options().dtype(at::kFloat));
  }

  std::vector<at::Tensor> outputs;
  outputs.reserve(num_chunks);
  for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
    const int64_t start = chunk * chunk_size;
    const int64_t stop = start + chunk_size;
    const int64_t valid_length =
        std::min(chunk_size, seqlen - start);

    auto q_factor_chunk = q_factor.slice(1, start, stop);
    auto k_factor_chunk = k_factor.slice(1, start, stop);
    auto q_state_chunk = q_state.slice(1, start, stop);
    auto v_chunk = vf.slice(1, start, stop);

    auto scores = tops_bmm(
        q_factor_chunk, k_factor_chunk.transpose(1, 2), stream);
    scores.masked_fill_(causal_mask.logical_not().unsqueeze(0), 0.0);
    auto chunk_output = tops_bmm(scores, v_chunk, stream);
    chunk_output.add_(tops_bmm(q_state_chunk, state, stream));
    outputs.push_back(chunk_output.slice(1, 0, valid_length));

    auto update = tops_bmm(
        k_factor_chunk.transpose(1, 2), v_chunk, stream);
    auto chunk_decay = decay.select(1, chunk);
    auto chunk_update_scale = update_scale.select(1, chunk);
    state = state * chunk_decay.unsqueeze(-1) +
            update * chunk_update_scale.unsqueeze(-1);
  }

  auto output = at::cat(outputs, 1)
                    .reshape({batch, heads, seqlen, value_dim})
                    .transpose(1, 2)
                    .to(q.scalar_type());

  if (!output_final_state) {
    return std::make_tuple(output, c10::nullopt);
  }
  auto final_state = state.reshape({batch, heads, key_dim, value_dim});
  if (state_v_first) {
    final_state = final_state.transpose(-1, -2).contiguous();
  }
  return std::make_tuple(output, final_state);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &chunk_gla_tops_forward,
             "Chunk GLA forward using batched TopsAten C++ kernels");
}
