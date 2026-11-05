#pragma once

// Scores one feature row with the ONNX model exported by ml/export_onnx.py.
//
// score   = -log10 P(normal)  (same as Python's ml/train.py; higher = more suspicious)
// pattern = the most likely non-normal class
//
// ONNX Runtime is hidden behind a pointer so other files don't need its headers, and so the
// engine still builds (without model scoring) when configured with -DTSA_WITH_ONNX=OFF.

#include <filesystem>
#include <memory>

#include "tsa/features.hpp"
#include "tsa/rules.hpp"

namespace tsa {

// False if the engine was built without ONNX Runtime.
bool model_support_available() noexcept;

struct ModelScore {
    double score = 0.0;
    Rule pattern = Rule::Spoofing;
};

class ModelScorer {
public:
    // Throws std::runtime_error if the file is missing, isn't a valid model, or has the wrong
    // input/output shape.
    explicit ModelScorer(const std::filesystem::path& onnx_path);
    ~ModelScorer();
    ModelScorer(const ModelScorer&) = delete;
    ModelScorer& operator=(const ModelScorer&) = delete;

    ModelScore score(const FeatureRow& row);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tsa
