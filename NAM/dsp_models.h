#pragma once

#include <atomic>
#include <filesystem>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "activations.h"
#include "compiler.h"
#include "model_config.h"
#include "dsp.h"

/// \brief Default max buffer size used by prewarm() when none has been set.
/// Override at compile time with -DNAM_DEFAULT_MAX_BUFFER_SIZE=N.
#ifndef NAM_DEFAULT_MAX_BUFFER_SIZE
  #define NAM_DEFAULT_MAX_BUFFER_SIZE 4096
#endif

/// \brief Use a sample rate of -1 if we don't know what the model expects to be run at
#define NAM_UNKNOWN_EXPECTED_SAMPLE_RATE -1.0

namespace nam
{
/// \brief Temporarily change the thread-local prewarm-on-reset default for newly constructed DSP objects
///
/// Existing DSP objects are not affected. DSP instances constructed while this object is alive on the current thread
/// copy the scoped default into their instance-level prewarm-on-reset setting.
class ScopedPrewarmOnResetDefault
{
public:
  explicit ScopedPrewarmOnResetDefault(const bool prewarmOnReset);
  ~ScopedPrewarmOnResetDefault();

  ScopedPrewarmOnResetDefault(const ScopedPrewarmOnResetDefault&) = delete;
  ScopedPrewarmOnResetDefault& operator=(const ScopedPrewarmOnResetDefault&) = delete;

  bool PreviousPrewarmOnReset() const { return mPreviousPrewarmOnReset; }

private:
  bool mPreviousPrewarmOnReset;
};

/// \brief Base class for DSP models that require input buffering
/// This class is deprecated and will be removed in a future version.
///
/// Class where an input buffer is kept so that long-time effects can be captured.
/// (e.g. conv nets or impulse responses, where we need history that's longer than
/// the sample buffer that's coming in.)
class Buffer : public DSP
{
public:
  /// \brief Constructor
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param receptive_field Size of the receptive field (buffer size needed)
  /// \param expected_sample_rate Expected sample rate in Hz (-1.0 if unknown)
  Buffer(const int in_channels, const int out_channels, const int receptive_field,
         const double expected_sample_rate = -1.0);

protected:
  int _receptive_field;
  // First location where we add new samples from the input (same for all channels)
  long _input_buffer_offset;
  // Per-channel input buffers
  std::vector<std::vector<float>> _input_buffers;
  std::vector<std::vector<float>> _output_buffers;

  void _advance_input_buffer_(const int num_frames);
  void _set_receptive_field(const int new_receptive_field, const int input_buffer_size);
  void _set_receptive_field(const int new_receptive_field);
  void _reset_input_buffer();
  // Use this->_input_post_gain
  virtual void _update_buffers_(NAM_SAMPLE** input, int num_frames);
  virtual void _rewind_buffers_();
};

// NN modules =================================================================

/// \brief 1x1 convolution (really just a fully-connected linear layer operating per-sample)
///
/// Performs a pointwise convolution, which is equivalent to a fully connected layer
/// applied independently to each time step. Supports grouped convolution for efficiency.
class Conv1x1
{
public:
  /// \brief Constructor
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param _bias Whether to use bias
  /// \param groups Number of groups for grouped convolution (default: 1)
  Conv1x1(const int in_channels, const int out_channels, const bool _bias, const int groups = 1);

  /// \brief Get the entire internal output buffer
  ///
  /// This is intended for internal wiring between layers/arrays; callers should treat
  /// the buffer as pre-allocated storage and only consider the first num_frames columns
  /// valid for a given processing call. Slice with .leftCols(num_frames) as needed.
  /// \return Reference to the output buffer
  Eigen::MatrixXf& GetOutput() { return _output; }

  /// \brief Get the entire internal output buffer (const version)
  /// \return Const reference to the output buffer
  const Eigen::MatrixXf& GetOutput() const { return _output; }

  /// \brief Resize the output buffer to handle maxBufferSize frames
  /// \param maxBufferSize Maximum number of frames to process in a single call
  void SetMaxBufferSize(const int maxBufferSize);

  /// \brief Set the parameters (weights) of this module
  /// \param weights Iterator to the weights vector. Will be advanced as weights are consumed.
  void set_weights_(std::vector<float>::iterator& weights);

  /// \brief Process input and return output matrix
  ///
  /// \param input Input matrix (channels x num_frames) or (channels,)
  /// \return Output matrix (channels x num_frames) or (channels,), respectively
  Eigen::MatrixXf process(const Eigen::MatrixXf& input) const { return process(input, (int)input.cols()); };

  /// \brief Process input and return output matrix
  /// \param input Input matrix (channels x num_frames)
  /// \param num_frames Number of frames to process
  /// \return Output matrix (channels x num_frames)
  Eigen::MatrixXf process(const Eigen::MatrixXf& input, const int num_frames) const;

  /// \brief Process input and store output to pre-allocated buffer
  ///
  /// Uses Eigen::Ref to accept matrices and block expressions without creating
  /// temporaries (real-time safe). Access output via GetOutput().
  /// \param input Input matrix (channels x num_frames)
  /// \param num_frames Number of frames to process
  void process_(const Eigen::Ref<const Eigen::MatrixXf>& input, const int num_frames);

  long get_out_channels() const;
  long get_in_channels() const;

protected:
  // Non-depthwise: full weight matrix (out_channels x in_channels)
  Eigen::MatrixXf _weight;
  // For depthwise convolution (groups == in_channels == out_channels):
  // stores one weight per channel
  Eigen::VectorXf _depthwise_weight;
  bool _is_depthwise = false;
  int _channels = 0; // Used for depthwise case (in_channels == out_channels)
  Eigen::VectorXf _bias;
  int _num_groups;

private:
  Eigen::MatrixXf _output;
  bool _do_bias;
};

// Utilities ==================================================================
// Implemented in get_dsp.cpp

/// \brief Verify that the config version is supported by this plugin version
/// \param version Config version string to verify
void verify_config_version(const std::string version);

/// \brief Legacy loader for directory-style DSPs
///
/// Loads models from a directory structure (older format).
/// \param dirname Path to the directory containing the model
/// \return Unique pointer to a DSP object
std::unique_ptr<DSP> get_dsp_legacy(const std::filesystem::path dirname);
}; // namespace nam

#include "linear.h"
