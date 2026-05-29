#pragma once

#ifdef NAM_SAMPLE_FLOAT
  #define NAM_SAMPLE float
#else
  #define NAM_SAMPLE double
#endif

namespace nam
{
namespace wavenet
{
/// Forward declaration to allow WaveNet to access protected members of DSP
class WaveNet;
} // namespace wavenet

/// \brief Base class for all DSP models
///
/// DSP provides the common interface for all neural network-based audio processing models.
/// It handles:
/// - Input/output channel management
/// - Sample rate tracking
/// - Level management (input/output levels and loudness)
/// - Prewarm functionality for settling initial conditions
/// - Buffer size management
///
/// Subclasses should override process() to implement the actual processing algorithm.
class DSP
{
public:
  /// \brief Constructor
  ///
  /// \param in_channels Number of input channels
  /// \param out_channels Number of output channels
  /// \param expected_sample_rate Expected sample rate in Hz (-1.0 if unknown)
  DSP(const int in_channels, const int out_channels, const double expected_sample_rate);

  /// \brief Virtual destructor
  virtual ~DSP() = default;

  /// \brief Prewarm the model to settle initial conditions
  ///
  /// This can be somewhat expensive, so should not be called during real-time audio processing.
  /// Important: don't expect the model to be outputting zeroes after this. Neural networks
  /// don't know that there's anything special about "zero", and forcing this gets rid of
  /// some possibilities (e.g. models that "are noisy").
  virtual void prewarm();

  /// \brief Process audio frames
  ///
  /// \param input Input audio buffers. Double pointer where the first pointer indexes channels
  ///              and the second indexes frames: input[channel][frame]
  /// \param output Output audio buffers. Same structure as input.
  /// \param num_frames Number of frames to process
  virtual void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames);
  /// \brief Get the expected sample rate
  /// \return Expected sample rate in Hz (-1.0 if unknown)
  double GetExpectedSampleRate() const { return mExpectedSampleRate; };

  /// \brief Get the number of input channels
  /// \return Number of input channels
  int NumInputChannels() const { return mInChannels; };

  /// \brief Get the number of output channels
  /// \return Number of output channels
  int NumOutputChannels() const { return mOutChannels; };

  /// \brief Get the input level
  ///
  /// Input level is in dBu RMS, corresponding to 0 dBFS peak for a 1 kHz sine wave.
  /// You should call HasInputLevel() first to be safe.
  /// Note: input level is assumed global over all inputs.
  /// \return Input level in dBu
  double GetInputLevel();

  /// \brief Get how loud this model's output is, in dB, if a "typical" input is processed.
  /// This can be used to normalize the output level of the object.
  ///
  /// Throws a std::runtime_error if the model doesn't know how loud it is.
  /// Note: loudness is assumed global over all outputs.
  /// \return Loudness in dB
  /// \throws std::runtime_error If the model doesn't know its loudness
  double GetLoudness() const;

  /// \brief Get the output level
  ///
  /// Output level is in dBu RMS, corresponding to 0 dBFS peak for a 1 kHz sine wave.
  /// You should call HasOutputLevel() first to be safe.
  /// Note: output level is assumed global over all outputs.
  /// \return Output level in dBu
  double GetOutputLevel();

  /// \brief Check if this model knows its input level
  ///
  /// Note: input level is assumed global over all inputs.
  /// \return true if input level is known, false otherwise
  bool HasInputLevel();

  /// \brief Check if the model knows how loud it is
  /// \return true if loudness is known, false otherwise
  bool HasLoudness() const { return mHasLoudness; };

  /// \brief Check if this model knows its output level
  ///
  /// Note: output level is assumed global over all outputs.
  /// \return true if output level is known, false otherwise
  bool HasOutputLevel();

  /// \brief General function for resetting the DSP unit
  ///
  /// This doesn't call prewarm(). If you want to do that, then you might want to use ResetAndPrewarm().
  /// See https://github.com/sdatkinson/NeuralAmpModelerCore/issues/96 for the reasoning.
  /// \param sampleRate Current sample rate
  /// \param maxBufferSize Maximum buffer size to process
  virtual void Reset(const double sampleRate, const int maxBufferSize);

  /// \brief Reset the DSP unit, then prewarm
  /// \param sampleRate Current sample rate
  /// \param maxBufferSize Maximum buffer size to process
  void ResetAndPrewarm(const double sampleRate, const int maxBufferSize)
  {
    Reset(sampleRate, maxBufferSize);
    prewarm();
  }

  /// \brief Set the input level
  /// \param inputLevel Input level in dBu
  void SetInputLevel(const double inputLevel);

  /// \brief Set the loudness
  ///
  /// This is usually defined to be the loudness to a standardized input. The trainer has its own,
  /// but you can always use this to define it a different way if you like yours better.
  /// Note: loudness is assumed global over all outputs.
  /// \param loudness Loudness in dB
  void SetLoudness(const double loudness);

  /// \brief Set the output level
  /// \param outputLevel Output level in dBu
  void SetOutputLevel(const double outputLevel);

protected:
  friend class wavenet::WaveNet; // Allow WaveNet to access protected members. Used in condition DSP.

  bool mHasLoudness = false;
  // How loud is the model? In dB
  double mLoudness = 0.0;
  // What sample rate does the model expect?
  double mExpectedSampleRate;
  // Have we been told what the external sample rate is? If so, what is it?
  bool mHaveExternalSampleRate = false;
  double mExternalSampleRate = -1.0;
  // The largest buffer I expect to be told to process:
  int mMaxBufferSize = 0;

  /// \brief Get how many samples should be processed for the model to be considered "warmed up"
  ///
  /// Override this in subclasses to specify prewarm requirements.
  /// \return Number of samples needed for prewarm
  virtual int PrewarmSamples() { return 0; };

  /// \brief Set the maximum buffer size
  /// \param maxBufferSize Maximum number of frames to process in a single call
  virtual void SetMaxBufferSize(const int maxBufferSize);

  /// \brief Get the maximum buffer size
  /// \return Maximum buffer size
  int GetMaxBufferSize() const { return mMaxBufferSize; };

private:
  const int mInChannels;
  const int mOutChannels;
  struct Level
  {
    bool haveLevel = false;
    float level = 0.0;
  };
  // Note: input/output levels are assumed global over all inputs/outputs
  Level mInputLevel;
  Level mOutputLevel;
};

}; // namespace nam
