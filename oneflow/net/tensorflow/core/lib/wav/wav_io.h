/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Functions to write audio in WAV format.

#ifndef TENSORFLOW_LIB_WAV_WAV_IO_H_
#define TENSORFLOW_LIB_WAV_WAV_IO_H_

#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace wav {

// Encode the provided interleaved buffer of audio as a signed 16-bit PCM
// little-endian WAV file.
//
// Example usage for 4 frames of an 8kHz stereo signal:
// First channel is -1, 1, -1, 1.
// Second channel is 0, 0, 0, 0.
//
// float audio_buffer[] = { -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f};
// string wav_string;
// if (EncodeAudioAsS16LEWav(audio_buffer, 8000, 2, 4, &wav_string).ok()) {
//   // Use wav_string.
// }
Status EncodeAudioAsS16LEWav(const float* audio, size_t sample_rate,
                             size_t num_channels, size_t num_frames,
                             string* wav_string);

}  // namespace wav
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_WAV_WAV_IO_H_
