#include <Arduino.h>
#include <arduinoFFT.h>
#include "driver/i2s.h"

#define I2S_WS   18
#define I2S_SD   38
#define I2S_SCK  17
#define I2S_PORT I2S_NUM_0

constexpr uint16_t SAMPLING_RATE = 16000;
constexpr uint16_t I2S_BUFFER_LEN  = 256;
constexpr uint16_t WINDOW_SIZE = 8192;

constexpr float DRONE_LOW  = 175.0f;
constexpr float DRONE_HIGH = 425.0f;

constexpr uint16_t RECORD_SECONDS = 2;
constexpr uint32_t RECORD_WINDOWS_TO_SEND = (SAMPLING_RATE * RECORD_SECONDS + WINDOW_SIZE - 1) / WINDOW_SIZE;

// I2S input buffer
int32_t i2sBuffer[I2S_BUFFER_LEN];


// arduinoFFT arrays
float vReal[WINDOW_SIZE];
float vImag[WINDOW_SIZE];

// buffers for detecting and recording audio
int32_t currentWindow[WINDOW_SIZE];
uint16_t currentWindowIndex = 0;
int32_t recordBuffer[WINDOW_SIZE];
uint32_t recordIndex = 0;
uint16_t recordWindowsSent = 0;

bool isRecording = false;
bool recordDone = false;
bool sentHeader = false;

// arduinoFFT instance
ArduinoFFT<float> FFT = ArduinoFFT<float>(vReal, vImag, WINDOW_SIZE, SAMPLING_RATE);

bool detectDroneWindow(const int32_t* signal, float& peakFreqOut) {
  // load FFT input
  for (uint16_t i = 0; i < WINDOW_SIZE; i++) {
    vReal[i] = (float)signal[i];
    vImag[i] = 0.0f;
  }

  // remove DC component
  FFT.dcRemoval();

  // apply Hamming window
  FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);

  // compute FFT
  FFT.compute(FFTDirection::Forward);

  // convert to magnitude
  FFT.complexToMagnitude();

  // find peak
  peakFreqOut = FFT.majorPeak();

  // check if peak frequency is within drone range
  return (peakFreqOut >= DRONE_LOW && peakFreqOut <= DRONE_HIGH);
}

void sendRecordingHeader() {
  uint32_t totalSamples = RECORD_WINDOWS_TO_SEND * WINDOW_SIZE;
  uint32_t totalBytes   = totalSamples * sizeof(int32_t);

  Serial.print("WAVSIZE ");
  Serial.println(totalBytes);

  sentHeader = true;
}

void transferWindow() {
  // send raw 32-bit PCM samples
  Serial.write((uint8_t*)recordBuffer, WINDOW_SIZE * sizeof(int32_t));
  recordIndex = 0;
}

void handleInput(const int32_t* samples, size_t sampleCount) {
  for (size_t i = 0; i < sampleCount; i++) {
    int32_t sample = samples[i];

    // if recording audio, store in record buffer
    if (isRecording && !recordDone) {
      if (!sentHeader) {
        sendRecordingHeader();
      }

      recordBuffer[recordIndex++] = sample;

      if (recordIndex >= WINDOW_SIZE) {
        transferWindow();
        recordWindowsSent++;

        if (recordWindowsSent >= RECORD_WINDOWS_TO_SEND) {
          isRecording = false;
          recordDone = true;
        }
      } 

      continue;
    }

    // store current sample for filter
    currentWindow[currentWindowIndex++] = sample;

    // pass through filter once window is full
    if (currentWindowIndex >= WINDOW_SIZE) {
      float peakFreq = 0.0f;
      bool detected = detectDroneWindow(currentWindow, peakFreq);

      if (detected) {
        isRecording = true;
        recordDone = false;
        recordIndex = 0;
        recordWindowsSent = 0;
        sentHeader = false;
      }

      // reset index for next window
      currentWindowIndex = 0;
    }
  }
}

void setup() {
  Serial.begin(921600);
  delay(1000);

  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLING_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = 0,
    .dma_buf_count = 8,
    .dma_buf_len = 64,
    .use_apll = false
  };

  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_PORT, &pin_config);
}

void loop() {
  size_t bytesRead = 0;
  i2s_read(I2S_PORT, i2sBuffer, sizeof(i2sBuffer), &bytesRead, portMAX_DELAY);

  size_t samplesRead = bytesRead / sizeof(i2sBuffer[0]);
  handleInput(i2sBuffer, samplesRead);

  if (recordDone) {
    recordDone = false;
  }
}