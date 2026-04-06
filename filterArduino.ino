#include <Arduino.h>
#include <arduinoFFT.h>
#include "driver/i2s.h"

#define I2S_WS   18
#define I2S_SD   38
#define I2S_SCK  17
#define I2S_PORT I2S_NUM_0

constexpr uint32_t SAMPLING_RATE = 16000;
constexpr uint16_t I2S_BUFFER_LEN = 256;
constexpr uint16_t WINDOW_SIZE = 8192;

constexpr float DRONE_LOW = 175.0f;
constexpr float DRONE_HIGH = 500.0f;

constexpr uint16_t RECORD_SECONDS = 3;
constexpr uint32_t RECORD_SAMPLES = SAMPLING_RATE * RECORD_SECONDS;

// I2S input buffer
int32_t i2sBuffer[I2S_BUFFER_LEN];

// arduinoFFT arrays
float vReal[WINDOW_SIZE];
float vImag[WINDOW_SIZE];

// buffer for detection
int32_t currentWindow[WINDOW_SIZE];
uint16_t currentWindowIndex = 0;

// buffer for recording output
int16_t recordBuffer[I2S_BUFFER_LEN];
uint16_t recordBufferIndex = 0;

// states
bool armed = false;
bool isRecording = false;
uint32_t samplesSent = 0;
uint32_t windowCounter = 0;

// arduinoFFT instance
ArduinoFFT<float> FFT = ArduinoFFT<float>(vReal, vImag, WINDOW_SIZE, SAMPLING_RATE);

void writeWavHeader(uint32_t dataSize) {
  uint32_t fileSize = dataSize + 36;
  uint16_t audioFormat = 1;
  uint16_t numChannels = 1;
  uint16_t bitsPerSample = 16;
  uint32_t byteRate = SAMPLING_RATE * numChannels * bitsPerSample / 8;
  uint16_t blockAlign = numChannels * bitsPerSample / 8;

  Serial.write("RIFF", 4);
  Serial.write((uint8_t*)&fileSize, 4);
  Serial.write("WAVE", 4);

  Serial.write("fmt ", 4);
  uint32_t subChunk1Size = 16;
  Serial.write((uint8_t*)&subChunk1Size, 4);
  Serial.write((uint8_t*)&audioFormat, 2);
  Serial.write((uint8_t*)&numChannels, 2);

  uint32_t sampleRateVar = SAMPLING_RATE;
  Serial.write((uint8_t*)&sampleRateVar, 4);
  Serial.write((uint8_t*)&byteRate, 4);
  Serial.write((uint8_t*)&blockAlign, 2);
  Serial.write((uint8_t*)&bitsPerSample, 2);

  Serial.write("data", 4);
  Serial.write((uint8_t*)&dataSize, 4);
}

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

void flushRecordBuffer() {
  if (recordBufferIndex > 0) {
    // send buffered samples as 16 bit PCM
    Serial.write((uint8_t*)recordBuffer, recordBufferIndex * sizeof(int16_t));
    recordBufferIndex = 0;
  }
}

void armDetector() {
  armed = true;
  isRecording = false;
  samplesSent = 0;
  recordBufferIndex = 0;
  currentWindowIndex = 0;

  // clear I2S buffer
  i2s_zero_dma_buffer(I2S_PORT);

  Serial.println("ARMED");
  Serial.print("Listening for peak in ");
  Serial.print(DRONE_LOW, 0);
  Serial.print(" - ");
  Serial.print(DRONE_HIGH, 0);
  Serial.println(" Hz");
}

void stopDetector() {
  armed = false;
  isRecording = false;
  samplesSent = 0;
  recordBufferIndex = 0;
  currentWindowIndex = 0;
  Serial.println("STOPPED");
}

void finishRecording() {
  flushRecordBuffer();
  isRecording = false;
  Serial.println("DONE");

  // go back to listening
  armDetector();
}

void pushRecordedSample(int32_t sample24) {
  // convert from 24 bit to 16 bit annd store in record buffer
  int16_t sample16 = (int16_t)(sample24 >> 8);
  recordBuffer[recordBufferIndex++] = sample16;
  samplesSent++;

  // flush buffer if full or enough samples sent
  if (recordBufferIndex >= I2S_BUFFER_LEN || samplesSent >= RECORD_SAMPLES) {
    flushRecordBuffer();
  }

  // if enough samples sent stop recording
  if (samplesSent >= RECORD_SAMPLES) {
    finishRecording();
  }
}

void startTriggeredRecording(float peakFreq) {
  uint32_t dataBytes = RECORD_SAMPLES * sizeof(int16_t);

  Serial.print("DETECTED ");
  Serial.println(peakFreq, 1);

  Serial.print("WAVSIZE ");
  Serial.println(dataBytes);

  writeWavHeader(dataBytes);

  samplesSent = 0;
  recordBufferIndex = 0;
  isRecording = true;
  armed = false;

  // include current window in recording
  uint32_t firstCount = (RECORD_SAMPLES < WINDOW_SIZE) ? RECORD_SAMPLES : WINDOW_SIZE;
  for (uint32_t i = 0; i < firstCount; i++) {
    pushRecordedSample(currentWindow[i]);
    if (!isRecording) {
      break;
    }
  }
}

void handleCommand() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == 'c') {
      armDetector();
    } else if (c == 's') {
      stopDetector();
    }
  }
}

void processMonoSample(int32_t sample24) {
  // if currently recording, push sample to record buffer
  if (isRecording) {
    pushRecordedSample(sample24);
    return;
  }

  if (!armed) {
    return;
  }

  currentWindow[currentWindowIndex++] = sample24;

  // detect once window is full
  if (currentWindowIndex >= WINDOW_SIZE) {
    float peakFreq = 0.0f;
    bool detected = detectDroneWindow(currentWindow, peakFreq);

    windowCounter++;
    Serial.print("PEAK ");
    Serial.println(peakFreq, 1);

    // if detected start recording
    if (detected) {
      startTriggeredRecording(peakFreq);
    }

    currentWindowIndex = 0;
  }
}

void setup() {
  Serial.begin(921600);
  delay(1000);

  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLING_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S_MSB,
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

  Serial.println("READY");
  armDetector();
}

void loop() {
  handleCommand();

  size_t bytesRead = 0;
  i2s_read(I2S_PORT, i2sBuffer, sizeof(i2sBuffer), &bytesRead, portMAX_DELAY);

  int totalWords = bytesRead / sizeof(i2sBuffer[0]);

  // process samples as mono
  for (int i = 0; i < totalWords; i += 2) {
    int32_t sample24 = i2sBuffer[i] >> 8;
    processMonoSample(sample24);
  }
}