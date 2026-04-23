#include <Arduino.h>
#include <arduinoFFT.h>
#include "driver/i2s.h"
#include <WiFi.h>

#define I2S_WS   18
#define I2S_SD   38
#define I2S_SCK  17
#define I2S_PORT I2S_NUM_0

constexpr uint32_t SAMPLING_RATE = 16000;
constexpr uint16_t I2S_BUFFER_LEN = 256;
constexpr uint16_t WINDOW_SIZE = 8192;

constexpr uint16_t RECORD_SECONDS = 5;
constexpr uint32_t RECORD_SAMPLES = SAMPLING_RATE * RECORD_SECONDS;

bool DEBUG_MODE = false;

// WiFi credentials
const char* ssid = "DroneWifi";
const char* password = "GuidingMoonlight";

WiFiServer server(12345);
WiFiClient client;

// Output macros
#define OUT_PRINT(x)    do { if (client && client.connected()) client.print(x); else Serial.print(x); } while(0)
#define OUT_PRINTLN(x)  do { if (client && client.connected()) client.println(x); else Serial.println(x); } while(0)
#define OUT_WRITE(buf,len) do { if (client && client.connected()) client.write(buf,len); else Serial.write(buf,len); } while(0)

// I2S input buffer
int32_t i2sBuffer[I2S_BUFFER_LEN];

// FFT arrays
float vReal[WINDOW_SIZE];
float vImag[WINDOW_SIZE];

// buffers
int32_t currentWindow[WINDOW_SIZE];
uint16_t currentWindowIndex = 0;

int16_t recordBuffer[I2S_BUFFER_LEN];
uint16_t recordBufferIndex = 0;

// states
bool armed = false;
bool isRecording = false;
uint32_t samplesSent = 0;
uint32_t windowCounter = 0;

ArduinoFFT<float> FFT = ArduinoFFT<float>(vReal, vImag, WINDOW_SIZE, SAMPLING_RATE);

void writeWavHeader(uint32_t dataSize) {
  uint32_t fileSize = dataSize + 36;
  uint16_t audioFormat = 1;
  uint16_t numChannels = 1;
  uint16_t bitsPerSample = 16;
  uint32_t byteRate = SAMPLING_RATE * numChannels * bitsPerSample / 8;
  uint16_t blockAlign = numChannels * bitsPerSample / 8;

  OUT_WRITE((uint8_t*)"RIFF", 4);
  OUT_WRITE((uint8_t*)&fileSize, 4);
  OUT_WRITE((uint8_t*)"WAVE", 4);

  OUT_WRITE((uint8_t*)"fmt ", 4);
  uint32_t subChunk1Size = 16;
  OUT_WRITE((uint8_t*)&subChunk1Size, 4);
  OUT_WRITE((uint8_t*)&audioFormat, 2);
  OUT_WRITE((uint8_t*)&numChannels, 2);

  OUT_WRITE((uint8_t*)&SAMPLING_RATE, 4);
  OUT_WRITE((uint8_t*)&byteRate, 4);
  OUT_WRITE((uint8_t*)&blockAlign, 2);
  OUT_WRITE((uint8_t*)&bitsPerSample, 2);

  OUT_WRITE((uint8_t*)"data", 4);
  OUT_WRITE((uint8_t*)&dataSize, 4);
}

bool detectDroneWindow(const int32_t* signal, float &centroidOut, float &bandwidthOut, float &rolloffOut, float &entropyOut) {

  for (uint16_t i = 0; i < WINDOW_SIZE; i++) {
    vReal[i] = (float)signal[i];
    vImag[i] = 0.0f;
  }

  FFT.dcRemoval();
  FFT.windowing(FFTWindow::Hamming, FFTDirection::Forward);
  FFT.compute(FFTDirection::Forward);
  FFT.complexToMagnitude();

  const uint16_t half = WINDOW_SIZE / 2;

  float sumMag = 0.0f;
  float weightedSum = 0.0f;

  for (uint16_t i = 0; i < half; i++) {
    float mag = vReal[i];
    sumMag += mag;
    weightedSum += i * mag;
  }

  if (sumMag < 1e-6) return false;

  float binToHz = (float)SAMPLING_RATE / WINDOW_SIZE;
  float centroid = (weightedSum / sumMag) * binToHz;

  float bwSum = 0.0f;
  for (uint16_t i = 0; i < half; i++) {
    float freq = i * binToHz;
    float diff = freq - centroid;
    bwSum += vReal[i] * diff * diff;
  }
  float bandwidth = sqrt(bwSum / sumMag);

  float threshold = 0.85f * sumMag;
  float cumulative = 0.0f;
  float rolloff = 0.0f;

  for (uint16_t i = 0; i < half; i++) {
    cumulative += vReal[i];
    if (cumulative >= threshold) {
      rolloff = i * binToHz;
      break;
    }
  }

  float entropy = 0.0f;
  for (uint16_t i = 0; i < half; i++) {
    float p = vReal[i] / sumMag;
    if (p > 0) entropy -= p * log(p);
  }

  centroidOut = centroid;
  bandwidthOut = bandwidth;
  rolloffOut = rolloff;
  entropyOut = entropy;

  return (
    bandwidth >= 1650 && bandwidth <= 4000 &&
    centroid >= 1500 && centroid <= 4600 &&
    entropy >= 6 && entropy <= 12 &&
    rolloff >= 4000 && rolloff <= 7000
  );
}

void flushRecordBuffer() {
  if (recordBufferIndex > 0) {
    OUT_WRITE((uint8_t*)recordBuffer, recordBufferIndex * sizeof(int16_t));
    recordBufferIndex = 0;
  }
}

void finishRecording() {
  flushRecordBuffer();
  isRecording = false;
  OUT_PRINTLN("DONE");
  armed = true;
}

void pushRecordedSample(int32_t sample24) {
  int16_t sample16 = (int16_t)(sample24 >> 8);
  recordBuffer[recordBufferIndex++] = sample16;
  samplesSent++;

  if (recordBufferIndex >= I2S_BUFFER_LEN || samplesSent >= RECORD_SAMPLES) {
    flushRecordBuffer();
  }

  if (samplesSent >= RECORD_SAMPLES) {
    finishRecording();
  }
}

void startTriggeredRecording() {
  uint32_t dataBytes = RECORD_SAMPLES * sizeof(int16_t);

  OUT_PRINT("WAVSIZE ");
  OUT_PRINTLN(dataBytes);

  writeWavHeader(dataBytes);

  samplesSent = 0;
  recordBufferIndex = 0;
  isRecording = true;
  armed = false;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nConnected!");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  server.begin();
  Serial.println("Server started");

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
  armed = true;
}

void loop() {
  // Accept client
  if (!client || !client.connected()) {
    client = server.available();
    if (client) {
      Serial.println("Client connected");
    }
  }

  size_t bytesRead = 0;
  i2s_read(I2S_PORT, i2sBuffer, sizeof(i2sBuffer), &bytesRead, portMAX_DELAY);

  int totalWords = bytesRead / sizeof(i2sBuffer[0]);

  for (int i = 0; i < totalWords; i += 2) {
    int32_t sample24 = i2sBuffer[i] >> 8;

    if (isRecording) {
      pushRecordedSample(sample24);
      continue;
    }

    if (!armed) continue;

    currentWindow[currentWindowIndex++] = sample24;

    if (currentWindowIndex >= WINDOW_SIZE) {
      float c, b, r, e;
      bool detected = detectDroneWindow(currentWindow, c, b, r, e);

      if (detected && !DEBUG_MODE) {
        startTriggeredRecording();
      }

      currentWindowIndex = 0;
    }
  }
}