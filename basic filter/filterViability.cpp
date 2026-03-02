#include <Arduino.h>
#include <arduinoFFT.h>
#include <math.h>

constexpr uint16_t SAMPLING_RATE = 16000;
constexpr uint16_t WINDOW_SIZE = 4096;

constexpr float DRONE_LOW  = 175.0f;
constexpr float DRONE_HIGH = 425.0f;

constexpr uint16_t RECORD_SECONDS = 2;
constexpr uint32_t RECORD_WINDOWS_SENT = (SAMPLING_RATE * RECORD_SECONDS) / WINDOW_SIZE + 1;

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

void transferWindow(uint16_t windowNumber) {
  Serial.print("Transferring window ");
  Serial.println(windowNumber + 1);
  recordDone = false;
  recordIndex = 0;
}

void handleSample(int32_t sample) {
  // if recording audio, store in record buffer
  if (isRecording && !recordDone) {
    recordBuffer[recordIndex++] = sample;

    if (recordIndex >= WINDOW_SIZE) {
      transferWindow(recordWindowsSent++);

      if (recordWindowsSent >= RECORD_WINDOWS_SENT) {
        isRecording = false;
        recordDone = true;
        Serial.println("Recording complete");
      }
    } 

    return;
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

      Serial.print("Drone detected. Peak frequency: ");
      Serial.println(peakFreq);
    }

    // reset index for next window
    currentWindowIndex = 0;
  }
}

int32_t getNextSample() {
  static uint32_t n = 0;
  float freq = 250.0f;
  float value = 10000.0f * sinf(2.0f * PI * freq * n / SAMPLING_RATE);
  n++;
  return (int32_t)value;
}

void setup() {
  Serial.begin(115200);
  Serial.println("Starting");
}

void loop() {
  int32_t sample = getNextSample();
  handleSample(sample);

  if (recordDone) {
    recordDone = false;
    Serial.println("Ready for next recording");
  }

  delayMicroseconds(1000000 / SAMPLING_RATE);
}