#include <Arduino.h>
#include <arduinoFFT.h>

constexpr uint16_t SAMPLING_RATE = 16000;
constexpr uint16_t WINDOW_SIZE = 16384;
constexpr uint16_t HOP_SIZE    = 8192;

constexpr float DRONE_LOW  = 175.0f;
constexpr float DRONE_HIGH = 425.0f;

// arduinoFFT arrays
float vReal[WINDOW_SIZE];
float vImag[WINDOW_SIZE];

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
  FFT.windowing(FFTWindow::Hamming, FFTOption::Forward);

  // compute FFT
  FFT.compute(FFTDirection::Forward);

  // convert to magnitude
  FFT.complexToMagnitude();

  // find peak
  peakFreqOut = FFT.majorPeak();

  // check if peak frequency is within drone range
  return (peakFreqOut >= DRONE_LOW && peakFreqOut <= DRONE_HIGH);
}