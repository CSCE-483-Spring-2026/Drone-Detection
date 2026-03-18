#include <Arduino.h>
#include "driver/i2s.h"

#define I2S_WS 18  // LEFT channel
#define I2S_SD 38  // DIN from mic
#define I2S_SCK 17 // BCLK

#define I2S_PORT I2S_NUM_0
#define SAMPLE_RATE 16000
#define RECORD_TIME 15 // seconds
#define BUFFER_LEN 256

int32_t buffer[BUFFER_LEN];

void writeWavHeader(uint32_t dataSize)
{
    uint32_t fileSize = dataSize + 36;
    uint16_t audioFormat = 1; // PCM
    uint16_t numChannels = 1;
    uint16_t bitsPerSample = 32; // 32-bit PCM
    uint32_t byteRate = SAMPLE_RATE * numChannels * bitsPerSample / 8;
    uint16_t blockAlign = numChannels * bitsPerSample / 8;

    Serial.write("RIFF", 4);
    Serial.write((uint8_t *)&fileSize, 4);
    Serial.write("WAVE", 4);

    Serial.write("fmt ", 4);
    uint32_t subChunk1Size = 16;
    Serial.write((uint8_t *)&subChunk1Size, 4);
    Serial.write((uint8_t *)&audioFormat, 2);
    Serial.write((uint8_t *)&numChannels, 2);

    uint32_t sampleRateVar = SAMPLE_RATE;
    Serial.write((uint8_t *)&sampleRateVar, 4);

    Serial.write((uint8_t *)&byteRate, 4);
    Serial.write((uint8_t *)&blockAlign, 2);
    Serial.write((uint8_t *)&bitsPerSample, 2);

    Serial.write("data", 4);
    Serial.write((uint8_t *)&dataSize, 4);
}

void setup()
{
    Serial.begin(921600);
    delay(1000);
    Serial.println("I2S WAV capture ready...");

    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = 0,
        .dma_buf_count = 8,
        .dma_buf_len = 64,
        .use_apll = false};

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = -1,
        .data_in_num = I2S_SD};

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);
}

void loop()
{
    if (Serial.available() > 0)
    {
        char cmd = Serial.read();
        if (cmd == 'c')
        { // capture command
            uint32_t totalSamples = SAMPLE_RATE * RECORD_TIME;
            uint32_t totalBytes = totalSamples * 4; // 32-bit PCM = 4 bytes per sample

            // Send WAVSIZE to Python
            Serial.print("WAVSIZE ");
            Serial.println(totalBytes);

            // Send WAV header
            writeWavHeader(totalBytes);

            uint32_t samplesSent = 0;

            while (samplesSent < totalSamples)
            {
                size_t bytesRead;
                i2s_read(I2S_PORT, buffer, sizeof(buffer), &bytesRead, portMAX_DELAY);
                int samples = bytesRead / 4;

                // Send raw 32-bit samples
                Serial.write((uint8_t *)buffer, samples * 4);
                samplesSent += samples;
            }

            Serial.println("DONE"); // optional end marker
        }
    }
}