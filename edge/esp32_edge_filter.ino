// SkyGuard AI - Tier-1 edge filter for ESP32 (Arduino framework).
// Runs cheap checks on-device (range, jump, frozen) and only wakes the radio when something is odd,
// or once per N readings as a heartbeat. Heavy ML (spatial + IsolationForest) runs on the server.
#include <Arduino.h>

struct Limits { float lo, hi, maxStep; };
const Limits LIM[3] = {{-20, 55, 4.0f}, {850, 1085, 2.5f}, {0, 100, 25.0f}};  // temp, pres, rh
const int FLAT_N = 6;                       // identical readings => frozen
const uint64_t SLEEP_US = 10ULL * 60 * 1000000ULL;   // 10-minute cycle

RTC_DATA_ATTR float last[3] = {NAN, NAN, NAN};        // survives deep sleep
RTC_DATA_ATTR uint8_t same[3] = {0, 0, 0};
RTC_DATA_ATTR uint8_t beat = 0;

extern float readTemp();  extern float readPres();  extern float readRH();   // your sensor drivers
extern void  sendMQTT(const float v[3], uint8_t flags);                       // your uplink

void setup() {
  float v[3] = {readTemp(), readPres(), readRH()};
  uint8_t flags = 0;                                    // bit0..2 range, bit3..5 jump, bit6 frozen
  for (int i = 0; i < 3; i++) {
    if (isnan(v[i]) || v[i] < LIM[i].lo || v[i] > LIM[i].hi) flags |= (1 << i);
    if (!isnan(last[i]) && fabsf(v[i] - last[i]) > LIM[i].maxStep) flags |= (1 << (3 + i));
    same[i] = (!isnan(last[i]) && v[i] == last[i]) ? same[i] + 1 : 0;
    if (same[i] >= FLAT_N) flags |= (1 << 6);
    last[i] = v[i];
  }
  if (flags || ++beat >= 6) { sendMQTT(v, flags); beat = 0; }   // radio only when needed (energy saving)
  esp_sleep_enable_timer_wakeup(SLEEP_US);
  esp_deep_sleep_start();
}
void loop() {}
