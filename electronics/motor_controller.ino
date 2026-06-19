int motorPins[4][4] = {
  {13, 14, 27, 26},   

  {25, 33, 32, 15},   

  {2,  4,  5,  18},   

  {19, 21, 22, 23}    

};

int stepSequence[8][4] = {
  {1,0,0,0},
  {1,1,0,0},
  {0,1,0,0},
  {0,1,1,0},
  {0,0,1,0},
  {0,0,1,1},
  {0,0,0,1},
  {1,0,0,1}
};

int currentStep[4] = {0, 0, 0, 0};

const int STEP_DELAY_US = 1200;

void setStep(int motor, int stepIndex) {
  for (int i = 0; i < 4; i++) {
    digitalWrite(motorPins[motor][i], stepSequence[stepIndex][i]);
  }
}

void releaseMotorCoils(int motor) {
  for (int i = 0; i < 4; i++) {
    digitalWrite(motorPins[motor][i], LOW);
  }
}

void releaseAllCoils() {
  for (int m = 0; m < 4; m++) {
    releaseMotorCoils(m);
  }
}

void moveMotors(int m0, int m1, int m2, int m3) {
  int target[4] = {m0, m1, m2, m3};
  int direction[4];
  int stepsRemaining[4];

  for (int i = 0; i < 4; i++) {
    if (target[i] >= 0) {
      direction[i] = 1;
      stepsRemaining[i] = target[i];
    } else {
      direction[i] = -1;
      stepsRemaining[i] = -target[i];
    }
  }

  bool running = true;

  while (running) {
    running = false;

    for (int m = 0; m < 4; m++) {
      if (stepsRemaining[m] > 0) {
        running = true;

        currentStep[m] += direction[m];

        if (currentStep[m] > 7) currentStep[m] = 0;
        if (currentStep[m] < 0) currentStep[m] = 7;

        setStep(m, currentStep[m]);
        stepsRemaining[m]--;
      }
    }

    delayMicroseconds(STEP_DELAY_US);
  }
}

bool parseCommand(String cmd, int &m0, int &m1, int &m2, int &m3) {
  cmd.trim();
  if (cmd.length() == 0) return false;

  int parsed = sscanf(cmd.c_str(), "%d,%d,%d,%d", &m0, &m1, &m2, &m3);
  return (parsed == 4);
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(20);

  for (int m = 0; m < 4; m++) {
    for (int p = 0; p < 4; p++) {
      pinMode(motorPins[m][p], OUTPUT);
      digitalWrite(motorPins[m][p], LOW);
    }
  }

  Serial.println("TDCR controller ready");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');

    int m0, m1, m2, m3;
    bool ok = parseCommand(cmd, m0, m1, m2, m3);

    if (!ok) {
      Serial.println("ERR");
      return;
    }

    moveMotors(m0, m1, m2, m3);

    Serial.println("DONE");
  }
}