#include <LowPower.h> // librairie pour le mode veille.
#include <Wire.h> // lib pour l'I2C.
#include <DS3231.h> // lib pour l'horloge externe.
#include <Adafruit_BME280.h> // lib pour le capteur température/humidité/pression.
#include <Adafruit_SSD1306.h> // lib pour l'écran.


Adafruit_BME280 bme;
Adafruit_SSD1306 oled(128, 64, &Wire, -1);

const int relay = 4; // borne pour activer le relai d'alimentation de la raspberry pi (démarre la raspberry pi).
const int wakeUpPin = 2; // borne de réveil arduino.
const int timeCounter = 180; // durée de collecte des données de l'arduino en période éveil en secondes concernant les compteurs anémo/pluvio.

int switch_counter_anemo = 0;  // variables utiles pour l'anémomètre.
byte state_anemo = HIGH; // utile pour éviter les problèmes de rebonds.
int wind_speed = 0; 

int switch_counter_pluie = 0; // variables utiles pour le pluviomètre.
byte state_pluie = HIGH; // utile pour éviter les problèmes de rebonds.
int pluie_volume = 0;


// Bits dispos pour les deux alarmes.
// They need to be combined into a single value (see below)
// Found here: https://github.com/mlepard/ArduinoChicken/blob/master/roboCoop/alarmControl.ino
#define ALRM1_MATCH_EVERY_SEC  0b1111  // once a second
#define ALRM1_MATCH_SEC        0b1110  // when seconds match
#define ALRM1_MATCH_MIN_SEC    0b1100  // when minutes and seconds match
#define ALRM1_MATCH_HR_MIN_SEC 0b1000  // when hours, minutes, and seconds match

#define ALRM2_ONCE_PER_MIN     0b111   // once per minute (00 seconds of every minute)
#define ALRM2_MATCH_MIN        0b110   // when minutes match
#define ALRM2_MATCH_HR_MIN     0b100   // when hours and minutes match

RTClib RTC;
DS3231 Clock;

 
void setup() {
  /*
  // Calibrage de l'horloge.
  Clock.setClockMode(false);// heures vont jusqu'à 24 au lieu de 12 am/12 pm.
  Clock.setYear(22);//définir l'année
  Clock.setMonth(5);//définir le mois
  Clock.setDate(21);//définir la date du mois
  Clock.setMinute(27);//définir les minutes
  Clock.setHour(12);//définir l'heure
  Clock.setSecond(00);//définir les secondes
  */
  Wire.begin(); // initialisation de la liaison I2C.
  
  bme.begin(0x76); // initialisation du BME280 via son adresse I2C.
  
  oled.begin(SSD1306_SWITCHCAPVCC, 0x3C); // itou pour l'écran.
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(SSD1306_BLACK, SSD1306_WHITE);
  oled.setTextColor(SSD1306_WHITE);
  oled.clearDisplay();
  oled.display();
  
  
  Serial.begin(115200); // initilisation liaison série pour communiquer avec la raspberrypi.
  
  pinMode(wakeUpPin, INPUT); // borne de reveil arduino.
  pinMode(relay, OUTPUT); // borne du relai d'activation raspberrypi.
  pinMode(A3, INPUT); // borne pour le photo transistor.
  pinMode(A0, INPUT); // borne pour le pluviomètre.
  pinMode(A1, INPUT); // borne pour la girouette.
  pinMode(A2, INPUT); // borne pour l'anémomètre.
  
  
  // règlage des bits de l'alarme.
  byte ALRM1_SET = ALRM1_MATCH_MIN_SEC; // enclenche A1 quand les minutes et secondes sont les mêmes.
  byte ALRM2_SET = ALRM2_MATCH_MIN;     // enclenche A2 quand les minutes correspondent (pas de secondes sur l'alarme A2).
  
  
  int ALARM_BITS = ALRM2_SET;
  ALARM_BITS <<= 4;
  ALARM_BITS |= ALRM1_SET;
  
  // enclenche une alarme quand les minutes == 0 
  
  Clock.setA1Time(0, 0, 0, 0, ALARM_BITS, false, false, false); 
  
   
  Clock.turnOnAlarm(1);
  
   
  if (Clock.checkAlarmEnabled(1)) {
    Serial.println("alarme activée"); // verifie que l'alarme 1 est bien activée
  }
  
  
  if (Clock.checkIfAlarm(1)) {
    Serial.println("Ok");
  }
  Serial.println("mise en veille");
   
  delay(1000); // petite pause sinon ça plante.
  
  ShutDown(); // fonction de mise en veille de l'arduino.
  
  
}
 
 
 
void loop() {

  digitalWrite(relay, HIGH); // activation du relai

  DateTime now = RTC.now(); // On récupère les données temporelles de l'horloge dans la boucle.
  int years = now.year();
  int months = now.month();
  int days = now.day();
  int hours = now.hour();
  int minutes = now.minute();
  int seconds = now.second();
 

  long pression = bme.readPressure()/100; // Convertit simplement les pascals en hectopascals.
  int temperatur = bme.readTemperature();
  int humidite = bme.readHumidity();

  unsigned long soleil = analogRead(A3);
  int sensor_gir = analogRead(A1);
  int sensor_anemo = digitalRead(A2);  
  int sensor_pluie = digitalRead(A0);
  
  
  
  if (sensor_gir > 940) 
  {
    sensor_gir = 180;
  }
  else if ((sensor_gir < 940) and (sensor_gir > 900))
  {
    sensor_gir = 90;
  }
  else if ((sensor_gir < 900) and (sensor_gir > 880))
  {
    sensor_gir = 0;
  }
  else if ((sensor_gir < 880) and (sensor_gir > 860))
  {
    sensor_gir = 135;
  }
  else if ((sensor_gir < 860) and (sensor_gir > 840))
  {
    sensor_gir = 270;
  }
  else if ((sensor_gir < 840) and (sensor_gir > 800))
  {
    sensor_gir = 45;
  }
  else if ((sensor_gir < 800) and (sensor_gir > 780))
  {
    sensor_gir = 225;
  }
  else
  {
    sensor_gir = 315;
  }

  
  if (minutes <= 2) { // trois minutes de collecte de données.

  

  if (sensor_anemo != state_anemo) {
      
      switch_counter_anemo++;
      delay(25); // pour éviter les rebonds, notamment par vent faible...Et du coup grandement biaiser les mesures.
      state_anemo=sensor_anemo;
  }
  

  if (sensor_pluie != state_pluie) {
      
      switch_counter_pluie++;
      delay(100); // pour également éviter les rebonds et les mesures à côté de la plaque.
      state_pluie=sensor_pluie;
  }
  

  
  
  oled.setCursor(0, 0);
  oled.print("collecte des");
  oled.setCursor(0, 7);
  oled.print("donnees...");
  oled.display();
  delay(20);
  oled.clearDisplay();
  
   
  
  }

  else if ((minutes>2) & (minutes<=4)) { // affichage et transmission des données à la raspberry pi pendant 2 min.
  
     

     wind_speed=round(((switch_counter_anemo) / timeCounter) * (2.4 / 1.852)) ; // donne la vitesse en noeuds
     pluie_volume=switch_counter_pluie;
    
     Serial.print(years); // transfert via la liaison série des données à la raspberry pi.
     Serial.print(" ");
     Serial.print(months);
     Serial.print(" ");
     Serial.print(days);
     Serial.print(" ");
     Serial.print(hours);
     Serial.print(" ");
     Serial.print(minutes);
     Serial.print(" ");
     Serial.print(seconds);
     Serial.print(" ");
     Serial.print(sensor_gir);
     Serial.print(" ");
     Serial.print(wind_speed);
     Serial.print(" ");
     Serial.print(humidite);
     Serial.print(" ");
     Serial.print(temperatur);
     Serial.print(" ");
     Serial.print(pression);
     Serial.print(" ");
     Serial.print(soleil);
     Serial.print(" ");
     Serial.println(pluie_volume);
     delay(100);
     
     
    
    oled.setCursor(30, 0);  // affichage données temps réel.
    oled.print(hours);
    oled.print("/");
    oled.print(minutes);
    oled.print("/");
    oled.print(seconds);
    oled.setCursor(0, 7);
    oled.print("direction=");
    oled.println(sensor_gir);
    oled.print("force=");
    oled.println(wind_speed);
    oled.print("pluie=");
    oled.println(pluie_volume);
    oled.print("humidite=");
    oled.println(humidite);
    oled.print("temperature=");
    oled.println(temperatur);
    oled.print("pression=");
    oled.println(pression);
    oled.print("soleil=");
    oled.println(soleil);
    oled.display();
    delay(20);
    oled.clearDisplay();
    
    
  }
  
  else if ((minutes > 4) & (minutes <= 10) ) { // temps de flottement pour permettre a la rasp de faire son job avant de l'éteindre.

    
    
  }
  
  else {
    
    switch_counter_anemo = 0; // on remet les compteurs à 0 pour la prochaine heure, le prochain cycle.
    switch_counter_pluie = 0; // idem pour la pluie.
    
    oled.clearDisplay(); // on éteint à nouveau l'écran pour économiser un max de jus.
    oled.display();
    
    Clock.turnOnAlarm(1); // on remet en route l'alarme avant la remise en veille de l'arduino.

    if (Clock.checkAlarmEnabled(1)) {

      Serial.println("alarme activée");
    }

    if (Clock.checkIfAlarm(1)) {

      Serial.println("ok");
    }

    Serial.println("mise en veille");

    delay(1000);
    
    ShutDown(); // on remet à nouveau en veille.
    
  }
  
  
  

}
   

 
 
void WakeUp () {
   
}

void ShutDown () {
  
  digitalWrite(relay, LOW);
  attachInterrupt(digitalPinToInterrupt(wakeUpPin), WakeUp, FALLING);
  LowPower.powerDown(SLEEP_FOREVER, ADC_OFF, BOD_OFF);
  detachInterrupt(digitalPinToInterrupt(wakeUpPin));

}
