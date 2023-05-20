
#include <Wire.h> // lib pour l'I2C.
#include <Adafruit_SSD1306.h> // lib pour l'écran.
#include <RCSwitch.h>

RCSwitch mySwitch = RCSwitch();
Adafruit_SSD1306 oled(128, 64, &Wire, -1);

int heure;
int directio;
int force;
int humidite;
int temperatur;
int pression;
int soleil;
int pluie;
int directio_pred;
String force_pred;
int humidite_pred;
int temperatur_pred;
int soleil_pred;
byte switcher_start;
byte switcher_end;


void setup() {

  Wire.begin();
  
  Serial.begin(9600);
  mySwitch.enableReceive(0);  // reception sur la borne 2 de l'arduino.

  oled.begin(SSD1306_SWITCHCAPVCC, 0x3C); // itou pour l'écran.
  oled.clearDisplay();
  oled.setTextSize(1);
  oled.setTextColor(SSD1306_BLACK, SSD1306_WHITE);
  oled.setTextColor(SSD1306_WHITE);
  oled.clearDisplay();
  oled.display();
}

void loop() {
  
    if ((mySwitch.available()==true) && (switcher_end==0)) {

    
    long value = mySwitch.getReceivedValue();

    if (value == 1) {

      switcher_start=1;
      
      
    }
    
    else if ((value >= 10000) && (value < 20000)) {

      heure = value - 10000;

      if (heure == 5555) {

        heure = 0;
        
      }
      
    }

    else if ((value >= 20000) && (value < 30000)) {

      directio = value - 20000;

      if (directio == 5555) {

        directio = 0;
        
      }
      
    }

    else if ((value >= 30000) && (value < 40000)) {

      force = value - 30000;

      if (force == 5555) {

        force = 0;
        
      }
      /*
      else if (fo == 1) {

        force= "0/5";
        
      }

      else if (fo == 2) {

        force="5/10";
      }

      else if (fo == 3) {

        force="10/15";
      }

      else if (fo == 4) {

        force="15/20";
      }

      else if (fo == 5) {

        force="20/25";
      }

      else if (fo ==6) {

        force="25/30";
      }

      else if (fo == 7) {

        force="30+";
      }

      */
    }

    else if ((value >= 40000) && (value < 50000)) {

      humidite = value - 40000;

      if (humidite == 5555) {

        humidite = 0;
        
      }
      
    }

    else if ((value >= 50000) && (value < 60000)) {

      temperatur = value - 50000;

      if (temperatur == 5555) {

        temperatur = 0;
        
      }
      
    }

    else if ((value >= 60000) && (value < 70000)) {

      pression = value - 60000;

      if (pression == 5555) {

        pression = 0;
        
      }
      
    }

    else if ((value >= 70000) && (value < 80000)) {

      soleil = value - 70000;

      if (soleil == 5555) {

        soleil = 0;
        
      }
      
    }

    else if ((value >= 80000) && (value < 90000)) {

      pluie = value - 80000;

      if (pluie == 5555) {

        pluie=0;
      }
      
    }

    else if ((value >= 90000) && (value < 100000)) {

      directio_pred = value - 90000;

      if (directio_pred == 5555) {

        directio_pred = 0;
      }
      
    }

    else if ((value >= 100000) && (value < 110000)) {

      int force_p = value - 100000;

      if (force_p == 5555) {

        force_pred = "0";
      }

      else if (force_p == 1) {

        force_pred = "0/5";
      }

      else if (force_p == 2) {

        force_pred = "5/10";
      }

      else if (force_p ==3) {

        force_pred = "10/15";
      }

      else if (force_p == 4) {

        force_pred = "15/20";
      }

      else if (force_p == 5) {

        force_pred = "20/25";
      }

      else if (force_p == 6) {

        force_pred = "25/30";
      }

      else if (force_p == 7) {

        force_pred = "30+";
      }
      
    }

    else if ((value >= 110000) && (value < 120000)) {

      humidite_pred = value - 110000;

      if (humidite_pred == 5555) {

        humidite_pred = 0;
      }
      
    }

    else if ((value >= 120000) && (value < 130000)) {

      temperatur_pred = value - 120000;

      if (temperatur_pred == 5555) {

        temperatur_pred = 0;
      }
      
    }

    else if ((value >= 130000) && (value < 140000)) {

      soleil_pred = value - 130000;

      if (soleil_pred == 5555) {

        soleil_pred = 0;
      }
      
    }


    else if (value == 2) {

      switcher_end=2;
      
    }

    Serial.println(value);
    

    oled.setCursor(0, 0);
    oled.print("reception meteo...");
    oled.display();
    oled.clearDisplay();

    mySwitch.resetAvailable();
    
    }

    else if ( (mySwitch.available()==false) && (switcher_start==1) && (switcher_end == 2)) {
    
    

    
    
    oled.setCursor(30, 0);  // affichage.
    oled.print(heure);
    oled.println("H now");
    oled.setCursor(0, 7);
    oled.print("direction=");
    oled.print(directio);
    oled.println(" deg");
    oled.print("force=");
    oled.print(force);
    oled.println(" nds");
    oled.print("pluie=");
    oled.println(pluie);
    oled.print("humidite=");
    oled.print(humidite);
    oled.println(" %");
    oled.print("temperature=");
    oled.print(temperatur);
    oled.println(" C");
    oled.print("pression=");
    oled.print(pression);
    oled.println(" hpa");
    oled.print("soleil=");
    oled.println(soleil);
    oled.display();
    delay(60000);
    oled.clearDisplay();

    

    
    oled.setCursor(30, 0);  // affichage.
    oled.print(heure);
    oled.println("H +24");
    oled.setCursor(0, 7);
    oled.print("direction=");
    oled.print(directio_pred);
    oled.println(" deg");
    oled.print("force=");
    oled.print(force_pred);
    oled.println(" nds");
    oled.print("humidite=");
    oled.print(humidite_pred);
    oled.println(" %");
    oled.print("temperature=");
    oled.print(temperatur_pred);
    oled.println(" C");
    oled.print("soleil=");
    oled.println(soleil_pred);
    oled.display();
    delay(60000);
    oled.clearDisplay();

    
    
  }
  
  else if ((mySwitch.available()==false) && (switcher_start==0) && (switcher_end==0)) {

    oled.setCursor(0, 0);
    oled.print("attente signal...");
    oled.display();
    oled.clearDisplay();
  }
  
  
  
  }

  

 

 
