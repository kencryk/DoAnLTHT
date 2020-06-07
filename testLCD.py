import Adafruit_CharLCD as LCD
import time

def DisplayLCD(name, count):

    LCD_RS = 25
    LCD_E = 24
    LCD_D4 = 23
    LCD_D5 = 17
    LCD_D6 = 18
    LCD_D7 = 22
    lcd_backlight = 2
    # set cursor to line 1

    lcd_columns = 16
    lcd_rows = 2
    lcd = LCD.Adafruit_CharLCD(LCD_RS, LCD_E, LCD_D4, LCD_D5, LCD_D6, LCD_D7, lcd_columns, 1, lcd_backlight)

    lcd.message(name)
    lcd.message("\n")
    lcd.message(str(count))

    # Wait 5 seconds


