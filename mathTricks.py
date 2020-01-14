import random
import sys
import math
import time

operators = ['+','-','*','/']

def round(num):
    wholeNum = math.floor(num)
    rem = num - wholeNum
    if(rem*2 < 1):
        return wholeNum
    elif(rem*2 > 1):
        return wholeNum+1

def ndecimalPlaces(num,n):
    whole = math.floor(num)
    rem = num - whole
    dp = round(10**n * rem) / 10**(n)
    return whole+dp

def add(num1,num2):

    return num1 + num2

def subtract(num1,num2):

    return num1 - num2

def multiply(num1,num2):

    return num1*num2

def divide(num1,num2):
    
    val = ndecimalPlaces(num1/num2,3)
    return val



while(True):
   i = 0
   maxval = int(input("How many questions do you want to answer: "))
   score = 0
   while(i < maxval):
        firstNum = random.randint(-100,100)
        secondNum = random.randint(-100,100)
        opNum = random.randint(0,3)
        print("{}. {} {} {} = ".format(i+1,firstNum,operators[opNum],secondNum))
        answer = float(input(""))

        funcDict = {'+' : add(firstNum,secondNum),
        '-' : subtract(firstNum,secondNum),
        '/' : divide(firstNum,secondNum), 
        '*' : multiply(firstNum,secondNum)
        }

        result = funcDict[operators[opNum]]

    
        if(answer == result):
             print("Correct!! well done" )
             score += 1
    
        else :
             print("Incorrect \n")
             print("Correct answer: {}".format(result))
        i+= 1
    

   print("You scored {}/{}".format(score,maxval))
   res = input("Do you want to try again ? y/n ")
   if(res == 'y'):
       continue

   else :
       break
      


print("See you soon")
time.sleep(2)
input("Press enter to exit ")
