import tensorflow as tf
import csv
import numpy as np

''' 
0 PassengerId - type should be integers
1 Survived - Survived or Not
2 Pclass - Class of Travel
3 Name - Name of Passenger
4 Sex - Gender
5 Age - Age of Passengers
6 SibSp - Number of Sibling/Spouse aboard
7 Parch - Number of Parent/Child aboard
8 Ticket
9 Fare
10 Cabin
11 Embarked - The port in which a passenger has embarked. C - Cherbourg, S - Southampton, Q = Queenstown
'''

# testcsv = open('test.csv', 'r', encoding='utf-8')
# traindb = csv.reader(testcsv)
# for line in traindb:
#     print(line)


testcsv = open('train.csv', 'r', encoding='utf-8')
traindb = csv.reader(testcsv, delimiter=',')
trainset=[]
for record in traindb:
    if record[0] != 'PassengerId':
        ID = float(record[0])
        Survived = float(record[1])
        PClass = record[2]
        if PClass == '':
            PClass = 0.0
        else:
            PClass = PClass
        PClass = float(PClass)
        Sex = record[4]
        if Sex == '':
            Sex = 0.0
        elif Sex == 'male':
            Sex = 0
        else:
            Sex = 1
        Sex = float(Sex)
        Age = record[5]
        if Age == '':
            Age = 0.0
        else:
            Age = Age
        Age = float(Age)
        Sib = record[6]
        Par = record[7]
        Fare = record[9]
        Destination = record[11]
        if Destination == '':
            Destination = 0
        elif Destination == 'C':
            Destination = 0
        elif Destination == 'Q':
            Destination = 1
        elif Destination == 'S':
            Destination = 2
        doc_info = [ID, Survived, PClass, Sex, Age, Sib, Par, Fare, Destination]
        trainset.append(doc_info)
print(trainset)


