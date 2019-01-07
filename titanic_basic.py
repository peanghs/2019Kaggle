import tensorflow as tf
import csv

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

#  전처리 - 학습 자료 처리
train_db = csv.reader(open('train.csv', 'r', encoding='utf-8'), delimiter=',')
train_set = []
train_label = []
for record in train_db:
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
        Sib = float(Sib)
        Par = record[7]
        Par = float(Par)
        Fare = record[9]
        if Fare == '':
            Fare = 0.0
        elif float(Fare) is ValueError:
            Fare = 0.0
        Fare = float(Fare)
        Destination = record[11]
        if Destination == '':
            Destination = 0
        elif Destination == 'C':
            Destination = 0
        elif Destination == 'Q':
            Destination = 1
        elif Destination == 'S':
            Destination = 2
        Destination = float(Destination)
        train_array_info = [ID, PClass, Sex, Age, Sib, Par, Fare, Destination]
        label_array_info = [Survived]
        train_set.append(train_array_info)
        train_label.append(label_array_info)

print('-----Train Set-----')
print(train_set)
print('-----Train Label-----')
print(train_label)
print('학습 자료 길이:', '%d' % (len(train_set)))

#  전처리 - 테스트 자료 처리
test_db = csv.reader(open('test.csv', 'r', encoding='utf-8'), delimiter=',')
test_label_db = csv.reader(open('gender_submission.csv', 'r', encoding='utf-8'), delimiter=',')
test_set = []
test_label = []
for record in test_db:
    if record[0] != 'PassengerId':
        ID = float(record[0])
        PClass = record[1]
        if PClass == '':
            PClass = 0.0
        else:
            PClass = PClass
        PClass = float(PClass)
        Sex = record[3]
        if Sex == '':
            Sex = 0.0
        elif Sex == 'male':
            Sex = 0
        else:
            Sex = 1
        Sex = float(Sex)
        Age = record[4]
        if Age == '':
            Age = 0.0
        else:
            Age = Age
        Age = float(Age)
        Sib = record[5]
        Sib = float(Sib)
        Par = record[6]
        Par = float(Par)
        Fare = record[8]
        if Fare == '':
            Fare = 0.0
        elif float(Fare) is ValueError:
            Fare = 0.0
        Fare = float(Fare)
        Destination = record[10]
        if Destination == '':
            Destination = 0
        elif Destination == 'C':
            Destination = 0
        elif Destination == 'Q':
            Destination = 1
        elif Destination == 'S':
            Destination = 2
        Destination = float(Destination)
        test_array_info = [ID, PClass, Sex, Age, Sib, Par, Fare, Destination]
        test_set.append(test_array_info)

for record in test_label_db:
    if record[0] != 'PassengerId':
        Survived = float(record[1])
        test_label_array_info = [Survived]
        test_label.append(test_label_array_info)

print('-----Test Set-----')
print(test_set)
print('-----Test Label-----')
print(test_label)
print('실험 자료 길이:', '%d' % (len(test_set)))

X = tf.placeholder(tf.float32, [None, 8])
Y = tf.placeholder(tf.float32, [None, 1])
b = tf.Variable(tf.random_normal([1]))

W1 = tf.Variable(tf.random_normal([8, 6], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1) + b)

W2 = tf.Variable(tf.random_normal([6, 4], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b)

W3 = tf.Variable(tf.random_normal([4, 1], stddev=0.01))
model = tf.matmul(L2, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 30
total_batch = int(len(train_set) / batch_size)
print('토탈 배치:', total_batch)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        # x_batch, y_batch = sess.run([train_set, train_label])
        # _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_batch, Y: y_batch})
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: train_set, Y: train_label})
        total_cost += cost_val

    print('Epoch:', '%d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict={X: test_set, Y: test_label}))
