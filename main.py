import tensorflow as tf 

#load mnist 

mnist = tf.keras.datasets.mnist

(xTrain , yTrain) , (xTest, yTest) = mnist.load_data(); 

xTrain , xText = xTrain / 255.0 , xTest / 255.0


#model 
model =tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape =(28,28)),
    tf.keras.layers.Dense(128, activation ='relu'),
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Dense(10)
])

predictions = model(xTrain[:1]).numpy()

print(tf.nn.softmax(predictions).numpy())

lossFN = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print(lossFN (yTrain[:1], predictions).numpy())


model.compile(optimizer ='adam',
              loss= lossFN,
              metrics= ['accuracy'])

#train model 
model.fit(xTrain, yTrain, epochs = 5)

model.evaluate(xTest, yTest , verbose=2)

print("good job forehead ")

