X_Train = data_train(:,2:end);
Y_Train = data_train(:,1);
Y_Train = categorical(data_train(:,1));
inputSize = 1;
numHiddenUnits = 100;
numClasses = 5;
layers = [ ...
sequenceInputLayer(inputSize)
bilstmLayer(numHiddenUnits,'OutputMode','last')
fullyConnectedLayer(numClasses)
softmaxLayer
classificationLayer]
maxEpochs = 100;
miniBatchSize = 27;
options = trainingOptions('adam', ...
'ExecutionEnvironment','cpu', ...
'GradientThreshold',1, ...
'MaxEpochs',maxEpochs, ...
'SequenceLength','longest', ...
'Shuffle','never', ...
'Verbose',0, ...
'Plots','training-progress');
X_Train = num2cell(X_Train,2);
net = trainNetwork(X_Train,Y_Train,layers,options);
X_Test = data_test(:,2:end);
Y_Test = data_test(:,1);
Y_Test = categorical(Y_Test)
X_Test = num2cell(X_Test,2);
miniBatchSize = 27;
YPred = classify(net,X_Test, ...
'MiniBatchSize',miniBatchSize, ...
'SequenceLength','longest');
acc = sum(YPred == Y_Test)./numel(Y_Test)

n_Train = dummyvar(Y_Train);
X_Train = data_train(:,2:end);
[net,tr] = train(net,X_Train',n_Train');
X_Test = data_test(:,2:end);
testY = net(X_Train');
Y_Test = data_test(:,1);
testClasses = vec2ind(testY);