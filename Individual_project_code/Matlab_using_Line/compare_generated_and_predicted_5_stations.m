clear all 

model_generated = Network('myModel_generated');

x = [150;0;0;0;0];
s = [200,100,50,20,10];
mu = [10,20,40,30,20];

%% generated model

%% Block 1: nodes
node{1} = Queue(model_generated, 'QueueStation1', SchedStrategy.FCFS);
node{2} = Queue(model_generated, 'QueueStation2', SchedStrategy.FCFS);
node{3} = Queue(model_generated, 'QueueStation3', SchedStrategy.FCFS);
node{4} = Queue(model_generated, 'QueueStation4', SchedStrategy.FCFS);
node{5} = Queue(model_generated, 'QueueStation5', SchedStrategy.FCFS);

%% Set number of servers for each queueing station node
node{1}.setNumberOfServers(s(1));
node{2}.setNumberOfServers(s(2));
node{3}.setNumberOfServers(s(3));
node{4}.setNumberOfServers(s(4));
node{5}.setNumberOfServers(s(5));

%% Block 2: Classes

cclass1 = ClosedClass(model_generated, 'class1', x(1), node{1});
% cclass2 = ClosedClass(model, 'class2', x(2), queue2);
% cclass3 = ClosedClass(model, 'class3', x(3), queue3);

node{2}.setState(x(2));
node{3}.setState(x(3));
node{4}.setState(x(4));
node{5}.setState(x(5));

%% Might have to set service rates like this and create loads of them so let's see

%% Set service rates for each queueing station node 

% First class starting at node 1

node{1}.setService(cclass1, Exp(mu(1)));
node{2}.setService(cclass1, Exp(mu(2)));
node{3}.setService(cclass1, Exp(mu(3)));
node{4}.setService(cclass1, Exp(mu(4)));
node{5}.setService(cclass1, Exp(mu(5)));

%% Block 3: Topology using Routing Matrix P

T = [0,0.2,0.2,0.1,0.5;0.2,0,0.1,0.5,0.2;0.2,0.4,0,0.2,0.2;0.4,0.2,0.1,0,0.3;0.3,0.2,0.1,0.4,0];

P = model_generated.initRoutingMatrix;
P{cclass1} = T;
% P{cclass2} = T;
% P{cclass3} = T;
model_generated.link(P);

[Qt_generated,Ut_generated,Tt_generated] = model_generated.getTranHandles();

[QNt_generated,UNt_generated,TNt_generated] = SolverJMT(model_generated,'force', true, 'timespan',[0,10]).getTranAvg(Qt_generated,Ut_generated,Tt_generated);

%% Predicted Model

model_predicted = Network('myModel_predicted');

% mu 8.826743 13.0607815 9.28407 10.970014 14.628479

mu = [8.827,13.0608,9.28407,10.970014,14.628479];

%% Block 1: nodes
node{1} = Queue(model_predicted, 'QueueStation1', SchedStrategy.FCFS);
node{2} = Queue(model_predicted, 'QueueStation2', SchedStrategy.FCFS);
node{3} = Queue(model_predicted, 'QueueStation3', SchedStrategy.FCFS);
node{4} = Queue(model_predicted, 'QueueStation4', SchedStrategy.FCFS);
node{5} = Queue(model_predicted, 'QueueStation5', SchedStrategy.FCFS);

%% Set number of servers for each queueing station node
node{1}.setNumberOfServers(s(1));
node{2}.setNumberOfServers(s(2));
node{3}.setNumberOfServers(s(3));
node{4}.setNumberOfServers(s(4));
node{5}.setNumberOfServers(s(5));

%% Block 2: Classes

cclass1 = ClosedClass(model_predicted, 'class1', x(1), node{1});
% cclass2 = ClosedClass(model, 'class2', x(2), queue2);
% cclass3 = ClosedClass(model, 'class3', x(3), queue3);

node{2}.setState(x(2))
node{3}.setState(x(3))
node{3}.setState(x(4))
node{3}.setState(x(5))

%% Might have to set service rates like this and create loads of them so let's see

%% Set service rates for each queueing station node 

% First class starting at node 1

node{1}.setService(cclass1, Exp(mu(1)));
node{2}.setService(cclass1, Exp(mu(2)));
node{3}.setService(cclass1, Exp(mu(3)));
node{4}.setService(cclass1, Exp(mu(4)));
node{5}.setService(cclass1, Exp(mu(5)));


%% Block 3: Topology using Routing Matrix P

% P 0.0 0.18930703 0.07670161 0.13043748 0.6035539 0.72601825 0.0 0.027256325 0.14107823 0.1056472 0.37243447 0.009886529 0.0 0.051320173 0.56635875 0.4433542 0.18314077 0.020263959 0.0 0.35324106 0.3569563 0.38013926 0.029297197 0.2336072 0.0

T = [0.0,0.18930703,0.07670161,0.13043748,0.6035539;0.72601825,0.0,0.027256325,0.14107823,0.1056472;0.37243447,0.009886529,0.0,0.051320173,0.56635875; 0.4433542,0.18314077,0.020263959,0.0,0.35324106; 0.3569563,0.38013926,0.029297197,0.2336072,0.0];

P = model_predicted.initRoutingMatrix;
P{cclass1} = T;
% P{cclass2} = T;
% P{cclass3} = T;
model_predicted.link(P);

[Qt_predicted,Ut_predicted,Tt_predicted] = model_predicted.getTranHandles();

[QNt_predicted,UNt_predicted,TNt_predicted] = SolverJMT(model_predicted,'force', true, 'timespan',[0,10]).getTranAvg(Qt_predicted,Ut_predicted,Tt_predicted);

%% Plot average queue length

% subplot(5,2,1); plot(QNt_generated{1,1}.t, QNt_generated{1,1}.metric)
% title('Generated Network Station 1')
% ylabel('Average Queue Length')
% xlabel('Time (s)')
% subplot(5,2,3); plot(QNt_generated{2,1}.t, QNt_generated{2,1}.metric)
% title('Generated Network Station 2')
% ylabel('Average Queue Length')
% xlabel('Time (s)')
% subplot(5,2,5); plot(QNt_generated{3,1}.t, QNt_generated{3,1}.metric)
% title('Generated Network Station 3')
% ylabel('Average Queue Length')
% xlabel('Time (s)')
% subplot(5,2,7); plot(QNt_generated{4,1}.t, QNt_generated{4,1}.metric)
% title('Generated Network Station 4')
% ylabel('Average Queue Length')
% xlabel('Time (s)')
% subplot(5,2,9); plot(QNt_generated{5,1}.t, QNt_generated{5,1}.metric)
% title('Generated Network Station 5')
% ylabel('Average Queue Length')
% xlabel('Time (s)')
% subplot(5,2,2); plot(QNt_predicted{1,1}.t, QNt_predicted{1,1}.metric)
% title('Predicted Network Station 1')
% ylabel('Average Queue Length')
% xlabel('Time (s)')
% subplot(5,2,4); plot(QNt_predicted{2,1}.t, QNt_predicted{2,1}.metric)
% title('Predicted Network Station 2')
% ylabel('Average Queue Length')
% xlabel('Time (s)')
% subplot(5,2,6); plot(QNt_predicted{3,1}.t, QNt_predicted{3,1}.metric)
% title('Predicted Network Station 3')
% ylabel('Average Queue Length')
% xlabel('Time (s)')
% subplot(5,2,8); plot(QNt_predicted{4,1}.t, QNt_predicted{4,1}.metric)
% title('Predicted Network Station 4')
% ylabel('Average Queue Length')
% xlabel('Time (s)')
% subplot(5,2,10); plot(QNt_predicted{5,1}.t, QNt_predicted{5,1}.metric)
% title('Predicted Network Station 5')
% ylabel('Average Queue Length')
% xlabel('Time (s)')

%% plot Accuracy and Mean absolute difference


QNt_predicted_interp_station1 = interp1(QNt_predicted{1,1}.t,QNt_predicted{1,1}.metric,QNt_generated{1,1}.t);
QNt_predicted_interp_station2 = interp1(QNt_predicted{2,1}.t,QNt_predicted{2,1}.metric,QNt_generated{2,1}.t);
QNt_predicted_interp_station3 = interp1(QNt_predicted{3,1}.t,QNt_predicted{3,1}.metric,QNt_generated{3,1}.t);
QNt_predicted_interp_station4 = interp1(QNt_predicted{4,1}.t,QNt_predicted{4,1}.metric,QNt_generated{4,1}.t);
QNt_predicted_interp_station5 = interp1(QNt_predicted{5,1}.t,QNt_predicted{5,1}.metric,QNt_generated{5,1}.t);

length_predict = length(QNt_predicted_interp_station1)-100;
pred_interp_1 = QNt_predicted_interp_station1(1:length_predict-100);
pred_interp_2 = QNt_predicted_interp_station2(1:length_predict-100);
pred_interp_3 = QNt_predicted_interp_station3(1:length_predict-100);
pred_interp_4 = QNt_predicted_interp_station4(1:length_predict-100);
pred_interp_5 = QNt_predicted_interp_station5(1:length_predict-100);

QNt_generated_1 = QNt_generated{1,1}.metric(1:length_predict-100);
QNt_generated_2 = QNt_generated{2,1}.metric(1:length_predict-100);
QNt_generated_3 = QNt_generated{3,1}.metric(1:length_predict-100);
QNt_generated_4 = QNt_generated{4,1}.metric(1:length_predict-100);
QNt_generated_5 = QNt_generated{5,1}.metric(1:length_predict-100);

mean_graph_values_1 = mean(abs(pred_interp_1 - QNt_generated_1))*ones(length_predict-100,1);
mean_graph_values_2 = mean(abs(pred_interp_2 - QNt_generated_2))*ones(length_predict-100,1);
mean_graph_values_3 = mean(abs(pred_interp_3 - QNt_generated_3))*ones(length_predict-100,1);
mean_graph_values_4 = mean(abs(pred_interp_4 - QNt_generated_4))*ones(length_predict-100,1);
mean_graph_values_5 = mean(abs(pred_interp_5 - QNt_generated_5))*ones(length_predict-100,1);

time_values = QNt_generated{1,1}.t(1:length_predict-100);

subplot(5,1,1);
plot(time_values, abs(pred_interp_1 - QNt_generated_1))
hold on
plot(time_values, mean_graph_values_1, 'r')
hold off
title('Absolute Difference in Average Queueing Lengths: Station 1')
ylabel('Absolute Difference')
xlabel('Time (s)')
subplot(5,1,2);
plot(time_values, abs(pred_interp_2 - QNt_generated_2))
hold on
plot(time_values, mean_graph_values_2, 'r')
hold off
title('Absolute Difference in Average Queueing Lengths: Station 2')
ylabel('Absolute Difference')
xlabel('Time (s)')
subplot(5,1,3);
plot(time_values, abs(pred_interp_3 - QNt_generated_3))
hold on
plot(time_values, mean_graph_values_3, 'r')
hold off
title('Absolute Difference in Average Queueing Lengths: Station 3')
ylabel('Absolute Difference')
xlabel('Time (s)')
subplot(5,1,4);
plot(time_values, abs(pred_interp_4 - QNt_generated_4))
hold on
plot(time_values, mean_graph_values_4, 'r')
hold off
title('Absolute Difference in Average Queueing Lengths: Station 4')
ylabel('Absolute Difference')
xlabel('Time (s)')
subplot(5,1,5);
plot(time_values, abs(pred_interp_5 - QNt_generated_5))
hold on
plot(time_values, mean_graph_values_5, 'r')
hold off
title('Absolute Difference in Average Queueing Lengths: Station 5')
ylabel('Absolute Difference')
xlabel('Time (s)')

QNt_predicted_truncated = [pred_interp_1, pred_interp_2, pred_interp_3, pred_interp_4, pred_interp_5];
QNt_generated_truncated = [QNt_generated_1, QNt_generated_2, QNt_generated_3, QNt_generated_4, QNt_generated_5];
% 
error = 100*max(sum(abs(QNt_predicted_truncated-QNt_generated_truncated),2))/2/150

%% Plot util

% subplot(5,2,1); plot(UNt_generated{1,1}.t, UNt_generated{1,1}.metric)
% title('Generated Network Station 1')
% ylabel('Utilization')
% xlabel('Time (s)')
% subplot(5,2,3); plot(UNt_generated{2,1}.t, UNt_generated{2,1}.metric)
% title('Generated Network Station 2')
% ylabel('Utilization')
% xlabel('Time (s)')
% subplot(5,2,5); plot(UNt_generated{3,1}.t, UNt_generated{3,1}.metric)
% title('Generated Network Station 3')
% ylabel('Utilization')
% xlabel('Time (s)')
% subplot(5,2,7); plot(UNt_generated{4,1}.t, UNt_generated{4,1}.metric)
% title('Generated Network Station 4')
% ylabel('Utilization')
% xlabel('Time (s)')
% subplot(5,2,9); plot(UNt_generated{5,1}.t, UNt_generated{5,1}.metric)
% title('Generated Network Station 5')
% ylabel('Utilization')
% xlabel('Time (s)')
% subplot(5,2,2); plot(UNt_predicted{1,1}.t, UNt_predicted{1,1}.metric)
% title('Predicted Network Station 1')
% ylabel('Utilization')
% xlabel('Time (s)')
% subplot(5,2,4); plot(UNt_predicted{2,1}.t, UNt_predicted{2,1}.metric)
% title('Predicted Network Station 2')
% ylabel('Utilization')
% xlabel('Time (s)')
% subplot(5,2,6); plot(UNt_predicted{3,1}.t, UNt_predicted{3,1}.metric)
% title('Predicted Network Station 3')
% ylabel('Utilization')
% xlabel('Time (s)')
% subplot(5,2,8); plot(UNt_predicted{4,1}.t, UNt_predicted{4,1}.metric)
% title('Predicted Network Station 4')
% ylabel('Utilization')
% xlabel('Time (s)')
% subplot(5,2,10); plot(UNt_predicted{5,1}.t, UNt_predicted{5,1}.metric)
% title('Predicted Network Station 5')
% ylabel('Utilization')
% xlabel('Time (s)')